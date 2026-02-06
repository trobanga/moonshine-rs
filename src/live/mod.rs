//! Live transcription module with audio capture and VAD.
//!
//! This module provides real-time speech-to-text transcription using
//! audio capture from the system microphone and voice activity detection.

mod capture;
mod vad;

pub use capture::AudioCapture;
pub use vad::{Vad, VadEvent};

use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;
use std::time::{Duration, Instant};

use crate::error::Result;
use crate::model::MoonshineModel;
use crate::tokenizer::load_tokenizer;

/// Sample rate for audio capture (must be 16kHz for Moonshine).
pub const SAMPLE_RATE: u32 = 16000;

/// Chunk size for VAD processing (Silero VAD requirement).
pub const CHUNK_SIZE: usize = 512;

/// Number of lookback chunks to keep before speech detection.
pub const LOOKBACK_CHUNKS: usize = 15;

/// Maximum speech duration in seconds before forced transcription.
pub const MAX_SPEECH_SECS: f32 = 15.0;

/// Minimum refresh interval for partial transcriptions.
pub const MIN_REFRESH_SECS: f32 = 0.2;

/// Events emitted during live transcription.
#[derive(Debug, Clone)]
pub enum TranscriptionEvent {
    /// Partial transcription result (may change).
    Partial(String),
    /// Final transcription result.
    Final(String),
    /// An error occurred.
    Error(String),
    /// Speech started.
    SpeechStart,
    /// Speech ended.
    SpeechEnd,
}

/// Default minimum silence duration (ms) before speech-end is confirmed.
pub const MIN_SILENCE_MS: u32 = 300;

/// Default offset below the VAD threshold for the negative (speech-end) trigger.
pub const NEG_THRESHOLD_OFFSET: f32 = 0.15;

/// Builder for configuring LiveTranscriber.
pub struct LiveTranscriberBuilder {
    model_name: String,
    vad_threshold: f32,
    max_speech_secs: f32,
    min_refresh_secs: f32,
    lookback_chunks: usize,
    min_silence_ms: u32,
    neg_threshold_offset: f32,
}

impl Default for LiveTranscriberBuilder {
    fn default() -> Self {
        Self {
            model_name: "moonshine/tiny".to_string(),
            vad_threshold: 0.5,
            max_speech_secs: MAX_SPEECH_SECS,
            min_refresh_secs: MIN_REFRESH_SECS,
            lookback_chunks: LOOKBACK_CHUNKS,
            min_silence_ms: MIN_SILENCE_MS,
            neg_threshold_offset: NEG_THRESHOLD_OFFSET,
        }
    }
}

impl LiveTranscriberBuilder {
    /// Sets the model name.
    pub fn model(mut self, name: &str) -> Self {
        self.model_name = name.to_string();
        self
    }

    /// Sets the VAD threshold (0.0 to 1.0).
    pub fn vad_threshold(mut self, threshold: f32) -> Self {
        self.vad_threshold = threshold;
        self
    }

    /// Sets the maximum speech duration before forced transcription.
    pub fn max_speech_secs(mut self, secs: f32) -> Self {
        self.max_speech_secs = secs;
        self
    }

    /// Sets the minimum refresh interval for partial transcriptions.
    pub fn min_refresh_secs(mut self, secs: f32) -> Self {
        self.min_refresh_secs = secs;
        self
    }

    /// Sets the number of audio chunks kept before speech onset.
    pub fn lookback_chunks(mut self, chunks: usize) -> Self {
        self.lookback_chunks = chunks;
        self
    }

    /// Sets the minimum silence duration (ms) before speech-end is confirmed.
    pub fn min_silence_ms(mut self, ms: u32) -> Self {
        self.min_silence_ms = ms;
        self
    }

    /// Sets the offset below the VAD threshold for the negative trigger.
    pub fn neg_threshold_offset(mut self, offset: f32) -> Self {
        self.neg_threshold_offset = offset;
        self
    }

    /// Builds the LiveTranscriber.
    pub fn build(self) -> Result<LiveTranscriber> {
        LiveTranscriber::new(
            &self.model_name,
            self.vad_threshold,
            self.max_speech_secs,
            self.min_refresh_secs,
            self.lookback_chunks,
            self.min_silence_ms,
            self.neg_threshold_offset,
        )
    }
}

/// Live transcriber for real-time speech-to-text.
pub struct LiveTranscriber {
    model: MoonshineModel,
    tokenizer: tokenizers::Tokenizer,
    vad_threshold: f32,
    max_speech_secs: f32,
    min_refresh_secs: f32,
    lookback_chunks: usize,
    min_silence_ms: u32,
    neg_threshold_offset: f32,
}

impl LiveTranscriber {
    /// Creates a builder for configuring the transcriber.
    pub fn builder() -> LiveTranscriberBuilder {
        LiveTranscriberBuilder::default()
    }

    /// Creates a new LiveTranscriber with the given configuration.
    fn new(
        model_name: &str,
        vad_threshold: f32,
        max_speech_secs: f32,
        min_refresh_secs: f32,
        lookback_chunks: usize,
        min_silence_ms: u32,
        neg_threshold_offset: f32,
    ) -> Result<Self> {
        let mut model = MoonshineModel::new(model_name)?;
        let tokenizer = load_tokenizer()?;

        // Warmup the model with 1 second of silence
        let warmup_audio = ndarray::Array2::zeros((1, SAMPLE_RATE as usize));
        let _ = model.generate(&warmup_audio, None);

        Ok(Self {
            model,
            tokenizer,
            vad_threshold,
            max_speech_secs,
            min_refresh_secs,
            lookback_chunks,
            min_silence_ms,
            neg_threshold_offset,
        })
    }

    /// Transcribes audio samples.
    fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        let audio = crate::audio::prepare_for_model(audio);
        let tokens = self.model.generate(&audio, None)?;
        Ok(self.tokenizer.decode(&tokens, true)?)
    }

    /// Starts live transcription and returns a receiver for events.
    ///
    /// This spawns a background thread that captures audio and emits
    /// transcription events.
    pub fn start(self) -> Result<Receiver<TranscriptionEvent>> {
        let (tx, rx) = channel();

        thread::spawn(move || {
            if let Err(e) = self.run_transcription_loop(tx.clone()) {
                let _ = tx.send(TranscriptionEvent::Error(e.to_string()));
            }
        });

        Ok(rx)
    }

    fn run_transcription_loop(mut self, tx: Sender<TranscriptionEvent>) -> Result<()> {
        let mut capture = AudioCapture::new(SAMPLE_RATE, CHUNK_SIZE)?;
        let mut vad = Vad::new(self.vad_threshold, self.min_silence_ms, self.neg_threshold_offset)?;

        let lookback_size = self.lookback_chunks * CHUNK_SIZE;
        let mut speech: Vec<f32> = Vec::new();
        let mut recording = false;
        let mut start_time = Instant::now();

        capture.start()?;

        loop {
            let chunk = match capture.read_chunk() {
                Some(c) => c,
                None => {
                    thread::sleep(Duration::from_millis(10));
                    continue;
                }
            };

            speech.extend_from_slice(&chunk);

            // Keep only lookback buffer when not recording
            if !recording && speech.len() > lookback_size {
                let drain_len = speech.len() - lookback_size;
                speech.drain(..drain_len);
            }

            // Run VAD on chunk
            match vad.process(&chunk) {
                VadEvent::SpeechStart => {
                    if !recording {
                        recording = true;
                        start_time = Instant::now();
                        let _ = tx.send(TranscriptionEvent::SpeechStart);
                    }
                }
                VadEvent::SpeechEnd => {
                    if recording {
                        recording = false;
                        let text = self.transcribe(&speech).unwrap_or_default();
                        let _ = tx.send(TranscriptionEvent::Final(text));
                        let _ = tx.send(TranscriptionEvent::SpeechEnd);
                        speech.clear();
                    }
                }
                VadEvent::None => {
                    if recording {
                        let speech_duration = speech.len() as f32 / SAMPLE_RATE as f32;

                        // Check for max duration
                        if speech_duration > self.max_speech_secs {
                            recording = false;
                            let text = self.transcribe(&speech).unwrap_or_default();
                            let _ = tx.send(TranscriptionEvent::Final(text));
                            let _ = tx.send(TranscriptionEvent::SpeechEnd);
                            speech.clear();
                            vad.soft_reset();
                            continue;
                        }

                        // Emit partial transcription periodically
                        if start_time.elapsed().as_secs_f32() > self.min_refresh_secs {
                            if let Ok(text) = self.transcribe(&speech) {
                                let _ = tx.send(TranscriptionEvent::Partial(text));
                            }
                            start_time = Instant::now();
                        }
                    }
                }
            }
        }
    }
}
