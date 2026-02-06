//! Voice Activity Detection (VAD) using Silero VAD ONNX model.
//!
//! This module provides a neural network-based VAD using the Silero VAD model
//! for accurate speech detection even in noisy environments.

use std::borrow::Cow;

use ndarray::{s, Array1, Array2, ArrayD, ArrayView, IxDyn};
use ort::session::{Session, SessionInputValue};
use ort::value::Value;

use crate::error::{MoonshineError, Result};

/// Sample rate required by Silero VAD at 16kHz.
const SAMPLE_RATE: u32 = 16000;

/// Chunk size for 16kHz audio (512 samples = 32ms).
const CHUNK_SIZE: usize = 512;

/// Context size for 16kHz audio.
const CONTEXT_SIZE: usize = 64;

/// Hidden state size for Silero VAD LSTM.
const STATE_SIZE: usize = 128;

/// VAD events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    /// No event.
    None,
    /// Speech started.
    SpeechStart,
    /// Speech ended.
    SpeechEnd,
}

/// Silero VAD ONNX model wrapper.
struct SileroModel {
    session: Session,
    state: ArrayD<f32>,
    context: Array2<f32>,
}

impl SileroModel {
    /// Creates a new Silero VAD model from an ONNX file path.
    fn from_path(path: &std::path::Path) -> Result<Self> {
        let session = Session::builder()?.commit_from_file(path)?;

        Ok(Self {
            session,
            state: ArrayD::<f32>::zeros(IxDyn(&[2, 1, STATE_SIZE])),
            context: Array2::<f32>::zeros((1, CONTEXT_SIZE)),
        })
    }

    /// Resets the internal LSTM state.
    fn reset(&mut self) {
        self.state = ArrayD::<f32>::zeros(IxDyn(&[2, 1, STATE_SIZE]));
        self.context = Array2::<f32>::zeros((1, CONTEXT_SIZE));
    }

    /// Runs inference on a single audio chunk and returns speech probability.
    fn forward(&mut self, chunk: &[f32]) -> Result<f32> {
        if chunk.len() != CHUNK_SIZE {
            return Err(MoonshineError::Audio(format!(
                "VAD chunk must be {} samples, got {}",
                CHUNK_SIZE,
                chunk.len()
            )));
        }

        // Concatenate context with input: [1, context_size + chunk_size]
        let mut input = Array2::<f32>::zeros((1, CONTEXT_SIZE + CHUNK_SIZE));
        input.slice_mut(s![.., ..CONTEXT_SIZE]).assign(&self.context);
        for (i, &sample) in chunk.iter().enumerate() {
            input[[0, CONTEXT_SIZE + i]] = sample;
        }

        // Sample rate tensor
        let sr = Array1::<i64>::from_elem(1, SAMPLE_RATE as i64);

        // Prepare inputs
        let inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![
            ("input".into(), Value::from_array(input.clone())?.into()),
            ("state".into(), Value::from_array(self.state.clone())?.into()),
            ("sr".into(), Value::from_array(sr)?.into()),
        ];

        let outputs = self.session.run(inputs)?;

        // Extract speech probability from output
        let output: ArrayView<f32, IxDyn> = outputs[0].try_extract_array()?;
        let speech_prob = output[[0, 0]];

        // Update state from stateN output
        if outputs.len() > 1 {
            let state_view: ArrayView<f32, IxDyn> = outputs[1].try_extract_array()?;
            self.state = state_view.to_owned();
        }

        // Update context with last CONTEXT_SIZE samples of input
        for i in 0..CONTEXT_SIZE {
            self.context[[0, i]] = chunk[CHUNK_SIZE - CONTEXT_SIZE + i];
        }

        Ok(speech_prob)
    }
}

/// Voice Activity Detection using Silero VAD model.
///
/// Uses neural network-based detection for accurate speech boundary detection
/// even in noisy conditions.
pub struct Vad {
    model: SileroModel,
    threshold: f32,
    neg_threshold: f32,
    triggered: bool,
    temp_end_samples: Option<usize>,
    current_sample: usize,
    min_silence_samples: usize,
}

impl Vad {
    /// Creates a new VAD instance.
    ///
    /// # Arguments
    /// * `threshold` - Speech probability threshold (0.0 to 1.0). Recommended: 0.5
    /// * `min_silence_ms` - Minimum silence duration (ms) before speech-end is confirmed
    /// * `neg_threshold_offset` - How far below `threshold` the probability must drop
    ///   to begin the silence timer (clamped so neg_threshold >= 0.01)
    pub fn new(threshold: f32, min_silence_ms: u32, neg_threshold_offset: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(MoonshineError::Audio(
                "VAD threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Download Silero VAD model from HuggingFace
        let model_path = download_silero_vad()?;
        let model = SileroModel::from_path(&model_path)?;

        Ok(Self {
            model,
            threshold,
            neg_threshold: (threshold - neg_threshold_offset).max(0.01),
            triggered: false,
            temp_end_samples: None,
            current_sample: 0,
            min_silence_samples: SAMPLE_RATE as usize * min_silence_ms as usize / 1000,
        })
    }

    /// Processes an audio chunk and returns a VAD event.
    ///
    /// The chunk should be 512 samples at 16kHz sample rate.
    pub fn process(&mut self, audio: &[f32]) -> VadEvent {
        let speech_prob = match self.model.forward(audio) {
            Ok(prob) => prob,
            Err(_) => return VadEvent::None,
        };

        let chunk_len = audio.len();
        self.current_sample += chunk_len;

        // Speech detected above threshold
        if speech_prob >= self.threshold {
            // Cancel any pending end
            if self.temp_end_samples.is_some() {
                self.temp_end_samples = None;
            }

            // Start of speech
            if !self.triggered {
                self.triggered = true;
                return VadEvent::SpeechStart;
            }
        }

        // Speech probability dropped below negative threshold while triggered
        if speech_prob < self.neg_threshold && self.triggered {
            // Mark potential end
            if self.temp_end_samples.is_none() {
                self.temp_end_samples = Some(self.current_sample);
            }

            let temp_end = self.temp_end_samples.unwrap();
            let silence_duration = self.current_sample.saturating_sub(temp_end);

            // Confirm end after minimum silence duration
            if silence_duration >= self.min_silence_samples {
                self.temp_end_samples = None;
                self.triggered = false;
                return VadEvent::SpeechEnd;
            }
        }

        VadEvent::None
    }

    /// Soft resets the VAD state without affecting the model's internal LSTM state.
    pub fn soft_reset(&mut self) {
        self.triggered = false;
        self.temp_end_samples = None;
        self.current_sample = 0;
    }

    /// Fully resets the VAD including the model's internal LSTM state.
    pub fn reset(&mut self) {
        self.model.reset();
        self.soft_reset();
    }

    /// Returns whether speech is currently detected.
    pub fn is_triggered(&self) -> bool {
        self.triggered
    }
}

/// Downloads the Silero VAD ONNX model from HuggingFace Hub.
fn download_silero_vad() -> Result<std::path::PathBuf> {
    use hf_hub::api::sync::Api;

    let api = Api::new()?;
    let repo = api.model("csukuangfj/vad".to_string());

    // Download Silero VAD v5 model
    let model_path = repo
        .get("silero_vad_v5.onnx")
        .map_err(|e| MoonshineError::Audio(format!("Failed to download Silero VAD model: {e}")))?;

    Ok(model_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_creation() {
        let vad = Vad::new(0.5, 300, 0.15).unwrap();
        assert!(!vad.is_triggered());
    }

    #[test]
    fn test_vad_invalid_threshold() {
        assert!(Vad::new(1.5, 300, 0.15).is_err());
        assert!(Vad::new(-0.1, 300, 0.15).is_err());
    }

    #[test]
    fn test_vad_silence() {
        let mut vad = Vad::new(0.5, 300, 0.15).unwrap();
        let silence = vec![0.0f32; 512];

        for _ in 0..20 {
            let event = vad.process(&silence);
            assert_eq!(event, VadEvent::None);
        }
        assert!(!vad.is_triggered());
    }

    #[test]
    fn test_vad_reset() {
        let mut vad = Vad::new(0.5, 300, 0.15).unwrap();
        let silence = vec![0.0f32; 512];

        // Process some audio
        for _ in 0..5 {
            vad.process(&silence);
        }

        // Reset and verify state is cleared
        vad.reset();
        assert!(!vad.is_triggered());
    }
}
