//! Moonshine speech-to-text inference library for Rust.
//!
//! This crate provides Rust bindings for Moonshine ASR models, optimized for
//! fast, accurate transcription on resource-constrained devices.
//!
//! # Quick Start
//!
//! ```no_run
//! use moonshine_rs::{transcribe, load_audio};
//!
//! // High-level API
//! let text = transcribe("audio.wav", "moonshine/tiny").unwrap();
//! println!("{}", text);
//! ```
//!
//! # Low-level API
//!
//! ```no_run
//! use moonshine_rs::{MoonshineModel, load_audio, prepare_for_model};
//!
//! let mut model = MoonshineModel::new("moonshine/base").unwrap();
//! let audio = load_audio("audio.wav").unwrap();
//! let audio = prepare_for_model(&audio);
//! let tokens = model.generate(&audio, None).unwrap();
//! let text = model.decode(&tokens).unwrap();
//! ```
//!
//! # Live Transcription
//!
//! ```no_run
//! use moonshine_rs::live::{LiveTranscriber, TranscriptionEvent};
//!
//! let transcriber = LiveTranscriber::builder()
//!     .model("moonshine/tiny")
//!     .vad_threshold(0.5)
//!     .build()
//!     .unwrap();
//!
//! for event in transcriber.start().unwrap() {
//!     match event {
//!         TranscriptionEvent::Partial(text) => print!("\r{}", text),
//!         TranscriptionEvent::Final(text) => println!("\n{}", text),
//!         TranscriptionEvent::Error(e) => eprintln!("Error: {}", e),
//!         TranscriptionEvent::SpeechStart | TranscriptionEvent::SpeechEnd => {}
//!     }
//! }
//! ```

pub mod audio;
pub mod error;
pub mod hub;
pub mod model;
pub mod tokenizer;

#[cfg(all(feature = "live", not(target_arch = "wasm32")))]
pub mod live;

pub use audio::{load_audio, prepare_for_model, validate_audio, SAMPLE_RATE};
pub use error::{MoonshineError, Result};
pub use model::get_supported_languages;
pub use model::{get_model_flavor, MoonshineModel, ModelFlavor};
pub use tokenizer::load_tokenizer;

/// Transcribes audio from a file path.
///
/// This is a high-level convenience function that loads the audio, runs the model,
/// and returns the transcribed text.
///
/// # Arguments
/// * `audio_path` - Path to a WAV audio file
/// * `model_name` - Model name (e.g., "moonshine/tiny", "moonshine/base")
///
/// # Example
/// ```no_run
/// let text = moonshine_rs::transcribe("audio.wav", "moonshine/tiny").unwrap();
/// ```
pub fn transcribe<P: AsRef<std::path::Path>>(audio_path: P, model_name: &str) -> Result<String> {
    let mut model = MoonshineModel::new(model_name)?;
    let audio = load_audio(audio_path)?;
    validate_audio(&audio)?;
    let audio = prepare_for_model(&audio);
    let tokens = model.generate(&audio, None)?;
    model.decode(&tokens)
}

/// Transcribes audio samples directly.
///
/// # Arguments
/// * `audio` - Audio samples at 16kHz sample rate
/// * `model` - A pre-loaded MoonshineModel
///
/// # Example
/// ```no_run
/// use moonshine_rs::{MoonshineModel, load_audio, transcribe_audio};
///
/// let mut model = MoonshineModel::new("moonshine/tiny").unwrap();
/// let audio = load_audio("audio.wav").unwrap();
/// let text = transcribe_audio(&audio, &mut model).unwrap();
/// ```
pub fn transcribe_audio(audio: &[f32], model: &mut MoonshineModel) -> Result<String> {
    validate_audio(audio)?;
    let audio = prepare_for_model(audio);
    let tokens = model.generate(&audio, None)?;
    model.decode(&tokens)
}
