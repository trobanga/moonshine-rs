//! Error types for moonshine-rs.

use std::path::PathBuf;

/// Errors that can occur during Moonshine operations.
#[derive(Debug, thiserror::Error)]
pub enum MoonshineError {
    /// Model not found or unknown model name.
    #[error("Unknown model: {0}. Supported models: tiny, tiny-ar, tiny-zh, tiny-ja, tiny-ko, tiny-uk, tiny-vi, base, base-es")]
    UnknownModel(String),

    /// Model file not found at specified path.
    #[error("Model file not found: {0}")]
    ModelFileNotFound(PathBuf),

    /// ONNX Runtime error.
    #[error("ONNX runtime error: {0}")]
    Ort(#[from] ort::Error),

    /// Tokenizer error.
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] Box<dyn std::error::Error + Send + Sync>),

    /// HuggingFace Hub error.
    #[error("HuggingFace Hub error: {0}")]
    HfHub(#[from] hf_hub::api::sync::ApiError),

    /// Audio loading error.
    #[error("Audio error: {0}")]
    Audio(String),

    /// Invalid audio duration.
    #[error("Invalid audio duration: {duration:.2}s. Moonshine supports audio between 0.1s and 64s.")]
    InvalidAudioDuration { duration: f32 },

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Audio capture error (live transcription).
    #[cfg(not(target_arch = "wasm32"))]
    #[error("Audio capture error: {0}")]
    AudioCapture(String),
}

/// Result type alias for Moonshine operations.
pub type Result<T> = std::result::Result<T, MoonshineError>;
