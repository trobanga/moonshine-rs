//! HuggingFace Hub model downloading.

use std::path::PathBuf;

use hf_hub::api::sync::Api;

use crate::error::{MoonshineError, Result};

const REPO_ID: &str = "UsefulSensors/moonshine";

/// Downloads the encoder and decoder ONNX models from HuggingFace Hub.
///
/// Returns paths to the cached model files: (encoder_path, decoder_path).
pub fn download_model(model_name: &str, precision: &str) -> Result<(PathBuf, PathBuf)> {
    let api = Api::new()?;
    let repo = api.model(REPO_ID.to_string());

    let subfolder = format!("onnx/merged/{model_name}/{precision}");

    let encoder_path = repo.get(&format!("{subfolder}/encoder_model.onnx"))?;
    let decoder_path = repo.get(&format!("{subfolder}/decoder_model_merged.onnx"))?;

    Ok((encoder_path, decoder_path))
}

/// Checks if a model name is valid.
pub fn is_valid_model(model_name: &str) -> bool {
    matches!(
        model_name,
        "tiny" | "tiny-ar" | "tiny-zh" | "tiny-ja" | "tiny-ko" | "tiny-uk" | "tiny-vi" | "base" | "base-es"
    )
}

/// Validates a model name and returns an error if invalid.
pub fn validate_model_name(model_name: &str) -> Result<()> {
    if is_valid_model(model_name) {
        Ok(())
    } else {
        Err(MoonshineError::UnknownModel(model_name.to_string()))
    }
}
