//! Moonshine ONNX model implementation.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use ndarray::{Array2, Array4, ArrayView, IxDyn, ShapeError};
use ort::session::{Session, SessionInputValue};
use ort::value::Value;

use crate::error::{MoonshineError, Result};
use crate::hub::{download_model, validate_model_name};

/// Model flavor configurations.
#[derive(Debug, Clone, Copy)]
pub struct ModelFlavor {
    pub language: &'static str,
    pub token_rate: u32,
    pub num_layers: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
}

/// Returns the model flavor configuration for a given model name.
pub fn get_model_flavor(model_name: &str) -> Option<ModelFlavor> {
    // Determine architecture based on model name
    let (num_layers, num_key_value_heads, head_dim) = if model_name.contains("base") {
        (8, 8, 52)
    } else {
        // tiny variants
        (6, 8, 36)
    };

    let (language, token_rate) = match model_name {
        "tiny" => ("English", 6),
        "tiny-ar" => ("Arabic", 13),
        "tiny-zh" => ("Chinese", 13),
        "tiny-ja" => ("Japanese", 13),
        "tiny-ko" => ("Korean", 13),
        "tiny-uk" => ("Ukrainian", 8),
        "tiny-vi" => ("Vietnamese", 13),
        "base" => ("English", 6),
        "base-es" => ("Spanish", 6),
        _ => return None,
    };

    Some(ModelFlavor {
        language,
        token_rate,
        num_layers,
        num_key_value_heads,
        head_dim,
    })
}

/// Returns a map of supported languages to their model names.
pub fn get_supported_languages() -> HashMap<&'static str, Vec<&'static str>> {
    let mut languages: HashMap<&'static str, Vec<&'static str>> = HashMap::new();

    let models = [
        "tiny", "tiny-ar", "tiny-zh", "tiny-ja", "tiny-ko", "tiny-uk", "tiny-vi", "base", "base-es",
    ];

    for model in models {
        if let Some(flavor) = get_model_flavor(model) {
            languages.entry(flavor.language).or_default().push(model);
        }
    }

    languages
}

/// Moonshine ONNX model for speech-to-text transcription.
pub struct MoonshineModel {
    encoder: Session,
    decoder: Session,
    flavor: ModelFlavor,
    encoder_input_names: Vec<String>,
    decoder_input_names: Vec<String>,
}

impl MoonshineModel {
    /// Creates a new MoonshineModel by downloading from HuggingFace Hub.
    ///
    /// # Arguments
    /// * `model_name` - Model name, e.g., "moonshine/tiny" or "tiny"
    ///
    /// # Example
    /// ```no_run
    /// use moonshine_rs::MoonshineModel;
    /// let model = MoonshineModel::new("moonshine/tiny").unwrap();
    /// ```
    pub fn new(model_name: &str) -> Result<Self> {
        // Handle "moonshine/tiny" -> "tiny"
        let model_name = model_name.split('/').last().unwrap_or(model_name);
        validate_model_name(model_name)?;

        let (encoder_path, decoder_path) = download_model(model_name, "float")?;

        Self::from_paths(&encoder_path, &decoder_path, model_name)
    }

    /// Creates a MoonshineModel from local ONNX files.
    ///
    /// # Arguments
    /// * `encoder_path` - Path to encoder_model.onnx
    /// * `decoder_path` - Path to decoder_model_merged.onnx
    /// * `model_name` - Model name for configuration (e.g., "tiny", "base")
    pub fn from_paths<P: AsRef<Path>>(
        encoder_path: P,
        decoder_path: P,
        model_name: &str,
    ) -> Result<Self> {
        let encoder_path = encoder_path.as_ref();
        let decoder_path = decoder_path.as_ref();

        if !encoder_path.exists() {
            return Err(MoonshineError::ModelFileNotFound(encoder_path.to_path_buf()));
        }
        if !decoder_path.exists() {
            return Err(MoonshineError::ModelFileNotFound(decoder_path.to_path_buf()));
        }

        let flavor = get_model_flavor(model_name)
            .ok_or_else(|| MoonshineError::UnknownModel(model_name.to_string()))?;

        let encoder = Session::builder()?.commit_from_file(encoder_path)?;
        let decoder = Session::builder()?.commit_from_file(decoder_path)?;

        let encoder_input_names: Vec<String> =
            encoder.inputs().iter().map(|i| i.name().to_string()).collect();

        let decoder_input_names: Vec<String> =
            decoder.inputs().iter().map(|i| i.name().to_string()).collect();

        Ok(Self {
            encoder,
            decoder,
            flavor,
            encoder_input_names,
            decoder_input_names,
        })
    }

    /// Returns the model's language.
    pub fn language(&self) -> &'static str {
        self.flavor.language
    }

    /// Returns the model's token rate.
    pub fn token_rate(&self) -> u32 {
        self.flavor.token_rate
    }

    /// Generates tokens from audio samples.
    ///
    /// # Arguments
    /// * `audio` - Audio samples as a 2D array of shape [1, num_samples]
    /// * `max_len` - Optional maximum number of tokens to generate
    ///
    /// # Returns
    /// Vector of token IDs (including start and end tokens)
    pub fn generate(&mut self, audio: &Array2<f32>, max_len: Option<usize>) -> Result<Vec<u32>> {
        let num_samples = audio.shape()[1];
        let max_len = max_len.unwrap_or_else(|| {
            ((num_samples as f32 / 16000.0) * self.flavor.token_rate as f32) as usize
        });

        // Run encoder
        let audio_attention_mask = Array2::<i64>::ones(audio.raw_dim());

        let mut encoder_inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![(
            "input_values".into(),
            Value::from_array(audio.clone())?.into(),
        )];

        if self
            .encoder_input_names
            .contains(&"attention_mask".to_string())
        {
            encoder_inputs.push((
                "attention_mask".into(),
                Value::from_array(audio_attention_mask.clone())?.into(),
            ));
        }

        let encoder_outputs = self.encoder.run(encoder_inputs)?;
        let last_hidden_state: ArrayView<f32, IxDyn> = encoder_outputs[0].try_extract_array()?;
        let last_hidden_state = last_hidden_state.to_owned();

        // Initialize KV cache
        let mut past_key_values: HashMap<String, Array4<f32>> = HashMap::new();
        for i in 0..self.flavor.num_layers {
            for location in ["decoder", "encoder"] {
                for kv in ["key", "value"] {
                    let name = format!("past_key_values.{i}.{location}.{kv}");
                    past_key_values.insert(
                        name,
                        Array4::zeros((
                            0,
                            self.flavor.num_key_value_heads,
                            1,
                            self.flavor.head_dim,
                        )),
                    );
                }
            }
        }

        const DECODER_START_TOKEN_ID: u32 = 1;
        const EOS_TOKEN_ID: u32 = 2;

        let mut tokens = vec![DECODER_START_TOKEN_ID];
        let mut input_ids = Array2::from_shape_vec((1, 1), vec![DECODER_START_TOKEN_ID as i64])
            .map_err(|e: ShapeError| MoonshineError::Audio(format!("Shape error: {e}")))?;

        for i in 0..max_len {
            let use_cache_branch = i > 0;

            // Build decoder inputs
            let mut decoder_inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![
                (
                    "input_ids".into(),
                    Value::from_array(input_ids.clone())?.into(),
                ),
                (
                    "encoder_hidden_states".into(),
                    Value::from_array(last_hidden_state.clone())?.into(),
                ),
                (
                    "use_cache_branch".into(),
                    Value::from_array(ndarray::arr1(&[use_cache_branch]))?.into(),
                ),
            ];

            if self
                .decoder_input_names
                .contains(&"encoder_attention_mask".to_string())
            {
                decoder_inputs.push((
                    "encoder_attention_mask".into(),
                    Value::from_array(audio_attention_mask.clone())?.into(),
                ));
            }

            // Add KV cache inputs
            let mut kv_names: Vec<String> = Vec::new();
            for j in 0..self.flavor.num_layers {
                for location in ["decoder", "encoder"] {
                    for kv in ["key", "value"] {
                        let name = format!("past_key_values.{j}.{location}.{kv}");
                        let cache = past_key_values.get(&name).unwrap();
                        decoder_inputs.push((
                            Cow::Owned(name.clone()),
                            Value::from_array(cache.clone())?.into(),
                        ));
                        kv_names.push(name);
                    }
                }
            }

            let decoder_outputs = self.decoder.run(decoder_inputs)?;

            // Get logits and find next token
            let logits: ArrayView<f32, IxDyn> = decoder_outputs[0].try_extract_array()?;
            let logits_shape = logits.shape();
            let seq_len = logits_shape[1];

            // Get the last token's logits
            let vocab_size = logits_shape[2];
            let offset = (seq_len - 1) * vocab_size;
            let last_logits = &logits.as_slice().unwrap()[offset..offset + vocab_size];

            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(EOS_TOKEN_ID);

            tokens.push(next_token);

            if next_token == EOS_TOKEN_ID {
                break;
            }

            // Update input_ids for next iteration
            input_ids = Array2::from_shape_vec((1, 1), vec![next_token as i64])
                .map_err(|e: ShapeError| MoonshineError::Audio(format!("Shape error: {e}")))?;

            // Update KV cache
            for (k, name) in kv_names.iter().enumerate() {
                if !use_cache_branch || name.contains("decoder") {
                    let new_kv: ArrayView<f32, IxDyn> =
                        decoder_outputs[k + 1].try_extract_array()?;
                    let new_kv_owned = new_kv.to_owned();
                    let new_kv_4d = new_kv_owned.into_dimensionality::<ndarray::Ix4>().map_err(
                        |e| MoonshineError::Audio(format!("Dimensionality error: {e}")),
                    )?;
                    past_key_values.insert(name.clone(), new_kv_4d);
                }
            }
        }

        Ok(tokens)
    }

    /// Decodes tokens to text using the bundled tokenizer.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let tokenizer = crate::tokenizer::load_tokenizer()?;
        Ok(tokenizer.decode(tokens, true)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_model_flavor() {
        let tiny = get_model_flavor("tiny").unwrap();
        assert_eq!(tiny.language, "English");
        assert_eq!(tiny.token_rate, 6);
        assert_eq!(tiny.num_layers, 6);

        let base = get_model_flavor("base").unwrap();
        assert_eq!(base.language, "English");
        assert_eq!(base.num_layers, 8);

        let tiny_zh = get_model_flavor("tiny-zh").unwrap();
        assert_eq!(tiny_zh.language, "Chinese");
        assert_eq!(tiny_zh.token_rate, 13);

        assert!(get_model_flavor("unknown").is_none());
    }

    #[test]
    fn test_get_supported_languages() {
        let langs = get_supported_languages();
        assert!(langs.contains_key("English"));
        assert!(langs["English"].contains(&"tiny"));
        assert!(langs["English"].contains(&"base"));
    }
}
