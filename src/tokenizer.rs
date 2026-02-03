//! Tokenizer wrapper for decoding model outputs.

use tokenizers::Tokenizer;

use crate::error::Result;

/// Bundled tokenizer JSON data.
const TOKENIZER_JSON: &str = include_str!("assets/tokenizer.json");

/// Loads the bundled tokenizer.
pub fn load_tokenizer() -> Result<Tokenizer> {
    let tokenizer = Tokenizer::from_bytes(TOKENIZER_JSON.as_bytes())?;
    Ok(tokenizer)
}

/// Decodes a batch of token sequences into text.
pub fn decode_batch(tokenizer: &Tokenizer, tokens: &[Vec<u32>]) -> Result<Vec<String>> {
    tokens
        .iter()
        .map(|seq| {
            tokenizer
                .decode(seq, true)
                .map_err(|e| e.into())
        })
        .collect()
}
