//! Audio loading and resampling utilities.

use std::path::Path;

use crate::error::{MoonshineError, Result};

/// Target sample rate for Moonshine models.
pub const SAMPLE_RATE: u32 = 16000;

/// Minimum audio duration in seconds.
pub const MIN_DURATION_SECS: f32 = 0.1;

/// Maximum audio duration in seconds.
pub const MAX_DURATION_SECS: f32 = 64.0;

/// Loads audio from a WAV file and returns samples as f32.
///
/// The audio is resampled to 16kHz if necessary and converted to mono.
pub fn load_audio<P: AsRef<Path>>(path: P) -> Result<Vec<f32>> {
    let path = path.as_ref();
    let reader = hound::WavReader::open(path)
        .map_err(|e| MoonshineError::Audio(format!("Failed to open WAV file: {e}")))?;

    let spec = reader.spec();
    let channels = spec.channels as usize;
    let source_rate = spec.sample_rate;

    // Read samples based on format
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| MoonshineError::Audio(format!("Failed to read samples: {e}")))?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1i32 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| MoonshineError::Audio(format!("Failed to read samples: {e}")))?
                .into_iter()
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Convert to mono by averaging channels
    let mono: Vec<f32> = if channels > 1 {
        samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    // Resample to 16kHz if necessary
    let resampled = if source_rate != SAMPLE_RATE {
        resample(&mono, source_rate, SAMPLE_RATE)?
    } else {
        mono
    };

    Ok(resampled)
}

/// Resamples audio from one sample rate to another.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    use rubato::Resampler;

    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    let params = rubato::FftFixedInOut::<f32>::new(from_rate as usize, to_rate as usize, 1024, 1)
        .map_err(|e| MoonshineError::Audio(format!("Failed to create resampler: {e}")))?;

    let mut resampler = params;
    let chunk_size = resampler.input_frames_max();

    let mut output = Vec::new();
    let mut input_pos = 0;

    while input_pos < samples.len() {
        let end = (input_pos + chunk_size).min(samples.len());
        let mut chunk = samples[input_pos..end].to_vec();

        // Pad last chunk if necessary
        if chunk.len() < chunk_size {
            chunk.resize(chunk_size, 0.0);
        }

        let result = resampler
            .process(&[chunk], None)
            .map_err(|e| MoonshineError::Audio(format!("Resampling failed: {e}")))?;

        output.extend_from_slice(&result[0]);
        input_pos += chunk_size;
    }

    // Trim to expected length
    let expected_len = (samples.len() as f64 * to_rate as f64 / from_rate as f64) as usize;
    output.truncate(expected_len);

    Ok(output)
}

/// Validates audio duration and returns the duration in seconds.
pub fn validate_audio(samples: &[f32]) -> Result<f32> {
    let duration = samples.len() as f32 / SAMPLE_RATE as f32;

    if duration < MIN_DURATION_SECS || duration > MAX_DURATION_SECS {
        return Err(MoonshineError::InvalidAudioDuration { duration });
    }

    Ok(duration)
}

/// Converts audio samples to the shape expected by the model: [1, num_samples].
pub fn prepare_for_model(samples: &[f32]) -> ndarray::Array2<f32> {
    ndarray::Array2::from_shape_vec((1, samples.len()), samples.to_vec())
        .expect("Shape should always be valid")
}
