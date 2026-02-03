//! Audio capture using cpal.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, Stream, StreamConfig};

use crate::error::{MoonshineError, Result};

/// Audio capture from the system's default input device.
pub struct AudioCapture {
    sample_rate: u32,
    chunk_size: usize,
    stream: Option<Stream>,
    rx: Option<Receiver<Vec<f32>>>,
    buffer: Arc<Mutex<Vec<f32>>>,
}

impl AudioCapture {
    /// Creates a new AudioCapture instance.
    pub fn new(sample_rate: u32, chunk_size: usize) -> Result<Self> {
        Ok(Self {
            sample_rate,
            chunk_size,
            stream: None,
            rx: None,
            buffer: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Starts audio capture.
    pub fn start(&mut self) -> Result<()> {
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or_else(|| {
            MoonshineError::AudioCapture("No input device available".to_string())
        })?;

        let supported_configs = device.supported_input_configs().map_err(|e| {
            MoonshineError::AudioCapture(format!("Failed to get input configs: {e}"))
        })?;

        // Find a config that matches our requirements
        let config = supported_configs
            .filter(|c| c.channels() == 1 || c.channels() == 2)
            .filter(|c| {
                c.min_sample_rate().0 <= self.sample_rate
                    && c.max_sample_rate().0 >= self.sample_rate
            })
            .next()
            .ok_or_else(|| {
                MoonshineError::AudioCapture(
                    "No suitable input configuration found".to_string(),
                )
            })?
            .with_sample_rate(cpal::SampleRate(self.sample_rate));

        let sample_format = config.sample_format();
        let stream_config: StreamConfig = config.into();
        let channels = stream_config.channels as usize;

        let (tx, rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = channel();
        let chunk_size = self.chunk_size;
        let buffer = self.buffer.clone();

        let stream = match sample_format {
            SampleFormat::F32 => self.build_stream::<f32>(&device, &stream_config, tx, channels, chunk_size, buffer)?,
            SampleFormat::I16 => self.build_stream::<i16>(&device, &stream_config, tx, channels, chunk_size, buffer)?,
            SampleFormat::U16 => self.build_stream::<u16>(&device, &stream_config, tx, channels, chunk_size, buffer)?,
            SampleFormat::I8 => self.build_stream::<i8>(&device, &stream_config, tx, channels, chunk_size, buffer)?,
            SampleFormat::U8 => self.build_stream::<u8>(&device, &stream_config, tx, channels, chunk_size, buffer)?,
            _ => {
                return Err(MoonshineError::AudioCapture(format!(
                    "Unsupported sample format: {:?}",
                    sample_format
                )))
            }
        };

        stream.play().map_err(|e| {
            MoonshineError::AudioCapture(format!("Failed to start stream: {e}"))
        })?;

        self.stream = Some(stream);
        self.rx = Some(rx);

        Ok(())
    }

    fn build_stream<T>(
        &self,
        device: &cpal::Device,
        config: &StreamConfig,
        tx: Sender<Vec<f32>>,
        channels: usize,
        chunk_size: usize,
        buffer: Arc<Mutex<Vec<f32>>>,
    ) -> Result<Stream>
    where
        T: cpal::Sample + cpal::SizedSample + 'static,
        f32: cpal::FromSample<T>,
    {
        let stream = device
            .build_input_stream(
                config,
                move |data: &[T], _: &cpal::InputCallbackInfo| {
                    let samples: Vec<f32> = if channels == 1 {
                        data.iter().map(|&s| f32::from_sample(s)).collect()
                    } else {
                        // Convert to mono by averaging channels
                        data.chunks(channels)
                            .map(|chunk| {
                                chunk.iter().map(|&s| f32::from_sample(s)).sum::<f32>()
                                    / channels as f32
                            })
                            .collect()
                    };

                    let mut buf = buffer.lock().unwrap();
                    buf.extend_from_slice(&samples);

                    // Send complete chunks
                    while buf.len() >= chunk_size {
                        let chunk: Vec<f32> = buf.drain(..chunk_size).collect();
                        let _ = tx.send(chunk);
                    }
                },
                |err| {
                    eprintln!("Audio capture error: {err}");
                },
                None,
            )
            .map_err(|e| {
                MoonshineError::AudioCapture(format!("Failed to build stream: {e}"))
            })?;

        Ok(stream)
    }

    /// Reads a chunk of audio samples.
    ///
    /// Returns `None` if no chunk is available yet.
    pub fn read_chunk(&self) -> Option<Vec<f32>> {
        self.rx.as_ref()?.try_recv().ok()
    }

    /// Stops audio capture.
    pub fn stop(&mut self) {
        self.stream = None;
        self.rx = None;
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        self.stop();
    }
}
