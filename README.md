# moonshine-rs

Rust implementation of Moonshine speech-to-text models for fast, accurate transcription on resource-constrained devices.

## Features

- ONNX Runtime-based inference
- Automatic model downloading from HuggingFace Hub
- Live transcription with Voice Activity Detection (VAD)
- Cross-platform audio capture

## Supported Models

| Model | Language | Token Rate |
|-------|----------|------------|
| `tiny` | English | 6 |
| `base` | English | 6 |
| `tiny-ar` | Arabic | 13 |
| `tiny-zh` | Chinese | 13 |
| `tiny-ja` | Japanese | 13 |
| `tiny-ko` | Korean | 13 |
| `tiny-uk` | Ukrainian | 8 |
| `tiny-vi` | Vietnamese | 13 |
| `base-es` | Spanish | 6 |

## Quick Start

```rust
use moonshine_rs::transcribe;

let text = transcribe("audio.wav", "moonshine/tiny").unwrap();
println!("{}", text);
```

## Low-level API

```rust
use moonshine_rs::{MoonshineModel, load_audio, prepare_for_model};

let mut model = MoonshineModel::new("moonshine/base").unwrap();
let audio = load_audio("audio.wav").unwrap();
let audio = prepare_for_model(&audio);
let tokens = model.generate(&audio, None).unwrap();
let text = model.decode(&tokens).unwrap();
```

## Live Transcription

```rust
use moonshine_rs::live::{LiveTranscriber, TranscriptionEvent};

let transcriber = LiveTranscriber::builder()
    .model("moonshine/tiny")
    .vad_threshold(0.5)
    .build()
    .unwrap();

for event in transcriber.start().unwrap() {
    match event {
        TranscriptionEvent::Partial(text) => print!("\r{}", text),
        TranscriptionEvent::Final(text) => println!("\n{}", text),
        TranscriptionEvent::Error(e) => eprintln!("Error: {}", e),
        TranscriptionEvent::SpeechStart | TranscriptionEvent::SpeechEnd => {}
    }
}
```

## Examples

```bash
# Transcribe a file
cargo run --example transcribe_file -- audio.wav moonshine/tiny

# Live captions from microphone
cargo run --example live_captions -- moonshine/tiny
```

## Building

```bash
cargo build --release
```

## License

MIT
