//! Example: Transcribe an audio file.
//!
//! Usage: cargo run --example transcribe_file -- <audio.wav> [model_name]

use std::env;
use std::process::exit;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <audio.wav> [model_name]", args[0]);
        eprintln!("  model_name: moonshine/tiny (default), moonshine/base");
        exit(1);
    }

    let audio_path = &args[1];
    let model_name = args.get(2).map(|s| s.as_str()).unwrap_or("moonshine/tiny");

    println!("Loading model '{model_name}'...");
    let start = Instant::now();

    let mut model = match moonshine_rs::MoonshineModel::new(model_name) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load model: {e}");
            exit(1);
        }
    };

    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f32());

    println!("Loading audio from '{audio_path}'...");
    let audio = match moonshine_rs::load_audio(audio_path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed to load audio: {e}");
            exit(1);
        }
    };

    let duration = audio.len() as f32 / moonshine_rs::SAMPLE_RATE as f32;
    println!("Audio duration: {duration:.2}s");

    if let Err(e) = moonshine_rs::validate_audio(&audio) {
        eprintln!("Audio validation failed: {e}");
        exit(1);
    }

    println!("Transcribing...");
    let start = Instant::now();

    let audio_arr = moonshine_rs::prepare_for_model(&audio);
    let tokens = match model.generate(&audio_arr, None) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Transcription failed: {e}");
            exit(1);
        }
    };

    let text = match model.decode(&tokens) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Decoding failed: {e}");
            exit(1);
        }
    };

    let elapsed = start.elapsed().as_secs_f32();
    let rtf = duration / elapsed;

    println!("\nTranscription ({elapsed:.2}s, {rtf:.1}x realtime):");
    println!("{text}");
}
