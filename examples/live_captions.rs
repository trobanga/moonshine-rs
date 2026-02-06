//! Example: Live captions from microphone.
//!
//! Usage: cargo run --example live_captions -- [model_name]

use std::env;
use std::io::{Write, stdout};
use std::process::exit;

use moonshine_rs::live::{LiveTranscriber, TranscriptionEvent};

const MAX_LINE_LENGTH: usize = 80;

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_name = args.get(1).map(|s| s.as_str()).unwrap_or("moonshine/base");

    println!("Loading Moonshine model '{model_name}' (using ONNX runtime)...");

    let transcriber = match LiveTranscriber::builder()
        .model(model_name)
        .vad_threshold(0.5)
        .build()
    {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to create transcriber: {e}");
            exit(1);
        }
    };

    println!("Press Ctrl+C to quit live captions.\n");

    let events = match transcriber.start() {
        Ok(rx) => rx,
        Err(e) => {
            eprintln!("Failed to start transcription: {e}");
            exit(1);
        }
    };

    let mut caption_cache: Vec<String> = Vec::new();

    print_captions("Ready...", &caption_cache);

    for event in events {
        match event {
            TranscriptionEvent::Partial(text) => {
                print_captions(&text, &caption_cache);
            }
            TranscriptionEvent::Final(text) => {
                print_captions(&text, &caption_cache);
                caption_cache.push(text);
            }
            TranscriptionEvent::SpeechStart => {
                // Optional: visual indicator that speech was detected
            }
            TranscriptionEvent::SpeechEnd => {
                println!(); // New line after final transcription
            }
            TranscriptionEvent::Error(e) => {
                eprintln!("\nError: {e}");
            }
        }
    }
}

fn print_captions(text: &str, cache: &[String]) {
    let mut display_text = text.to_string();

    // Prepend cached captions if there's room
    if display_text.len() < MAX_LINE_LENGTH {
        for caption in cache.iter().rev() {
            let combined = format!("{} {}", caption, display_text);
            if combined.len() > MAX_LINE_LENGTH {
                break;
            }
            display_text = combined;
        }
    }

    // Truncate or pad to MAX_LINE_LENGTH
    if display_text.len() > MAX_LINE_LENGTH {
        display_text = display_text[display_text.len() - MAX_LINE_LENGTH..].to_string();
    } else {
        display_text = format!("{:>width$}", display_text, width = MAX_LINE_LENGTH);
    }

    // Clear line and print
    print!("\r{}\r{}", " ".repeat(MAX_LINE_LENGTH), display_text);
    stdout().flush().unwrap();
}
