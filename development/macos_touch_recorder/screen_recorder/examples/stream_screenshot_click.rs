use crabgrab::prelude::*;
use device_query::{DeviceEvents, DeviceState};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

#[tokio::main]
async fn main() {
    let filter = CapturableContentFilter::NORMAL_WINDOWS;
    let content = CapturableContent::new(filter).await.unwrap();
    
    // Find specific window (modify the filter as needed)
    let window = content.windows()
        .find(|window| {
            let app_identifier = window.application().identifier();
            window.title().len() != 0 && window.title().contains("iPhone Mirroring")
        });

    if let Some(window) = window {
        println!("Capturing window: {}", window.title());
        record(window).await;
    } else {
        println!("Failed to find target window");
    }
}

async fn record(window: CapturableWindow) {
    let (sender, receiver) = mpsc::channel();
    let frame_sender = Arc::new(Mutex::new(sender));
    let click_sender = frame_sender.clone();
    
    let device_state = DeviceState::new();
    let _mouse_guard = device_state.on_mouse_down(move |button| {
        if *button == 1 { // Left click - dereferenced the button value
            if let Ok(sender) = click_sender.lock() {
                sender.send(true).unwrap();
            }
        }
    });

    let token = match CaptureStream::test_access(true) {
        Some(token) => token,
        None => CaptureStream::request_access(true)
            .await
            .expect("Expected capture access"),
    };

    let config = CaptureConfig::with_window(window, CapturePixelFormat::Bgra8888).unwrap();
    let mut capture_requested = false;

    let mut stream = CaptureStream::new(token, config, move |result| {
        if let StreamEvent::Video(frame) = result.expect("Expected stream event") {
            if let Ok(mut sender) = frame_sender.lock() {
                if let Ok(should_capture) = receiver.try_recv() {
                    println!("Capture requested: {}", should_capture);
                    capture_requested = should_capture;
                }

                if capture_requested {
                    if let FrameBitmap::BgraUnorm8x4(data) = frame.get_bitmap().unwrap() {
                        let (width, height) = (data.width as u32, data.height as u32);
                        let bgra_data = data.data.clone();
                        
                        thread::spawn(move || {
                            let mut rgba_buffer = image::ImageBuffer::new(width, height);
                            
                            for (x, y, pixel) in rgba_buffer.enumerate_pixels_mut() {
                                let index = (y * width + x) as usize;
                                let [b, g, r, a] = bgra_data[index];
                                *pixel = image::Rgba([r, g, b, a]);
                            }
                            
                            let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
                            rgba_buffer.save(format!("screenshots/screenshot_{}.png", timestamp)).unwrap();
                        });
                        
                        capture_requested = false;
                    }
                }
            }
        }
    })
    .unwrap();

    // Keep the stream running
    tokio::time::sleep(std::time::Duration::from_secs(3600)).await; // Run for 1 hour
    stream.stop().unwrap();
}


