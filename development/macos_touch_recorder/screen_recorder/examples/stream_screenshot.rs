use crabgrab::prelude::*;
use std::sync::{mpsc, Arc, Mutex};

#[tokio::main]
async fn main() {
    let filter = CapturableContentFilter::DISPLAYS;
    let content = CapturableContent::new(filter).await.unwrap();
    for display in content.displays() {
        record(display).await;
    }
}

async fn record(display: CapturableDisplay) {
    let (sender, receiver) = mpsc::channel();
    let sender = Arc::new(Mutex::new(sender));
    let token = match CaptureStream::test_access(false) {
        Some(token) => token,
        None => CaptureStream::request_access(false)
            .await
            .expect("Expected capture access"),
    };
    let config = CaptureConfig::with_display(display.clone(), CapturePixelFormat::Bgra8888);
    let mut stream = CaptureStream::new(token, config, move |result| {
        if let StreamEvent::Video(frame) = result.expect("Expected stream event") {
            let bitmap = frame.get_bitmap().unwrap();
            if let FrameBitmap::BgraUnorm8x4(data) = bitmap {
                let (width, height) = (data.width as u32, data.height as u32);
                let bgra_data = data.data;
                let mut rgba_buffer = image::ImageBuffer::new(width, height);
                for (x, y, pixel) in rgba_buffer.enumerate_pixels_mut() {
                    let index = (y * (width) + x) as usize;
                    let [b, g, r, a] = bgra_data[index];
                    *pixel = image::Rgba([r, g, b, a]);
                }
                sender.lock().unwrap().send(rgba_buffer).unwrap();
            }
        }
    })
    .unwrap();
    if let Ok(data) = receiver.recv() {
        stream.stop().unwrap();
        data.save("screenshot.png").unwrap();
    }
}


