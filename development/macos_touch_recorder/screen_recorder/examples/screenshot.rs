use crabgrab::{feature::bitmap::VideoFrameBitmap as _, prelude::*};
use futures::executor::block_on;
use image::{ImageBuffer, Rgba};


fn main() { 
    block_on(async {
        let token = match CaptureStream::test_access(false) {
            Some(token) => token,
            None => CaptureStream::request_access(false).await.expect("Expected capture access")
        };
        let filter = CapturableContentFilter::NORMAL_WINDOWS;
        let content = CapturableContent::new(filter).await.unwrap();

        println!("Found {} windows", content.windows().count());

        for window in content.windows() {
            println!("{}", window.title());
        }

        let window = content.windows().filter(|window| {
            let app_identifier = window.application().identifier();
            // println!("{}", app_identifier);
            window.title().len() != 0 && window.title().contains("iPhone Mirroring")
        }).next();

        match window {
            Some(window) => {
                println!("screenshotting window: {}", window.title()); 
                let config = CaptureConfig::with_window(window, CaptureStream::supported_pixel_formats()[0]).unwrap();

                match crabgrab::feature::screenshot::take_screenshot(token, config).await {
                    Ok(frame) => { 
                        println!("Got frame: {}", frame.frame_id());
                        println!("Frame size: {:?}", frame.size());
                        
                        let bitmap = frame.get_bitmap().unwrap();
                        if let FrameBitmap::BgraUnorm8x4(data) = bitmap {
                            let (width, height) = (data.width as u32, data.height as u32);
                            let bgra_data = data.data;
                            let mut rgba_buffer = ImageBuffer::new(width, height);
                            for (x, y, pixel) in rgba_buffer.enumerate_pixels_mut() {
                                let index = (y * width + x) as usize;
                                let [b, g, r, a] = bgra_data[index];
                                *pixel = Rgba([r, g, b, a]);
                            }
                            rgba_buffer.save("screenshot_1.png").unwrap();
                            println!("Saved screenshot as screenshot_1.png");
                        }

                        else {
                            println!("Failed to get bitmap");
                        }
                    }
                    Err(_) => println!("screenshot failed!"),
                }
            }
            None => println!("Failed to find window"),
        }
    });
}