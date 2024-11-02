use device_query::{DeviceEvents, DeviceState};
use std::time::Instant;
use xcap::Window;

fn normalized(filename: &str) -> String {
    filename
        .replace("|", "")
        .replace("\\", "")
        .replace(":", "")
        .replace("/", "")
}

fn take_screenshot() {
    let start = Instant::now();

    if let Ok(windows) = Window::all() {
        windows
            .into_iter()
            .filter(|w| !w.is_minimized() && w.title().contains("iPhone Mirroring"))
            .enumerate()
            .for_each(|(i, window)| {
                if let Ok(image) = window.capture_image() {
                    println!("Time elapsed: {:?}", start.elapsed());

                    // add timestamp to filename
                    let timestamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");

                    let _ = image.save(format!(
                        "screenshots/xcap/window-{}-{}-{}.png",
                        i,
                        normalized(window.title()),
                        timestamp
                    ));
                }
            });
    }
}

fn main() {
    let device_state = DeviceState::new();

    let _guard = device_state.on_mouse_down(|button| {
        if *button == 1 {
            println!("Take screenshot");
            take_screenshot();
        }
    });

    // Keep the program running
    loop {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}
