use enigo::{
    Button, Coordinate,
    Direction::{Click, Press, Release},
    Enigo, Key, Keyboard, Mouse, Settings,
};
// use screenshots::Screen;
//use winit::event_loop::EventLoop;
use std::thread;
use std::time::Duration;
use winit::window::WindowId;

struct WindowInfo {
    name: String,
    width: u32,
    height: u32,
    last_cursor_x: i32,
    last_cursor_y: i32,
    id: WindowId,
}

struct WindowCapture {
    window_name: String,
    scale_factor: f64,
    current_window_info: Option<WindowInfo>,
    enigo: Enigo,
}

impl WindowCapture {
    fn new(window_name: &str, scale_factor: f64) -> Self {
        WindowCapture {
            window_name: window_name.to_string(),
            scale_factor,
            current_window_info: None,
            enigo: Enigo::new(&Settings::default()).expect("Failed to create Enigo instance"),
        }
    }

    fn get_window_info(&mut self) -> Option<WindowInfo> {
        // use winit::event_loop::EventLoop;
        // use winit::window::WindowId;

        // let event_loop = EventLoop::new().expect("Failed to create event loop");
        // let window_list = event_loop.available_monitors();

        // for window in window_list {
        //     if self.window_name == window.name() {
        //         let size = window.size();
        //         let position = window.position();
        //         return Some(WindowInfo {
        //             name: window.name(),
        //             width: size.width,
        //             height: size.height,
        //             last_cursor_x: position.x,
        //             last_cursor_y: position.y,
        //             id: WindowId::dummy(), // You might need to adjust this based on how you're handling WindowId
        //         });
        //     }
        // }

        // println!("Unable to find window");
        None
    }

    fn move_cursor(&mut self, x: f64, y: f64) {
        let pixel_x = (x / 100.0 * 1920.0) as i32; // Assuming a default width of 1920
        let pixel_y = (y / 100.0 * 1080.0) as i32; // Assuming a default height of 1080
        self.enigo.move_mouse(pixel_x, pixel_y, Coordinate::Abs);
        println!("Cursor moved to {}, {}", pixel_x, pixel_y);
    }

    fn click(&mut self, x: Option<f64>, y: Option<f64>) {
        if let (Some(x), Some(y)) = (x, y) {
            self.move_cursor(x, y);
        }

        println!("Window name: {}", self.window_name);

        // Activate the window using AppleScript
        let script = format!(
            r#"tell application "System Events"
                set frontmost of process "{}" to true
            end tell"#,
            self.window_name
        );
        std::process::Command::new("osascript")
            .arg("-e")
            .arg(&script)
            .output()
            .expect("Failed to execute AppleScript");

        thread::sleep(Duration::from_millis(10));

        // self.enigo.button(Button::Left, Click);

        self.enigo.button(Button::Left, Press);
        thread::sleep(Duration::from_millis(200));
        self.enigo.button(Button::Left, Release);
    }

    fn double_click(&mut self, x: Option<f64>, y: Option<f64>) {
        if let (Some(x), Some(y)) = (x, y) {
            self.move_cursor(x, y);
        }
        self.enigo.button(Button::Left, Click);
        thread::sleep(Duration::from_millis(10));
        self.enigo.button(Button::Left, Release);
    }

    fn paste(&mut self) {
        self.enigo.key(Key::Control, Press);
        self.enigo.key(Key::Unicode('v'), Click);
        self.enigo.key(Key::Control, Release);
    }

    fn enter_text(&mut self, text: &str) {
        self.enigo.text(text);
    }
}

fn main() {
    let mut window_capture = WindowCapture::new("Ryujinx", 0.5);

    // window_capture.click(Some(20.0), Some(55.0));

    window_capture.click(Some(20.0), Some(55.0));
}
