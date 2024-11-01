use device_query::{DeviceQuery, DeviceState, MouseState};
use std::{thread, time::Duration};

fn main() {
    let device_state = DeviceState::new();
    let mut last_position = (0, 0);
    let mut last_button_state = vec![false; 3]; // Track left, right, middle buttons

    loop {
        let mouse: MouseState = device_state.get_mouse();
        let current_position = mouse.coords;
        let button_state = mouse.button_pressed;

        // Check for mouse movement
        if current_position != last_position {
            println!(
                "Mouse moved to ({}, {})",
                current_position.0, current_position.1
            );
            last_position = current_position;
        }

        // Check for button clicks
        for (index, &is_pressed) in button_state.iter().enumerate().take(3) {
            if is_pressed != last_button_state[index] {
                let button_name = match index {
                    0 => "Left",
                    1 => "Right",
                    2 => "Middle",
                    _ => "Unknown",
                };
                let state = if is_pressed { "pressed" } else { "released" };
                println!(
                    "Mouse {} {} at ({}, {})",
                    button_name, state, current_position.0, current_position.1
                );
            }
        }
        last_button_state = button_state;

        // Small sleep to prevent high CPU usage
        thread::sleep(Duration::from_millis(10));
    }
}
