import Cocoa
import Quartz
import sys
import signal

def create_custom_cursor(image_path, hotspot_x, hotspot_y):
    # Load the image
    image = Cocoa.NSImage.alloc().initWithContentsOfFile_(image_path)
    if not image:
        print(f"Failed to load image from {image_path}")
        return None

    # Create the cursor
    cursor = Cocoa.NSCursor.alloc().initWithImage_hotSpot_(
        image,
        Cocoa.NSPoint(hotspot_x, hotspot_y)
    )
    return cursor

def set_cursor(cursor):
    cursor.set()

def main():
    if len(sys.argv) != 4:
        print("Usage: python custom_cursor.py <image_path> <hotspot_x> <hotspot_y>")
        sys.exit(1)

    image_path = sys.argv[1]
    hotspot_x = float(sys.argv[2])
    hotspot_y = float(sys.argv[3])

    custom_cursor = create_custom_cursor(image_path, hotspot_x, hotspot_y)
    if not custom_cursor:
        print("Failed to create custom cursor.")
        sys.exit(1)

    set_cursor(custom_cursor)
    print("Custom cursor set. Move your mouse to see it. Press Ctrl+C to exit.")

    def signal_handler(sig, frame):
        print("\nExiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            Cocoa.NSRunLoop.currentRunLoop().runMode_beforeDate_(
                Cocoa.NSDefaultRunLoopMode,
                Cocoa.NSDate.dateWithTimeIntervalSinceNow_(0.1)
            )
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
