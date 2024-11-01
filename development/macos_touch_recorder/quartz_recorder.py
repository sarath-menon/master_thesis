#%%
from Quartz import *
import time

def callback(proxy, event_type, event, refcon):
    # Get touch event details
    if event_type == 29: # NSEventTypeGesture
        phase = "Touch Begin"
    elif event_type == 30: # NSEventTypeMagnify
        phase = "Touch Moved"
    elif event_type == 31: # NSEventTypeSwipe
        phase = "Touch End"
    else:
        return None

    # Get touch location
    point = CGEventGetLocation(event)
    print(f"{phase} at x: {point.x}, y: {point.y}")
    
    return event

def main():
    # Create event tap for touch events
    mask = (
        CGEventMaskBit(29) |  # NSEventTypeGesture
        CGEventMaskBit(30) |  # NSEventTypeMagnify
        CGEventMaskBit(31)    # NSEventTypeSwipe
    )
    
    tap = CGEventTapCreate(
        kCGSessionEventTap,
        kCGHeadInsertEventTap,
        kCGEventTapOptionDefault,
        mask,
        callback,
        None
    )
    
    if tap is None:
        print("Failed to create event tap. Try running with sudo.")
        return

    # Create a run loop source and add it to the current run loop
    run_loop_source = CFMachPortCreateRunLoopSource(None, tap, 0)
    CFRunLoopAddSource(CFRunLoopGetCurrent(), run_loop_source, kCFRunLoopDefaultMode)
    
    # Enable the event tap
    CGEventTapEnable(tap, True)
    
    # Start the run loop
    CFRunLoopRun()

#%%
if __name__ == "__main__":
    main()

# %%
