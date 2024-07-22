// Select all images within #galleryOverlay that have a .jpg source
let images = document.querySelectorAll('#galleryOverlay img[src$=".jpg"]');

let img = images[2];
if (img) {
    // Create a link and set the href to the image source
    var link = document.createElement('a');
    link.href = img.src; // Use the actual image source
    link.download = `downloaded-image-0.jpg`; // Names files with an index
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// If you need to simulate right arrow key press for navigation, consider doing it outside the loop
let e = new KeyboardEvent("keydown", {bubbles: true, cancelable: true, key: "ArrowRight", keyCode: 39});
document.dispatchEvent(e);