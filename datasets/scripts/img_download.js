// Select all images within #galleryOverlay that have a .jpg source
let images = document.querySelectorAll('#galleryOverlay img[src$=".jpg"]');

let img = images[2];
if (img) {
    var link = document.createElement('a');
    link.href = img.src;

    // Extract the filename from the image source
    var filename = img.src.split('/').pop(); // Get the last segment after the last slash
    link.download = filename; // Use the actual image filename

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// If you need to simulate right arrow key press for navigation, consider doing it outside the loop
let e = new KeyboardEvent("keydown", {bubbles: true, cancelable: true, key: "ArrowRight", keyCode: 39});
document.dispatchEvent(e);