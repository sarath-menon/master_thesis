
// Function to create a blocking delay
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

for (let i = 0; i < 4; i++) {
    
// Select all images within #galleryOverlay that have a .jpg source
let images = document.querySelectorAll('img[src$=".jpg"]');

for (let img of images) {
    if (img.naturalWidth === 1920) {
        console.log(img.src);

        // Print the resolution of the image
        console.log(`Resolution: ${img.naturalWidth} x ${img.naturalHeight}`);

        var link = document.createElement('a');
        link.href = img.src;

        // Extract the filename from the image source
        var filename = img.src.split('/').pop(); // Get the last segment after the last slash
        link.download = filename; // Use the actual image filename

        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
}

// If you need to simulate right arrow key press for navigation, consider doing it outside the loop
let e = new KeyboardEvent("keydown", {bubbles: true, cancelable: true, key: "ArrowRight", keyCode: 39});
document.dispatchEvent(e);

// Add a delay of 1 second
setTimeout(function() {}, 1000);

await delay(1000);
}