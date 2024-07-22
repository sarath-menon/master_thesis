
    // Function to create a blocking delay
    function delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    let download_count = 40;
    let cache_count = 10;
    let prev_filenames = [];

    for (let i = 0; i < download_count; i++) {
        console.log("Iteration " + i);
        // Select all images within #galleryOverlay that have a .jpg source
        let images = document.querySelectorAll('img[src$=".jpg"]');

        // clear cache periodically
        if (i % cache_count === 0) {
            prev_filenames = [];
            console.log("Cleared cache");
        }
        
        for (let img of images) {
            if (img.naturalWidth === 1920) {
                // console.log(img.src);

                // // Print the resolution of the image
                // console.log(`Resolution: ${img.naturalWidth} x ${img.naturalHeight}`);

                var link = document.createElement('a');
                link.href = img.src;

                // Extract the filename from the image source
                let filename = img.src.split('/').pop(); // Get the last segment after the last slash

                
                if (prev_filenames.includes(filename)) {
                    // console.log("Same filename as last time");
                    break;
                }

                console.log("Current filename: " + filename);

                link.download = filename; // Use the actual image filename
                prev_filenames.push(filename);

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