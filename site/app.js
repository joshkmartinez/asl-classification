const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

// Options for handtrack.js
const modelParams = {
    flipHorizontal: true,   // flip e.g for video 
    maxNumBoxes: 20,        // maximum number of boxes to detect
    iouThreshold: 0.5,      // ioU threshold for non-max suppression
    scoreThreshold: 0.6,    // confidence threshold for predictions.
}

// Load the model
handTrack.load(modelParams).then(model => {
    // Start video
    handTrack.startVideo(video).then(status => {
        if (status) {
            console.log("Video started");
            runDetection(model);
        } else {
            console.log("Please enable your webcam");
        }
    });
});

function runDetection(model) {
    model.detect(video).then(predictions => {
        model.renderPredictions(predictions, canvas, context, video);
        requestAnimationFrame(() => {
            runDetection(model);
        });
    });
}

// function runDetection(model) {
//     model.detect(video).then(predictions => {
//         // Filter out predictions that are not hands
//         const handPredictions = predictions.filter(prediction => prediction.label !== 'face');

//         // Clear previous drawings
//         context.clearRect(0, 0, canvas.width, canvas.height);

//         // Draw bounding box for each hand detection
//         handPredictions.forEach(prediction => {
//             const [x, y, width, height] = prediction.bbox;
//             // Draw a rectangle around the detected hand
//             context.strokeStyle = '#ff0000'; // Set bounding box color
//             context.lineWidth = 4; // Set bounding box line width
//             context.strokeRect(x, y, width, height);
//         });

//         // Request the next animation frame
//         requestAnimationFrame(() => {
//             runDetection(model);
//         });
//     });
// }



// Make sure the video stream is stopped when the page is unloaded
window.addEventListener("beforeunload", function() {
    handTrack.stopVideo(video);
});
