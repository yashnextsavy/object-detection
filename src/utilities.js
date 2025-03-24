export const drawRect = (detections, ctx, skipClear = false) => {
    if (!skipClear) {
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    }
    
    // Update detections list
    const detectionsDiv = document.getElementById('detectionsList');
    if (detectionsDiv) {
        detectionsDiv.innerHTML = detections
            .map(prediction => `
                <div class="detection-item">
                    <span class="item-name">${prediction.class}</span>
                    <span class="confidence">${Math.round(prediction.score * 100)}%</span>
                </div>
            `)
            .join('');
    }

    detections.forEach((prediction) => {
        const [x, y, width, height] = prediction.bbox;
        const text = prediction.class;

        // Draw box
        ctx.strokeStyle = "#00ff00";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);

        // Draw background for text
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        ctx.fillRect(x, y > 20 ? y - 20 : y, ctx.measureText(text).width + 10, 20);

        // Draw text
        ctx.fillStyle = "#00ff00";
        ctx.font = "16px Arial";
        ctx.fillText(text, x + 5, y > 20 ? y - 5 : y + 15);
    });
};
