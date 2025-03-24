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
        const text = `${prediction.class} ${Math.round(prediction.score * 100)}%`;

        // Draw box with thicker border
        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);

        // Draw background for text
        ctx.font = "18px Arial";
        const textWidth = ctx.measureText(text).width;
        ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
        ctx.fillRect(x, y - 25, textWidth + 10, 25);

        // Draw text
        ctx.fillStyle = "#FFF";
        ctx.fillText(text, x + 5, y - 7);
    });
};
