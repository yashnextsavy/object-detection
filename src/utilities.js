export const drawRect = (detections, ctx) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    detections.forEach((prediction) => {
        const [x, y, width, height] = prediction.bbox;
        const text = prediction.class;

        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);

        ctx.fillStyle = "red";
        ctx.font = "18px Arial";
        ctx.fillText(text, x, y > 10 ? y - 5 : 10);
    });
};
