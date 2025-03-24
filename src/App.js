
import React, { useRef, useCallback, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as cocossd from "@tensorflow-models/coco-ssd";
import Webcam from "react-webcam";
import "./App.css";
import { drawRect } from "./utilities";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState('webcam');
  const [model, setModel] = useState(null);

  useEffect(() => {
    const setupTf = async () => {
      try {
        setIsLoading(true);
        await tf.ready();
        await tf.setBackend('webgl');
        const loadedModel = await cocossd.load();
        setModel(loadedModel);
      } catch (err) {
        setError('Failed to initialize TensorFlow.js');
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };
    setupTf();
  }, []);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const fileType = file.type.split('/')[0];
    setMode(fileType);

    if (fileType === 'image') {
      const img = new Image();
      img.src = URL.createObjectURL(file);
      img.onload = async () => {
        const canvas = canvasRef.current;
        const displayWidth = Math.min(640, img.width);
        const scale = displayWidth / img.width;
        const displayHeight = img.height * scale;
        
        canvas.width = displayWidth;
        canvas.height = displayHeight;
        const ctx = canvas.getContext('2d');
        
        // Draw the image first
        ctx.drawImage(img, 0, 0, displayWidth, displayHeight);
        
        if (model) {
          const predictions = await model.detect(img);
          // Scale the predictions to match the displayed image size
          const scaledPredictions = predictions.map(pred => ({
            ...pred,
            bbox: [
              pred.bbox[0] * scale,
              pred.bbox[1] * scale,
              pred.bbox[2] * scale,
              pred.bbox[3] * scale
            ]
          }));
          drawRect(scaledPredictions, ctx, true);
        }
      };
    } else if (fileType === 'video') {
      const video = videoRef.current;
      video.src = URL.createObjectURL(file);
      video.onloadedmetadata = () => {
        canvasRef.current.width = video.videoWidth;
        canvasRef.current.height = video.videoHeight;
      };
    }
  };

  const detectFromVideo = useCallback(async () => {
    if (!model || !videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    if (video.paused || video.ended) return;

    const predictions = await model.detect(video);
    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.drawImage(video, 0, 0);
    drawRect(predictions, ctx);
    requestAnimationFrame(detectFromVideo);
  }, [model]);

  useEffect(() => {
    if (mode === 'video') {
      videoRef.current?.play();
      detectFromVideo();
    }
  }, [mode, detectFromVideo]);

  const detectFromWebcam = useCallback(async () => {
    if (!model || !webcamRef.current || !canvasRef.current) return;

    if (webcamRef.current.video.readyState === 4) {
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      const predictions = await model.detect(video);
      const ctx = canvasRef.current.getContext('2d');
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      drawRect(predictions, ctx, true);
    }

    requestAnimationFrame(detectFromWebcam);
  }, [model]);

  useEffect(() => {
    if (mode === 'webcam' && !isLoading) {
      detectFromWebcam();
    }
  }, [mode, detectFromWebcam, isLoading]);

  return (
    <div className="App">
      <header className="App-header">
        {error && <div className="error-message">{error}</div>}
        {isLoading && <div className="loading-message">Loading TensorFlow.js...</div>}
        <h1>Object Detection</h1>
        
        <div className="controls">
          <input
            type="file"
            accept="image/*,video/*"
            onChange={handleFileUpload}
            className="file-input"
          />
          <button onClick={() => setMode('webcam')} className="mode-button">
            Use Webcam
          </button>
        </div>
        <div className="detections-box">
          <h3>Detected Items</h3>
          <div id="detectionsList"></div>
        </div>

        {mode === 'webcam' && (
          <Webcam
            ref={webcamRef}
            muted={true}
            style={{
              position: "absolute",
              marginLeft: "auto",
              marginRight: "auto",
              left: 0,
              right: 0,
              textAlign: "center",
              zIndex: 9,
              width: 640,
              height: 480,
              visibility: "hidden"
            }}
          />
        )}

        {mode === 'video' && (
          <video
            ref={videoRef}
            style={{
              position: "absolute",
              marginLeft: "auto",
              marginRight: "auto",
              left: 0,
              right: 0,
              textAlign: "center",
              zIndex: 9,
              maxWidth: '100%',
              height: 'auto',
            }}
          />
        )}

        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 8,
            maxWidth: '100%',
            height: 'auto',
          }}
        />
      </header>
    </div>
  );
}

export default App;
