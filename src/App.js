
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
  const [webcamError, setWebcamError] = useState(null);

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
      if (videoRef.current) {
        const video = videoRef.current;
        const videoURL = URL.createObjectURL(file);
        video.src = videoURL;
        video.onloadedmetadata = () => {
          canvasRef.current.width = video.videoWidth;
          canvasRef.current.height = video.videoHeight;
        };
        video.onloadeddata = () => {
          detectFromVideo();
        };
      }
    }
  };

  const detectFromVideo = useCallback(async () => {
    if (!model || !videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    if (video.paused || video.ended || video.readyState < 2) return;

    try {
      // Make sure video is playing and ready
      if (video.readyState === video.HAVE_ENOUGH_DATA) {
        // Set canvas dimensions to match video
        const videoWidth = video.videoWidth || 640;
        const videoHeight = video.videoHeight || 480;
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;

        const ctx = canvasRef.current.getContext('2d');
        ctx.clearRect(0, 0, videoWidth, videoHeight);

        // Draw video frame
        ctx.drawImage(video, 0, 0, videoWidth, videoHeight);

        // Detect objects with lower threshold for better detection
        const predictions = await model.detect(video, undefined, 0.3);

        // Draw bounding boxes
        ctx.lineWidth = 4;
        predictions.forEach(prediction => {
          const [x, y, width, height] = prediction.bbox;
          ctx.strokeStyle = "#00FF00";
          ctx.strokeRect(x, y, width, height);

          // Draw label
          const text = `${prediction.class} ${Math.round(prediction.score * 100)}%`;
          ctx.font = '18px Arial';
          ctx.fillStyle = "#00FF00";
          ctx.fillText(text, x, y > 20 ? y - 5 : y + 20);
        });
      }
    } catch (err) {
      console.error('Detection error:', err);
    }
    requestAnimationFrame(detectFromVideo);
  }, [model]);

  useEffect(() => {
    let animationFrame: number;
    if (mode === 'video' && videoRef.current) {
      videoRef.current.addEventListener('loadeddata', () => {
        if (videoRef.current && canvasRef.current) {
          canvasRef.current.width = videoRef.current.videoWidth;
          canvasRef.current.height = videoRef.current.videoHeight;
          videoRef.current.play();
          animationFrame = requestAnimationFrame(detectFromVideo);
        }
      });
    }
    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
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

      const predictions = await model.detect(video, undefined, 0.5);
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, videoWidth, videoHeight);
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      drawRect(predictions, ctx, true);
    }

    setTimeout(() => requestAnimationFrame(detectFromWebcam), 10);
  }, [model]);

  useEffect(() => {
    if (mode === 'webcam' && !isLoading) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(() => {
          setWebcamError(null);
          detectFromWebcam();
        })
        .catch(err => {
          setWebcamError("Webcam access denied or not available. Please check your camera permissions.");
          console.error(err);
        });
    }
  }, [mode, detectFromWebcam, isLoading]);

  return (
    <div className="App">
      <header className="App-header">
        {error && <div className="error-message">{error}</div>}
        {isLoading && <div className="loading-message">Loading TensorFlow.js...</div>}
        <div className="controller-wrapper">
          <h1>Object Detection</h1>
          <div className="controls">
            <input
              type="file"
              accept="image/*,video/*"
              onChange={handleFileUpload}
              className="file-input"
            />
            <button onClick={() => {
              setMode('webcam');
              navigator.mediaDevices.getUserMedia({ video: true })
                .catch(err => {
                  setWebcamError("Webcam access denied or not available");
                  console.error(err);
                });
            }} className="mode-button">
              Use Webcam
            </button>
          </div>
        </div>
        {webcamError && <div className="error-message">{webcamError}</div>}
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
              zIndex: 8,
              width: 640,
              height: 480,
              display: mode === 'webcam' ? 'block' : 'none'
            }}
          />
        )}

        <video
          ref={videoRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 8,
            maxWidth: '100%',
            width: '1280px',
            height: 'auto',
            display: mode === 'video' ? 'block' : 'none'
          }}
          controls
        />

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
            width: '1280px',
            height: 'auto',
          }}
        />
      </header>
    </div>
  );
}

export default App;
