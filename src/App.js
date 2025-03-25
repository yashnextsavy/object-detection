import React, { useRef, useCallback, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as cocossd from "@tensorflow-models/coco-ssd";
import Webcam from "react-webcam";
import "./App.css";
import { drawRect, cropImage } from "./utilities";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState('none'); // Changed default mode to 'none'
  const [model, setModel] = useState(null);
  const [webcamError, setWebcamError] = useState(null);
  const [croppedObjects, setCroppedObjects] = useState([]); // Store cropped objects
  const [maskedRegions, setMaskedRegions] = useState([]); // Track masked areas
  const [detectedObjects, setDetectedObjects] = useState([]); // Track unique objects by class and position

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

    // Clear old data before processing new upload
    setCroppedObjects([]);
    setMaskedRegions([]);
    setDetectedObjects([]); // Clear detected objects

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

        ctx.drawImage(img, 0, 0, displayWidth, displayHeight);

        if (model) {
          const predictions = await model.detect(img, undefined, 0.6);
          const scaledPredictions = predictions.map(pred => ({
            ...pred,
            bbox: [
              pred.bbox[0] * scale,
              pred.bbox[1] * scale,
              pred.bbox[2] * scale,
              pred.bbox[3] * scale
            ]
          }));

          // Crop detected objects
          const newCroppedObjects = scaledPredictions
            .filter(pred => pred.score > 0.6)
            .map(pred => {
              const [x, y, width, height] = pred.bbox;
              return cropImage(ctx, x, y, width, height, pred.class, pred.score);
            });

          if (newCroppedObjects.length > 0) {
            setCroppedObjects(newCroppedObjects);
          }

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
      if (video.readyState === video.HAVE_ENOUGH_DATA) {
        const videoWidth = video.videoWidth || 640;
        const videoHeight = video.videoHeight || 480;
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;

        const ctx = canvasRef.current.getContext('2d');
        ctx.clearRect(0, 0, videoWidth, videoHeight);
        ctx.drawImage(video, 0, 0, videoWidth, videoHeight);

        // Apply masks to previously detected regions
        maskedRegions.forEach(region => {
          ctx.fillStyle = 'black';
          ctx.fillRect(region.x, region.y, region.width, region.height);
        });

        const predictions = await model.detect(video, undefined, 0.6);
        console.log('Video Predictions:', predictions); // Debug
        const newCroppedObjects = [];
        const newDetectedObjects = [...detectedObjects];

        predictions.forEach(prediction => {
          const { bbox, class: className, score } = prediction;
          const [x, y, width, height] = bbox;

          // Check if this region is already masked
          const isMasked = maskedRegions.some(region => {
            const overlapX = Math.max(0, Math.min(x + width, region.x + region.width) - Math.max(x, region.x));
            const overlapY = Math.max(0, Math.min(y + height, region.y + region.height) - Math.max(y, region.y));
            const overlapArea = overlapX * overlapY;
            const boxArea = width * height;
            return overlapArea > boxArea * 0.1; // Stricter threshold: 10%
          });

          // Check for duplicates based on class and position
          const centerX = x + width / 2;
          const centerY = y + height / 2;
          const isDuplicate = detectedObjects.some(obj => {
            if (obj.className !== className) return false;
            const objCenterX = obj.centerX;
            const objCenterY = obj.centerY;
            const distance = Math.sqrt(
              Math.pow(centerX - objCenterX, 2) + Math.pow(centerY - objCenterY, 2)
            );
            return distance < 50; // Stricter distance threshold: 50px
          });

          if (!isMasked && !isDuplicate && score > 0.6) {
            const croppedImage = cropImage(ctx, x, y, width, height, className, score);
            newCroppedObjects.push(croppedImage);

            // Add to detected objects
            newDetectedObjects.push({
              className,
              centerX,
              centerY
            });

            // Add to masked regions with expanded area
            const padding = 30; // Padding of 30px
            setMaskedRegions(prev => [...prev, {
              x: Math.max(0, x - padding),
              y: Math.max(0, y - padding),
              width: width + 2 * padding,
              height: height + 2 * padding
            }]);
          }
        });

        if (newCroppedObjects.length > 0) {
          setCroppedObjects(prev => [...prev, ...newCroppedObjects]);
          setDetectedObjects(newDetectedObjects);
        }

        drawRect(predictions.filter(pred => !maskedRegions.some(r =>
          Math.max(0, Math.min(pred.bbox[0] + pred.bbox[2], r.x + r.width) - Math.max(pred.bbox[0], r.x)) *
          Math.max(0, Math.min(pred.bbox[1] + pred.bbox[3], r.y + r.height) - Math.max(pred.bbox[1], r.y)) >
          (pred.bbox[2] * pred.bbox[3] * 0.1))), ctx, false);
      }
    } catch (err) {
      console.error('Detection error:', err);
    }
    requestAnimationFrame(detectFromVideo);
  }, [model, maskedRegions, detectedObjects]);

  useEffect(() => {
    let animationFrame;
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
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;

      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, videoWidth, videoHeight);
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);

      // Apply masks
      maskedRegions.forEach(region => {
        ctx.fillStyle = 'black';
        ctx.fillRect(region.x, region.y, region.width, region.height);
      });

      const predictions = await model.detect(canvasRef.current, undefined, 0.6);
      console.log('Webcam Predictions:', predictions); // Debug

      const newCroppedObjects = [];
      predictions.forEach(prediction => {
        const { bbox, class: className, score } = prediction;
        const [x, y, width, height] = bbox;

        const isMasked = maskedRegions.some(region => {
          const overlapX = Math.max(0, Math.min(x + width, region.x + region.width) - Math.max(x, region.x));
          const overlapY = Math.max(0, Math.min(y + height, region.y + region.height) - Math.max(y, region.y));
          const overlapArea = overlapX * overlapY;
          const boxArea = width * height;
          return overlapArea > boxArea * 0.1;
        });

        if (!isMasked && score > 0.6) {
          const croppedImage = cropImage(ctx, x, y, width, height, className, score);
          newCroppedObjects.push(croppedImage);
          setMaskedRegions(prev => [...prev, { x, y, width, height }]);
        }
      });

      console.log('New Cropped Objects:', newCroppedObjects); // Debug
      if (newCroppedObjects.length > 0) {
        setCroppedObjects(prev => [...prev, ...newCroppedObjects]);
      }

      drawRect(predictions.filter(pred => !maskedRegions.some(r =>
        Math.max(0, Math.min(pred.bbox[0] + pred.bbox[2], r.x + r.width) - Math.max(pred.bbox[0], r.x)) *
        Math.max(0, Math.min(pred.bbox[1] + pred.bbox[3], r.y + r.height) - Math.max(pred.bbox[1], r.y)) >
        (pred.bbox[2] * pred.bbox[3] * 0.1))), ctx, true);
    }

    setTimeout(() => requestAnimationFrame(detectFromWebcam), 500);
  }, [model, maskedRegions]);

  // Removed automatic webcam initialization on app load
  const handleWebcamActivation = () => {
    setMode('webcam');
    setCroppedObjects([]);
    setMaskedRegions([]);
    setDetectedObjects([]);
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(() => {
        setWebcamError(null);
        detectFromWebcam();
      })
      .catch(err => {
        setWebcamError("Webcam access denied or not available.");
        console.error(err);
      });
  };

  return (
    <div className="App">
      <header className="App-header">
        {error && <div className="error-message">{error}</div>}
        {isLoading && <div className="loading-message">Loading TensorFlow.js...</div>}
        <div className="header-line-wrapper">
          <div className="controller-wrapper">
            <h1>Object Detection</h1>
            <div className="controls">
              <input
                type="file"
                accept="image/*,video/*"
                onChange={handleFileUpload}
                className="file-input"
              />
              <button onClick={handleWebcamActivation} className="mode-button">
                Use Webcam
              </button>
            </div>
          </div>
          {webcamError && <div className="error-message">{webcamError}</div>}
          <div className="detections-box">
            <h3>Detected Items</h3>
            <div id="detectionsList"></div>
          </div>
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

        {croppedObjects.length > 0 && (
          <div className="cropped-objects" style={{ marginTop: '53%', marginBottom: '20px', display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
            <h3>Cropped Objects ({croppedObjects.length})</h3>
            {croppedObjects.map((obj, index) => (
              <div key={index} style={{ textAlign: 'center' }}>
                <img src={obj.src} alt={obj.className} style={{ maxWidth: '100px', border: '1px solid #00FF00' }} />
                <p>{obj.className} - {obj.score}%</p>
                <button
                  onClick={() => {
                    const link = document.createElement('a');
                    link.href = obj.src;
                    link.download = `${obj.className}-${obj.score}-${Date.now()}.png`;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                  }}
                  style={{ marginTop: '5px', padding: '5px 10px', background: '#4CAF50', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
                >
                  Download
                </button>
              </div>
            ))}
          </div>
        )}
      </header>
    </div>
  );
}

export default App;