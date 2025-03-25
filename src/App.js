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
  const [mode, setMode] = useState('webcam');
  const [model, setModel] = useState(null);
  const [webcamError, setWebcamError] = useState(null);
  const [croppedObjects, setCroppedObjects] = useState([]); // Store cropped objects
  const [maskedRegions, setMaskedRegions] = useState([]); // Track masked areas

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
          const predictions = await model.detect(img);
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
            .filter(pred => pred.score > 0.3)
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

        const predictions = await model.detect(video, undefined, 0.3);
        const newCroppedObjects = [];

        predictions.forEach(prediction => {
          const { bbox, class: className, score } = prediction;
          const [x, y, width, height] = bbox;

          // Check if this region is already masked
          const isMasked = maskedRegions.some(region => {
            const overlapX = Math.max(0, Math.min(x + width, region.x + region.width) - Math.max(x, region.x));
            const overlapY = Math.max(0, Math.min(y + height, region.y + region.height) - Math.max(y, region.y));
            const overlapArea = overlapX * overlapY;
            const boxArea = width * height;
            return overlapArea > boxArea * 0.3; // Lowered threshold to 30% for stricter masking
          });

          if (!isMasked && score > 0.3) {
            // Check for duplicates in croppedObjects based on class and bounding box similarity
            const isDuplicate = croppedObjects.some(obj => {
              if (obj.className !== className) return false;
              const [objX, objY, objWidth, objHeight] = [obj.x, obj.y, obj.width, obj.height];
              const overlapX = Math.max(0, Math.min(x + width, objX + objWidth) - Math.max(x, objX));
              const overlapY = Math.max(0, Math.min(y + height, objY + objHeight) - Math.max(y, objY));
              const overlapArea = overlapX * overlapY;
              const boxArea = width * height;
              return overlapArea > boxArea * 0.3; // Same 30% threshold for duplicates
            });

            if (!isDuplicate) {
              const croppedImage = cropImage(ctx, x, y, width, height, className, score);
              // Add bounding box info to croppedImage for duplicate checking
              croppedImage.x = x;
              croppedImage.y = y;
              croppedImage.width = width;
              croppedImage.height = height;
              newCroppedObjects.push(croppedImage);

              // Add to masked regions to prevent re-detection
              setMaskedRegions(prev => [...prev, { x, y, width, height }]);
            }
          }
        });

        if (newCroppedObjects.length > 0) {
          setCroppedObjects(prev => [...prev, ...newCroppedObjects]);
        }

        drawRect(predictions.filter(pred => !maskedRegions.some(r =>
          Math.max(0, Math.min(pred.bbox[0] + pred.bbox[2], r.x + r.width) - Math.max(pred.bbox[0], r.x)) *
          Math.max(0, Math.min(pred.bbox[1] + pred.bbox[3], r.y + r.height) - Math.max(pred.bbox[1], r.y)) >
          (pred.bbox[2] * pred.bbox[3] * 0.3))), ctx, false);
      }
    } catch (err) {
      console.error('Detection error:', err);
    }
    requestAnimationFrame(detectFromVideo);
  }, [model, maskedRegions, croppedObjects]); // Added croppedObjects to dependencies

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

      const predictions = await model.detect(canvasRef.current, undefined, 0.3);
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
          return overlapArea > boxArea * 0.3;
        });

        if (!isMasked && score > 0.3) {
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
        (pred.bbox[2] * pred.bbox[3] * 0.3))), ctx, true);
    }

    setTimeout(() => requestAnimationFrame(detectFromWebcam), 500);
  }, [model, maskedRegions]);

  useEffect(() => {
    if (mode === 'webcam' && !isLoading) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(() => {
          setWebcamError(null);
          detectFromWebcam();
        })
        .catch(err => {
          setWebcamError("Webcam access denied or not available.");
          console.error(err);
        });
    }
  }, [mode, detectFromWebcam, isLoading]);

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
              <button onClick={() => {
                setMode('webcam');
                setCroppedObjects([]);
                setMaskedRegions([]);
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
        </div>

        {croppedObjects.length > 0 && (
          <div className="cropped-objects" style={{ marginTop: '20px', marginBottom: '20px', display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
            <h3>Cropped Objects</h3>
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