import React, { useRef, useCallback, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as cocossd from "@tensorflow-models/coco-ssd";
import Webcam from "react-webcam";
import "./App.css";
import { drawRect, cropImage } from "./utilities";

// Utility function to calculate Intersection over Union (IoU) between two bounding boxes
const calculateIoU = (box1, box2) => {
  const [x1, y1, w1, h1] = box1;
  const [x2, y2, w2, h2] = box2;

  const xLeft = Math.max(x1, x2);
  const yTop = Math.max(y1, y2);
  const xRight = Math.min(x1 + w1, x2 + w2);
  const yBottom = Math.min(y1 + h1, y2 + h2);

  if (xRight < xLeft || yBottom < yTop) return 0;

  const intersectionArea = (xRight - xLeft) * (yBottom - yTop);
  const box1Area = w1 * h1;
  const box2Area = w2 * h2;
  const unionArea = box1Area + box2Area - intersectionArea;

  return intersectionArea / unionArea;
};

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const videoRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState('none');
  const [model, setModel] = useState(null);
  const [webcamError, setWebcamError] = useState(null);
  const [croppedObjects, setCroppedObjects] = useState([]);
  const [maskedRegions, setMaskedRegions] = useState([]);
  const [trackedObjects, setTrackedObjects] = useState([]);
  const [previousCenters, setPreviousCenters] = useState([]); // Track centers of objects in the previous frame

  // Define thresholds as variables
  const IOU_THRESHOLD = 0.3; // Lowered for more leniency with camera movement
  const TEMPORAL_THRESHOLD = 10000; // Increased to 10 seconds for longer tracking
  const CONFIDENCE_THRESHOLD = 0.1; // Confidence difference threshold for deduplication

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

    setCroppedObjects([]);
    setMaskedRegions([]);
    setTrackedObjects([]);
    setPreviousCenters([]);

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

        maskedRegions.forEach(region => {
          ctx.fillStyle = 'black';
          ctx.fillRect(region.x, region.y, region.width, region.height);
        });

        const predictions = await model.detect(video, undefined, 0.6);
        console.log('Video Predictions:', predictions);
        const newCroppedObjects = [];
        const currentTime = Date.now();
        const updatedTrackedObjects = [];

        // Calculate centers of current predictions
        const currentCenters = predictions.map(pred => {
          const [x, y, width, height] = pred.bbox;
          return { x: x + width / 2, y: y + height / 2 };
        });

        // Estimate camera motion by calculating average shift in centers
        let avgShiftX = 0;
        let avgShiftY = 0;
        if (previousCenters.length > 0 && currentCenters.length > 0) {
          const pairedCenters = Math.min(previousCenters.length, currentCenters.length);
          if (pairedCenters > 0) {
            let totalShiftX = 0;
            let totalShiftY = 0;
            for (let i = 0; i < pairedCenters; i++) {
              totalShiftX += currentCenters[i].x - previousCenters[i].x;
              totalShiftY += currentCenters[i].y - previousCenters[i].y;
            }
            avgShiftX = totalShiftX / pairedCenters;
            avgShiftY = totalShiftY / pairedCenters;
          }
        }
        setPreviousCenters(currentCenters);

        // Clean up old tracked objects
        const trackedObjectsFiltered = trackedObjects.filter(obj => currentTime - obj.lastSeen < TEMPORAL_THRESHOLD);

        predictions.forEach(prediction => {
          const { bbox, class: className, score } = prediction;
          let [x, y, width, height] = bbox;

          // Adjust bounding box for camera motion
          x -= avgShiftX;
          y -= avgShiftY;

          const isMasked = maskedRegions.some(region => {
            const overlapX = Math.max(0, Math.min(x + width, region.x + region.width) - Math.max(x, region.x));
            const overlapY = Math.max(0, Math.min(y + height, region.y + region.height) - Math.max(y, region.y));
            const overlapArea = overlapX * overlapY;
            const boxArea = width * height;
            return overlapArea > boxArea * 0.1;
          });

          if (!isMasked && score > 0.6) {
            let isDuplicate = false;
            let matchedObject = null;

            for (const trackedObj of trackedObjectsFiltered) {
              if (trackedObj.className !== className) continue;

              const iou = calculateIoU([x, y, width, height], trackedObj.bbox);
              const confidenceDiff = Math.abs(score - trackedObj.score);
              if (iou > IOU_THRESHOLD && confidenceDiff < CONFIDENCE_THRESHOLD) {
                isDuplicate = true;
                matchedObject = trackedObj;
                break;
              }
            }

            if (!isDuplicate) {
              const croppedImage = cropImage(ctx, x + avgShiftX, y + avgShiftY, width, height, className, score);
              newCroppedObjects.push(croppedImage);

              const newId = trackedObjects.length > 0 ? Math.max(...trackedObjects.map(obj => obj.id)) + 1 : 1;
              updatedTrackedObjects.push({
                id: newId,
                className,
                bbox: [x, y, width, height],
                score,
                lastSeen: currentTime
              });

              const padding = 30;
              setMaskedRegions(prev => [...prev, {
                x: Math.max(0, (x + avgShiftX) - padding),
                y: Math.max(0, (y + avgShiftY) - padding),
                width: width + 2 * padding,
                height: height + 2 * padding
              }]);
            } else if (matchedObject) {
              updatedTrackedObjects.push({
                ...matchedObject,
                bbox: [x, y, width, height],
                score,
                lastSeen: currentTime
              });
            }
          }
        });

        const nonMatchedObjects = trackedObjectsFiltered.filter(obj => !updatedTrackedObjects.some(updated => updated.id === obj.id));
        setTrackedObjects([...nonMatchedObjects, ...updatedTrackedObjects]);

        if (newCroppedObjects.length > 0) {
          setCroppedObjects(prev => [...prev, ...newCroppedObjects]);
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
  }, [model, maskedRegions, trackedObjects, previousCenters]);

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

      maskedRegions.forEach(region => {
        ctx.fillStyle = 'black';
        ctx.fillRect(region.x, region.y, region.width, region.height);
      });

      const predictions = await model.detect(canvasRef.current, undefined, 0.6);
      console.log('Webcam Predictions:', predictions);

      const newCroppedObjects = [];
      const currentTime = Date.now();
      const updatedTrackedObjects = [];

      const currentCenters = predictions.map(pred => {
        const [x, y, width, height] = pred.bbox;
        return { x: x + width / 2, y: y + height / 2 };
      });

      let avgShiftX = 0;
      let avgShiftY = 0;
      if (previousCenters.length > 0 && currentCenters.length > 0) {
        const pairedCenters = Math.min(previousCenters.length, currentCenters.length);
        if (pairedCenters > 0) {
          let totalShiftX = 0;
          let totalShiftY = 0;
          for (let i = 0; i < pairedCenters; i++) {
            totalShiftX += currentCenters[i].x - previousCenters[i].x;
            totalShiftY += currentCenters[i].y - previousCenters[i].y;
          }
          avgShiftX = totalShiftX / pairedCenters;
          avgShiftY = totalShiftY / pairedCenters;
        }
      }
      setPreviousCenters(currentCenters);

      const trackedObjectsFiltered = trackedObjects.filter(obj => currentTime - obj.lastSeen < TEMPORAL_THRESHOLD);

      predictions.forEach(prediction => {
        const { bbox, class: className, score } = prediction;
        let [x, y, width, height] = bbox;

        x -= avgShiftX;
        y -= avgShiftY;

        const isMasked = maskedRegions.some(region => {
          const overlapX = Math.max(0, Math.min(x + width, region.x + region.width) - Math.max(x, region.x));
          const overlapY = Math.max(0, Math.min(y + height, region.y + region.height) - Math.max(y, region.y));
          const overlapArea = overlapX * overlapY;
          const boxArea = width * height;
          return overlapArea > boxArea * 0.1;
        });

        if (!isMasked && score > 0.6) {
          let isDuplicate = false;
          let matchedObject = null;

          for (const trackedObj of trackedObjectsFiltered) {
            if (trackedObj.className !== className) continue;

            const iou = calculateIoU([x, y, width, height], trackedObj.bbox);
            const confidenceDiff = Math.abs(score - trackedObj.score);
            if (iou > IOU_THRESHOLD && confidenceDiff < CONFIDENCE_THRESHOLD) {
              isDuplicate = true;
              matchedObject = trackedObj;
              break;
            }
          }

          if (!isDuplicate) {
            const croppedImage = cropImage(ctx, x + avgShiftX, y + avgShiftY, width, height, className, score);
            newCroppedObjects.push(croppedImage);

            const newId = trackedObjects.length > 0 ? Math.max(...trackedObjects.map(obj => obj.id)) + 1 : 1;
            updatedTrackedObjects.push({
              id: newId,
              className,
              bbox: [x, y, width, height],
              score,
              lastSeen: currentTime
            });

            setMaskedRegions(prev => [...prev, { x: x + avgShiftX, y: y + avgShiftY, width, height }]);
          } else if (matchedObject) {
            updatedTrackedObjects.push({
              ...matchedObject,
              bbox: [x, y, width, height],
              score,
              lastSeen: currentTime
            });
          }
        }
      });

      const nonMatchedObjects = trackedObjectsFiltered.filter(obj => !updatedTrackedObjects.some(updated => updated.id === obj.id));
      setTrackedObjects([...nonMatchedObjects, ...updatedTrackedObjects]);

      if (newCroppedObjects.length > 0) {
        setCroppedObjects(prev => [...prev, ...newCroppedObjects]);
      }

      drawRect(predictions.filter(pred => !maskedRegions.some(r =>
        Math.max(0, Math.min(pred.bbox[0] + pred.bbox[2], r.x + r.width) - Math.max(pred.bbox[0], r.x)) *
        Math.max(0, Math.min(pred.bbox[1] + pred.bbox[3], r.y + r.height) - Math.max(pred.bbox[1], r.y)) >
        (pred.bbox[2] * pred.bbox[3] * 0.1))), ctx, true);
    }

    setTimeout(() => requestAnimationFrame(detectFromWebcam), 500);
  }, [model, maskedRegions, trackedObjects, previousCenters]);

  const handleWebcamActivation = () => {
    setMode('webcam');
    setCroppedObjects([]);
    setMaskedRegions([]);
    setTrackedObjects([]);
    setPreviousCenters([]);
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

  const handleReset = () => {
    setMode('none');
    setCroppedObjects([]);
    setMaskedRegions([]);
    setTrackedObjects([]);
    setPreviousCenters([]);
    setWebcamError(null);
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.src = '';
    }
  };

  const handleClearCroppedObjects = () => {
    setCroppedObjects([]);
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
                disabled={isLoading}
              />
              <button
                onClick={handleWebcamActivation}
                className="mode-button"
                disabled={isLoading}
              >
                Use Webcam
              </button>
              <button
                onClick={handleReset}
                className="reset-button"
                disabled={isLoading}
                style={{
                  marginLeft: '10px',
                  padding: '5px 10px',
                  background: '#f44336',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: isLoading ? 'not-allowed' : 'pointer'
                }}
              >
                Reset
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
            <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3>Cropped Objects ({croppedObjects.length})</h3>
              <button
                onClick={handleClearCroppedObjects}
                style={{
                  padding: '5px 10px',
                  background: '#f44336',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Clear
              </button>
            </div>
            <div className="img-wrapper-objects" style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', maxHeight: '650px', overflowY: 'scroll' }}>
              {croppedObjects.map((obj, index) => (
                <div key={index} style={{ textAlign: 'center' }}>
                  <img src={obj.src} alt={obj.className} style={{ maxWidth: '100px', border: '1px solid #00FF00' }} />
                  <p>{obj.className} - {Math.round(obj.score * 100)}%</p>
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
          </div>
        )}
      </header>
    </div>
  );
}

export default App;