/**
 * Sign to Voice - Frontend Application Logic
 * Exclusively Client-Side In-Browser Inference with MediaPipe Hands & Web Speech API
 */

(function () {
  'use strict';

  // --- DOM Elements ---
  const videoElement = document.getElementById('webcam');
  const canvasElement = document.getElementById('output-canvas');
  const canvasCtx = canvasElement.getContext('2d');
  const cameraLoading = document.getElementById('camera-loading');
  const statusDot = document.getElementById('status-dot');
  const statusText = document.getElementById('status-text');
  
  const currentSignEl = document.getElementById('current-sign');
  const confidenceBadgeEl = document.getElementById('confidence-badge');
  const confidenceBarEl = document.getElementById('confidence-bar');
  const lastConfirmedSignEl = document.getElementById('last-confirmed-sign');
  
  const sentenceBox = document.getElementById('sentence-box');
  const charCountEl = document.getElementById('char-count');
  
  const btnSpeak = document.getElementById('btn-speak');
  const btnSpace = document.getElementById('btn-space');
  const btnBackspace = document.getElementById('btn-backspace');
  const btnClear = document.getElementById('btn-clear');
  const btnCopy = document.getElementById('btn-copy');
  
  const fpsCounterEl = document.getElementById('fps-counter');
  const latencyCounterEl = document.getElementById('latency-counter');
  const toggleSkeletonBtn = document.getElementById('toggle-skeleton-btn');
  const toggleCameraBtn = document.getElementById('toggle-camera-btn');
  const handDotEl = document.getElementById('hand-dot');
  const handTextEl = document.getElementById('hand-text');
  const labelsGrid = document.getElementById('labels-grid');

  // --- Configuration & State ---
  const CONFIDENCE_THRESHOLD = 0.90;
  const PREDICTION_DELAY_MS = 1500; // 1.5 seconds between predictions as in predict_keras_live.py
  const BUFFER_STABILITY_COUNT = 3;  // Consecutive frames needed for stable gesture

  let modelWeights = null;
  let labels = [];
  let showSkeleton = true;
  let isCameraPaused = false;
  let lastPrediction = null;
  let lastPredTime = 0;

  // Smoothing buffer
  let predictionBuffer = [];
  
  // FPS calculation
  let lastFrameTime = performance.now();
  let frameCount = 0;
  let fps = 0;

  // Web Speech API
  const synth = window.speechSynthesis;

  // ==========================================
  // 1. Model Weights & Client-side Forward Pass
  // ==========================================
  async function loadStaticAssets() {
    try {
      statusText.textContent = 'Loading neural network...';
      
      // Load labels
      const labelsRes = await fetch('/static/labels.json');
      if (labelsRes.ok) {
        labels = await labelsRes.json();
      } else {
        // Fallback 28 classes
        labels = ['A','B','C','D','E','F','G','H','I','J','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space'];
      }
      renderLabelsGrid(labels);

      // Load precomputed model weights for 100% in-browser matrix forward pass
      const weightsRes = await fetch('/static/model_weights.json');
      if (weightsRes.ok) {
        modelWeights = await weightsRes.json();
        console.log('✅ In-Browser model loaded:', modelWeights.length, 'layers');
      }

      statusDot.className = 'w-2 h-2 rounded-full bg-emerald-500';
      statusText.textContent = 'AI Ready';
    } catch (err) {
      console.error('Failed to load static assets:', err);
      statusDot.className = 'w-2 h-2 rounded-full bg-rose-500';
      statusText.textContent = 'Model Load Error';
    }
  }

  // Pure In-Browser Forward Pass: Dense layers with ReLU and Softmax
  function predictClientSide(features) {
    if (!modelWeights || modelWeights.length === 0) return null;

    let current = features; // 63 features

    for (let l = 0; l < modelWeights.length; l++) {
      const layer = modelWeights[l];
      const W = layer.weights; // shape: (in_dim, out_dim)
      const b = layer.bias;    // shape: (out_dim)
      const inDim = W.length;
      const outDim = b.length;
      const next = new Array(outDim).fill(0);

      // Matrix multiplication: next[j] = sum(current[i] * W[i][j]) + b[j]
      for (let j = 0; j < outDim; j++) {
        let sum = b[j];
        for (let i = 0; i < inDim; i++) {
          sum += current[i] * W[i][j];
        }
        
        // Activation
        if (layer.activation === 'relu') {
          next[j] = Math.max(0, sum);
        } else {
          next[j] = sum;
        }
      }

      // Softmax on last layer
      if (layer.activation === 'softmax' || l === modelWeights.length - 1) {
        let maxVal = Math.max(...next);
        let expSum = 0;
        const expArr = new Array(outDim);
        for (let j = 0; j < outDim; j++) {
          expArr[j] = Math.exp(next[j] - maxVal);
          expSum += expArr[j];
        }
        for (let j = 0; j < outDim; j++) {
          next[j] = expArr[j] / (expSum || 1);
        }
      }

      current = next;
    }

    // Find argmax & confidence
    let maxIdx = 0;
    let maxProb = current[0];
    for (let i = 1; i < current.length; i++) {
      if (current[i] > maxProb) {
        maxProb = current[i];
        maxIdx = i;
      }
    }

    const predictedLabel = labels[maxIdx] || `Class_${maxIdx}`;
    return {
      prediction: maxProb > CONFIDENCE_THRESHOLD ? predictedLabel : null,
      confidence: maxProb,
      label: predictedLabel,
      threshold: CONFIDENCE_THRESHOLD
    };
  }

  // ==========================================
  // 2. MediaPipe Hands & Video Processing
  // ==========================================
  function onResults(results) {
    if (isCameraPaused) return;

    // Calculate FPS
    frameCount++;
    const now = performance.now();
    if (now - lastFrameTime >= 1000) {
      fps = frameCount;
      frameCount = 0;
      lastFrameTime = now;
      fpsCounterEl.textContent = `${fps} FPS`;
    }

    // Set canvas dimensions
    canvasElement.width = videoElement.videoWidth || 640;
    canvasElement.height = videoElement.videoHeight || 480;

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // Draw camera image onto canvas
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    // Process Hand Landmarks
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const handLandmarks = results.multiHandLandmarks[0];

      // Update Hand Status Indicator
      handDotEl.className = 'w-1.5 h-1.5 rounded-full bg-emerald-400';
      handTextEl.textContent = 'Hand detected';

      // Skeleton Drawing Overlay
      if (showSkeleton) {
        drawConnectors(canvasCtx, handLandmarks, HAND_CONNECTIONS, {
          color: '#0ea5e9', // Sky blue
          lineWidth: 2.5
        });
        drawLandmarks(canvasCtx, handLandmarks, {
          color: '#ffffff',
          fillColor: '#0284c7',
          lineWidth: 1.5,
          radius: 3
        });
      }

      // Extract 63 landmarks in exact order [lm.x, lm.y, lm.z]
      const landmarkPoints = [];
      for (let i = 0; i < handLandmarks.length; i++) {
        const lm = handLandmarks[i];
        landmarkPoints.push(lm.x, lm.y, lm.z);
      }

      if (landmarkPoints.length === 63) {
        handleInference(landmarkPoints);
      }

    } else {
      // No hand in frame
      handDotEl.className = 'w-1.5 h-1.5 rounded-full bg-slate-400';
      handTextEl.textContent = 'No hand detected';
      updatePredictionDisplay(null, 0, '-');
      predictionBuffer = [];
    }

    canvasCtx.restore();
  }

  // Handle client-side inference & temporal smoothing
  function handleInference(landmarkPoints) {
    const t0 = performance.now();
    const result = predictClientSide(landmarkPoints);
    const elapsed = (performance.now() - t0).toFixed(1);
    latencyCounterEl.textContent = `${elapsed}ms`;

    if (!result) return;

    const rawLabel = result.label || '-';
    const confidence = result.confidence || 0;
    const confirmedPrediction = result.prediction; // Non-null only if confidence > 0.9

    // Update real-time UI gauge
    updatePredictionDisplay(confirmedPrediction, confidence, rawLabel);

    // Apply temporal stability buffer & sentence builder logic
    if (confirmedPrediction) {
      predictionBuffer.push(confirmedPrediction);
      if (predictionBuffer.length > BUFFER_STABILITY_COUNT) {
        predictionBuffer.shift();
      }

      // Check if last N predictions are all the same confirmed letter
      const isStable = predictionBuffer.length === BUFFER_STABILITY_COUNT &&
                       predictionBuffer.every(p => p === confirmedPrediction);

      if (isStable) {
        const currentTime = Date.now();
        const timeSinceLast = currentTime - lastPredTime;

        if (confirmedPrediction !== lastPrediction && timeSinceLast > PREDICTION_DELAY_MS) {
          applyGestureToSentence(confirmedPrediction);
          lastPrediction = confirmedPrediction;
          lastPredTime = currentTime;
          lastConfirmedSignEl.textContent = confirmedPrediction;
        }
      }
    } else {
      predictionBuffer = [];
    }
  }

  // Update UI with current gesture & confidence
  function updatePredictionDisplay(prediction, confidence, rawLabel) {
    const percent = Math.round(confidence * 100);
    confidenceBadgeEl.textContent = `${percent}%`;
    confidenceBarEl.style.width = `${percent}%`;

    if (confidence > CONFIDENCE_THRESHOLD) {
      currentSignEl.textContent = prediction || rawLabel;
      currentSignEl.className = 'text-2xl font-bold text-blue-600 leading-none';
      confidenceBarEl.className = 'bg-blue-600 h-full rounded-full transition-all duration-150';
      confidenceBadgeEl.className = 'text-xs px-2 py-0.5 rounded-md bg-blue-50 text-blue-700 border border-blue-200 font-mono font-medium';
    } else {
      currentSignEl.textContent = rawLabel !== '-' ? rawLabel : '-';
      currentSignEl.className = 'text-2xl font-bold text-slate-400 leading-none';
      confidenceBarEl.className = 'bg-slate-400 h-full rounded-full transition-all duration-150';
      confidenceBadgeEl.className = 'text-xs px-2 py-0.5 rounded-md bg-slate-100 text-slate-500 border border-slate-200 font-mono';
    }
  }

  // Gesture actions: Del, Space, Nothing, Letters
  function applyGestureToSentence(gesture) {
    let currentText = sentenceBox.value;

    if (gesture === 'del' || gesture === 'backspace') {
      currentText = currentText.slice(0, -1);
    } else if (gesture === 'space') {
      currentText += ' ';
    } else if (gesture === 'nothing') {
      // Ignored
      return;
    } else {
      // Regular character
      currentText += gesture;
    }

    sentenceBox.value = currentText;
    updateCharCount();
  }

  function updateCharCount() {
    const count = sentenceBox.value.length;
    charCountEl.textContent = `${count} character${count === 1 ? '' : 's'}`;
  }

  // ==========================================
  // 3. Web Speech API (Text-to-Speech)
  // ==========================================
  function speakText(text) {
    if (!synth || !text || !text.trim()) return;
    synth.cancel(); // Cancel any existing speech utterance

    const utterance = new SpeechSynthesisUtterance(text.trim());
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    synth.speak(utterance);
  }

  // ==========================================
  // 4. UI Controls & Event Listeners
  // ==========================================
  function setupEventListeners() {
    // Speak button
    btnSpeak.addEventListener('click', () => {
      speakText(sentenceBox.value);
    });

    // Space button
    btnSpace.addEventListener('click', () => {
      sentenceBox.value += ' ';
      updateCharCount();
    });

    // Backspace button
    btnBackspace.addEventListener('click', () => {
      sentenceBox.value = sentenceBox.value.slice(0, -1);
      updateCharCount();
    });

    // Clear button
    btnClear.addEventListener('click', () => {
      sentenceBox.value = '';
      lastPrediction = null;
      lastConfirmedSignEl.textContent = '-';
      updateCharCount();
    });

    // Copy button
    btnCopy.addEventListener('click', async () => {
      if (sentenceBox.value) {
        try {
          await navigator.clipboard.writeText(sentenceBox.value);
          const originalHTML = btnCopy.innerHTML;
          btnCopy.innerHTML = '<span>✅</span><span>Copied!</span>';
          setTimeout(() => { btnCopy.innerHTML = originalHTML; }, 1500);
        } catch (e) {
          console.error('Clipboard copy error:', e);
        }
      }
    });

    // Toggle skeleton overlay
    toggleSkeletonBtn.addEventListener('click', () => {
      showSkeleton = !showSkeleton;
      toggleSkeletonBtn.textContent = showSkeleton ? '🦴 Skeleton: On' : '🦴 Skeleton: Off';
    });

    // Pause/Resume Camera
    toggleCameraBtn.addEventListener('click', () => {
      isCameraPaused = !isCameraPaused;
      toggleCameraBtn.textContent = isCameraPaused ? '▶️ Resume' : '⏸️ Pause';
    });

    // Textarea manual typing listener
    sentenceBox.addEventListener('input', updateCharCount);

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      // Do not trigger shortcuts when manually editing inside textarea
      if (document.activeElement === sentenceBox) return;

      if (e.code === 'Space') {
        e.preventDefault();
        btnSpace.click();
      } else if (e.code === 'Backspace') {
        e.preventDefault();
        btnBackspace.click();
      } else if (e.key === 's' || e.key === 'S') {
        e.preventDefault();
        btnSpeak.click();
      } else if (e.key === 'c' || e.key === 'C') {
        e.preventDefault();
        btnClear.click();
      } else if (e.key === 'p' || e.key === 'P') {
        e.preventDefault();
        toggleCameraBtn.click();
      }
    });
  }

  function renderLabelsGrid(classesList) {
    if (!labelsGrid) return;
    labelsGrid.innerHTML = '';
    classesList.forEach(cls => {
      const badge = document.createElement('span');
      badge.className = 'px-2 py-0.5 bg-slate-100 border border-slate-200 rounded text-slate-700 font-mono text-[11px] hover:border-blue-400 transition cursor-default';
      badge.textContent = cls;
      labelsGrid.appendChild(badge);
    });
  }

  // ==========================================
  // 5. Initialize Camera & MediaPipe Hands
  // ==========================================
  async function initCameraAndHands() {
    try {
      const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      });

      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.7
      });

      hands.onResults(onResults);

      const camera = new Camera(videoElement, {
        onFrame: async () => {
          if (!isCameraPaused) {
            await hands.send({ image: videoElement });
          }
        },
        width: 640,
        height: 480
      });

      await camera.start();
      cameraLoading.classList.add('hidden');
    } catch (err) {
      console.error('Camera initialization failed:', err);
      cameraLoading.innerHTML = `
        <div class="text-center p-4">
          <p class="text-rose-600 font-semibold mb-1">⚠️ Camera Access Error</p>
          <p class="text-xs text-slate-300">Please allow webcam access in your browser to use Sign-to-Voice.</p>
        </div>
      `;
    }
  }

  // ==========================================
  // Main Entry Point
  // ==========================================
  async function init() {
    setupEventListeners();
    await loadStaticAssets();
    await initCameraAndHands();
  }

  window.addEventListener('DOMContentLoaded', init);
})();
