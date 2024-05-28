// script.js

// Elements
const video = document.getElementById('video');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const speakButton = document.getElementById('speakButton');
const transcriptionText = document.getElementById('results');

let stream;
let recognition;
let isRecognitionRunning = false;
let finalTranscript = '';

// Start the video stream
startButton.addEventListener('click', async () => {
  if (!stream) {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  }
});

// Stop the video stream
stopButton.addEventListener('click', () => {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    video.srcObject = null;
    stream = null;
  }
});

// Function to capture image from video
const captureImage = () => {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return canvas;
};

const speakText = (text) => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    speechSynthesis.speak(utterance);
  };

// Send image and text to server
const sendImageToServer = async (canvas, text) => {
  const dataUrl = canvas.toDataURL('image/jpeg');
  const blob = await fetch(dataUrl).then(res => res.blob());
  const formData = new FormData();
  formData.append('file', blob, 'capture.jpg');
  formData.append('input', "You are Eyestral, an AI assistant designed to help visually impaired individuals by providing accurate and concise information based on visual inputs. Your main goal is to describe images, objects, and scenes succinctly and clearly to aid visually impaired users in understanding their surroundings. Keep your responses brief, informative, and to the point. Always prioritize clarity and usefulness in your descriptions. <text>" + text + "<text>, explain this image.");
  formData.append('temperature', "0.2");
  formData.append('max_new_tokens', "512");


  const response = await fetch('http://20.246.100.13:5000/predict', {
    method: 'POST',
    body: formData
  });

  const data = await response.json();
  transcriptionText.textContent = data.response;

  speakText(data.response);
};

// Speech to text
function toggleDictation() {
  if (!('webkitSpeechRecognition' in window)) {
    alert('Üzgünüm, tarayıcınız ses tanımayı desteklemiyor.');
    return;
  }

  if (!recognition) {
    recognition = new window.webkitSpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      isRecognitionRunning = true;
      speakButton.textContent = 'Stop';
    };

    recognition.onend = () => {
      isRecognitionRunning = false;
      speakButton.textContent = 'Speak';

      // Konuşma bitince görüntü yakala ve sunucuya gönder
      const canvas = captureImage();
      sendImageToServer(canvas, finalTranscript);
      finalTranscript = ''; // Reset the transcript after sending
    };

    recognition.onresult = (event) => {
      let interimTranscript = '';
      
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }

      transcriptionText.textContent = finalTranscript + interimTranscript;
    };
  }

  if (isRecognitionRunning) {
    recognition.stop();
  } else {
    recognition.start();
  }
}

// Add event listener to the speak button
speakButton.addEventListener('click', toggleDictation);
