// Basic Express server setup for image upload and YOLO inference

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const cors = require('cors');

const app = express();
app.use(cors());
const PORT = process.env.PORT || 3000;

// Set up multer for file uploads
const upload = multer({
  dest: 'uploads/',
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
});

app.use(express.static('public'));
// (do NOT add app.use(cors()) here again)

// Path to YOLO model weights
const MODEL_PATH = path.join(__dirname, 'model', 'yolo11s.pt');

app.post('/upload', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }
  const imagePath = req.file.path;
  try {
    // Call Python script for YOLO inference
    const pythonProcess = spawn('python3', [
      path.join(__dirname, 'detect.py'),
      MODEL_PATH,
      imagePath
    ]);
    let result = '';
    let error = '';
    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });
    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
    });
    pythonProcess.on('close', (code) => {
      // Remove uploaded file after processing
      fs.unlink(imagePath, () => {});
      if (code !== 0 || error) {
        return res.status(500).json({ error: error || 'Detection failed', code });
      }
      if (!result) {
        return res.status(500).json({ error: 'No output from detection script', code });
      }
      try {
        const json = JSON.parse(result);
        res.json(json);
      } catch (e) {
        res.status(500).json({ error: 'Invalid detection output', raw: result });
      }
    });
  } catch (err) {
    fs.unlink(imagePath, () => {});
    res.status(500).json({ error: 'Server error', details: err.message });
  }
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on http://0.0.0.0:${PORT}`);
});
