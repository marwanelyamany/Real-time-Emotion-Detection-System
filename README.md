# Real-time Emotion Detection System

A deep learning project that detects human emotions in real-time using a webcam.  
Built with Python, OpenCV, and TensorFlow/Keras.

---

## Emotions Detected

| Emotion     | Emotion    | Emotion |
| ----------- | ---------- | ------- |
| 😠 Angry    | 🤢 Disgust | 😨 Fear |
| 😄 Happy    | 😐 Neutral | 😢 Sad  |
| 😲 Surprise |            |         |

---

## Project Structure

```
emotion-detection/
├── notebook/
│   └── emotion_detection.ipynb   ← Main notebook (run this)
├── model/                        ← Saved model goes here after training
├── data/                         ← Put FER-2013 dataset here
│   ├── train/
│   └── test/
├── .gitignore
└── README.md
```

---

## Setup & Run

### 1. Download the dataset

- Go to: https://www.kaggle.com/datasets/msambare/fer2013
- Download and extract it
- Place the `train/` and `test/` folders inside a `data/` folder in the project root

### 2. Open the notebook

```bash
jupyter notebook notebook/emotion_detection.ipynb
```

### 3. Run cells in order

The notebook is split into 10 steps — just run them top to bottom.

---

## Notebook Steps

| Step | What it does                   |
| ---- | ------------------------------ |
| 1    | Install libraries              |
| 2    | Import libraries               |
| 3    | Set configuration              |
| 4    | Load dataset                   |
| 5    | Preview dataset                |
| 6    | Build CNN model                |
| 7    | Train model                    |
| 8    | Plot accuracy & loss           |
| 9    | Evaluate & confusion matrix    |
| 10   | Run real-time webcam detection |

---

## How it Works

1. **Face Detection** — OpenCV's Haar Cascade finds faces in each webcam frame
2. **Preprocessing** — The face is cropped, converted to grayscale, and resized to 48×48 pixels
3. **CNN Prediction** — The model outputs a probability for each of the 7 emotions
4. **Display** — A colored bounding box and emotion label are drawn on the frame

---

## Model Architecture

```
Input (48×48 grayscale)
    ↓
Conv Block 1 — 32 filters
    ↓
Conv Block 2 — 64 filters
    ↓
Conv Block 3 — 128 filters
    ↓
Dense (256) → Softmax (7 emotions)
```

Each conv block uses: `Conv2D → BatchNorm → Conv2D → MaxPool → Dropout`

---

## Expected Accuracy

- **~62–68%** on the FER-2013 test set
- This is normal — FER-2013 is a challenging dataset even for advanced models
- Real-world performance on a clear webcam feed is typically better

---

## Requirements

- Python 3.8+
- Webcam (for real-time detection)
- GPU recommended for training (CPU works but is slower)

---

## Tech Stack

- **Python** — core language
- **TensorFlow / Keras** — model building and training
- **OpenCV** — face detection and webcam feed
- **NumPy** — numerical operations
- **Matplotlib / Seaborn** — charts and confusion matrix
- **Scikit-learn** — evaluation metrics
