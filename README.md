# Signwave 🤟

**Signwave** is a deep learning-powered system that translates Indian Sign Language (ISL) sentence videos into text and audio, bridging the communication gap for hearing and speech-impaired individuals.
---

## What It Does

A user uploads or records a video of an ISL sentence being signed. Signwave processes the video frame by frame, classifies each frame using a trained CNN-ViT model, determines the most likely sentence via majority voting, and returns both the predicted text and an audio playback of that sentence.

---

## Features

- **ISL Video-to-Text** — Classifies sign language videos into one of 65 sentence classes
- **Text-to-Speech** — Converts the predicted sentence to audio using gTTS
- **Frame-level Inference** — Processes up to 450 frames per video, picks the majority prediction
- **Web Interface** — Simple browser-based UI (`signwave.html`) to upload videos and view results
- **REST API** — Flask backend with `/predict` and `/play-audio` endpoints

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Python, Flask |
| Deep Learning | TensorFlow / Keras (CNN + Vision Transformer) |
| Video Processing | OpenCV |
| Text-to-Speech | gTTS (Google Text-to-Speech) |
| Frontend | HTML/CSS/JS |
| Model Training | Jupyter Notebook (`cnn_vit_model.ipynb`) |
| Preprocessing | Jupyter Notebook (`preprocessing.ipynb`) |

---

## Project Structure

```
Signwave/
├── app.py                    # Flask backend — prediction and audio endpoints
├── model.h5                  # Trained CNN-ViT model weights
├── cnn_vit_model.ipynb       # Model architecture and training notebook
├── preprocessing.ipynb       # Dataset preprocessing notebook
├── signwave.html             # Web UI for video upload and result display
└── uploads/                  # Temp folder for uploaded videos (auto-created)
```

---

## Supported Sign Classes (65 Sentences)

The model classifies ISL videos into 65 common conversational sentences, including:

- Greetings: `Good Morning`, `Good Afternoon`, `Hello`, `How are you`
- Expressions: `I am very happy`, `I am tired`, `I am hungry`, `I am crying`
- Questions: `What are you doing`, `Who are you`, `Where are you from`, `How old are you`
- Responses: `Thank you`, `You are welcome`, `I do not agree`, `I promise`
- And 50+ more everyday conversational phrases

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/prakrutikalagudi/Signwave-.git
cd Signwave-
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install flask tensorflow opencv-python numpy gtts
```

### 4. Update the Model Path

In `app.py`, update the model path to point to your local `model.h5`:

```python
model = load_model("path/to/your/model.h5")
```

### 5. Run the App

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

---

## How It Works

```
Input Video
     │
     ▼
Extract Frames (up to 450)
     │
     ▼
Preprocess Each Frame
  → Grayscale conversion
  → Resize to 224×224
  → Normalize to [0, 1]
     │
     ▼
CNN-ViT Model → Per-frame class prediction
     │
     ▼
Majority Voting → Final predicted sentence
     │
     ├──► Return text to frontend
     └──► gTTS → Generate MP3 audio
```

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serve the web UI |
| `POST` | `/predict` | Upload a video, get predicted text + audio path |
| `GET` | `/play-audio?audio_path=<path>` | Stream the generated audio file |

### `/predict` Request & Response

```json
// Request: multipart/form-data with 'file' (video)
// Response:
{
  "prediction": "How are you",
  "audio": "/tmp/tmpXYZ.mp3"
}
```

---

## Model Details

The model is a hybrid **CNN + Vision Transformer (ViT)** architecture trained on ISL sentence videos:

- **Input:** Grayscale video frames, resized to `224×224`
- **Architecture:** CNN layers for local feature extraction + ViT attention for temporal/spatial context
- **Output:** Softmax over 65 ISL sentence classes
- **Inference strategy:** Frame-level predictions aggregated by majority vote

Training and architecture details are in `cnn_vit_model.ipynb`. Preprocessing steps (frame extraction, normalization, dataset splits) are in `preprocessing.ipynb`.

---

## Limitations & Future Scope

- Currently supports 65 fixed sentence classes — expanding vocabulary is a future goal
- Model path is hardcoded; should be made configurable via environment variable
- Real-time webcam streaming inference not yet implemented
- No authentication or rate limiting on the prediction endpoint
---

## License

This project is for academic and research use. Please cite the associated IEEE paper if you use this work.
