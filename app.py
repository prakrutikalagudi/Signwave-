'''from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter

app = Flask(__name__)

# Load your trained model (update the path to your actual model file)
model = load_model("D:/Capstone Project/New folder/model.h5")

# Define your class labels
class_labels = {
    0: "Good Morning",
    1: "Good Afternoon",
    2: "Hello",
    3: "How are you",
    4: "Thank you",
    5: "Are you free today",
    6: "Are you hiding something",
    7: "Help me",
    8: "How are things",
    9: "How can I help you",
    10: "How can I trust you",
    11: "How dare you",
    12: "How old are you",
    13: "I am (age)",
    14: "I am afraid of that",
    15: "I am crying",
    16: "I am feeling bored",
    17: "I am feeling cold",
    18: "I am fine. Thank you sir",
    19: "I am hungry",
    20: "I am in dilemma what to do",
    21: "I am not really sure",
    22: "I am really grateful",
    23: "I am sitting in the class",
    24: "I am so sorry to hear that",
    25: "I am suffering from fever",
    26: "I am tired",
    27: "I am very happy",
    28: "I can not help you there",
    29: "I do not agree",
    30: "I do not like it",
    31: "I do not mean it",
    32: "I don't agree",
    33: "I enjoyed a lot",
    34: "I got hurt",
    35: "I like you, I love you",
    36: "I need water",
    37: "I promise",
    38: "I really appreciate it",
    39: "I somehow got to know about it",
    40: "I was stopped by someone",
    41: "It does not make any difference to me",
    42: "It was nice chatting with you",
    43: "Let him take time",
    44: "My name is XXXXXXXX",
    45: "What are you doing",
    46: "What did you tell him",
    47: "What do you do",
    48: "What do you think",
    49: "What do you want to become",
    50: "What happened",
    51: "What have you planned for your career",
    52: "What is your phone number",
    53: "What do you want",
    54: "When will the train leave",
    55: "Where are you from",
    56: "Which college/school are you from",
    57: "Who are you",
    58: "Why are you angry",
    59: "Why are you crying",
    60: "Why are you disappointed",
    61: "You are bad",
    62: "You are good",
    63: "You are welcome",
    64: "You can do it"
}

def preprocess_frame(frame, target_size=(224, 224)):
    # Convert to grayscale if your model expects it, or remove if not
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, target_size)
    frame = frame.astype("float32") / 255.0
    frame = np.expand_dims(frame, axis=-1)  # For grayscale channel
    return frame

def extract_frames(video_path, max_frames=450):  
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = preprocess_frame(frame)
        frames.append(frame)
    cap.release()
    return np.array(frames)

def predict_video(video_path):
    frames = extract_frames(video_path)
    predictions = []
    for frame in frames:
        frame_expanded = np.expand_dims(frame, axis=0)  # batch dimension
        pred = model.predict(frame_expanded)
        pred_class = np.argmax(pred, axis=1)[0]
        predictions.append(pred_class)
    most_common_class = Counter(predictions).most_common(1)[0][0]
    return class_labels.get(most_common_class, "Unknown")

@app.route('/')
def index():
    return render_template('signwave.html')  # Your HTML with webcam and buttons

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    os.makedirs('uploads', exist_ok=True)
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    try:
        predicted_class = predict_video(file_path)
        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run()  '''

from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter
from gtts import gTTS
import tempfile

app = Flask(__name__)

# Load your trained model (update the path to your actual model file)
model = load_model("D:/Capstone Project/New folder/model.h5")

# Define your class labels
class_labels = {
    0: "Good Morning",
    1: "Good Afternoon",
    2: "Hello",
    3: "How are you",
    4: "Thank you",
    5: "Are you free today",
    6: "Are you hiding something",
    7: "Help me",
    8: "How are things",
    9: "How can I help you",
    10: "How can I trust you",
    11: "How dare you",
    12: "How old are you",
    13: "I am (age)",
    14: "I am afraid of that",
    15: "I am crying",
    16: "I am feeling bored",
    17: "I am feeling cold",
    18: "I am fine. Thank you sir",
    19: "I am hungry",
    20: "I am in dilemma what to do",
    21: "I am not really sure",
    22: "I am really grateful",
    23: "I am sitting in the class",
    24: "I am so sorry to hear that",
    25: "I am suffering from fever",
    26: "I am tired",
    27: "I am very happy",
    28: "I can not help you there",
    29: "I do not agree",
    30: "I do not like it",
    31: "I do not mean it",
    32: "I don't agree",
    33: "I enjoyed a lot",
    34: "I got hurt",
    35: "I like you, I love you",
    36: "I need water",
    37: "I promise",
    38: "I really appreciate it",
    39: "I somehow got to know about it",
    40: "I was stopped by someone",
    41: "It does not make any difference to me",
    42: "It was nice chatting with you",
    43: "Let him take time",
    44: "My name is XXXXXXXX",
    45: "What are you doing",
    46: "What did you tell him",
    47: "What do you do",
    48: "What do you think",
    49: "What do you want to become",
    50: "What happened",
    51: "What have you planned for your career",
    52: "What is your phone number",
    53: "What do you want",
    54: "When will the train leave",
    55: "Where are you from",
    56: "Which college/school are you from",
    57: "Who are you",
    58: "Why are you angry",
    59: "Why are you crying",
    60: "Why are you disappointed",
    61: "You are bad",
    62: "You are good",
    63: "You are welcome",
    64: "You can do it"
}

def preprocess_frame(frame, target_size=(224, 224)):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, target_size)
    frame = frame.astype("float32") / 255.0
    frame = np.expand_dims(frame, axis=-1)  # For grayscale channel
    return frame

def extract_frames(video_path, max_frames=450):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = preprocess_frame(frame)
        frames.append(frame)
    cap.release()
    return np.array(frames)

def predict_video(video_path):
    frames = extract_frames(video_path)
    predictions = []
    for frame in frames:
        frame_expanded = np.expand_dims(frame, axis=0)  # batch dimension
        pred = model.predict(frame_expanded)
        pred_class = np.argmax(pred, axis=1)[0]
        predictions.append(pred_class)
    most_common_class = Counter(predictions).most_common(1)[0][0]
    return class_labels.get(most_common_class, "Unknown")

def convert_text_to_speech(text):
    tts = gTTS(text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file.close()
    tts.save(temp_file.name)
    return temp_file.name

@app.route('/')
def index():
    return render_template('signwave.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    os.makedirs('uploads', exist_ok=True)
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    try:
        predicted_class = predict_video(file_path)
        audio_path = convert_text_to_speech(predicted_class)
        return jsonify({'prediction': predicted_class, 'audio': audio_path})
    except Exception as e:
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/play-audio', methods=['GET'])
def play_audio():
    audio_path = request.args.get('audio_path')
    if audio_path and os.path.exists(audio_path):
        return send_file(audio_path, as_attachment=True)
    return jsonify({'error': 'Audio file not found'}), 404

if __name__ == '__main__':
    app.run()
