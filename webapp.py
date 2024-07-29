import streamlit as st
import numpy as np
import librosa
import sounddevice as sd
import queue
import threading
import soundfile as sf
from keras.models import load_model
import tempfile
from pydub import AudioSegment

# Load the trained RNN model for emotion detection
emotion_model = load_model('trained_rnn_model01.h5')

# Load the gender detection model
gender_model = load_model('genderDetection02.h5')

# Define global variables
q = queue.Queue()
recording = False

# Function to extract features for the gender model
def extract_gender_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    features = {
        'meanfreq': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'sd': np.std(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'median': np.median(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'Q25': np.percentile(librosa.feature.spectral_centroid(y=y, sr=sr), 25),
        'Q75': np.percentile(librosa.feature.spectral_centroid(y=y, sr=sr), 75),
        'IQR': np.percentile(librosa.feature.spectral_centroid(y=y, sr=sr), 75) - np.percentile(librosa.feature.spectral_centroid(y=y, sr=sr), 25),
        'skew': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'kurt': np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'sp.ent': np.mean(librosa.feature.spectral_flatness(y=y)),
        'sfm': np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)),
        'mode': np.median(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'meanfun': np.mean(librosa.feature.rms(y=y)),
        'minfun': np.min(librosa.feature.rms(y=y)),
        'maxfun': np.max(librosa.feature.rms(y=y)),
        'meandom': np.mean(librosa.feature.delta(y)),
        'mindom': np.min(librosa.feature.delta(y)),
        'maxdom': np.max(librosa.feature.delta(y)),
        'dfrange': np.max(librosa.feature.delta(y)) - np.min(librosa.feature.delta(y)),
        'modindx': np.std(librosa.feature.delta(y))
    }

    return np.array(list(features.values())).reshape(1, -1)

# Function to extract features for the emotion model
def extract_emotion_features(audio_path):
    audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

# Function to determine the gender of the voice
def detect_gender(audio_path):
    features = extract_gender_features(audio_path)
    prediction = gender_model.predict(features)
    gender_label = np.argmax(prediction)
    genders = ['Male', 'Female']
    return genders[gender_label]

# Function to detect emotion
def detect_emotion(features):
    features = np.expand_dims(features, axis=0)
    prediction = emotion_model.predict(features)
    emotion_label = np.argmax(prediction)
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    return emotions[emotion_label]

# Function to convert audio to wav format
def convert_audio_to_wav(uploaded_file, output_path):
    audio = AudioSegment.from_file(uploaded_file)
    audio.export(output_path, format='wav')

# Function to upload and process audio file
def upload_file():
    uploaded_file = st.file_uploader("Upload Voice Note", type=["wav", "mp3", "opus"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file_path = temp_file.name
            if uploaded_file.type == "audio/opus":
                convert_audio_to_wav(uploaded_file, temp_file_path)
            else:
                temp_file.write(uploaded_file.read())
        return temp_file_path
    return None

# Function to start recording
def start_recording():
    global recording
    recording = True
    t = threading.Thread(target=record_audio)
    t.start()
    st.experimental_rerun()

# Function to stop recording
def stop_recording():
    global recording
    recording = False
    st.experimental_rerun()

# Function to record audio
def record_audio():
    with sf.SoundFile('recorded_audio.wav', mode='w', samplerate=44100, channels=1) as file:
        with sd.InputStream(callback=callback, samplerate=44100, channels=1):
            while recording:
                if not q.empty():
                    data = q.get()
                    file.write(data)

# Function to callback audio data
def callback(indata, frames, time, status):
    if status:
        st.warning(f"Error during recording: {status}")
    q.put(indata.copy())

# Function to process recorded audio
def process_recorded_audio():
    try:
        audio_path = 'recorded_audio.wav'
        return audio_path
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# Function to predict emotion and gender
def predict_emotion_and_gender(audio_path):
    gender = detect_gender(audio_path)
    gender_color = "blue" if gender == "Male" else "pink"
    st.markdown(f'<p style="color:{gender_color}; font-size: 20px">Detected Gender: {gender}</p>', unsafe_allow_html=True)

    features = extract_emotion_features(audio_path)
    emotion = detect_emotion(features)
    emotion_colors = {
        'Angry': 'red',
        'Disgust': 'green',
        'Fear': 'purple',
        'Happy': 'yellow',
        'Neutral': 'gray',
        'Sad': 'blue',
        'Surprise': 'orange'
    }
    st.markdown(f'<p style="color:{emotion_colors[emotion]}; font-size: 20px">The detected emotion is: {emotion}</p>', unsafe_allow_html=True)

# Streamlit app
st.title("Emotion Detection through Voice")

# Upload button
uploaded_audio_path = upload_file()

# Start Recording button
if st.button("Start Recording"):
    start_recording()

# Stop Recording button
if st.button("Stop Recording"):
    stop_recording()

# Process Recorded Audio button
if st.button("Process Recorded Audio"):
    recorded_audio_path = process_recorded_audio()
    if recorded_audio_path:
        st.write("Audio recorded successfully.")

# Predict button
if st.button("Predict"):
    if uploaded_audio_path:
        predict_emotion_and_gender(uploaded_audio_path)
    elif 'recorded_audio_path' in locals():
        predict_emotion_and_gender(recorded_audio_path)
    else:
        st.error("Please upload or record an audio first.")
