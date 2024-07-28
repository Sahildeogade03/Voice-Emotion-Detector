# Voice-Emotion-Detector

## Project Overview
This project aims to detect emotions through voice analysis. It leverages deep learning models to identify both the gender of the speaker and the emotion conveyed in the audio. The project supports both uploaded audio files and real-time audio recording.

## Features
- Detects gender of the speaker (Male/Female).
- Identifies emotions such as Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
- Supports audio file uploads in various formats (wav, mp3, opus).
- Real-time audio recording and analysis.
- Color-coded results for easy interpretation.

## Datasets Used
- **CREMA-D**: The Crowd-sourced Emotional Multimodal Actors Dataset.
- **TESS**: Toronto Emotional Speech Set.
- **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song.

These datasets were combined and augmented to increase the diversity and size of the training set, ensuring better model performance.

## Output
Here is an example of the output displayed by the application:

![Screenshot 2024-07-28 114419](https://github.com/user-attachments/assets/987bbe96-f957-4482-87e5-626f20f5eb7c)

## Challenges
- **Dataset Size**: The initial dataset size was small. To address this, three datasets were combined, and data augmentation techniques were applied.
- **Gender Classification**: Developing a robust gender classification model required multiple iterations and fine-tuning to achieve satisfactory results.
