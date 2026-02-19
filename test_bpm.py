import librosa
import numpy as np
import os
import matplotlib.pyplot as plt

# Path to a sample file
file_path = os.path.join('HS', 'HS', 'F_N_RC.wav')

if not os.path.exists(file_path):
    # Try finding any wav file
    print(f"File {file_path} not found. Searching for others...")
    for root, dirs, files in os.walk('HS'):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                break
        if file_path.endswith('.wav'): 
            break

print(f"Analyzing {file_path}...")

try:
    y, sr = librosa.load(file_path, duration=10)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    print(f"Estimated Tempo (BPM): {tempo}")
except Exception as e:
    print(f"Error: {e}")
