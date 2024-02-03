import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio file
y, sr = librosa.load('A_acousticguitar_Mari_1.wav')

# Extract chroma features
chromagram = librosa.feature.chroma_stft(y=y, sr=sr , hop_length=512)
print(chromagram)
print(chromagram.shape)

# Simple chord recognition example (just selecting the most prominent chroma)
chroma_max = np.argmax(chromagram, axis=0)
# Assuming 12-note chroma representation
chord_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
chords = [chord_labels[note] for note in chroma_max]

print(chords)

# Visualize chromagram
plt.figure(figsize=(10, 4))
plt.imshow(chromagram, cmap='coolwarm', origin='lower', aspect='auto', extent=[0, len(y)/sr, 0, 12])
plt.title('Chromagram')
plt.xlabel('Time (s)')
plt.ylabel('Pitch Class')
plt.yticks(range(12), ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.show()
