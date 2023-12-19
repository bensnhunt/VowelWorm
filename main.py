import pyaudio
import numpy as np
import librosa
import os
import pygame
from pygame.locals import QUIT
from sklearn.ensemble import RandomForestRegressor
from collections import deque

MASTER = 2

# Constants
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050
CHUNK = int(4096 * (1/MASTER))
FFT_SIZE = int(2048 * (1/MASTER))
MEDIA_FOLDER = "media" 

# Pygame setup
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

def compute_mfcc(audio_data):

    mfccs = librosa.feature.mfcc(y=audio_data, sr=RATE, n_fft=FFT_SIZE, hop_length=CHUNK // 2, n_mfcc=13)

    mfccs = np.mean(mfccs, axis=1)

    return mfccs

def load_data():

    vowels = ['a', 'e', 'i', 'o', 'u']

    X, y = [], []

    for vowel in vowels:

        vowel_path = os.path.join(MEDIA_FOLDER, vowel)

        mp3_files = [f for f in os.listdir(vowel_path) if f.endswith('.mp3')]

        for mp3_file in mp3_files:

            file_path = os.path.join(vowel_path, mp3_file)

            audio_data, _ = librosa.load(file_path, sr=RATE)

            mfccs = compute_mfcc(audio_data)

            label = [0.0, 0.0, 0.0]

            # openness
            if vowel == 'a':
                label[0] = 1.0
            elif vowel in ['i', 'u']:
                label[0] = -1.0
            
            # frontness
            if vowel in ['e', 'i']:
                label[1] = 1.0
            elif vowel in ['o','u']:
                label[1] = -1.0
            
            # roundedness
            if vowel in ['o', 'u']:
                label[2] = 1.0

            X.append(mfccs)
            y.append(label)

    return np.array(X), np.array(y)

def train_model(X, y):
    """
    X: MFCC frames
    X = [[-4.5622260e+02  8.5003677e+01 -1.0475142e+01 ...  2.9455202e+01 -1.4943029e+01  2.5797539e+00] ... [-5.1069302e+02  4.5632874e+01  1.5567463e+01 ... -3.0503311e+00 -1.6901724e+00 -2.5506253e+00]]
    
    y: Vowel features i.e. [openness, frontness, roundedness] for each vowel
    y = [[ 1.  0.  0.] ... [-1. -1.  1.]]
    """
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X, y)
    return rf

# Create a queue to store previous positions and their colors
positions_queue = deque(maxlen=10)

def plot_vowel_prediction(prediction):
    openness, frontness, roundedness = prediction

    screen.fill((255, 255, 255))

    # print vowels
    vowels = {'a': (1, 0), 
              'e': (0, 1), 
              'i': (-1, 1), 
              'o': (0, -1), 
              'u': (-1, -1)}

    for i in vowels:
        x = vowels[i][1] * .7
        x = WINDOW_WIDTH - int((x + 1) * WINDOW_WIDTH / 2)
        y = vowels[i][0] * .7
        y = int((y + 1) * WINDOW_HEIGHT / 2)
        coords = (x, y)
        text_surface = font.render(i, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=coords)
        screen.blit(text_surface, text_rect)


    # Map the prediction values to screen coordinates
    mapped_frontness = WINDOW_WIDTH - int((frontness + 1) * WINDOW_WIDTH / 2)
    mapped_openness = int((openness + 1) * WINDOW_HEIGHT / 2)

    # Append current position and color to the queue
    positions_queue.append((mapped_frontness, mapped_openness))

    for n, (x, y) in enumerate(positions_queue):
        alpha = int(255 - (25.5*n))
        pygame.draw.circle(screen, (alpha, alpha, alpha), (x, y), 30)

    pygame.display.update()

    return

def predict_vowel(model):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Listening for vowels... Press ^C to stop.")

    try:

        while True:
            audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
            mfccs = compute_mfcc(audio_data)
            prediction = model.predict([mfccs])[0]
            print(prediction)
            plot_vowel_prediction(prediction)
    
    except KeyboardInterrupt:
        print("Stopped listening.")
        stream.stop_stream()
        stream.close()
        audio.terminate()

def main():
    X, y = load_data()
    model = train_model(X, y)
    global plot_function
    plot_function = plot_vowel_prediction([0, 0, 0])  # Initial placeholder prediction
    predict_vowel(model)

if __name__ == "__main__":
    main()
