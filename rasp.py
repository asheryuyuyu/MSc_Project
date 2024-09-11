import sounddevice as sd
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, Resample
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
import tkinter as tk
from gpiozero import LED
import threading
import signal
import sys
import time

led_green = LED(27, initial_value=False)
led_blue = LED(22, initial_value=False)

for _ in range(3):
    led_green.on()
    led_blue.off()
    time.sleep(1)

    led_green.off()
    led_blue.on()
    time.sleep(1)

led_green.off()
led_blue.off()

print('IMPORT SUCCESSFULLY')

class CNN(nn.Module):
    def __init__(self, class_dim=2):
        super(CNN, self).__init__()

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.depthwise_conv1 = nn.Conv2d(1, 1 * 16, kernel_size=7, padding=3, stride=1, dilation=1, groups=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=7, padding=3, stride=1, dilation=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=(5, 5), stride=(5, 5))
        self.dropout1 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=7, padding=3, stride=1, dilation=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 10), stride=(4, 10))
        self.dropout2 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(32 * 2 * 1, 100)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(100, class_dim)

    def forward(self, x):
        x = self.quant(x)
        x = self.depthwise_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # flatten
        x = x.view(x.size(0), -1)

        # full connect
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        x = self.dequant(x)
        return x


# convert to log_mel
def extract_log_mel(audio_chunk):
    # convert to tensor
    audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)
    # calculate mel spectrogram
    mel_spec = mel_spectrogram(audio_tensor)
    # convert to log scale
    log_mel_spec = db_transform(mel_spec)
    return log_mel_spec


def detect_drone(log_mel_spec):
    # change size for model input
    log_mel_spec = log_mel_spec.unsqueeze(0)  # add batch dimension
    log_mel_spec = log_mel_spec.unsqueeze(0)  # add channel dimension

    with torch.no_grad():
        output = model(log_mel_spec)
    probabilities = output.squeeze().numpy()

    drone_probability = probabilities[1]  # 1 represents drone presence
    if drone_probability > 0.5:
        return True, drone_probability
    else:
        return False, drone_probability


# realtime audio processing callback
def audio_callback(indata, frames, time, status, factor=1):
    if status:
        print(status, flush=True)
    audio_chunk = indata[:, 0] * factor  # use the first channel

    # Range normalization to [-1, 1]
    audio_chunk = 2.0 * (audio_chunk - np.min(audio_chunk)) / (np.max(audio_chunk) - np.min(audio_chunk)) - 1.0

    # resample audio from 44100Hz to 16000Hz
    resampled_audio = resampler(torch.tensor(audio_chunk, dtype=torch.float32)).numpy()

    # ensure the audio chunk is of correct length
    if len(resampled_audio) == int(sample_rate * block_duration):
        log_mel_spec = extract_log_mel(resampled_audio)
        drone_detected, drone_probability = detect_drone(log_mel_spec)
        update_led(drone_detected, drone_probability)


led = LED(17, initial_value=False)


def update_led(drone_detected, drone_probability):
    if drone_detected:
        led.on()
        print('drone')
    else:
        led.off()
        print('no')
    print(f'{drone_probability}')


# main program execution starts here

# load the pre-trained model
model = CNN(class_dim=2)
model.load_state_dict(
    torch.load('/path/to/your/model', map_location=torch.device('cpu')))
model.eval()

# mel spectrogram settings
mel_spectrogram = MelSpectrogram(sample_rate=16000, n_mels=40, win_length=640, n_fft=1024, hop_length=320)
db_transform = AmplitudeToDB()

# initialize resampler (from 44100Hz to 16000Hz)
resampler = Resample(orig_freq=44100, new_freq=16000)

# audio parameters
sample_rate = 16000
block_duration = 1

# start the audio stream
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=44100, blocksize=int(44100 * block_duration))
stream.start()

print("START DETECTING...")


def signal_handler(sig, frame):
    print("STOPPING...")
    stream.stop()
    stream.close()
    led.close()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    while True:
        pass
except KeyboardInterrupt:
    signal_handler(None, None)
