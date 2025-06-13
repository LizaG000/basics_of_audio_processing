import librosa
import shutil
import time
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import speech_recognition  as sr
from pydub import AudioSegment


def create_audio_copy(destination_file):
    audio = 'Monetochka_-_Padat_v_gryaz_64293978.wav'
    destination_file += ".wav"
    try:
        shutil.copy(audio, destination_file)
    except FileNotFoundError:
        print(f"Файл не найден.")
    except Exception as e:
        print(f"Ошибка при копировании файла: {e}")



audio = 'Monetochka_-_Padat_v_gryaz_64293978.wav'
waveform, sample_rate = librosa.load(audio, sr=None)
sd.play(waveform, sample_rate)
time.sleep(5.0)

# Визуализация звуковой волны
fig = plt.figure(figsize=(14, 5))
librosa.display.waveshow(waveform, sr=sample_rate)
plt.title('Звуковая волна')
plt.xlabel('Время (сек)')
plt.ylabel('Амплитуда')
fig.savefig('звуковая волна.png')
plt.close(fig)


# разделение гармонических и ударных
waveform, sample_rate = librosa.load(audio, sr=None)
y_harmonic, y_percussive = librosa.effects.hpss(waveform)
fig = plt.figure(figsize=(15, 5))
librosa.display.waveshow(y_harmonic, sr=sample_rate, alpha=1)
librosa.display.waveshow(y_percussive, sr=sample_rate, color='r', alpha=0.5)
plt.title('Гармонические звуки (синий) и ударные звуки (красный)')
plt.xlabel('Время (сек)')
fig.savefig('разделение гармонических и ударных.png')
plt.close(fig)

# гармонические звуки
create_audio_copy("гармонические звуки")
harmonic = 'гармонические звуки.wav'
sf.write(harmonic, y_harmonic, sample_rate)
waveform, sample_rate = librosa.load(harmonic, sr=None)
sd.play(waveform, sample_rate)
time.sleep(5.0)
# ударные звуки
create_audio_copy("ударные звуки")
percussive = 'ударные звуки.wav'
sf.write(percussive, y_percussive, sample_rate)
waveform, sample_rate = librosa.load(percussive, sr=None)
sd.play(waveform, sample_rate)
time.sleep(5.0)

#  Создание тона частотой 440 Гц.
tone_file = "Создание тона частотой 440 Гц.wav"
waveform, sample_rate = librosa.load(tone_file, sr=None)
duration = 5
frequency = 440
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
tone = np.sin(2 * np.pi * frequency * t)

sf.write(tone_file, tone, sample_rate)
sf.write(tone_file, tone, sample_rate)
waveform, sample_rate = librosa.load(tone_file, sr=None)
sd.play(waveform, sample_rate)
time.sleep(5.0)

# ДО РЕ МИ ФА СО ЛЯ СИ
notes_freqs = {
    "C": 261.63,
    "D": 293.66,
    "E": 329.63,
    "F": 349.23,
    "G": 392.00,
    "A": 440.00,
    "H": 493.88
}
sample_rate = 44100
note_duration = 1.0 

notes_sequence = ["C", "D", "E", "F", "G", "A", "H", "C"]
music = np.array([], dtype=np.float32)

for note in notes_sequence:
    frequency = notes_freqs[note]
    t = np.linspace(0, note_duration, int(sample_rate * note_duration), endpoint=False)
    tone = np.sin(2 * np.pi * frequency * t)
    music = np.append(music, tone)

filename = "ДО РЕ МИ ФА СО ЛЯ СИ.wav"
sf.write(filename, music, sample_rate)

waveform, sr = librosa.load(filename, sr=None)
sd.play(waveform, sr)
sd.wait()

# спектрограмма
audio = 'Monetochka_-_Padat_v_gryaz_64293978.wav'
y, sr = librosa.load(audio, sr=None)
X = librosa.stft(y)
Xdb = librosa.amplitude_to_db(abs(X))
fig = plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
plt.title('Спектрограмма')
plt.xlabel('Время (сек)')
plt.ylabel('Частота (Гц)')
fig.savefig("спектрограмма.png")
plt.close(fig)
#распознование речи

r = sr.Recognizer()
audio_file = "converted_mono.wav"  # Преобразованный файл

try:
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
        print("Audio записан")
        text = r.recognize_google(audio, language="ru-RU")
        print("Распознанный текст: " + text)
except sr.UnknownValueError:
    print("Google Speech Recognition не смог распознать аудио.")
except sr.RequestError as e:
    print(f"Ошибка сервиса Google Speech Recognition: {e}")