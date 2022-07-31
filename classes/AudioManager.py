# AudioManager.py
# Title: AudioManager
# Description: AudioManager to control devices and load and save audio files

# import libraries
import sounddevice as sd
import soundfile as sf
import numpy as np

# define the audio file manager class
class AudioFileManager:
  def __init__(self) -> None:
    pass
  
  # load an audio file from disk
  def load_audio_file(self, file_path: str) -> np.ndarray:
    data = sf.read(file_path, dtype='float32', always_2d=True)
    return np.array(data)        

  # recalculate audio from float32 to pcm_16 and save it to disk
  def save_audio_file(self, audio: np.ndarray, file_path: str, sample_rate: int = 44100) -> None:
    sf.write(file_path, audio, sample_rate, 'FLOAT')
    return None

# define the audio device manager class
class AudioDeviceManager:
  def __init__(self) -> None:
    pass

  # print the available devices
  def print_devices(self) -> None:
    for i, device in enumerate(sd.query_devices()):
      print(f'{i}: {device["name"]}')
    pass