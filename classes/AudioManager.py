# import libraries
import sounddevice as sd
import soundfile as sf
import numpy as np

# define the audio file manager class
class AudioFileManager:
  def __init__(self) -> None:
    pass
  
  def load_audio_file(self, file_path: str) -> np.ndarray:
    data = sf.read(file_path, dtype='float32', always_2d=True)
    return np.array(data)        

  def save_audio_file(self, audio: np.ndarray, file_path: str, sample_rate: int = 44100) -> None:
    sf.write(file_path, audio, sample_rate, 'FLOAT')
    return None

# define the audio device manager class
class AudioDeviceManager:
  def __init__(self) -> None:
    pass

  def print_devices(self) -> None:
    for i, device in enumerate(sd.query_devices()):
      print(f'{i}: {device["name"]}')
    pass
