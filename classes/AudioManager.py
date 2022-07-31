# AudioManager.py
# Title: AudioManager
# Description: AudioManager to control devices and load and save audio files

# import libraries
import sounddevice as sd
import soundfile as sf
import numpy as np
import os

# define the audio file manager class
class AudioFileManager:
  def __init__(self) -> None:
    pass

  # scan directory for audio files
  def scan_directory(self, directory: str, file_types: list = ['wav']) -> list:
    audio_files = []
    for file in os.listdir(directory):
      if file.endswith(tuple(file_types)):
        audio_files.append(file)
    return audio_files
  
  # load an audio file from disk
  def load_audio_file(self, file_path: str) -> np.ndarray:
    data = sf.read(file_path, dtype='float32', always_2d=True)
    return np.array(data)

  # load all audio files from a directory
  def load_all_audio_files(self, directory: str, file_types: list = None) -> np.ndarray:
    # if no file types are specified don't pass an argument for file_types
    audio_files = self.scan_directory(directory, file_types)
    audio_data = []
    for file in audio_files:
      audio_data.append(self.load_audio_file(os.path.join(directory, file)))   

    return np.array(audio_data)

  # load a random audio file from a directory
  def load_random_audio_file(self, directory: str, file_types: list = None) -> np.ndarray:
    audio_files = self.scan_directory(directory, file_types)
    file = audio_files[np.random.randint(0, len(audio_files))]
    return self.load_audio_file(os.path.join(directory, file))

  # load random files from a directory
  def load_random_audio_files(self, directory: str, n_files: int, file_types: list = None) -> np.ndarray:
    audio_files_array = []
    for file in range(n_files):
      random_file = self.load_random_audio_file(directory, file_types)
      audio_files_array.append(random_file)
    return np.array(audio_files_array)
        

  # recalculate audio from float32 to pcm_16 and save it to disk
  def save_audio_file(self, audio: np.ndarray, file_path: str, sample_rate: int = 44100) -> None:
    sf.write(file_path, audio, sample_rate, 'FLOAT')
    return None

  # save all audio files to a directory
  def save_all_audio_files(self, audio: np.ndarray, directory: str, sample_rate: int = 44100, file_type: str = 'wav', file_name: str = 'file_') -> None:
    for i in range(len(audio)):
      self.save_audio_file(audio[i], os.path.join(directory, file_name + str(i) + '.' + file_type), sample_rate)
    return None
  
  # pick n seconds from a random time in the loaded audio file
  def pick_random_time(self, audio: np.ndarray, duration: int, sample_rate: int) -> np.ndarray:
    if (len(audio) < (duration * sample_rate)):
      return None
    start_time = np.random.randint(0, len(audio) - (duration * sample_rate))
    end_time = start_time + (duration * sample_rate)
    return audio[start_time:end_time]

  # pick 'duration' seconds from a random time in 'n_files' random files from a directory
  def load_random_times_from_random_files_in_a_directory(self, directory: str, duration: int, sample_rate: int, n_files: int, file_types: list = None) -> np.ndarray:
    new_audio=[]
    audio_files_array = self.load_random_audio_files(directory, n_files, file_types)
    for audio_file_index in range(len(audio_files_array)):
      if (len(audio_files_array[audio_file_index]) >= (duration * sample_rate)):
        picked = self.pick_random_time(audio_files_array[audio_file_index], duration, sample_rate)
        new_audio.append(picked)
    audio_array = np.array(new_audio)
    return audio_array


    return self.pick_random_time(np.array(audio_data), duration, sample_rate)
# define the audio device manager class
class AudioDeviceManager:
  def __init__(self) -> None:
    pass

  # print the available devices
  def print_devices(self) -> None:
    for i, device in enumerate(sd.query_devices()):
      print(f'{i}: {device["name"]}')
    pass

  # select a device
  def select_device(self, device_index: int) -> None:
    sd.default.device = device_index
    print(f'Selected device: {sd.query_devices()[device_index]["name"]}')
    pass

  # check if the device is available
  def is_device_selected(self) -> bool:
    return sd.default.device is not None

  # record from the device
  def record_from_device(self, duration: float, sample_rate: int = 44100, blocking=True, out=None) -> np.ndarray:
    return sd.rec(int(duration * sample_rate), channels=1, dtype='float32', blocking=blocking, out=out)

  def wait_for_recording(self) -> np.ndarray:
    return sd.wait()

# define the audio format manager class
class AudioFormatManager:
  def __init__(self) -> None:
    pass

  # convert an audio file to an image representation in float32
  def convert_audio_file_to_image(self, audio: np.ndarray, width: int, height: int) -> np.ndarray:
    # reshape the audio file to a 3D array
    image = audio.reshape(1, width, height)

    # convert the image to float32
    image = image.astype(np.float32)

    # recalculate from pcm_16 to float32
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    return image

  # convert multiple audio files to an image representation in float32
  def convert_multiple_audio_files_to_images(self, audio: np.ndarray, n_channels: int, width: int, height: int) -> np.ndarray:
    # reshape the audio file to a 4D array
    image = audio.reshape(len(audio), n_channels, width, height)

    return image

  # convert a float32 image representation to a pcm_16 audio file
  def convert_image_to_audio_file(self, image: np.ndarray) -> np.ndarray:
    # reshape the image to a 1D array
    audio = image.reshape(-1, 1)

    return audio

  # convert multiple float32 image representations to audio files
  def convert_multiple_images_to_audio_files(self, image_array: np.ndarray) -> np.ndarray:
    # reshape the image to a 3D array
    audio_files = []
    for image in image_array:
      audio_files.append(self.convert_image_to_audio_file(image))

    return np.array(audio_files)
    