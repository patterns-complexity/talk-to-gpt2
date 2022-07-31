import numpy as np
from multiprocessing import Process, Queue
from sounddevice import InputStream
from torch import device, Tensor
from torch.cuda import is_available as is_cuda_available
import pyttsx3

from classes.AudioManager import AudioDeviceManager as adm, AudioFileManager as afm
from classes.SpeechRecognizer import SpeechRecognizer as sr
from classes.GPT2Large import GPT2Large as gpt2

# define the audio recording function
print('Getting the recording process ready...')
def record_audio(q: Queue, device_index: int):
  process_q: Queue = Queue()

  def record_callback(indata, frames, time, status):
    process_q.put(indata.copy())
  
  print('initializing audio capture')
  with InputStream(samplerate=44100, channels=1, dtype='float32', device=device_index, callback=record_callback):
    print('recording')
    while True:
      if (process_q.qsize() > 0):
        q.put(process_q.get())

if __name__ == '__main__':
  # define constants
  print('Defining constants...')
  THRESHOLD = 0.003
  NOISE_LEVEL_CAPTURE_TIMEOUT = 10
  TALKING_TIMEOUT = 120
  LANG_ID = "en"
  SR_MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
  TG_MODEL_ID = "gpt2-large"
  FILE_NAME = "audio.wav"

  # set the computing device to cuda if available
  print('Setting the computing device to cuda if available...')
  device = device("cuda" if is_cuda_available() else "cpu")

  # instantiate audio managers
  print('Instantiating audio managers...')
  adm = adm()
  afm = afm()

  # pick the audio device
  print('Picking the audio device...')
  adm.print_devices()
  device_index = int(input('Select a device: '))

  # initialize the speech recognizer model and the text generator model
  print('Initializing the speech recognizer model and the text generator model...')
  sr = sr(LANG_ID, SR_MODEL_ID)
  tg = gpt2(TG_MODEL_ID)

  # initialize the text-to-speech engine
  print('Initializing the text-to-speech engine...')
  engine = pyttsx3.init()

  # initialize main loop variables
  print('Initializing main loop variables...')
  counter = 0

  # start the main loop
  print('Starting the main loop...')
  while True:
    # start the audio capture
    print('Starting the audio capture...')
    q: Queue = Queue()
    process: Process = Process(target=record_audio, args=(q, device_index,))
    process.start()

    # initialize speech recognition variables
    print('Initializing speech recognition variables...')
    sr_counter: int = 0
    timeout: int = 0
    spoken: bool = False
    noise_array = np.zeros((0, 1))
    audio_array = np.zeros((0, 1))

    # start the speech recognition
    print('Starting the speech recognition...')
    while True:
      if(q.qsize() > 0):
        sr_counter += 1
        
        audio: np.ndarray = q.get()

        if (sr_counter < NOISE_LEVEL_CAPTURE_TIMEOUT and counter == 0):
          print('Collecting noise, please do not speak...')
          noise_array = np.append(noise_array, audio, axis=0)
          continue

        if (sr_counter == NOISE_LEVEL_CAPTURE_TIMEOUT and counter == 0):
          noise_abs = np.abs(noise_array)
          noise_mean = np.mean(noise_abs)
          min_audio_level = (noise_mean + THRESHOLD)
          print('Done!')

        audio_abs = np.abs(audio)
        audio_mean = np.mean(audio_abs)

        threshold_reached = audio_mean > min_audio_level

        if(threshold_reached):
          spoken = True
          print('I hear you!', audio_mean, '>' , min_audio_level)
          audio_array = np.append(audio_array, audio, axis=0)
          timeout = 0

        if(spoken):
          timeout += 1

        if((timeout > TALKING_TIMEOUT) and spoken and len(audio_array[:]) > 0):
          afm.save_audio_file(audio_array, FILE_NAME)
          audio_array = np.zeros((0, 1))
          spoken = False
          print('Audio saved...')
          process.terminate()
          q.close()
          break

    audio_file: Tensor = sr.load_audio_file(FILE_NAME)
    audio_file: Tensor = sr.resample_audio_file(audio_file, original_sample_rate=44100, new_sample_rate=16000)
    text: str = sr.predict(audio_file)

    print('Loading conversation script...')
    with open('conversation-script.txt', 'r') as f:
      conversation_script = f.read()

    question, answer = tg.predict(
      text,
      initial_script=conversation_script,
      top_p=98,
      temperature=0.6,
      top_k=0,
      do_sample=True,
    )

    print(answer)
    engine.say(answer)
    engine.runAndWait()

    counter += 1
