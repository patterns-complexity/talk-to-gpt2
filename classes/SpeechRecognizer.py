from torch import Tensor, argmax, no_grad
from torchaudio.transforms import Resample
from torchaudio import load
from transformers import (
  Wav2Vec2ForCTC,
  Wav2Vec2Processor,
)

class SpeechRecognizer:
  def __init__(self, lang: str, model_id: str):
    self.lang: str = lang
    self.model_id: str = model_id
    self.processor: Wav2Vec2Processor = Wav2Vec2Processor.from_pretrained(model_id)
    self.model: Wav2Vec2ForCTC = Wav2Vec2ForCTC.from_pretrained(model_id).to("cuda:0")
    pass

  def load_audio_file(self, file_path: str) -> Tensor:
    self.audio_file = load(file_path)
    return self.remove_batch_and_channel_dimensions(self.audio_file[0])

  def resample_audio_file(self, audio_file: Tensor, original_sample_rate: int, new_sample_rate: int) -> Tensor:
    return Resample(original_sample_rate, new_sample_rate)(audio_file)

  def remove_batch_and_channel_dimensions(self, audio_file: Tensor) -> Tensor:
    return audio_file.squeeze(0).squeeze(0)

  def get_inputs(self, audio_file: Tensor, sampling_rate: int = 16_000) -> Tensor:
    return self.processor(audio_file, sampling_rate=sampling_rate, return_tensors="pt", padding=True).to("cuda:0")

  def get_logits(self, inputs: Tensor) -> Tensor:
    with no_grad():
      logits = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    return logits
  
  def get_predicted_ids(self, logits: Tensor) -> Tensor:
    return argmax(logits, dim=-1)

  def get_predicted_sentences(self, predicted_ids: Tensor) -> Tensor:
    return self.processor.batch_decode(predicted_ids)

  def get_text(self, predicted_sentences: Tensor) -> str:
    return predicted_sentences[0]

  def predict(self, audio_file: Tensor, sampling_rate: int = 16_000) -> str:
    inputs = self.get_inputs(audio_file, sampling_rate=sampling_rate)
    logits = self.get_logits(inputs)
    predicted_ids = self.get_predicted_ids(logits)
    predicted_sentences = self.get_predicted_sentences(predicted_ids)
    text = self.get_text(predicted_sentences)
    return text
