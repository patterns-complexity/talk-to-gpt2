from torch import no_grad, Tensor, cat, zeros, float32, long, tensor
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

class GPT2Large:
  def __init__(self, model_id: str):
    self.tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    self.model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_id).to(dtype=float32).to(device="cuda:0")
    self.first_message: str = ''
    self.history: list = []
    self.step: int = 0
    # create an empty history tensor

  def predict(self, text_input: str, initial_script: str = None,  **kwargs) -> list [str, str]:
    break_token = '\n'

    if (self.step == 0):
      self.history = initial_script.split(break_token)

    # get the length of history
    history_length = len(self.history)

    history_with_question: str = (break_token).join(self.history) + break_token + 'Q: ' + text_input + break_token + 'A:'

    encoded_history_with_question: Tensor = self.tokenizer.encode(history_with_question, return_tensors='pt').to(device='cuda:0')

    encoded_response: Tensor = self.model.generate(encoded_history_with_question, max_new_tokens=len(encoded_history_with_question) + 200, **kwargs)

    response: list [str] = self.tokenizer.decode(encoded_response[0].cpu(), skip_special_tokens=False).split(break_token)

    print('response', response)

    only_new = response[history_length:history_length + 2]

    response_without_history = (break_token).join(only_new)

    self.history = ((break_token).join(self.history) + break_token + response_without_history).split(break_token)

    print('history', self.history)

    self.step += 1

    # remove the first token from the history

    return [text_input.replace('Q: ', ''), only_new[1].replace('A:', '')]

