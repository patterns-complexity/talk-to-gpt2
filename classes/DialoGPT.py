from torch import no_grad, Tensor, cat, zeros, float32, long, tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class TextGenerator:
  def __init__(self, model_id: str):
    self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_id)
    self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model_id).to(dtype=float32).to(device="cuda:0")
    self.history: None
    self.step: int = 0
    # create an empty history tensor

  def predict(self, text_input: str, initial_script: str = None,  **kwargs) -> None:
    question = text_input + self.tokenizer.eos_token

    if initial_script is not None:
      self.history = self.tokenizer.encode(initial_script, return_tensors="pt").to(device="cuda:0")

    # encode the new user input, add the eos_token and return a tensor in Pytorch
    encoded_question = self.tokenizer.encode(question, return_tensors='pt').to(device="cuda:0")

    # append the new user input tokens to the chat history
    history_with_question = cat([self.history, encoded_question], dim=-1) if self.step > 0 else encoded_question

    # generated a response while limiting the total chat history to 1000 tokens, 
    response = self.model.generate(history_with_question, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)

    decoded_response = self.tokenizer.decode(response[0].cpu()).replace('<|endoftext|>', '<|endoftext|>\n')

    # encode the response and add it to the chat history
    response = self.tokenizer.encode('\n' + decoded_response, return_tensors='pt').to(device="cuda:0")
    self.history = cat([self.history, response], dim=-1)
    # self.history = response

    # pretty print last ouput tokens from bot

    self.step += 1

    print(self.history.shape)
    

    return [text_input, decoded_response.split('<|endoftext|>')[1].replace('<|endoftext|>', '\n')]

