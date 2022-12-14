# Talk to GPT-2 by Patterns-Complexity

## Installation

**This project was tested on Python 3.9.**

Clone the repository:
```
git clone https://github.com/patterns-complexity/talk-to-gpt2.git
```
Create a `conversation-script.txt` file in the root folder of the project.

This project relies on the conversation script having a Q&A format. It reads it as a history of your previous conversations. Make sure to guide GPT-2 by filling that file in with some fake history in this format:
```
Q: Some question... (you)
A: An answer to that question... (GPT-2 response)
```

Avoid whitespace between Q&A blocks, except for a single line break after each question and answer. Try not to leave an empty line break at the end of the file.

## Example `conversation-script.txt` file
Here's a fictional setup for some creative conversations:
```
# Conversation between: A user and the ship's AI.
# Location: Biolab full of old and rusty equipment.
# Mood: Creepy, Tense, Careful
# Date: 23/12/2075
# Time: 12:00 PM

Q: who are you
A: I am the ship's AI.
Q: I just woke up I think 50 years have passed since we launched
A: That is correct. It's been a long time. Today is the 23rd of December 2075.
```

You probably noticed that there is no punctuation or capitalization in those questions. This is on purpose. Remember that this model is supposed to transcribe your speech into text before it communicates with GPT-2. It doesn't really know where to put commas, periods and which words to capitalize.

## Usage
Run the `app.py` file in the root folder of the project.

```
python3 app.py
```

This will run the project with default parameters of:
- top_p: 0.9
- top_k: 0
- temperature: 1.0

You can change these parameters by passing them as arguments to the program. For example:

```
python3 app.py --top-p 0.98 --top-k 0 --temp 0.6
```

Here's a list of all parameters and their default values:

```
  --al / --audio-level-threshold = 0.003
  --nlt / --noise-evaluation-time = 10
  --ttt / --time-to-talk = 120
  --lang / --language = "en"
  --srm / --speech-recognition-model = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
  --tgm / --text-generation-model = "gpt2-large"
  --tempfile / --audio-temp-file-path = "audio.wav"
  --top-p = 0.9
  --top-k = 0
  --temp / --temperature = 1.0
```

The application will ask you to specify the audio device you want to use to communicate. Pick the device by typing its index.

Example below:

```
0: Microsoft Sound Mapper - Input
1: Analogue 1 + 2 (Focusrite Usb A
2: Microsoft Sound Mapper - Output
3: Speakers (Focusrite Usb Audio)
Select a device:
```

If I want to pick the `Speakers (Focusrite Usb Audio)` device I type in `3` and press enter.

## Credits
This project wouldn't even be possible if not for the amazing work of [huggingface/transformers](https://github.com/huggingface/transformers).
```
@inproceedings{Wolf_Transformers_State-of-the-Art_Natural_2020,
  author = {Wolf, Thomas and Debut, Lysandre and Sanh, Victor and Chaumond, Julien and Delangue, Clement and Moi, Anthony and Cistac, Perric and Ma, Clara and Jernite, Yacine and Plu, Julien and Xu, Canwen and Le Scao, Teven and Gugger, Sylvain and Drame, Mariama and Lhoest, Quentin and Rush, Alexander M.},
  month = {10},
  pages = {38--45},
  publisher = {Association for Computational Linguistics},
  title = {{Transformers: State-of-the-Art Natural Language Processing}},
  url = {https://www.aclweb.org/anthology/2020.emnlp-demos.6},
  year = {2020}
}
```


## Requirements
Those are the main requirements for this project (`requirements.txt`):
```
numpy==1.22.3
pyttsx3==2.90
sounddevice==0.4.4
SoundFile==0.10.3.post1
torch==1.11.0
torchaudio==0.11.0+cu113
transformers==4.21.0
```