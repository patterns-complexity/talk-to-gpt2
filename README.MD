# Talk to GPT-2 by Patterns-Complexity

## Installation

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
@TODO: Add credits (work in progress)

## Requirements
@TODO: Add requirements (work in progress)