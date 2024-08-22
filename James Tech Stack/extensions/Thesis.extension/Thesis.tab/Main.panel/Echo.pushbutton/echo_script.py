# !python3

import re
from word2number import w2n
import nltk
from nltk import word_tokenize, pos_tag, sent_tokenize
import json
import base64
import speech_recognition as sr
from pydub import AudioSegment
import io

#RECORD VOICE
class Recorder:
    def __init__(self, silence_timeout=15):
        self.recognizer = sr.Recognizer()
        self.silence_timeout = silence_timeout  # Time to wait for silence before stopping

    def record_audio(self):
        with sr.Microphone() as source:
            print("Adjusting for ambient noise, please wait...")
            self.recognizer.adjust_for_ambient_noise(source)
            print("Recording... Speak now.")
            
            # Record audio until silence is detected
            audio_data = self.recognizer.listen(source, timeout=self.silence_timeout, phrase_time_limit=self.silence_timeout)
            print("Recording stopped.")
            
            return audio_data
        

class Transcriber:
    def __init__(self):
        self.recognizer = sr.Recognizer()
    
    def transcribe_audio(self, audio_data):
        try:
            print("Transcribing...")
            prompt = self.recognizer.recognize_google(audio_data)
            print("You said: " + prompt)
            return prompt
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return ""
        
def main():
    # Create a Recorder instance with a silence timeout (e.g., 5 seconds)
    recorder = Recorder(silence_timeout=15)
    transcriber = Transcriber()

    audio_data = recorder.record_audio()
    prompt = transcriber.transcribe_audio(audio_data)
    
    # Check if prompt is empty
    if not prompt:
        print("No transcription available.")
    else:
        print(f"Transcribed Prompt: {prompt}")

    # 'prompt' is now defined and can be used in further processing
    return prompt

# Run the main function and capture the prompt
prompt = main()

print(prompt)


#SET UP PARAMETERS

# Define possible synonyms for each parameter
parameter_synonyms = {
    "Height": ["height", "tall", "elevation", "high"],
    "Corner Fillet": ["corner fillet", "fillet", "rounded corner", "corner fillets", "fillets"],
    "Leg Insert": ["leg insert", "leg distance", "leg spacing", "inserts", "leg insert distance of "],
    "Sides": ["sides", "edges"],
    "Thickness": ["thickness", "thick", "depth"]
}

# Function to find the closest number to a keyword, considering proximity and grammar
def find_closest_value(keyword, text):
    pattern = re.compile(r'(\b\d+(\.\d+)?\b)\s*(feet|foot|inches|inch|ft|in|inches.|sides)?')
    matches = pattern.finditer(text)
    keyword_position = text.find(keyword)

    closest_distance = float('inf')
    closest_value = None

    for match in matches:
        number, _, unit = match.groups()
        start, end = match.span()

        # Calculate the distance between the keyword and the number
        distance = abs(keyword_position - start)

        # Determine the direction (before or after) of the keyword relative to the number
        if keyword_position < start:  # Keyword before number
            direction = "after"
        else:  # Keyword after number
            direction = "before"

        # Only consider this match if it's closer than any previously found
        if distance < closest_distance:
            closest_distance = distance
            closest_value = f"{number} {unit}".strip()

            # Ensure grammatical correctness by checking direction
            if direction == "after" and closest_distance > len(keyword):
                continue

    return closest_value

# Extract parameters
parameters = {}

for parameter, synonyms in parameter_synonyms.items():
    for synonym in synonyms:
        if synonym in prompt:
            value = find_closest_value(synonym, prompt)
            if value:
                parameters[parameter] = value
                break



print(json.dumps(parameters, indent=2))