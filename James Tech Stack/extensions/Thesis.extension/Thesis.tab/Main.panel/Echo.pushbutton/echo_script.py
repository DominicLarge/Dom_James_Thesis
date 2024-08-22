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

import os
import shutil
import glob

import Autodesk.Revit.DB as DB
import Autodesk.Revit.UI as UI
import Autodesk.Revit.ApplicationServices as AS
import Autodesk.Revit.Creation as CR

# Define Revit Doc
doc = __revit__.ActiveUIDocument.Document
ui = __revit__.ActiveUIDocument

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


# Define keywords for actions
action_keywords = {
    "Create": ["create", "make", "build", "design", "generate"],
    "Modify": ["modify", "change", "adjust", "alter", "update"]
}


# Function to detect action
def detect_action(text):
    # Convert text to lowercase for case-insensitive comparison
    text = text.lower()
    print(f"Processing text: '{text}'")  # Debugging: print the processed text

    # Initialize action variable
    detected_action = "UNKNOWN"

    # Check for CREATE keywords
    create_keywords = action_keywords["Create"]
    if any(keyword in text for keyword in create_keywords):
        detected_action = "Create"

    # Check for MODIFY keywords
    modify_keywords = action_keywords["Modify"]
    if any(keyword in text for keyword in modify_keywords):
        detected_action = "Modify"

    return detected_action


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


def extract_ids(text):
    # Normalize the text to lowercase
    text = text.lower()

    # Define keywords indicating ID context
    id_keywords = ['table', 'tables']

    # Initialize a list to store IDs
    ids = []

    # Split the text into words
    words = text.split()

    # Iterate through the words to find IDs after the keywords
    for i, word in enumerate(words):
        if word == "selected":
                ids.append("selected")
        if word in id_keywords:
            # Collect numbers that follow the keyword, handling cases with "and"
            j = i + 1
            while j < len(words):
                if words[j].isdigit():
                    ids.append(int(words[j]))
                elif 'and' in words[j]:
                    # Continue if "and" is present to capture the number after it
                    j += 1
                    if j < len(words) and words[j].isdigit():
                        ids.append(int(words[j]))
                elif ',' in words[j]:
                    # Handle cases where numbers are followed by a comma
                    comma_separated_ids = [int(num) for num in words[j].split(',') if num.isdigit()]
                    ids.extend(comma_separated_ids)
                else:
                    break
                j += 1

    # Remove duplicates and sort IDs
    ids = sorted(set(ids))

    return ids

# Test the function with the example prompt
# Function to extract just the numeric part of a value
def extract_number(value):
    # Regular expression to find the numeric part of the string
    match = re.search(r'\d+(\.\d+)?', value)
    if match:
        return match.group()
    return "Not specified"

# Prepare Revit parameters including action type
def prepare_revit_parameters(parameters, action, ids):
    # Define the parameter dictionary with mappings to Revit family parameters
    revit_parameters = {
        "Action": action,
        "Table IDs": ids,
        "Parameters": {
            "Height": {
                "ParameterName": "Height",
                "Value": extract_number(parameters.get("Height", "Not specified")),
                "Unit": "feet"  # You can adjust or remove this if needed
            },
            "Thickness": {
                "ParameterName": "Thickness",
                "Value": extract_number(parameters.get("Thickness", "Not specified")),
                "Unit": "inches"
            },
            "Sides": {
                "ParameterName": "Number of Sides",
                "Value": extract_number(parameters.get("Sides", "Not specified")),
                "Unit": "feet"  # You can adjust or remove this if needed
            },
            "Leg Insert": {
                "ParameterName": "Leg Insert",
                "Value": extract_number(parameters.get("Leg Insert", "Not specified")),
                "Unit": "inches"
            },
            "Corner Fillet": {
                "ParameterName": "Corner Fillet",
                "Value": extract_number(parameters.get("Corner Fillet", "Not specified")),
                "Unit": "inches"
            }
        }
    }

    return revit_parameters

def modify_parameters(ids, parameters):
    elements = []
    selected = ui.Selection.GetElementIds()
    filtered_elements = DB.FilteredElementCollector(doc)

    if "selected" in ids:
        elements = selected
    else:
        elements = filtered_elements

    t = DB.Transaction(doc, "Voice Command: Modify Parameters")
    t.Start()

    for elementid in elements:

        element = doc.GetElement(elementid)
        
        if element: 
                
            for param, value in parameters.items():
                
                parameter = element.LookupParameter(param)

                if parameter:                    
                    parameter.Set(int(value.split()[0]))
                parameters = element.GetParameters(param)

                if parameters:
                 for parameter in parameters:
                        parameter.Set(value)   
    t.Commit()

# Run the main function and capture the prompt
prompt = main()

# Extract parameters
parameters = {}


for parameter, synonyms in parameter_synonyms.items():
    for synonym in synonyms:
        if synonym in prompt:
            value = find_closest_value(synonym, prompt)
            if value:
                parameters[parameter] = value
                break


# Use the prompt to determine action
action = detect_action(prompt)
print(f"Detected Action: {action}")


ids = extract_ids(prompt)


# Prepare Revit parameters
revit_parameters = prepare_revit_parameters(parameters, action, ids)


print(action)
print(ids)
print(parameters)


if action == "Modify":
    modify_parameters(ids, parameters)