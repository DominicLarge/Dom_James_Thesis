# !python3
"Prompts voice command for Revit"

from pydub import AudioSegment
import re
from word2number import w2n
import nltk
from nltk import word_tokenize, pos_tag, sent_tokenize
import json
import base64
import speech_recognition as sr
from pydub import AudioSegment



import Autodesk.Revit.DB as DB
import Autodesk.Revit.UI as UI

# Define Revit Doc
doc = __revit__.ActiveUIDocument.Document
ui = __revit__.ActiveUIDocument

#Get parameters from voice input
#[INSERT DOM'S CODE HERE]

voice_input = {'Height' :4.5 , 'Thickness': 0.25, 'Sides' : 5.0, 'Corner Fillet': 0.41, 'Edge Fillet' : 0.1, 'Leg Radius' : 0.2, 'Skew' : None, 'Taper' : 0.25}


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
                
            for param, value in voice_input.items():
                
                parameter = element.LookupParameter(param)

                if parameter:
                    parameter.Set(value)
                parameters = element.GetParameters(param)

                if parameters:
                 for parameter in parameters:
                        parameter.Set(value)   
    t.Commit()

