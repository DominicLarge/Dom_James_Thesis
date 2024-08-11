"Prompts voice command for Revit"

import Autodesk.Revit.DB as DB
import Autodesk.Revit.UI as UI

# Define Revit Doc
doc = __revit__.ActiveUIDocument.Document
ui = __revit__.ActiveUIDocument

#Get parameters from voice input
#[INSERT DOM'S CODE HERE]

voice_input = {'Height' :3.5 , 'Thickness': 0.25, 'Sides' : 5.0, 'Corner Fillet': 0.41, 'Edge Fillet' : 0.1, 'Leg Radius' : 0.2, 'Skew' : None, 'Taper' : 0.25}

selected = ui.Selection.GetElementIds()

elements = DB.FilteredElementCollector(doc)

t = DB.Transaction(doc, "Voice Command")
t.Start()

for elementid in selected:

    element = doc.GetElement(elementid)
        
    if element: 
                
        for param, value in voice_input.items():
                
            parameter = element.LookupParameter(param)

            if parameter:
                parameter.Set(value)
            # parameters = element.GetParameters(param)

            # if parameters:
            #     for parameter in parameters:
            #         parameter.Set(value)
    



t.Commit()

