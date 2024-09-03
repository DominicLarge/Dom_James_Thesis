# !python3

from Autodesk.Revit.DB import *
from Autodesk.Revit.UI import *

import json
from collections import defaultdict
import os
from config import *

doc = __revit__.ActiveUIDocument.Document
uidoc = __revit__.ActiveUIDocument
app = __revit__.Application

def parameter_to_string(param):
    """Convert a parameter to a string representation."""
    if not param.HasValue:
        return "N/A"
    
    if param.StorageType == StorageType.String:
        return param.AsString()
    elif param.StorageType == StorageType.Integer:
        return str(param.AsInteger())
    elif param.StorageType == StorageType.Double:
        return str(param.AsDouble())
    elif param.StorageType == StorageType.ElementId:
        return str(param.AsElementId().IntegerValue)
    else:
        return "Unsupported Type"

def element_to_string(element, doc):
    """Convert an element to a string representation."""
    lines = [
        f"ID: {element.Id.IntegerValue}",
        f"Name: {element.Name}",
        f"Category: {element.Category.Name if element.Category else 'Uncategorized'}"
    ]
    
    lines.append("Parameters:")
    for param in element.Parameters:
        lines.append(f"  {param.Definition.Name}: {parameter_to_string(param)}")
    
    element_type = doc.GetElement(element.GetTypeId())
    if element_type:
        lines.append("Type Parameters:")
        for param in element_type.Parameters:
            lines.append(f"  {param.Definition.Name}: {parameter_to_string(param)}")
    
    return "\n".join(lines)

def get_script_directory():
    """Get the directory of the current script."""
    return os.path.dirname(os.path.realpath(__file__))

def get_safe_file_path(directory, base_name='revit_export'):
    """Generate a safe file path in the specified directory."""
    file_name = f"{base_name}.txt"
    return os.path.join(directory, file_name)

def export_revit_to_text(doc):
    """Export all elements, types, and parameters to a text file in the script's directory."""
    # Get the script's directory
    export_dir = get_script_directory()

    # Get a safe file path
    output_file = get_safe_file_path(export_dir)

    try:
        # Collect all elements
        collector = FilteredElementCollector(doc).WhereElementIsNotElementType()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for element in collector:
                try:
                    element_string = element_to_string(element, doc)
                    f.write(element_string + "\n\n" + "-"*50 + "\n\n")
                except Exception as e:
                    print(f"Error processing element {element.Id}: {str(e)}")
        
    except Exception as e:
        error_message = f"Error writing to file: {str(e)}\nPlease check your permissions and try again."
        TaskDialog.Show("Export Error", error_message)

class RevitTextExporter(IExternalCommand):
    def Execute(self, commandData, message, elements):
        app = commandData.Application
        doc = app.ActiveUIDocument.Document

        export_revit_to_text(doc)

        return Result.Succeeded

# If running in Revit Python Shell, you can use this:
export_revit_to_text(__revit__.ActiveUIDocument.Document)


# Specify the correct path to the file
absolute_path = os.path.dirname(__file__)
print(absolute_path)
document_to_embed = os.path.join(absolute_path, "./revit_export.txt")
print(document_to_embed)

def get_embedding(text, model=embedding_model):
   text = text.replace("\n", " ")
   return local_client.embeddings.create(input = [text], model=model).data[0].embedding

# Read the text document
with open(document_to_embed, 'r', encoding='utf-8', errors='ignore') as file:
    text_file = file.read()

# OPTION1 Split the text into lines (each line = 1 vector). Pick this or the following chunking strategy.
chunks = text_file.split("\n")
chunks = [line for line in chunks if line.strip() and line.strip() != '---']

# OPTION2 Split the text into chunks based on newlines and remove lines containing patterns like "== some text ==" FOR WIKIPEDIA ARTICLES
# chunks = text_file.split("\n")
# chunks = [line.strip() for line in text_file.split("\n") if not line.strip().startswith("==") and not line.strip().endswith("==")]

# OPTION3 Alternetively, split the text into paragraphs by using the empty lines in between them
# Figure out your own strategy according to the structure of the txt you have.
# chunks = text_file.split("\n\n")
        
# Create the embeddings
embeddings = []
for i, line in enumerate(chunks):
    print(f'{i} / {len(chunks)}')
    vector = get_embedding(line.encode(encoding='utf-8').decode())
    database = {'content': line, 'vector': vector}
    embeddings.append(database)

# Save the embeddings to a json file
output_filename = os.path.splitext(document_to_embed)[0]
output_path = f"{output_filename}.json"

with open(output_path, 'w', encoding='utf-8') as outfile:
    json.dump(embeddings, outfile, indent=2, ensure_ascii=False)

print(f"Finished vectorizing. Created {document_to_embed}")
