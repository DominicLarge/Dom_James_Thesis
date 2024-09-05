# !python3

from Autodesk.Revit.DB import *
from Autodesk.Revit.UI import *

import subprocess
import json
from collections import defaultdict
import os

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


def run_curl_command(curl_command):
    print("Executing curl command:", " ".join(curl_command))
    try:
        result = subprocess.run(curl_command, capture_output=True, text=True, check=True)
        print("Raw response from server:")
        print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while calling the API: {e}")
        print(f"Stderr: {e.stderr}")
        return None


def get_embedding(text, url="http://localhost:1234/v1/embeddings"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer not-needed"
    }
    
    data = {
        "input": text,
        "model": "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
    }
    
    curl_command = [
        "curl", "-X", "POST", url,
        "-H", f"Content-Type: {headers['Content-Type']}",
        "-H", f"Authorization: {headers['Authorization']}",
        "-d", json.dumps(data)
    ]

    print(curl_command)
    
    response_text = run_curl_command(curl_command)
    
    if response_text is None:
        return None
    
    try:
        response = json.loads(response_text)
        if 'error' in response:
            print("Server returned an error:")
            print(json.dumps(response['error'], indent=2))
            return None
        embedding = response['data'][0]['embedding']
        return embedding
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        print("Raw response:", response_text)
    except KeyError as e:
        print(f"Unexpected response format: {e}")
        print("Response structure:", json.dumps(response, indent=2))
    return None


def split_text_file(file_path):
    chunks = []
    current_chunk = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() == '-' * 50:  # Check for separator line
                if current_chunk:  # If we have a non-empty chunk
                    chunks.append('\n'.join(current_chunk).strip())
                    current_chunk = []
            else:
                current_chunk.append(line.strip())

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append('\n'.join(current_chunk).strip())

    return chunks

# EMBEDDING MODEL
model_name = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"

def embed_document(document_to_embed):

    current_dir = get_script_directory()
    file_path = os.path.join(current_dir, document_to_embed)

    chunks = split_text_file(file_path)
        
    embeddings = []
    for i, line in enumerate(chunks):
        print(f'{i} / {len(chunks)}')
        vector = get_embedding(line.encode(encoding='utf-8').decode())
        database = {'content': line, 'vector': vector}
        embeddings.append(database)


    output_filename = os.path.splitext(document_to_embed)[0]
    output_path = f"{output_filename}.json"

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(embeddings, outfile, indent=2, ensure_ascii=False)

    print(f"Finished vectorizing. Created {document_to_embed}")


txt_directory = get_safe_file_path(get_script_directory(), "revit_export.txt")
embed_document("revit_export.txt")