# !python3

from Autodesk.Revit.DB import *
from Autodesk.Revit.UI import *

import json
from collections import defaultdict
import os
import aiohttp
import asyncio
from typing import List, Dict, Optional


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




class AsyncEmbeddingRetriever:
    def __init__(self, url: str = "http://localhost:1234/v1/embeddings", batch_size: int = 10):
        self.url = url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer not-needed"
        }
        self.model = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
        self.batch_size = batch_size

    async def get_embeddings_batch(self, session: aiohttp.ClientSession, texts: List[str]) -> List[Optional[List[float]]]:
        data = {
            "input": texts,
            "model": self.model
        }

        try:
            async with session.post(self.url, headers=self.headers, json=data) as response:
                response_text = await response.text()

                if response.status != 200:
                    print(f"Error occurred while calling the API. Status: {response.status}")
                    return [None] * len(texts)

                response_json = json.loads(response_text)
                if 'error' in response_json:
                    print("Server returned an error:")
                    print(json.dumps(response_json['error'], indent=2))
                    return [None] * len(texts)

                embeddings = [item['embedding'] for item in response_json['data']]
                return embeddings

        except Exception as e:
            print(f"Error occurred while calling the API: {e}")
            return [None] * len(texts)

    async def get_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        async with aiohttp.ClientSession() as session:
            batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
            results = []
            for batch in batches:
                batch_results = await self.get_embeddings_batch(session, batch)
                results.extend(batch_results)
            return results

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

def get_writable_directory():
    """
    Try to find a writable directory, falling back to the temp directory if needed.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_file = os.path.join(current_dir, 'test_write.tmp')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        return current_dir
    except PermissionError:
        return tempfile.gettempdir()

async def embed_document(document_to_embed):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, document_to_embed)

    if not os.path.exists(file_path):
        print(f"Error: Input file '{document_to_embed}' not found.")
        return

    chunks = split_text_file(file_path)
    
    retriever = AsyncEmbeddingRetriever(batch_size=100)  # Adjust batch_size as needed
    embeddings = await retriever.get_embeddings(chunks)

    result = []
    for chunk, vector in zip(chunks, embeddings):
        if vector is not None:
            result.append({'content': chunk, 'vector': vector})
        else:
            print(f"Failed to get embedding for chunk: {chunk[:50]}...")

    output_filename = os.path.splitext(document_to_embed)[0]
    writable_dir = get_writable_directory()
    output_path = os.path.join(writable_dir, f"{output_filename}.json")

    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(result, outfile, indent=2, ensure_ascii=False)
        print(f"Finished vectorizing. Created {output_path}")
    except PermissionError:
        print(f"Error: Unable to write to {output_path}. Please check your permissions.")
    except Exception as e:
        print(f"An unexpected error occurred while writing the output: {str(e)}")

async def main():
    document_to_embed = "revit_export.txt"
    await embed_document(document_to_embed)

if __name__ == "__main__":
    asyncio.run(main())