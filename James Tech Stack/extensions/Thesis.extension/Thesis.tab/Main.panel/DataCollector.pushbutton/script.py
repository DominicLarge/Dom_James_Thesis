# !python3

from Autodesk.Revit.DB import *
from Autodesk.Revit.UI import *

import clr
clr.AddReference('System.Core')

import json
import os
import tempfile
import hashlib
from typing import Dict, List, Optional
import aiohttp
import asyncio
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import time
import aiofiles
import System
from System.Collections.Concurrent import ConcurrentDictionary

doc = __revit__.ActiveUIDocument.Document
uidoc = __revit__.ActiveUIDocument
app = __revit__.Application

from pyrevit import revit, DB
from pyrevit import script

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




class RevitCompatibleEmbeddingCache:
    def __init__(self, cache_name: str = "embedding_cache"):
        self.cache_file = f"{cache_name}.json"
        self.cache = ConcurrentDictionary[str, List[float]]()
        self.load_cache()

    def load_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
                for key, value in cache_data.items():
                    self.cache[key] = value
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump({k: list(v) for k, v in self.cache.items()}, f)

    def get(self, text: str) -> Optional[List[float]]:
        value = self.cache.get(text)
        return list(value) if value else None

    def set(self, text: str, embedding: List[float]):
        self.cache[text] = System.Array[float](embedding)

class RevitCompatibleEmbeddingRetriever:
    def __init__(self, url: str = "http://localhost:1234/v1/embeddings", batch_size: int = 50):
        self.url = url
        self.headers = {"Content-Type": "application/json", "Authorization": "Bearer not-needed"}
        self.model = "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf"
        self.batch_size = batch_size
        self.cache = RevitCompatibleEmbeddingCache("revit_embedding_cache")

    def get_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        results = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cached_embedding = self.cache.get(text)
            if cached_embedding:
                results[i] = cached_embedding
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            # Here you would normally make an API call to get embeddings
            # For now, we'll just use dummy embeddings
            dummy_embeddings = [[0.1, 0.2, 0.3] for _ in uncached_texts]
            
            for embedding, index, text in zip(dummy_embeddings, uncached_indices, uncached_texts):
                if embedding:
                    self.cache.set(text, embedding)
                    results[index] = embedding

        self.cache.save_cache()
        return results

def process_chunks(chunks: List[str]) -> List[str]:
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def embed_document(document_to_embed: str):
    # Try to find the file in different locations
    possible_paths = [
        document_to_embed,
        os.path.join(os.path.dirname(__file__), document_to_embed),
        os.path.join(os.path.expanduser("~"), document_to_embed),
        os.path.join(os.getcwd(), document_to_embed)
    ]

    file_path = next((path for path in possible_paths if os.path.exists(path)), None)

    if not file_path:
        raise FileNotFoundError(f"Could not find {document_to_embed} in any expected locations")

    print(f"Reading file from: {file_path}")

    with open(file_path, mode='r') as f:
        content = f.read()
    
    chunks = content.split('-' * 50)
    processed_chunks = process_chunks(chunks)
    
    retriever = RevitCompatibleEmbeddingRetriever(batch_size=50)
    embeddings = retriever.get_embeddings(processed_chunks)

    result = [{'content': chunk, 'vector': vector} for chunk, vector in zip(processed_chunks, embeddings) if vector is not None]

    output_file = os.path.splitext(file_path)[0] + "_embedded.json"
    with open(output_file, 'w') as outfile:
        json.dump(result, outfile, indent=2)
    
    print(f"Embedded document saved to: {output_file}")

def script_execute():
    document_to_embed = "revit_export.txt"
    try:
        embed_document(document_to_embed)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the file exists and you have the correct path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        script.get_logger().error(str(e))

script_execute()