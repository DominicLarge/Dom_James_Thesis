# !python3

"""
IMPORTS
------------------------------------------------------------------------------------
"""

from Autodesk.Revit.DB import *
from Autodesk.Revit.UI import *

import traceback
import json
import os
import tempfile
import hashlib
from typing import Dict, List, Optional
import aiohttp
import asyncio
import multiprocessing
from functools import partial
import concurrent.futures
import time
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
import sys
import ast
import re


"""
GLOBAL VARIABLES
------------------------------------------------------------------------------------
"""

doc = __revit__.ActiveUIDocument.Document
uidoc = __revit__.ActiveUIDocument
app = __revit__.Application


"""
EXPORT REVIT TO TXT FILE
------------------------------------------------------------------------------------
"""

# List of categories that we want to include
model_categories = [
    BuiltInCategory.OST_Walls,
    BuiltInCategory.OST_Floors,
    BuiltInCategory.OST_Ceilings,
    BuiltInCategory.OST_Roofs,
    BuiltInCategory.OST_Doors,
    BuiltInCategory.OST_Windows,
    BuiltInCategory.OST_Stairs,
    BuiltInCategory.OST_Ramps,
    BuiltInCategory.OST_Railings,
    BuiltInCategory.OST_Columns,
    BuiltInCategory.OST_StructuralColumns,
    BuiltInCategory.OST_StructuralFraming,
    BuiltInCategory.OST_Furniture,
    BuiltInCategory.OST_FurnitureSystems,
    BuiltInCategory.OST_Casework,
    BuiltInCategory.OST_CurtainWallPanels,
    BuiltInCategory.OST_CurtainWallMullions,
    BuiltInCategory.OST_Rooms,
    BuiltInCategory.OST_Areas,
    BuiltInCategory.OST_StructuralFoundation,
    BuiltInCategory.OST_Rebar,
    BuiltInCategory.OST_PlumbingFixtures,
    BuiltInCategory.OST_MechanicalEquipment,
    BuiltInCategory.OST_ElectricalEquipment,
    BuiltInCategory.OST_ElectricalFixtures,
    BuiltInCategory.OST_LightingFixtures,
    BuiltInCategory.OST_SpecialityEquipment,
    BuiltInCategory.OST_GenericModel,
    BuiltInCategory.OST_Massing,
    BuiltInCategory.OST_Site,
    BuiltInCategory.OST_Topography,
    BuiltInCategory.OST_Parking,
    BuiltInCategory.OST_StructuralStiffener,
    BuiltInCategory.OST_StructuralTruss,
    BuiltInCategory.OST_Assemblies,
    BuiltInCategory.OST_CurtaSystem
]
    

def is_model_element(element):
    """Check if an element is a model element of specified categories."""
    if element.Category is None:
        return False    

    return element.Category.Id.IntegerValue in [int(category) for category in model_categories]



def is_type_of_model_element(element_type):
    """Check if an element type belongs to specified categories."""
    if element_type.Category is None:
        return False
    
    return element_type.Category.Id.IntegerValue in [int(category) for category in model_categories]


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
    """Convert an element to a string representation, including type parameters."""
    lines = [
        f"ID: {element.Id.IntegerValue}",
        f"Name: {element.Name}",
        f"Category: {element.Category.Name if element.Category else 'Uncategorized'}"
    ]
    
    lines.append("Instance Parameters:")
    for param in element.Parameters:
        lines.append(f"  {param.Definition.Name}: {parameter_to_string(param)}")
    
    element_type = doc.GetElement(element.GetTypeId())
    if element_type:
        lines.append("\nType Parameters:")
        for param in element_type.Parameters:
            lines.append(f"  {param.Definition.Name}: {parameter_to_string(param)}")
    
    return "\n".join(lines)


def type_to_string(element_type):
    """Convert an element type to a string representation with detailed error handling."""
    lines = []
    try:
        # Basic properties
        properties = [
            ("Type ID", lambda: str(element_type.Id.IntegerValue)),
            ("Type Name", lambda: element_type.Name),
            ("Family Name", lambda: getattr(element_type, 'FamilyName', 'N/A')),
            ("Category", lambda: element_type.Category.Name if element_type.Category else 'Uncategorized')
        ]
        
        for prop_name, prop_func in properties:
            try:
                value = prop_func()
                lines.append(f"{prop_name}: {value}")
            except Exception as e:
                lines.append(f"Error reading {prop_name}: {str(e)}")
                print(f"Debug: Error reading {prop_name} for element type {element_type.Id.IntegerValue}: {str(e)}")
        
        # Parameters
        lines.append("Type Parameters:")
        for param in element_type.Parameters:
            try:
                param_value = parameter_to_string(param)
                lines.append(f"  {param.Definition.Name}: {param_value}")
            except Exception as e:
                lines.append(f"  Error reading parameter {param.Definition.Name}: {str(e)}")
                print(f"Debug: Error reading parameter {param.Definition.Name} for element type {element_type.Id.IntegerValue}: {str(e)}")
        
        return "\n".join(lines)
    except Exception as e:
        error_msg = f"Error processing element type {element_type.Id.IntegerValue}: {str(e)}\n{traceback.format_exc()}"
        print(f"Debug: {error_msg}")
        return error_msg



def get_script_directory():
    """Get the directory of the current script."""
    return os.path.dirname(os.path.realpath(__file__))

def get_safe_file_path(directory, base_name='revit_export'):
    """Generate a safe file path in the specified directory."""
    file_name = f"{base_name}.txt"
    return os.path.join(directory, file_name)


def export_revit_to_text(doc):
    """Export model elements (with their type parameters) and filtered element types to a text file."""
    export_dir = get_script_directory()
    output_file = get_safe_file_path(export_dir)
    error_log_file = get_safe_file_path(export_dir, 'revit_export_errors')
    debug_log_file = get_safe_file_path(export_dir, 'revit_export_debug')

    try:
        # Collect and filter model elements
        all_elements = FilteredElementCollector(doc).WhereElementIsNotElementType().ToElements()
        model_elements = [elem for elem in all_elements if is_model_element(elem)]

        # Collect and filter all element types
        all_types = FilteredElementCollector(doc).WhereElementIsElementType().ToElements()
        filtered_types = [elem_type for elem_type in all_types if is_type_of_model_element(elem_type)]
        
        with open(output_file, 'w', encoding='utf-8') as f, \
             open(error_log_file, 'w', encoding='utf-8') as error_f, \
             open(debug_log_file, 'w', encoding='utf-8') as debug_f:
            
            # ... [Code for model elements remains unchanged] ...
            
            f.write("\n\nFILTERED ELEMENT TYPES:\n")
            f.write("=" * 50 + "\n\n")
            for element_type in filtered_types:
                try:
                    type_string = type_to_string(element_type)
                    f.write(type_string + "\n\n" + "-"*50 + "\n\n")
                except Exception as e:
                    error_message = f"Error processing element type {element_type.Id.IntegerValue}: {str(e)}"
                    print(error_message)
                    error_f.write(error_message + "\n")
                    debug_f.write(f"Debug info for element type {element_type.Id.IntegerValue}:\n")
                    debug_f.write(traceback.format_exc() + "\n\n")
        
        print(f"Export completed. File saved at: {output_file}")
        print(f"Error log saved at: {error_log_file}")
        print(f"Debug log saved at: {debug_log_file}")
    
    except Exception as e:
        error_message = f"Error writing to file: {str(e)}\nPlease check your permissions and try again."
        print(error_message)

# If running in Revit Python Shell, you can use this:
export_revit_to_text(__revit__.ActiveUIDocument.Document)


"""
GET LLM EMBEDDINGS FOR REVIT TXT FILE WITH CACHEING SYSTEM
------------------------------------------------------------------------------------
"""

class EmbeddingCache:
    def __init__(self, cache_name: str = "embedding_cache"):
        self.cache_dir = os.path.dirname(os.path.abspath(__file__))  # Changed to script directory
        self.cache_file = os.path.join(self.cache_dir, f"{cache_name}.json")
        self.cache = self.load_cache()
        print(f"Cache file location: {self.cache_file}")

    def load_cache(self) -> Dict[str, List[float]]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                print(f"Loaded existing cache from {self.cache_file}")
                print(f"Cache contains {len(cache_data)} entries")
                return cache_data
            except json.JSONDecodeError:
                print(f"Error reading cache file. Starting with an empty cache.")
                return {}
        else:
            print(f"No existing cache found. Starting with an empty cache.")
        return {}

    def save_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
            print(f"Cache saved to {self.cache_file}")
            print(f"Cache now contains {len(self.cache)} entries")
        except Exception as e:
            print(f"Error saving cache: {str(e)}")

    def get(self, text: str) -> Optional[List[float]]:
        result = self.cache.get(self.hash_text(text))
        if result:
            print(f"Cache hit for text: {text[:50]}...")
        else:
            print(f"Cache miss for text: {text[:50]}...")
        return result

    def set(self, text: str, embedding: List[float]):
        self.cache[self.hash_text(text)] = embedding
        print(f"Added new entry to cache for text: {text[:50]}...")

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get_cache_file_path(self):
        return self.cache_file
    

class AsyncEmbeddingRetriever:
    def __init__(self, url: str = "http://localhost:1234/v1/embeddings", batch_size: int = 100):
        self.url = url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer not-needed"
        }
        self.model = "second-state/All-MiniLM-L6-v2-Embedding-GGUF/all-MiniLM-L6-v2-Q4_0.gguf"
        self.batch_size = batch_size
        self.cache = EmbeddingCache("revit_embedding_cache")

    async def check_cache(self, text: str) -> Optional[List[float]]:
        return self.cache.get(text)

    async def check_cache_batch(self, texts: List[str]) -> Dict[int, Optional[List[float]]]:
        tasks = [self.check_cache(text) for text in texts]
        results = await asyncio.gather(*tasks)
        return {i: result for i, result in enumerate(results) if result is not None}

    async def get_embeddings_batch(self, session: aiohttp.ClientSession, texts: List[str]) -> List[Optional[List[float]]]:
        cached_results = await self.check_cache_batch(texts)
        
        results = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if i in cached_results:
                results[i] = cached_results[i]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            data = {
                "input": uncached_texts,
                "model": self.model
            }

            try:
                async with session.post(self.url, headers=self.headers, json=data) as response:
                    response_text = await response.text()

                    if response.status != 200:
                        print(f"Error occurred while calling the API. Status: {response.status}")
                        return results

                    response_json = json.loads(response_text)
                    if 'error' in response_json:
                        print("Server returned an error:")
                        print(json.dumps(response_json['error'], indent=2))
                        return results

                    embeddings = [item['embedding'] for item in response_json['data']]
                    for text, embedding, index in zip(uncached_texts, embeddings, uncached_indices):
                        self.cache.set(text, embedding)
                        results[index] = embedding

            except Exception as e:
                print(f"Error occurred while calling the API: {e}")

        return results

    async def get_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        async with aiohttp.ClientSession() as session:
            batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
            results = []
            for batch in batches:
                batch_results = await self.get_embeddings_batch(session, batch)
                results.extend(batch_results)
            return results



def process_chunk(chunk: str) -> str:
    """Process a single chunk of text."""
    return chunk.strip().replace('\n', ' ')


def split_text_file(file_path: str) -> List[str]:
    chunks = []
    current_chunk = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip() == '-' * 50:  # Check for separator line
                    if current_chunk:  # If we have a non-empty chunk
                        chunks.append(' '.join(current_chunk).strip())
                        current_chunk = []
                else:
                    current_chunk.append(line.strip())

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk).strip())

    except UnicodeDecodeError:
        print(f"Error: Unable to read the file using UTF-8 encoding. Trying with 'iso-8859-1'...")
        with open(file_path, 'r', encoding='iso-8859-1') as file:
            for line in file:
                if line.strip() == '-' * 50:  # Check for separator line
                    if current_chunk:  # If we have a non-empty chunk
                        chunks.append(' '.join(current_chunk).strip())
                        current_chunk = []
                else:
                    current_chunk.append(line.strip())

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk).strip())

    return chunks


def parallel_process_chunks(chunks: List[str], max_workers: int = None) -> List[str]:
    """Use threading to process chunks in parallel."""
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)  # This is ThreadPoolExecutor's default

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
        processed_chunks = []
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                result = future.result()
                if result:  # Only add non-empty results
                    processed_chunks.append(result)
            except Exception as exc:
                print(f'Generated an exception: {exc}')

    return processed_chunks


def get_writable_directory():
    """
    Use the script's directory as the writable directory.
    """
    return os.path.dirname(os.path.abspath(__file__))


async def embed_document(document_to_embed: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, document_to_embed)

    if not os.path.exists(file_path):
        print(f"Error: Input file '{document_to_embed}' not found.")
        return

    print("Reading and splitting the file...")
    chunks = split_text_file(file_path)
    
    print("Processing chunks in parallel...")
    processed_chunks = parallel_process_chunks(chunks)
    print(f"Processed {len(processed_chunks)} chunks.")

    print("Getting embeddings...")
    retriever = AsyncEmbeddingRetriever(batch_size=100)
    print(f"Cache file is located at: {retriever.cache.get_cache_file_path()}")
    
    start_time = time.time()
    embeddings = await retriever.get_embeddings(processed_chunks)
    end_time = time.time()
    
    print(f"Embedding retrieval took {end_time - start_time:.2f} seconds")
    print(f"Retrieved {len(embeddings)} embeddings")

    result = []
    for chunk, vector in zip(processed_chunks, embeddings):
        if vector is not None:
            result.append({'content': chunk, 'vector': vector})
        else:
            print(f"Failed to get embedding for chunk: {chunk[:50]}...")

    output_filename = os.path.splitext(document_to_embed)[0]
    writable_dir = get_writable_directory()  # Use script directory for output
    output_path = os.path.join(writable_dir, f"{output_filename}.json")

    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(result, outfile, indent=2, ensure_ascii=False)
        print(f"Finished vectorizing. Created {output_path}")
    except Exception as e:
        print(f"An unexpected error occurred while writing the output: {str(e)}")

    # Save the cache after processing
    retriever.cache.save_cache()



async def main():
    document_to_embed = "revit_export.txt"
    start_time = time.time()
    await embed_document(document_to_embed)
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows support
    asyncio.run(main())


"""
QUERY LLM WITH RAG
------------------------------------------------------------------------------------
"""

def find_json_file(filename='revit_export.json'):
    """Find the JSON file in the script's directory or its parent directories."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        file_path = os.path.join(current_dir, filename)
        if os.path.exists(file_path):
            return file_path
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            # We've reached the root directory
            return None
        current_dir = parent_dir

# Load the embeddings and text chunks
try:
    json_file_path = find_json_file()
    print(f"Loading data from: {json_file_path}")
    with open(json_file_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure that 'revit_export.json' is in the script's directory or any parent directory.")
    exit(1)


def load_data(json_file_path=None):
    if json_file_path is None:
        json_file_path = find_json_file()
    
    if json_file_path is None:
        print("Error: Could not find 'revit_export.json' in the current directory or any parent directory.")
        print(f"Current working directory: {os.getcwd()}")
        print("Please ensure that 'revit_export.json' exists and you have the necessary permissions.")
        return None, None

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        chunks = [item['content'] for item in data]
        embeddings = np.array([item['vector'] for item in data])
        print(f"Successfully loaded data from: {json_file_path}")
        return chunks, embeddings
    except Exception as e:
        print(f"Error loading data from {json_file_path}: {str(e)}")
        return None, None


def get_most_similar_chunks(query_embedding, embeddings, chunks, top_k=5):
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]


def query_llm(prompt):
    """Query the LLM API with the given prompt."""
    API_URL = "http://localhost:1234/v1/chat/completions"  # Update this with your LLM API endpoint
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer not-needed"
    }
    data = {
        "model": "second-state/all-MiniLM-L6-v2-Embedding-GGUF",  # Update with your model name
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(API_URL, headers=headers, json=data)
    return response.json()['choices'][0]['message']['content']

def get_query_embedding(query):
    API_URL = "http://localhost:1234/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer not-needed"
    }
    data = {
        "model": "second-state/all-MiniLM-L6-v2-Embedding-GGUF",
        "input": query
    }
    response = requests.post(API_URL, headers=headers, json=data)
    return response.json()['data'][0]['embedding']


def query_llm(prompt):
    API_URL = "http://localhost:1234/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer not-needed"
    }
    data = {
        "model": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    response = requests.post(API_URL, headers=headers, json=data)
    return response.json()['choices'][0]['message']['content']


def extract_dictionary(text):
    """Extract a dictionary from the text if present."""
    try:
        # Find the start and end of the dictionary in the text
        start = text.index('{')
        end = text.rindex('}') + 1
        dict_str = text[start:end]
        # Parse the dictionary string
        return ast.literal_eval(dict_str)
    except (ValueError, SyntaxError):
        return None


def extract_element_info(chunks, element_id):
    """Extract specific element information from chunks."""
    for chunk in chunks:
        if f"ID: {element_id}" in chunk:
            lines = chunk.split('\n')
            element_info = {}
            for line in lines:
                if line.startswith("ID:"):
                    element_info['ID'] = line.split(': ')[1]
                elif line.startswith("Name:"):
                    element_info['Name'] = line.split(': ')[1]
                elif line.startswith("Category:"):
                    element_info['Category'] = line.split(': ')[1]
                elif "Type ID:" in line:
                    element_info['Type ID'] = line.split(': ')[1]
            return element_info
    return None


def rag_query(query, chunks, embeddings):
    if chunks is None or embeddings is None:
        return {"Error": "Unable to load necessary data for querying."}
    
    query_embedding = get_query_embedding(query)
    relevant_chunks = get_most_similar_chunks(query_embedding, embeddings, chunks)

    
    # Extract element ID from the query
    element_id_match = re.search(r'\b(\d+)\b', query)
    if element_id_match:
        element_id = element_id_match.group(1)
        element_info = extract_element_info(chunks, element_id)
    else:
        element_id = None
        element_info = None

    if element_info:
        relevant_chunks.append(f"ELEMENT INFO: {json.dumps(element_info)}")

    context = "\n\n".join(relevant_chunks)
    prompt = f"""Based on the following context and verbal command, output ONLY a Python dictionary with this exact format:
    {{
        "Command": "command_here",
        "ElemendID: "elementid_here",
        "Family: "element_family_here",
        "Type": "type_id_here",
        "Parameters": {{
            "Parameter1": "Value1",
            "Parameter2": "Value2"
        }}
    }}
    IMPORTANT:
    1. Use the exact Element ID and Type ID from the ELEMENT INFO in the context.
    2. If an Element ID is not found in the context, use the one from the verbal command.
    3. Ensure all measurements are converted to the project units specified in the context.
    4. Do not include any explanatory text. The output should be a valid Python dictionary and nothing else.

    Context:
    {context}

    Verbal Command: {query}

    Python Dictionary Output:"""
    
    response = query_llm(prompt)
    dict_output = extract_dictionary(response)
    
    
    if dict_output:
        return dict_output
    else:
        return {"Error": "Unable to generate a valid dictionary response"}



chunks, embeddings = load_data()
if chunks is not None and embeddings is not None:
    user_query = "Change dining room tables to be 4ft wide"
    result = rag_query(user_query, chunks, embeddings)
    print(json.dumps(result, indent=2))
else:
    print("Unable to proceed due to data loading error.")