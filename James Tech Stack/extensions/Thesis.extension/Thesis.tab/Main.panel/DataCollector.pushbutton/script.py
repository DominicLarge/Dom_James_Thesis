# !python3

"""
IMPORTS
------------------------------------------------------------------------------------
"""

from Autodesk.Revit.DB import *
from Autodesk.Revit.UI import *
from Autodesk.Revit.DB import UnitUtils, UnitTypeId, StorageType, ForgeTypeId, SpecTypeId

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
    
    # Add element parameters
    lines.append("Instance Parameters:")
    for param in element.Parameters:
        lines.append(f"  {param.Definition.Name}: {parameter_to_string(param)}")
    
    # Add type parameters
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
    export_dir = get_script_directory()
    output_file = get_safe_file_path(export_dir)
    error_log_file = get_safe_file_path(export_dir, 'revit_export_errors')
    debug_log_file = get_safe_file_path(export_dir, 'revit_export_debug')

    try:

        # Collect all model elements
        all_elements = FilteredElementCollector(doc).WhereElementIsNotElementType().ToElements()
        model_elements = [elem for elem in all_elements if is_model_element(elem)]

        # Collect all element types
        all_types = FilteredElementCollector(doc).WhereElementIsElementType().ToElements()
        filtered_types = [elem_type for elem_type in all_types if is_type_of_model_element(elem_type)]
        
        with open(output_file, 'w', encoding='utf-8') as f, \
             open(error_log_file, 'w', encoding='utf-8') as error_f, \
             open(debug_log_file, 'w', encoding='utf-8') as debug_f:
            

            # Write model elements
            f.write("MODEL ELEMENTS:\n")
            f.write("=" * 50 + "\n\n")
            for element in model_elements:
                try:
                    element_string = element_to_string(element, doc)
                    f.write(element_string + "\n\n" + "-"*50 + "\n\n")
                except Exception as e:
                    error_message = f"Error processing element {element.Id.IntegerValue}: {str(e)}"
                    print(error_message)
                    error_f.write(error_message + "\n")
                    debug_f.write(f"Debug info for element {element.Id.IntegerValue}:\n")
                    debug_f.write(traceback.format_exc() + "\n\n")

            # Write element types
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
        self.model = "nomic-ai/nomic-embed-text-v1.5-GGUF"
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
    return chunk.strip().replace('\n', ', ')


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


def get_relevant_chunks(query_embedding, embeddings, chunks, top_k=5):
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices], similarities[top_indices]


def format_context(relevant_chunks, similarities):
    context = "RELEVANT INFORMATION:\n"
    for chunk, similarity in zip(relevant_chunks, similarities):
        context += f"[Similarity: {similarity:.2f}]\n"
        
        # Split the chunk into lines
        lines = chunk.split('\n')
        formatted_lines = []

        for line in lines:
            # Split the line into key-value pairs
            pairs = re.findall(r'(\w+(?:\s+\w+)*?):\s*(.*?)(?=\s+\w+(?:\s+\w+)*?:|$)', line)
            
            # Format each key-value pair
            formatted_pairs = [f"{key}: {value}" for key, value in pairs]
            
            # Join the formatted pairs with commas
            formatted_line = ', '.join(formatted_pairs)
            formatted_lines.append(formatted_line)

        # Join all formatted lines
        context += '\n'.join(formatted_lines) + "\n" + '='*50 + "\n"
    
    return context


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
        "model": "nomic-ai/nomic-embed-text-v1.5-GGUF",
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
        "temperature": 0.0
    }
    response = requests.post(API_URL, headers=headers, json=data)
    return response.json()['choices'][0]['message']['content']


def extract_dictionary(text):
    try:
        # Find the start and end of the dictionary in the text
        start = text.index('{')
        end = text.rindex('}') + 1
        dict_str = text[start:end]
        # Parse the dictionary string
        return eval(dict_str)
    except:
        return None


def find_element_by_id(chunks, element_type, element_id):
    for chunk in chunks:
        if f"ID: {element_id}" in chunk and element_type.lower() in chunk.lower():
            return chunk
    return None



def rag_query(query, chunks, embeddings):
    query_embedding = get_query_embedding(query)
    
    # Extract element type and ID from the query
    match = re.search(r'(\w+)\s+(\d+)', query)
    if match:
        element_type, element_id = match.groups()
        specific_element = find_element_by_id(chunks, element_type, element_id)
    else:
        specific_element = None
    
    relevant_chunks, similarities = get_relevant_chunks(query_embedding, embeddings, chunks)
    
    # If a specific element was found, add it to the relevant chunks
    if specific_element:
        relevant_chunks = [specific_element] + [chunk for chunk in relevant_chunks if chunk != specific_element]
        similarities = [1.0] + similarities[:len(relevant_chunks)-1]  # Assign highest similarity to specific element
    
    context = format_context(relevant_chunks, similarities)

    print("CONTEXT: " + context)
    
    prompt = f"""Based on the following context and verbal command,  output ONLY a Python dictionary with this exact format:
    {{
        "Command": "command_here",
        "ID": "id_here",
        "ElementType": "elementtype_here",
        "Family": "element_family_here",
        "Type": "type_here",
        "TypeID": "type_id_here",        
        "Parameters": {{
            "ParameterName1": "Value1_units",
            "ParameterName2": "Value2_units"
        }}
    }}

    IMPORTANT:

    1. Use the exact ID, Type ID, and Parameter names from the context if available.
    2. Use context provided by the user to determine which elements are being referred to.
    3. If a specific parameter is mentioned in the command, include it in the Parameters dictionary.
    4. Only include parameters that are explicitly mentioned in the command or are directly relevant to the command.
    5. Do not include any explanatory text. The output should be a valid Python dictionary and nothing else.
    6. Always include units after a parameter if the user specifies them.

    Context:
    {context}

    Verbal Command: {query}

    Python Dictionary Output:"""
    
    response = query_llm(prompt)
    dict_output = extract_dictionary(response)
    
    return dict_output if dict_output else {"Error": "Unable to generate a valid dictionary response"}


"""
UPDATE THE REVIT ELEMENTS
------------------------------------------------------------------------------------
"""


def parse_and_convert_value(doc, param, value_string):
    # Regular expression to separate number and unit
    match = re.match(r"([0-9.]+)\s*([a-zA-Z'\"]+)?", value_string)
    if not match:
        raise ValueError(f"Unable to parse value: {value_string}")

    value, unit = match.groups()
    value = float(value)

    # Define unit conversions
    unit_conversions = {
        'm': UnitTypeId.Meters,
        'mm': UnitTypeId.Millimeters,
        'cm': UnitTypeId.Centimeters,
        'ft': UnitTypeId.Feet,
        'feet': UnitTypeId.Feet,
        "'": UnitTypeId.Feet,
        'in': UnitTypeId.Inches,
        'inch': UnitTypeId.Inches,
        '"': UnitTypeId.Inches
    }

    # Get the project's length unit type
    project_unit_type = doc.GetUnits().GetFormatOptions(SpecTypeId.Length).GetUnitTypeId()

    # Convert the value to the project's unit type
    if unit and unit.lower() in unit_conversions:
        from_unit = unit_conversions[unit.lower()]
        value = UnitUtils.ConvertToInternalUnits(value, from_unit)
        value = UnitUtils.ConvertFromInternalUnits(value, project_unit_type)
    else:
        # If no unit is specified, assume it's already in project units
        pass

    return value

def update_parameter(doc, param, param_value):
    try:
        if param.StorageType == StorageType.Double:
            if param.Definition.GetDataType() == SpecTypeId.Length:
                converted_value = parse_and_convert_value(doc, param, param_value)
                param.Set(UnitUtils.ConvertToInternalUnits(converted_value, doc.GetUnits().GetFormatOptions(SpecTypeId.Length).GetUnitTypeId()))
            else:
                param.Set(float(param_value))
        elif param.StorageType == StorageType.Integer:
            param.Set(int(param_value))
        elif param.StorageType == StorageType.String:
            param.Set(str(param_value))
        else:
            print(f"Unsupported parameter type for '{param.Definition.Name}'")
            return False
        return True
    except ValueError as ve:
        print(f"Error converting value for parameter '{param.Definition.Name}': {str(ve)}")
        return False
    except Exception as e:
        print(f"Error updating parameter '{param.Definition.Name}': {str(e)}")
        return False

def extract_id(id_string):
    return int(''.join(filter(str.isdigit, id_string)))

def update_revit_element(doc, uidoc, llm_output):
    element_id = extract_id(llm_output.get('ID')) if llm_output.get('ID') else None
    type_id = extract_id(llm_output.get('TypeID')) if llm_output.get('TypeID') else None
    parameters_to_update = llm_output.get('Parameters', {})

    if not element_id and not type_id:
        print("No ElementID or TypeID provided in LLM output.")
        return

    from Autodesk.Revit.DB import ElementId
    element_id = ElementId(element_id) if element_id is not None else None
    type_id = ElementId(type_id) if type_id is not None else None

    element = doc.GetElement(element_id) if element_id else None
    element_type = doc.GetElement(type_id) if type_id else None

    if not element and not element_type:
        print(f"No element or element type found with provided IDs.")
        return

    t = Transaction(doc, "Update Element Parameters")
    t.Start()

    try:
        for param_name, param_value in parameters_to_update.items():
            # Try instance parameter first
            if element:
                instance_param = element.LookupParameter(param_name)
                if instance_param and not instance_param.IsReadOnly:
                    if update_parameter(doc, instance_param, param_value):
                        print(f"Updated instance parameter '{param_name}' for element {element_id}")
                        continue

            # If instance parameter not found or update failed, try type parameter
            if not element_type and element:
                element_type = doc.GetElement(element.GetTypeId())

            if element_type:
                type_param = element_type.LookupParameter(param_name)
                if type_param and not type_param.IsReadOnly:
                    if update_parameter(doc, type_param, param_value):
                        print(f"Updated type parameter '{param_name}' for type {type_id or element_type.Id}")
                        continue

            print(f"Parameter '{param_name}' not found or could not be updated.")

        t.Commit()
        print(f"Update process completed for element {element_id} and/or type {type_id}")
    except Exception as e:
        t.RollBack()
        print(f"Error updating element/type: {str(e)}")



"""
RUN SCRIPT
------------------------------------------------------------------------------------
"""

chunks, embeddings = load_data()
user_query = "Change kitchen room number to be 504"
result = rag_query(user_query, chunks, embeddings)
print(result)
update_revit_element(doc, uidoc, result)

