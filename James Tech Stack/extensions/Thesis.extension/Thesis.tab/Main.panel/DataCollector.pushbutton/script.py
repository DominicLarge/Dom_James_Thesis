# !python3

"""
IMPORTS
------------------------------------------------------------------------------------
"""

from Autodesk.Revit.DB import *
from Autodesk.Revit.UI import *
from Autodesk.Revit.DB import UnitUtils, UnitTypeId, StorageType, SpecTypeId

import traceback
import json
import os
import hashlib
from typing import Dict, List, Optional
import aiohttp
import asyncio
import concurrent.futures
import time
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
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
    BuiltInCategory.OST_CurtaSystem,
    BuiltInCategory.OST_Sheets,
    BuiltInCategory.OST_Levels
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


def safe_get_property(obj, prop_name, default="N/A"):
    try:
        value = getattr(obj, prop_name, default)
        return value if value is not None else default
    except Exception:
        return default




def element_to_dict(element, doc):
    """Convert an element to a dictionary representation, including type parameters."""
    element_dict = {
        "ID": element.Id.IntegerValue,
        "Name": safe_get_property(element, 'Name'),
        "Category": safe_get_property(element.Category, 'Name') if element.Category else 'Uncategorized',
        "Instance_Parameters": {},
        "Type_Parameters": {}
    }
    
    # Add element parameters
    for param in element.Parameters:
        try:
            element_dict["Instance_Parameters"][param.Definition.Name] = parameter_to_string(param)
        except Exception:
            continue
    
    # Add type parameters
    element_type = doc.GetElement(element.GetTypeId())
    if element_type:
        for param in element_type.Parameters:
            try:
                element_dict["Type_Parameters"][param.Definition.Name] = parameter_to_string(param)
            except Exception:
                continue
    
    return element_dict



def type_to_dict(element_type):
    """Convert an element type to a dictionary representation."""
    try:
        type_dict = {
            "Type_ID": element_type.Id.IntegerValue,
            "Type_Name": safe_get_property(element_type, 'Name'),
            "Family_Name": safe_get_property(element_type, 'FamilyName'),
            "Category": safe_get_property(element_type.Category, 'Name') if element_type.Category else 'Uncategorized',
            "Parameters": {}
        }
        
        # Parameters
        for param in element_type.Parameters:
            try:
                type_dict["Parameters"][param.Definition.Name] = parameter_to_string(param)
            except Exception as e:
                print(f"Error reading parameter {param.Definition.Name}: {str(e)}")
                continue
        
        return type_dict
    except Exception as e:
        print(f"Error processing element type {element_type.Id.IntegerValue}: {str(e)}")
        return None



def get_script_directory():
    """Get the directory of the current script."""
    return os.path.dirname(os.path.realpath(__file__))

def get_safe_file_path(directory, base_name='revit_export', extension='.json'):
    """Generate a safe file path in the specified directory."""
    file_name = f"{base_name}{extension}"
    return os.path.join(directory, file_name)



def export_revit_to_json(doc):
    export_dir = get_script_directory()
    output_file = get_safe_file_path(export_dir, 'revit_export', '.json')
    
    try:
        all_elements = FilteredElementCollector(doc).WhereElementIsNotElementType().ToElements()
        model_elements = [elem for elem in all_elements if is_model_element(elem)]

        all_types = FilteredElementCollector(doc).WhereElementIsElementType().ToElements()
        filtered_types = [elem_type for elem_type in all_types if is_type_of_model_element(elem_type)]
        
        export_data = {
            "MODEL_ELEMENTS": [element_to_dict(element, doc) for element in model_elements if element_to_dict(element, doc) is not None],
            "ELEMENT_TYPES": [type_to_dict(element_type) for element_type in filtered_types if type_to_dict(element_type) is not None]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Export completed. JSON file saved at: {output_file}")
        return output_file
    
    except Exception as e:
        error_message = f"Error during export: {str(e)}\n"
        error_message += f"Traceback: {traceback.format_exc()}"
        print(error_message)
        return None












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


def process_json_for_embedding(file_path: str) -> List[Dict[str, str]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON file.")
        return []
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return []
    
    chunks = []
    
    # Process MODEL_ELEMENTS
    model_elements = data.get("MODEL_ELEMENTS", [])
    if isinstance(model_elements, list):
        for element in model_elements:
            chunk = f"Element ID: {element.get('ID', 'N/A')}\n"
            chunk += f"Name: {element.get('Name', 'N/A')}\n"
            chunk += f"Category: {element.get('Category', 'N/A')}\n"
            chunk += "Instance Parameters:\n"
            for name, value in element.get('Instance_Parameters', {}).items():
                chunk += f"  {name}: {value}\n"
            chunk += "Type Parameters:\n"
            for name, value in element.get('Type_Parameters', {}).items():
                chunk += f"  {name}: {value}\n"
            chunks.append({"content": chunk.strip(), "metadata": {"type": "MODEL_ELEMENT", "id": element.get('ID', 'N/A')}})
    else:
        print("Warning: MODEL_ELEMENTS is not a list as expected.")
    
    # Process ELEMENT_TYPES
    element_types = data.get("ELEMENT_TYPES", [])
    if isinstance(element_types, list):
        for elem_type in element_types:
            chunk = f"Type ID: {elem_type.get('Type_ID', 'N/A')}\n"
            chunk += f"Type Name: {elem_type.get('Type_Name', 'N/A')}\n"
            chunk += f"Family Name: {elem_type.get('Family_Name', 'N/A')}\n"
            chunk += f"Category: {elem_type.get('Category', 'N/A')}\n"
            chunk += "Parameters:\n"
            for name, value in elem_type.get('Parameters', {}).items():
                chunk += f"  {name}: {value}\n"
            chunks.append({"content": chunk.strip(), "metadata": {"type": "ELEMENT_TYPE", "id": elem_type.get('Type_ID', 'N/A')}})
    else:
        print("Warning: ELEMENT_TYPES is not a list as expected.")
    
    return chunks


async def embed_document(document_to_embed: str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, document_to_embed)

    if not os.path.exists(file_path):
        print(f"Error: Input file '{document_to_embed}' not found.")
        return

    print("Processing JSON file...")
    chunks = process_json_for_embedding(file_path)
    print(f"Processed {len(chunks)} chunks.")

    print("Getting embeddings...")
    retriever = AsyncEmbeddingRetriever(batch_size=100)
    
    start_time = time.time()
    embeddings = await retriever.get_embeddings([chunk['content'] for chunk in chunks])
    end_time = time.time()
    
    print(f"Embedding retrieval took {end_time - start_time:.2f} seconds")
    print(f"Retrieved {len(embeddings)} embeddings")

    result = []
    for chunk, vector in zip(chunks, embeddings):
        if vector is not None:
            result.append({'content': chunk['content'], 'vector': vector, 'metadata': chunk['metadata']})
        else:
            print(f"Failed to get embedding for chunk: {chunk['content'][:50]}...")

    output_filename = os.path.splitext(document_to_embed)[0]
    writable_dir = get_writable_directory()
    output_path = os.path.join(writable_dir, f"{output_filename}_embedded.json")

    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(result, outfile, indent=2, ensure_ascii=False)
        print(f"Finished vectorizing. Created {output_path}")
    except Exception as e:
        print(f"An unexpected error occurred while writing the output: {str(e)}")

    # Save the cache after processing
    retriever.cache.save_cache()




async def main():
    try:
        # First, export Revit data to JSON
        output_file = export_revit_to_json(__revit__.ActiveUIDocument.Document)
        
        if output_file is None or not os.path.exists(output_file):
            print("Failed to export Revit data or create JSON file. Aborting embedding process.")
            return
        
        # Then, process and embed the JSON file
        start_time = time.time()
        await embed_document(output_file)
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"An error occurred in the main function: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())














"""
QUERY LLM WITH RAG
------------------------------------------------------------------------------------
"""


def find_json_file(filename='revit_export_embedded.json'):
    """Find the JSON file with embedded vectors in the script's directory or its parent directories."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        file_path = os.path.join(current_dir, filename)
        if os.path.exists(file_path):
            return file_path
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            return None
        current_dir = parent_dir


def load_data(json_file_path=None):
    if json_file_path is None:
        json_file_path = find_json_file()
    
    if json_file_path is None:
        print("Error: Could not find 'revit_export_embedded.json' in the current directory or any parent directory.")
        print(f"Current working directory: {os.getcwd()}")
        print("Please ensure that 'revit_export_embedded.json' exists and you have the necessary permissions.")
        return None, None

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        chunks = [item['content'] for item in data]
        embeddings = np.array([item['vector'] for item in data])
        metadata = [item['metadata'] for item in data]

        print(f"Successfully loaded data from: {json_file_path}")
        return chunks, embeddings, metadata
    except Exception as e:
        print(f"Error loading data from {json_file_path}: {str(e)}")
        return None, None, None



def get_embedding(text):
    API_URL = "http://localhost:1234/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer not-needed"
    }
    data = {
        "model": "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "input": text
    }
    response = requests.post(API_URL, headers=headers, json=data)
    return response.json()['data'][0]['embedding']



def get_relevant_chunks(query_embedding, embeddings, chunks, top_k=5):
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices], similarities[top_indices]


def format_context(relevant_chunks, similarities):
    context = "RELEVANT INFORMATION:\n"
    for chunk, similarity in zip(relevant_chunks, similarities):
        chunk_data = json.loads(chunk)  # Assuming chunk is a JSON string
        element_type = chunk_data.get('ElementType', 'Unknown Type')
        element_id = chunk_data.get('ID', 'Unknown ID')
        
        context += f"[Similarity: {similarity:.2f}] [Type: {element_type}] [ID: {element_id}]\n"
        context += chunk + "\n" + '='*50 + "\n"
    return context


def query_llm(prompt):
    API_URL = "http://localhost:1234/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer not-needed"
    }
    data = {
        "model": "lmstudio-community/Mistral-Nemo-Instruct-2407-GGUF",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    response = requests.post(API_URL, headers=headers, json=data)
    return response.json()['choices'][0]['message']['content']


def extract_dictionary(text):
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        dict_str = text[start:end]
        return json.loads(dict_str)
    except:
        return None


def find_element_by_id(chunks, element_type, element_id):
    for chunk in chunks:
        if chunk['type'].lower() == element_type.lower() and chunk['id'] == element_id:
            return chunk
    return None


def parse_chunk(chunk):
    """Parse the chunk string to extract relevant information."""
    element_type = re.search(r'Category:\s*(\w+)', chunk)
    element_id = re.search(r'ID:\s*(\d+)', chunk)
    return {
        'ElementType': element_type.group(1) if element_type else 'Unknown Type',
        'ID': element_id.group(1) if element_id else 'Unknown ID'
    }


def preprocess_chunks(chunks, metadata):
    """Preprocess chunks to extract potential identifiers."""
    processed_chunks = []
    for chunk, meta in zip(chunks, metadata):
        chunk_data = parse_chunk(chunk)
        identifiers = []
        
        # Check common identifier parameters
        for param in ['Mark', 'Family and Type', 'Type Mark', 'Comments', 'Name', 'ID', 'Type ID']:
            value = (chunk_data.get('Instance_Parameters', {}).get(param) or 
                     chunk_data.get('Type_Parameters', {}).get(param) or 
                     chunk_data.get(param))
            if value and value != 'N/A':
                identifiers.append(f"{param}: {value}")
        
        # Add any other potential identifiers here
        
        processed_chunks.append({
            'content': chunk,
            'metadata': meta,
            'identifiers': identifiers
        })
    return processed_chunks


def find_element_by_identifier(processed_chunks, query):
    """Find an element based on various potential identifiers."""
    for chunk in processed_chunks:
        for identifier in chunk['identifiers']:
            if identifier.lower() in query.lower():
                return chunk
    return None


def format_context(relevant_chunks, similarities):
    context = "RELEVANT INFORMATION:\n"
    for chunk, similarity in zip(relevant_chunks, similarities):
        chunk_data = parse_chunk(chunk)
        element_type = chunk_data['ElementType']
        element_id = chunk_data['ID']
        
        context += f"[Similarity: {similarity:.2f}] [Type: {element_type}] [ID: {element_id}]\n"
        context += chunk + "\n" + '='*50 + "\n"
    return context


def rag_query(query, processed_chunks, embeddings, top_k=5):
    query_embedding = get_embedding(query)
    
    # Try to find a specific element based on identifiers
    specific_element = find_element_by_identifier(processed_chunks, query)
    
    # Get top-k relevant chunks
    chunk_contents = [chunk['content'] for chunk in processed_chunks]
    chunk_embeddings = embeddings
    relevant_chunks, similarities = get_relevant_chunks(query_embedding, chunk_embeddings, chunk_contents, top_k)
    
    # If a specific element was found and it's not in the relevant chunks, add it
    if specific_element and specific_element['content'] not in relevant_chunks:
        relevant_chunks = [specific_element['content']] + relevant_chunks[:top_k-1]
        similarities = [1.0] + similarities[:top_k-1]
    
    context = format_context(relevant_chunks, similarities)

    print(f"Number of chunks in context: {len(relevant_chunks)}")
    print("CONTEXT: " + context)
    
    prompt = f"""Based on the following context and verbal command, output ONLY a Python dictionary with this exact format:
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
        }},
        "Identifiers": ["identifier1", "identifier2"]
    }}

    IMPORTANT:

    1. Use the exact ID, Type ID, and Parameter names from the context if available.
    2. Use any identifiers (such as Mark, Family and Type, Type Mark, etc.) provided in the context to determine which element is being referred to.
    4. If a parameter value is changed by the command, reflect this in the output.
    5. Do not include any explanatory text. The output should be a valid Python dictionary and nothing else.
    6. Always include units after a parameter value if they are provided in the context.
    7. In the "Identifiers" list, include any identifying information used to select this element.
    8. Units should be either feet (ft), Inches (in), Meters (m) or Millimeters (mm)

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
    if isinstance(id_string, list):
        return [int(''.join(filter(str.isdigit, str(id)))) for id in id_string]
    elif isinstance(id_string, str):
        return [int(''.join(filter(str.isdigit, id))) for id in id_string.split(',')]
    return [int(''.join(filter(str.isdigit, str(id_string))))]



def update_revit_element(doc, uidoc, llm_output):
    element_ids = extract_id(llm_output.get('ID')) if llm_output.get('ID') else None
    type_ids = extract_id(llm_output.get('TypeID')) if llm_output.get('TypeID') else None
    parameters_to_update = llm_output.get('Parameters', {})

    if not element_ids and not type_ids:
        print("No ElementID or TypeID provided in LLM output.")
        return

    from Autodesk.Revit.DB import ElementId, Transaction

    element_ids = [ElementId(id) for id in element_ids]
    type_ids = [ElementId(id) for id in type_ids] if type_ids else []

    elements = [doc.GetElement(id) for id in element_ids if doc.GetElement(id) is not None]
    element_types = [doc.GetElement(id) for id in type_ids if doc.GetElement(id) is not None]

    if not elements and not element_types:
        print(f"No elements or element types found with provided IDs.")
        return

    t = Transaction(doc, "Update Element Parameters")
    t.Start()

    try:
        for element in elements:
            update_element_parameters(doc, element, parameters_to_update)

        for element_type in element_types:
            update_element_parameters(doc, element_type, parameters_to_update)

        t.Commit()
        print(f"Update process completed for {len(elements)} elements and {len(element_types)} element types")
    except Exception as e:
        t.RollBack()
        print(f"Error updating elements/types: {str(e)}")

def update_element_parameters(doc, element, parameters_to_update):
    for param_name, param_value in parameters_to_update.items():
        param = element.LookupParameter(param_name)
        if param and not param.IsReadOnly:
            if update_parameter(doc, param, param_value):
                print(f"Updated parameter '{param_name}' for element/type {element.Id}")
            else:
                print(f"Failed to update parameter '{param_name}' for element/type {element.Id}")
        else:
            print(f"Parameter '{param_name}' not found or is read-only for element/type {element.Id}")




"""
RUN SCRIPT
------------------------------------------------------------------------------------
"""


user_input = "Change skylight width to 4ft and height to 2ft"


# Example usage
chunks, embeddings, metadata = load_data()
if chunks is not None and embeddings is not None and metadata is not None:
    processed_chunks = preprocess_chunks(chunks, metadata)
    result = rag_query(user_input, processed_chunks, embeddings, top_k=5)
    print(json.dumps(result, indent=2))
else:
    print("Failed to load data. Please check the error messages above.")


update_revit_element(doc, uidoc, result)

