# !python3

"""
IMPORTS
------------------------------------------------------------------------------------
"""

from Autodesk.Revit.DB import *
from Autodesk.Revit.UI import *

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

    with open(file_path, 'r') as file:
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