# !python3

import subprocess
import json
import os

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

    with open("./revit_export.txt", 'r', encoding='utf-8', errors='ignore') as infile:
        text_file = infile.read()

    chunks = split_text_file("./revit_export.txt")

        
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