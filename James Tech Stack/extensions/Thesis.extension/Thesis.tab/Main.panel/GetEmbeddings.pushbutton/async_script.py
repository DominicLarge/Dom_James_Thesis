# !python3

import aiohttp
import asyncio
from typing import List, Dict
import json

class BatchEmbeddingProcessor:
    def __init__(self, server_url: str):
        self.server_url = server_url

    async def get_embeddings(self, chunks: List[str]) -> List[List[float]]:
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_embedding(session, chunk) for chunk in chunks]
            embeddings = await asyncio.gather(*tasks)
        return embeddings

    async def fetch_embedding(self, session: aiohttp.ClientSession, text: str) -> List[float]:
        async with session.post(self.server_url, json={"text": text}) as response:
            if response.status == 200:
                data = await response.json()
                return data['embedding']
            else:
                print(f"Error: {response.status}")
                return []

async def main():
    # Replace with your LM Studio server URL
    server_url = "http://localhost:1234/v1/embeddings"
    
    processor = BatchEmbeddingProcessor(server_url)

    # Example chunks
    chunks = [
        "This is the first chunk of text.",
        "Here's the second chunk for embedding.",
        "And this is the third and final chunk."
    ]

    embeddings = await processor.get_embeddings(chunks)

    # Print results
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        print(f"Chunk {i + 1}: '{chunk[:20]}...'")
        print(f"Embedding (first 5 values): {embedding[:5]}")
        print()

# if __name__ == "__main__":
#     asyncio.run(main())