import requests
from dotenv import load_dotenv
import os
from openai import OpenAI
import json
from pinecone import Pinecone
load_dotenv()
API_KEY = os.getenv("PINECONE_API_KEY")

def get_response(question: str):
    from pinecone import Pinecone
    file_path = "/Users/haileyoon/code/comfortwomen_text.txt"
    with open(file_path, encoding="utf-8") as f:
        raw = f.read()
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    records = []
    for i, para in enumerate(paragraphs, start=1):
        records.append({
            "_id": f"rec{i}",
            "chunk_text": para
        })
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "data-py"
    # Create index if it doesn't exist
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"}
            }
        )
    index = pc.Index(index_name)
    # Upsert records (for demo; in production, upsert only once or when updating data)
    index.upsert_records("ns1", records)
    # Semantic search
    results = index.search(
        namespace="ns1",
        query={
            "top_k": 5,
            "inputs": {"text": question}
        }
    )
    # Rerank results
    reranked_results = index.search(
        namespace="ns1",
        query={
            "top_k": 5,
            "inputs": {"text": question}
        },
        rerank={
            "model": "bge-reranker-v2-m3",
            "top_n": 5,
            "rank_fields": ["chunk_text"]
        }
    )
    # Return only the reranked results
    return reranked_results

def chatbot_response(user_input: str, chat_history=None):
    if chat_history is None:
        chat_history = []
    load_dotenv()
    client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
    functions = [{
        "type": "function",
        "name": "get_response",
        "description": "Get the information about comfort women that is most appropriate for the user's question",
        "parameters":{
            "type":"object",
            "properties": {
                "question" : {"type":"string","description":"question about comfort women"}
            },
            "required": ["question"],
        },
    }]
    # Use chat_history for context
    input_messages = chat_history + [
        {"role":"user","content": user_input},
        {"role":"system","content":"You can call a function to return information about comfort women that is most appropriate for the user's question"}
    ]
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = input_messages,
        functions = functions,
        function_call = "auto",
    )
    message = response.choices[0].message
    if message.function_call:
        func_name = message.function_call.name
        args = json.loads(message.function_call.arguments)
        if func_name=="get_response":
            pinecone_results = get_response(**args)
            top_chunks = []
            for match in pinecone_results.get('matches', []):
                chunk = match.get('metadata', {}).get('chunk_text', '')
                if chunk:
                    top_chunks.append(chunk)
            function_response = {"response": top_chunks}
        else:
            return "Unknown function call: " + func_name
        followup_messages = chat_history + [{
            "role": "system",
            "content": "Reply to the user with the most helpful and relevant information. Use the information provided by the function call as your main source. try not to use external knowledge. answer in terms of comfort women and the entire issue."
        },
        {"role": "user", "content": user_input},
        message,
        {
            "role": "function",
            "name": func_name,
            "content": json.dumps(function_response)
        }
        ]
        followup = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = followup_messages,
        )
        return followup.choices[0].message.content
    else:
        return message.content

if __name__ == "__main__":
    chat_history = []
    while True:
        user_input = input("User:")
        if(user_input.lower() in ["exit", "end","bye"]):
            print("Bot: Goodbye!")
            break
        chat_history.append({"role": "user", "content": user_input})
        response = chatbot_response(user_input, chat_history)
        chat_history.append({"role": "assistant", "content": response})
        print("Bot:", response)