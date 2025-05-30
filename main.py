import os
import uuid
import time
import json
import asyncio
import base64
from typing import List, Dict, Any, Union, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx

# FastAPI app
app = FastAPI(title="Gemini Backend for Vercel - Real Stream")

# API Keys
API_KEYS = [
    "AIzaSyCT1PXjhup0VHx3Fz4AioHbVUHED0fVBP4",
    "AIzaSyArNqpA1EeeXBx-S3EVnP0tzao6r4BQnO0",
    "AIzaSyCXICPfRTnNAFwNQMmtBIb3Pi0pR4SydHg",
    "AIzaSyDiLvp7CU443luErAz3Ck0B8zFdm8UvNRs",
    "AIzaSyBzqJebfbVPcBXQy7r4Y5sVgC499uV85i0",
    "AIzaSyD6AFGKycSp1glkNEuARknMLvo93YbCqH8",
    "AIzaSyBTara5UhTbLR6qnaUI6nyV4wugycoABRM",
    "AIzaSyBI2Jc8mHJgjnXnx2udyibIZyNq8SGlLSY",
    "AIzaSyAcgdqbZsX9UOG4QieFSW7xCcwlHzDSURY",
    "AIzaSyAwOawlX-YI7_xvXY-A-3Ks3k9CxiTQfy4",
    "AIzaSyCJVUeJkqYeLNG6UsF06Gasn4mvMFfPhzw",
    "AIzaSyBFOK0YgaQOg5wilQul0P2LqHk1BgeYErw",
    "AIzaSyBQRsGHOhaiD2cNb5F68hI6BcZR7CXqmwc",
    "AIzaSyCIC16VVTlFGbiQtq7RlstTTqPYizTB7yQ",
    "AIzaSyCIlfHXQ9vannx6G9Pae0rKwWJpdstcZIM",
    "AIzaSyAUIR9gx08SNgeHq8zKAa9wyFtFu00reTM",
    "AIzaSyAST1jah1vAcnLfmofR4DDw0rjYkJXJoWg",
    "AIzaSyAV8OU1_ANXTIvkRooikeNrI1EMR3IbTyQ"
]

# Simple state management
current_key_index = 0
key_usage = {key: 0 for key in API_KEYS}

# Pydantic Models
class ImageUrl(BaseModel):
    url: str

class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentItem]]

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False

def get_next_key():
    """Simple round-robin key selection"""
    global current_key_index
    key = API_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    key_usage[key] = key_usage.get(key, 0) + 1
    return key

def process_content(content):
    """Convert OpenAI format to Gemini format"""
    if isinstance(content, str):
        return [{"text": content}]
    
    parts = []
    for item in content:
        if item.type == "text" and item.text:
            parts.append({"text": item.text})
        elif item.type == "image_url" and item.image_url:
            try:
                if item.image_url.url.startswith("data:"):
                    header, base64_data = item.image_url.url.split(",", 1)
                    mime_type = header.split(";")[0].split(":")[1]
                    parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64_data
                        }
                    })
            except:
                parts.append({"text": "[Image processing error]"})
    
    return parts or [{"text": ""}]

def convert_messages(messages):
    """Convert OpenAI messages to Gemini format"""
    return [
        {
            "role": "user" if msg.role == "user" else "model",
            "parts": process_content(msg.content)
        }
        for msg in messages
    ]

async def real_gemini_stream(api_key: str, messages, generation_config, model: str):
    """Gerçek Gemini stream API'sini kullan"""
    
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())
    
    # Initial chunk with role
    initial_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n"
    
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            print(f"Sending request to Gemini API with model: {model}, key: {api_key[:8]}...")
            async with client.stream(
                'POST',
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent",
                json={
                    "contents": messages,
                    "generationConfig": generation_config
                },
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key
                }
            ) as response:
                
                if response.status_code != 200:
                    error_text = await response.aread()
                    print(f"Gemini API error - Status: {response.status_code}, Response: {error_text.decode()}")
                    error_chunk = {
                        'id': chunk_id,
                        'object': 'chat.completion.chunk',
                        'created': created_time,
                        'model': model,
                        'choices': [{
                            'index': 0,
                            'delta': {'content': f"API error: {error_text.decode()}"},
                            'finish_reason': 'error'
                        }]
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                
                # Buffer to accumulate raw response lines
                buffer = ""
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    
                    print(f"Raw Gemini response: {line}")
                    
                    # Accumulate lines into buffer
                    buffer += line
                    
                    # Try to parse the accumulated buffer as a complete JSON
                    try:
                        # Add closing bracket if buffer starts with '[' and doesn't end with ']'
                        test_buffer = buffer if not buffer.startswith('[') else (buffer + ']' if not buffer.endswith(']') else buffer)
                        chunks = json.loads(test_buffer) if test_buffer else []
                        print(f"Processed chunks: {test_buffer}")
                        
                        # Process each chunk in the array or object
                        if isinstance(chunks, list):
                            for chunk_data in chunks:
                                if 'candidates' in chunk_data:
                                    candidate = chunk_data['candidates'][0]
                                    if 'content' in candidate and 'parts' in candidate['content']:
                                        for part in candidate['content']['parts']:
                                            if 'text' in part:
                                                content = part['text']
                                                openai_chunk = {
                                                    'id': chunk_id,
                                                    'object': 'chat.completion.chunk',
                                                    'created': created_time,
                                                    'model': model,
                                                    'choices': [{
                                                        'index': 0,
                                                        'delta': {'content': content},
                                                        'finish_reason': None
                                                    }]
                                                }
                                                yield f"data: {json.dumps(openai_chunk)}\n\n"
                        elif isinstance(chunks, dict) and 'candidates' in chunks:
                            candidate = chunks['candidates'][0]
                            if 'content' in candidate and 'parts' in candidate['content']:
                                for part in candidate['content']['parts']:
                                    if 'text' in part:
                                        content = part['text']
                                        openai_chunk = {
                                            'id': chunk_id,
                                            'object': 'chat.completion.chunk',
                                            'created': created_time,
                                            'model': model,
                                            'choices': [{
                                                'index': 0,
                                                'delta': {'content': content},
                                                'finish_reason': None
                                            }]
                                        }
                                        yield f"data: {json.dumps(openai_chunk)}\n\n"
                        
                        # Reset buffer after successful parsing
                        buffer = ""
                    
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error (incomplete JSON, continuing): {e}, Buffer: {buffer}")
                        # Continue accumulating lines if JSON is incomplete
                        continue
                    except Exception as e:
                        print(f"Stream processing error: {e}, Buffer: {buffer}")
                        buffer = ""
                        continue
                
                # Handle any remaining buffer data
                if buffer:
                    print(f"Final buffer before closing: {buffer}")
                    try:
                        test_buffer = buffer if not buffer.startswith('[') else (buffer + ']' if not buffer.endswith(']') else buffer)
                        chunks = json.loads(test_buffer) if test_buffer else []
                        print(f"Processed final chunks: {test_buffer}")
                        
                        if isinstance(chunks, list):
                            for chunk_data in chunks:
                                if 'candidates' in chunk_data:
                                    candidate = chunk_data['candidates'][0]
                                    if 'content' in candidate and 'parts' in candidate['content']:
                                        for part in candidate['content']['parts']:
                                            if 'text' in part:
                                                content = part['text']
                                                openai_chunk = {
                                                    'id': chunk_id,
                                                    'object': 'chat.completion.chunk',
                                                    'created': created_time,
                                                    'model': model,
                                                    'choices': [{
                                                        'index': 0,
                                                        'delta': {'content': content},
                                                        'finish_reason': None
                                                    }]
                                                }
                                                yield f"data: {json.dumps(openai_chunk)}\n\n"
                        elif isinstance(chunks, dict) and 'candidates' in chunks:
                            candidate = chunks['candidates'][0]
                            if 'content' in candidate and 'parts' in candidate['content']:
                                for part in candidate['content']['parts']:
                                    if 'text' in part:
                                        content = part['text']
                                        openai_chunk = {
                                            'id': chunk_id,
                                            'object': 'chat.completion.chunk',
                                            'created': created_time,
                                            'model': model,
                                            'choices': [{
                                                'index': 0,
                                                'delta': {'content': content},
                                                'finish_reason': None
                                            }]
                                        }
                                        yield f"data: {json.dumps(openai_chunk)}\n\n"
                    except json.JSONDecodeError as e:
                        print(f"Final JSON decode error: {e}, Buffer: {buffer}")
                        error_chunk = {
                            'id': chunk_id,
                            'object': 'chat.completion.chunk',
                            'created': created_time,
                            'model': model,
                            'choices': [{
                                'index': 0,
                                'delta': {'content': f"Stream error: Incomplete JSON response from API"},
                                'finish_reason': 'error'
                            }]
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"
    
    except BrokenPipeError:
        print("Client closed connection prematurely (BrokenPipeError)")
        error_chunk = {
            'id': chunk_id,
            'object': 'chat.completion.chunk',
            'created': created_time,
            'model': model,
            'choices': [{
                'index': 0,
                'delta': {'content': "Client disconnected, streaming aborted"},
                'finish_reason': 'error'
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return
    
    except Exception as e:
        print(f"Exception caught: {str(e)}")
        error_chunk = {
            'id': chunk_id,
            'object': 'chat.completion.chunk',
            'created': created_time,
            'model': model,
            'choices': [{
                'index': 0,
                'delta': {'content': f"Stream error: {str(e)}"},
                'finish_reason': 'error'
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    
    # Final chunk - Ensure proper termination
    final_chunk = {
        'id': chunk_id,
        'object': 'chat.completion.chunk',
        'created': created_time,
        'model': model,
        'choices': [{
            'index': 0,
            'delta': {},
            'finish_reason': 'stop'
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
    print("Streaming completed with final chunk and [DONE]")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        gemini_messages = convert_messages(request.messages)
        generation_config = {"temperature": request.temperature}
        if request.max_tokens:
            generation_config["max_output_tokens"] = request.max_tokens
        
        api_key = get_next_key()
        
        if request.stream:
            return StreamingResponse(
                real_gemini_stream(api_key, gemini_messages, generation_config, request.model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Access-Control-Allow-Origin": "*",
                    "Transfer-Encoding": "chunked",
                    "Content-Type": "text/event-stream"
                }
            )
        else:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{request.model}:generateContent",
                    json={
                        "contents": gemini_messages,
                        "generationConfig": generation_config
                    },
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": api_key
                    }
                )
                
                if not response.is_success:
                    raise HTTPException(status_code=response.status_code, detail="Gemini API error")
                
                result = response.json()
                text = ""
                if result.get("candidates"):
                    candidate = result["candidates"][0]
                    if candidate.get("content", {}).get("parts"):
                        text = "".join(part.get("text", "") for part in candidate["content"]["parts"])
                
                if not text:
                    text = "No response generated"
                
                return {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                }
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "key_usage": sum(key_usage.values()),
        "stream_type": "real_gemini_stream"
    }

@app.middleware("http")
async def cors_handler(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.options("/{path:path}")
async def options_handler():
    return {"message": "OK"}