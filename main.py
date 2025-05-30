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
        async with httpx.AsyncClient(timeout=60) as client:
            # Gemini stream endpoint kullan
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
                    raise HTTPException(
                        status_code=response.status_code, 
                        detail=f"Gemini API error: {error_text.decode()}"
                    )
                
                # Track if we've received any content
                content_received = False
                
                # Process real stream chunks
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Gemini stream format: data: {...}
                    if line.startswith('data: '):
                        try:
                            json_str = line[6:]  # Remove 'data: ' prefix
                            if json_str == '[DONE]':
                                break
                                
                            chunk_data = json.loads(json_str)
                            
                            # Check if this is the final chunk from Gemini
                            is_final_chunk = False
                            if 'candidates' in chunk_data:
                                candidate = chunk_data['candidates'][0]
                                
                                # Check finish reason from Gemini
                                if 'finishReason' in candidate:
                                    is_final_chunk = True
                                
                                # Extract text if available
                                if 'content' in candidate and 'parts' in candidate['content']:
                                    for part in candidate['content']['parts']:
                                        if 'text' in part:
                                            content = part['text']
                                            content_received = True
                                            
                                            # Send OpenAI format chunk
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
                            # Skip invalid JSON lines
                            continue
                        except Exception as e:
                            # Log error but continue streaming
                            print(f"Stream processing error: {e}")
                            continue
    
    except Exception as e:
        # Send error as content
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
    
    # CRITICAL: Send final chunk with empty delta BEFORE finish_reason
    empty_chunk = {
        'id': chunk_id,
        'object': 'chat.completion.chunk',
        'created': created_time,
        'model': model,
        'choices': [{
            'index': 0,
            'delta': {},
            'finish_reason': None
        }]
    }
    yield f"data: {json.dumps(empty_chunk)}\n\n"
    
    # Final chunk with finish_reason - CRITICAL for Cline
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

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        # Convert messages
        gemini_messages = convert_messages(request.messages)
        generation_config = {"temperature": request.temperature}
        if request.max_tokens:
            generation_config["max_output_tokens"] = request.max_tokens
        
        api_key = get_next_key()
        
        if request.stream:
            # Gerçek stream kullan
            return StreamingResponse(
                real_gemini_stream(api_key, gemini_messages, generation_config, request.model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Access-Control-Allow-Origin": "*",
                    "Transfer-Encoding": "chunked"
                }
            )
        else:
            # Non-stream için standart endpoint
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

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "gemini-pro", "object": "model", "created": int(time.time()), "owned_by": "google"},
            {"id": "gemini-pro-vision", "object": "model", "created": int(time.time()), "owned_by": "google"},
            {"id": "gemini-1.5-flash", "object": "model", "created": int(time.time()), "owned_by": "google"},
            {"id": "gemini-1.5-pro", "object": "model", "created": int(time.time()), "owned_by": "google"},
            {"id": "gemini-2.0-flash-exp", "object": "model", "created": int(time.time()), "owned_by": "google"}
        ]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "key_usage": sum(key_usage.values()),
        "stream_type": "real_gemini_stream"
    }

# CORS Middleware
@app.middleware("http")
async def cors_handler(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# Handle preflight requests
@app.options("/{path:path}")
async def options_handler():
    return {"message": "OK"}