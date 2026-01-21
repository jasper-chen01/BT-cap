"""
Chat agent endpoints for conversational interface
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, List, Dict
import tempfile
import os
import uuid
import json

from backend.models.schemas import ChatMessage, ChatResponse, ChatSession
from backend.services.chat_agent import ChatAgent

router = APIRouter()

# In-memory session storage (in production, use Redis or database)
chat_sessions: Dict[str, ChatSession] = {}


@router.post("/chat/session", response_model=ChatSession)
async def create_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    session = ChatSession(
        session_id=session_id,
        messages=[],
        uploaded_files=[]
    )
    chat_sessions[session_id] = session
    return session


@router.get("/chat/session/{session_id}", response_model=ChatSession)
async def get_session(session_id: str):
    """Get chat session by ID"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return chat_sessions[session_id]


@router.post("/chat/{session_id}/message", response_model=ChatResponse)
async def send_message(
    session_id: str,
    message: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Send a message to the chat agent
    
    Args:
        session_id: Chat session ID
        message: User message text
        file: Optional file upload (h5ad format)
    
    Returns:
        ChatResponse with agent's reply
    """
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = chat_sessions[session_id]
    chat_agent = ChatAgent()
    
    # Handle file upload if present
    file_path = None
    if file:
        if not file.filename.endswith('.h5ad'):
            raise HTTPException(
                status_code=400,
                detail="File must be in h5ad format"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5ad') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            file_path = tmp_file.name
        
        session.uploaded_files.append({
            "filename": file.filename,
            "path": file_path,
            "session_id": session_id
        })
    
    # Add user message to session
    user_msg = ChatMessage(
        role="user",
        content=message,
        file_uploaded=file.filename if file else None
    )
    session.messages.append(user_msg)
    
    # Get agent response
    try:
        response = await chat_agent.process_message(
            session=session,
            user_message=message,
            file_path=file_path
        )
        
        # Add agent response to session
        session.messages.append(response.message)
        
        return response
        
    except Exception as e:
        error_msg = ChatMessage(
            role="assistant",
            content=f"I encountered an error: {str(e)}. Please try again or rephrase your question."
        )
        session.messages.append(error_msg)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )


@router.post("/chat/{session_id}/annotate", response_model=ChatResponse)
async def annotate_via_chat(
    session_id: str,
    file_index: int = Form(0),
    top_k: Optional[int] = Form(10),
    similarity_threshold: Optional[float] = Form(0.7)
):
    """
    Annotate an uploaded file via chat interface
    
    Args:
        session_id: Chat session ID
        file_index: Index of file in session's uploaded_files
        top_k: Number of nearest neighbors
        similarity_threshold: Minimum similarity score
    """
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = chat_sessions[session_id]
    
    if file_index >= len(session.uploaded_files):
        raise HTTPException(status_code=400, detail="Invalid file index")
    
    file_info = session.uploaded_files[file_index]
    file_path = file_info["path"]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    chat_agent = ChatAgent()
    
    try:
        response = await chat_agent.annotate_file(
            session=session,
            file_path=file_path,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        session.messages.append(response.message)
        return response
        
    except Exception as e:
        error_msg = ChatMessage(
            role="assistant",
            content=f"Error annotating file: {str(e)}"
        )
        session.messages.append(error_msg)
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )


@router.delete("/chat/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session and clean up files"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = chat_sessions[session_id]
    
    # Clean up uploaded files
    for file_info in session.uploaded_files:
        if os.path.exists(file_info["path"]):
            try:
                os.unlink(file_info["path"])
            except:
                pass
    
    del chat_sessions[session_id]
    return {"status": "deleted"}
