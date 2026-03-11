"""
Chat agent service for conversational interface
"""
import asyncio
import re
from typing import Optional, List, Dict
from datetime import datetime
from collections import Counter

from backend.models.schemas import ChatMessage, ChatResponse, ChatSession
from backend.services.annotation_service import AnnotationService
from backend.config import settings
from pathlib import Path
from google.cloud import storage
import openpyxl
import os
import json



try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional dependency
    genai = None

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    from google.oauth2 import service_account
except Exception:  # pragma: no cover - optional dependency
    vertexai = None
    GenerativeModel = None
    service_account = None


class ChatAgent:
    """AI agent for conversational interaction"""
    
    def __init__(self):
        self.annotation_service = AnnotationService()
        self.greetings = [
            "hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"
        ]
        self.help_keywords = ["help", "guide", "tutorial", "instructions"]
        self.annotate_keywords = ["annotate", "analyze", "process", "classify", "label", "predict"]
        self.status_keywords = ["status", "health", "ready", "available"]
        self.gemini_model = None
        self.gemini_enabled = False
        self._init_gemini()

    def _init_gemini(self) -> None:
        """Initialize Gemini client if configured."""
        if settings.VERTEX_PROJECT_ID and vertexai is not None and GenerativeModel is not None:
            credentials = None
            if settings.GOOGLE_APPLICATION_CREDENTIALS and service_account is not None:
                credentials = service_account.Credentials.from_service_account_file(
                    settings.GOOGLE_APPLICATION_CREDENTIALS
                )
            vertexai.init(
                project=settings.VERTEX_PROJECT_ID,
                location=settings.VERTEX_LOCATION,
                credentials=credentials
            )
            self.gemini_model = GenerativeModel(settings.GEMINI_MODEL)
            self.gemini_enabled = True
            return

        if settings.GEMINI_API_KEY and genai is not None:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL)
            self.gemini_enabled = True
    
    async def process_message(
        self,
        session: ChatSession,
        user_message: str,
        file_path: Optional[str] = None
    ) -> ChatResponse:
        """
        Process user message and generate appropriate response
        
        Args:
            session: Current chat session
            user_message: User's message text
            file_path: Optional path to uploaded file
        
        Returns:
            ChatResponse with agent's reply
        """
        message_lower = user_message.lower().strip()
        

        
        # Handle greetings
        if any(self._has_keyword(message_lower, greeting) for greeting in self.greetings):
            return self._greeting_response(session.session_id)
        
        # Handle help requests
        if self._is_help_intent(message_lower):
            return self._help_response(session.session_id)

        
        # Handle status checks
        if any(self._has_keyword(message_lower, keyword) for keyword in self.status_keywords):
            return await self._status_response(session.session_id)
        
        # Handle file upload
        if file_path:
            return await self._handle_file_upload(session, file_path, user_message)
        
        # Handle annotation requests
        if any(self._has_keyword(message_lower, keyword) for keyword in self.annotate_keywords):
            if session.uploaded_files:
                # Annotate the most recent file
                return await self._annotate_latest_file(session)
            else:
                return self._request_file_upload(session.session_id)
        
        # Handle explicit annotation requests with file reference
        file_ref_match = re.search(r'file\s*(\d+)', message_lower)
        if file_ref_match and session.uploaded_files:
            file_index = int(file_ref_match.group(1)) - 1
            if 0 <= file_index < len(session.uploaded_files):
                return await self._annotate_specific_file(session, file_index)
        
        # Extract parameters from message
        top_k = self._extract_number(message_lower, r'top[_\s]?k[:\s]?(\d+)', default=10)
        threshold = self._extract_number(message_lower, r'threshold[:\s]?([\d.]+)', default=0.7, is_float=True)

        if getattr(session, "analysis_context", None) and self.gemini_enabled:
            return await self._gemini_response(session, user_message)

        direct = self._maybe_answer_from_analysis_context(session, user_message)
        if direct:
            return direct

        if self.gemini_enabled:
            return await self._gemini_response(session, user_message)
        return self._default_response(session.session_id, session.uploaded_files)
    
    async def annotate_file(
        self,
        session: ChatSession,
        file_path: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7
    ) -> ChatResponse:
        """Annotate a file and return chat response"""
        try:
            result = await self.annotation_service.annotate_file(
                file_path,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            # Generate summary
            annotations = result.annotations
            annotation_counts = {}
            total_confidence = 0
            
            for cell in annotations:
                ann = cell.predicted_annotation
                annotation_counts[ann] = annotation_counts.get(ann, 0) + 1
                total_confidence += cell.confidence_score
            
            avg_confidence = total_confidence / len(annotations) if annotations else 0

            sorted_counts = sorted(annotation_counts.items(), key=lambda x: -x[1])
            top_types = [
                {"cell_type": ann, "count": count, "pct": round((count / result.total_cells) * 100, 1)}
                for ann, count in sorted_counts[:5]
            ]

            session.analysis_context = {
                "type": "annotation_summary",
                "total_cells": result.total_cells,
                "avg_confidence": round(avg_confidence, 4),
                "top_cell_types": top_types,
            }

            # Plain-text user-facing message (no emojis/markdown)
            lines = []
            lines.append(f"Annotated {result.total_cells} cells.")
            lines.append("Top cell types:")
            for item in top_types:
                lines.append(f"{item['cell_type']}: {item['count']} ({item['pct']}%)")
            lines.append(f"Average confidence: {avg_confidence * 100:.1f}%")

            content = "\n".join(lines)
            
            message = ChatMessage(
                role="assistant",
                content=content,
                annotation_results={
                    "total_cells": result.total_cells,
                    "annotation_counts": annotation_counts,
                    "average_confidence": avg_confidence,
                    "annotations": [
                        {
                            "cell_id": cell.cell_id,
                            "predicted_annotation": cell.predicted_annotation,
                            "confidence_score": cell.confidence_score
                        }
                        for cell in annotations[:10]  
                    ]
                }
            )
            
            return ChatResponse(
                message=message,
                session_id=session.session_id,
                suggestions=[
                    "Download full results",
                    "Visualize annotations",
                    "Upload another file"
                ]
            )
            
        except Exception as e:
            error_message = ChatMessage(
                role="assistant",
                content=f"❌ Error annotating file: {str(e)}\n\nPlease check that:\n- The file is in h5ad format\n- The file contains valid single-cell data\n- Reference embeddings are loaded"
            )
            return ChatResponse(
                message=error_message,
                session_id=session.session_id
            )
    
    def _greeting_response(self, session_id: str) -> ChatResponse:
        """Generate greeting response"""
        content = """Hello! I'm your Brain Tumor Annotation Assistant.\n
I can help you:\n
- Upload and annotate single-cell glioma data\n
- Analyze your data using our reference embeddings\n
- Provide detailed annotation results\n

You can upload a file by dragging it into the chat or typing "upload file".\n How can I help you today?"""
        
        message = ChatMessage(role="assistant", content=content)
        return ChatResponse(
            message=message,
            session_id=session_id,
            suggestions=["Upload a file", "How does this work?", "Check system status"]
        )
    
    def _help_response(self, session_id: str) -> ChatResponse:
        """Generate help response"""
        content = """ How to use the Brain Tumor Annotation Portal:\n

1. Upload Data: Drag and drop a `.h5ad` file or type "upload file"\n
2. Annotate: Say "annotate" or "analyze my data" to process uploaded files\n
3. Customize: Specify parameters like "top k: 20" or "threshold: 0.8"\n
4. Download: Request to download results after annotation\n

Example Commands:\n
- "Upload my glioma data"
- "Annotate with top k 15"
- "Analyze file 1 with threshold 0.75"
- "What's the system status?"

Supported Formats:
- Input: `.h5ad` files (AnnData format)
- Output: CSV with annotations and confidence scores

Need more help? Just ask!"""
        
        message = ChatMessage(role="assistant", content=content)
        return ChatResponse(
            message=message,
            session_id=session_id,
            suggestions=["Upload a file", "Check system status"]
        )
    
    async def _status_response(self, session_id: str) -> ChatResponse:
        """Check system status"""
        status = self.annotation_service.get_status()
        
        if status["reference_loaded"] and status["index_loaded"]:
            content = f"""System Status: Ready

- Reference data: Loaded
- Embeddings index: Ready ({status['index_size']:,} reference cells)
- System: Operational

You can upload and annotate files now!"""
        else:
            ref_status = "Loaded" if status['reference_loaded'] else "Not loaded"
            idx_status = " Ready" if status['index_loaded'] else "Not ready"
            
            content = f"""System Status: Not Ready

- Reference data: {ref_status}
- Embeddings index: {idx_status}

Please ensure reference data is prepared. Run `python scripts/prepare_reference_embeddings.py` if needed."""
        
        message = ChatMessage(role="assistant", content=content)
        return ChatResponse(
            message=message,
            session_id=session_id
        )
    
    async def _handle_file_upload(
        self,
        session: ChatSession,
        file_path: str,
        user_message: str
    ) -> ChatResponse:
        """Handle file upload"""
        filename = session.uploaded_files[-1]["filename"] if session.uploaded_files else "file"
        
        content = f"""📁 **File uploaded successfully!**

File: `{filename}`

You can now:
- Say "annotate" to analyze this file
- Specify parameters: "annotate with top k 15 and threshold 0.8"
- Upload more files for batch processing

Ready to annotate?"""
        
        message = ChatMessage(
            role="assistant",
            content=content,
            file_uploaded=filename
        )
        
        return ChatResponse(
            message=message,
            session_id=session.session_id,
            suggestions=["Annotate this file", "Upload another file"],
            requires_action="annotate"
        )
    
    async def _annotate_latest_file(self, session: ChatSession) -> ChatResponse:
        """Annotate the most recently uploaded file"""
        if not session.uploaded_files:
            return self._request_file_upload(session.session_id)
        
        file_info = session.uploaded_files[-1]
        return await self.annotate_file(session, file_info["path"])
    
    async def _annotate_specific_file(self, session: ChatSession, file_index: int) -> ChatResponse:
        """Annotate a specific file by index"""
        file_info = session.uploaded_files[file_index]
        return await self.annotate_file(session, file_info["path"])
    
    def _request_file_upload(self, session_id: str) -> ChatResponse:
        """Request file upload"""
        content = """No file uploaded yet

Please upload a `.h5ad` file to get started. You can:\n
- Drag and drop a file into the chat\n
- Click the upload button\n
- Or type "upload file" and select a file\n

Once uploaded, I can annotate it for you!"""
        
        message = ChatMessage(role="assistant", content=content)
        return ChatResponse(
            message=message,
            session_id=session_id,
            requires_action="upload_file"
        )
    
    def _default_response(self, session_id: str, uploaded_files: List) -> ChatResponse:
        """Default response when intent is unclear"""
        if uploaded_files:
            content = """I'm not sure what you're asking. Here's what I can help with:

- Annotate files: Say "annotate" or "analyze my data"
- Get help: Ask "how does this work?" or "help"
- Check status: Ask "what's the system status?"

You have uploaded files ready to annotate. Would you like me to analyze them?"""
        else:
            content = """I'm here to help you annotate glioma single-cell data!

Try saying:
- "Upload a file" to get started
- "How does this work?" for instructions
- "What can you do?" to see my capabilities

What would you like to do?"""
        
        message = ChatMessage(role="assistant", content=content)
        suggestions = ["Upload a file", "How does this work?", "Check system status"] if not uploaded_files else ["Annotate files", "Get help"]
        
        return ChatResponse(
            message=message,
            session_id=session_id,
            suggestions=suggestions
        )

    def _build_gemini_prompt(self, session: ChatSession, user_message: str) -> str:
        """Build a prompt with analysis/visualization context for Gemini."""
        recent_messages = session.messages[-6:] if session.messages else []
        history_lines = [
            f"{msg.role}: {msg.content}" for msg in recent_messages if msg.content
        ]
        history_block = "\n".join(history_lines) if history_lines else "No prior messages."

        uploaded_names = [f["filename"] for f in session.uploaded_files] if session.uploaded_files else []
        uploaded_text = ", ".join(uploaded_names) if uploaded_names else "none"

        system_prompt = (
            "You are the Brain Tumor Annotation Assistant for a web app that annotates and visualizes "
            "single-cell data. Be concise, natural, and helpful.\n"
            "If the user asks about upload, annotation, or status, answer directly.\n"
            "If a latest analysis summary is provided, treat it as the source of truth for the current dataset.\n"
            "Do not tell the user to upload or annotate first if analysis summary context is already present.\n"
            "For gene-program score questions, explain that scores reflect relative enrichment, not definitive cell identity.\n"
            "Use exact numeric values from the summary when available, but also interpret them in plain language or using your own resources.\n"
            "If information is not present in the summary, say so clearly.\n"
            "Keep the tone conversational and natural.\n"
            "No profanity or swearing."
        )

        analysis_block = "none"
        ctx = getattr(session, "analysis_context", None)

        if ctx:
            total_cells = ctx.get("total_cells")
            cluster_counts = ctx.get("cluster_counts") or {}
            meta = ctx.get("metadata") or {}
            program_details = ctx.get("program_details") or {}
            cell_types = ctx.get("cell_types") or []

            top_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            top_clusters_str = ", ".join([f"{k}={v}" for k, v in top_clusters]) if top_clusters else "none"

            available_programs = meta.get("available_programs") or list(program_details.keys())
            selected_program = meta.get("selected_program") or "none"

            program_preview_lines = []
            for program_name in available_programs[:12]:
                detail = program_details.get(program_name) or {}
                score_stats = detail.get("score_stats") or {}
                top_prog_clusters = detail.get("top_clusters") or {}
                top_cluster_items = list(top_prog_clusters.items())[:3]
                top_cluster_str = ", ".join([f"{k}:{v}" for k, v in top_cluster_items]) if top_cluster_items else "none"

                program_preview_lines.append(
                    f"{program_name}: "
                    f"matched={detail.get('present_genes')}/{detail.get('total_genes')}, "
                    f"missing={detail.get('missing_genes')}, "
                    f"p05={score_stats.get('p05')}, "
                    f"p95={score_stats.get('p95')}, "
                    f"top_clusters={top_cluster_str}"
                )

            program_preview = "\n".join(program_preview_lines) if program_preview_lines else "none"

            full_program_details_block = json.dumps(
                program_details,
                ensure_ascii=False
            ) if program_details else "none"

            analysis_block = (
                f"total_cells={total_cells}\n"
                f"cluster_count={len(cluster_counts)}\n"
                f"top_clusters={top_clusters_str}\n"
                f"cell_type_summary_count={len(cell_types) if isinstance(cell_types, list) else 0}\n"
                f"selected_program={selected_program}\n"
                f"available_programs={', '.join(available_programs[:50]) if available_programs else 'none'}\n"
                f"program_preview:\n{program_preview}\n"
                f"full_program_details:\n{full_program_details_block}"
            )
        else:
            latest_analysis = None
            for msg in reversed(session.messages):
                if msg.role == "assistant" and getattr(msg, "annotation_results", None):
                    latest_analysis = msg.annotation_results
                    break

            if latest_analysis:
                counts = latest_analysis.get("annotation_counts", {}) or {}
                total_cells = latest_analysis.get("total_cells")
                avg_conf = latest_analysis.get("average_confidence")

                top5 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
                top5_str = ", ".join([f"{k}={v}" for k, v in top5]) if top5 else "none"

                analysis_block = f"total_cells={total_cells}, avg_confidence={avg_conf}, top_celltypes={top5_str}"

        return (
            f"{system_prompt}\n\n"
            f"Uploaded files: {uploaded_text}\n\n"
            f"Latest analysis summary:\n{analysis_block}\n\n"
            f"When the user asks for interpretation, give aninterpretation based on the analysis summary and your knowledge.\n"
            f"Conversation:\n{history_block}\n\n"
            f"User: {user_message}\nAssistant:"
        )

    async def _gemini_response(self, session: ChatSession, user_message: str) -> ChatResponse:
        """Generate a Gemini response for general conversation."""
        prompt = self._build_gemini_prompt(session, user_message)
        try:
            response = await asyncio.to_thread(self.gemini_model.generate_content, prompt)
            content = (response.text or "").strip()
            if not content:
                return self._default_response(session.session_id, session.uploaded_files)
            message = ChatMessage(role="assistant", content=content)
            return ChatResponse(message=message, session_id=session.session_id)
        except Exception as e:
            print("GEMINI ERROR:", repr(e))
            return self._default_response(session.session_id, session.uploaded_files)
    
    def _has_keyword(self, text: str, keyword: str) -> bool:
        """Match keyword as a whole word (or exact phrase if multi-word)."""
        keyword = keyword.strip().lower()
        if not keyword:
            return False
        if " " in keyword:
            return keyword in text
        return re.search(rf"\b{re.escape(keyword)}\b", text) is not None

    
    def _extract_number(self, text: str, pattern: str, default: float, is_float: bool = False) -> float:
        """Extract number from text using regex pattern"""
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1)) if is_float else int(match.group(1))
        return default
    
    def _is_help_intent(self, text: str) -> bool:
        """
        True only when the message is basically asking for help, not just containing the word.
        Examples that return True: "help", "help me", "instructions", "tutorial", "guide"
        Examples that return False: "how to bake a cake", "can you help me find weights for CHRM2"
        """
        t = (text or "").strip().lower()
        if not t:
            return False

        # exact commands
        if t in {"help", "guide", "tutorial", "instructions"}:
            return True

        # short help-like phrases only
        if re.fullmatch(r"(help|help me|need help|show help|show instructions|instructions|tutorial|guide)\b.*", t):
            # If it's long and clearly not about the app, don't hijack it
            # (you can remove this if you want help to still trigger more aggressively)
            if len(t.split()) > 6 and ("portal" not in t and "annotat" not in t and "upload" not in t):
                return False
            return True

        return False
   
    def _maybe_answer_from_analysis_context(self, session: ChatSession, user_message: str) -> Optional[ChatResponse]:
        """
        Minimal fallback only when Gemini is unavailable.
        Keep this intentionally small so conversational analysis stays natural.
        """
        ctx = getattr(session, "analysis_context", None)
        if not ctx:
            return None

        text = (user_message or "").strip().lower()
        total_cells = ctx.get("total_cells")
        cluster_counts = ctx.get("cluster_counts") or {}
        path = ctx.get("_analysis_summary_path") or ctx.get("analysis_summary_path")

        if "path" in text and ("summary" in text or "analysis" in text):
            return ChatResponse(
                message=ChatMessage(role="assistant", content=f"analysis_summary_path = {path}"),
                session_id=session.session_id,
            )

        if "total cells" in text:
            return ChatResponse(
                message=ChatMessage(role="assistant", content=f"total_cells = {total_cells}"),
                session_id=session.session_id,
            )

        if "cluster counts" in text:
            items = sorted(cluster_counts.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]))
            preview = ", ".join([f"{k}:{v}" for k, v in items[:10]])
            return ChatResponse(
                message=ChatMessage(role="assistant", content=f"cluster_counts (first 10): {preview}"),
                session_id=session.session_id,
            )

        return None



    
