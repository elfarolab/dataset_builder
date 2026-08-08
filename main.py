# main.py
import os
import json
import time
import glob
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import re
import string

import asyncio
import pdfplumber
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import yaml
import certifi

import base64
from image_extractor import (
    extract_images_from_pdf,
    extract_images_from_web_dir,
    image_to_base64,
    get_image_context_text,
)


# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# App configuration
APP_CONFIG = config.get("app", {})
APP_NAME = APP_CONFIG.get("name", "Dataset Builder")
APP_SUBTITLE = APP_CONFIG.get("subtitle", "Review and refine AI training data")
PERSONA = APP_CONFIG.get("persona", "expert AI assistant")
AUDIENCE = APP_CONFIG.get("audience", "general public")


# Load prompt templates
with open("prompts.yaml", "r") as f:
    prompt_config = yaml.safe_load(f)

TEXT_QA_TEMPLATE = string.Template(prompt_config["text_qa_prompt"])
IMAGE_QA_TEMPLATE = string.Template(prompt_config["image_qa_prompt"])
IMAGE_CONTEXT_TEMPLATE = string.Template(prompt_config["image_context_prompt"])
SEMANTIC_CHUNK_TEMPLATE = string.Template(prompt_config["semantic_chunk_prompt"])


@asynccontextmanager
async def lifespan(app):
    load_state()
    yield


app = FastAPI(title=APP_NAME, lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration from config.yaml
LLAMA_SERVER_URL = config["llama_server"]["url"]
LLAMA_TIMEOUT = config["llama_server"].get("timeout", 120)
LLAMA_MAX_TOKENS = config["llama_server"].get("max_tokens", 5000)
LLAMA_TEMPERATURE = config["llama_server"].get("temperature", 0.5)
LLAMA_TOP_P = config["llama_server"].get("top_p", 0.9)
LLAMA_MODEL_NAME = config["llama_server"].get("name", "llm")

PDF_DIR = config["paths"]["pdf_dir"]
WEB_DIR = config["paths"]["web_dir"]
RESULT_DIR = config["paths"]["result_dir"]
STATE_FILE = config["paths"]["state_file"]

CHUNK_SIZE = config["processing"]["chunk_size"]
QA_PER_CHUNK = config["processing"].get("qa_per_chunk", 5)
BATCH_DELAY = config["processing"].get("batch_delay", 5.0)


# Ensure directories exist
for d in [PDF_DIR, WEB_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)


# Globals ------------------


# Global state
processed_documents: List[Dict] = []  # List of all Q&A pairs with metadata
is_processing = False

# Add this global to track the active processing task
_active_processing_task = None

# Global progress tracking state
# Global progress tracking state
processing_state = {
    "is_processing": False,
    "current_source": None,
    "progress_percent": 0,
    "queue": [],
    "completed": [],
    # Per-phase tracking
    "phase": None,               # "text" or "images"
    "text_chunks_completed": 0,
    "text_chunks_total": 0,
    "images_completed": 0,
    "images_total": 0,
}
should_cancel_current = False
should_cancel_all = False



class QAEntry(BaseModel):
    id: str
    instruction: str
    input: str
    output: str
    source_file: str
    chunk_index: int
    enabled: bool = True
    edited: bool = False
    reviewed: bool = False


class ReviewUpdate(BaseModel):
    id: str
    instruction: Optional[str] = None
    input: Optional[str] = None
    output: Optional[str] = None
    enabled: Optional[bool] = None
    reviewed: Optional[bool] = None


class URLSourceRequest(BaseModel):
    url: str


class TextSourceRequest(BaseModel):
    text: str

class ProcessRequest(BaseModel):
    sources: Optional[List[str]] = None  # List of source filenames to process
    images_only: Optional[List[str]] = None


# Functions ----------------

@app.get("/api/processing-status")
async def get_processing_status():
    """Return current processing state for frontend polling"""
    return processing_state

@app.post("/api/cancel-processing")
async def cancel_processing(req: dict = None):
    global should_cancel_current, should_cancel_all
    req = req or {}

    if req.get("all"):
        should_cancel_all = True
        print("🛑 Global cancellation flag set")
    elif req.get("current"):
        should_cancel_current = True
        print("🛑 Current source cancellation flag set")

    # No /abort call needed. Cancellation is handled inside the streaming loops.
    return {"status": "cancelled"}


@app.get("/api/available-sources")
async def get_available_sources():
    """Get ALL source files on disk (processed and unprocessed) with their status"""
    sources = []

    # Scan PDF directory
    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith('.pdf'):
            filepath = os.path.join(PDF_DIR, filename)
            
            # Count existing Q&A entries for this file
            file_entries = [e for e in processed_documents if e.source_file == filename]
            total_qa = len(file_entries)
            reviewed_qa = sum(1 for e in file_entries if getattr(e, 'reviewed', False))
            
            sources.append({
                "filename": filename,
                "type": "PDF",
                "exists": True,
                "total_qa": total_qa,
                "reviewed_qa": reviewed_qa,
                "is_processed": total_qa > 0,
            })

    # Scan text directory
    for filename in os.listdir(WEB_DIR):
        if filename.lower().endswith('.txt'):
            filepath = os.path.join(WEB_DIR, filename)
            
            file_entries = [e for e in processed_documents if e.source_file == filename]
            total_qa = len(file_entries)
            reviewed_qa = sum(1 for e in file_entries if getattr(e, 'reviewed', False))
            
            sources.append({
                "filename": filename,
                "type": "Text",
                "exists": True,
                "total_qa": total_qa,
                "reviewed_qa": reviewed_qa,
                "is_processed": total_qa > 0,
            })

    # Sort: unprocessed first, then alphabetically
    sources.sort(key=lambda s: (s["is_processed"], s["filename"]))

    return {
        "sources": sources,
        "total": len(sources),
        "unprocessed": sum(1 for s in sources if not s["is_processed"]),
        "processed": sum(1 for s in sources if s["is_processed"]),
    }

async def generate_qa_from_image(image_info: Dict, source_file: str, image_index: int, text_context: str = "") -> List[QAEntry]:
    image_b64 = image_to_base64(image_info["image_bytes"])
    img_format = image_info.get("format", "png").lower()
    verbose = config.get("processing", {}).get("debug_print", False)

    page_info = f"Page: {image_info.get('page', 'N/A')}" if image_info.get("page", 0) > 0 else ""

    system_prompt = IMAGE_QA_TEMPLATE.safe_substitute(
        persona=PERSONA,
        source_file=source_file,
        page_info=page_info,
        width=image_info.get("width", 0),
        height=image_info.get("height", 0),
        image_qa_per_image=config.get('processing', {}).get('image_qa_per_image', 3),
        audience=AUDIENCE
    )

    if text_context:
        system_prompt += IMAGE_CONTEXT_TEMPLATE.safe_substitute(text_context=text_context)

    if verbose:
        print(f"\n🐛 [DEBUG] RAW IMAGE PROMPT SENT TO LLM (first 2000 chars):\n{system_prompt[:2000]}...\n")

    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        if should_cancel_all or (should_cancel_current and processing_state.get("current_source") == source_file):
            return []

        try:
            async with httpx.AsyncClient(timeout=LLAMA_TIMEOUT * 2) as client:
                # Use /v1/chat/completions for images (better multimodal support)
                payload = {
                    "model": LLAMA_MODEL_NAME,
                    "messages": [
                        {"role": "user", "content": [
                            {"type": "text", "text": system_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/{img_format};base64,{image_b64}"}}
                        ]}
                    ],
                    "max_tokens": LLAMA_MAX_TOKENS,
                    "temperature": LLAMA_TEMPERATURE,
                    "top_p": LLAMA_TOP_P,
                    "stream": True
                }

                print(f"Calling llama multimodal server (attempt {attempt + 1}/{max_retries})...")
                async with client.stream("POST", f"{LLAMA_SERVER_URL}/v1/chat/completions", json=payload) as response:
                    if response.status_code not in [200, 201]:
                        print(f"Error from llama server: {response.text[:500]}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                        continue

                    content = ""
                    raw_lines_debug = []  # Collect raw lines for debugging
                    try:
                        async for line in response.aiter_lines():
                            if should_cancel_all or (should_cancel_current and processing_state.get("current_source") == source_file):
                                print(f"🛑 Cancelled generation for {source_file}")
                                break

                            if not line or line.startswith(":"):
                                continue

                            if verbose:
                                raw_lines_debug.append(line[:200])  # Cap length for readability

                            if line.startswith("data: "):
                                data = line[6:].strip()
                                if data == "[DONE]": break
                                try:
                                    chunk = json.loads(data)
                                    # Handle both chat and completions streaming formats
                                    if "choices" in chunk:
                                        choice = chunk["choices"][0]
                                        # Chat completions format
                                        delta = choice.get("delta", {})
                                        content += delta.get("content", "") or ""
                                        # Completions format (non-chat)
                                        if "text" in choice:
                                            content += choice["text"] or ""
                                    # Some servers return at root level
                                    elif "content" in chunk:
                                        content += chunk["content"] or ""
                                except json.JSONDecodeError as e:
                                    if verbose:
                                        print(f"   ⚠️ JSON parse error on line: {line[:150]}")
                                    continue

                    except asyncio.CancelledError:
                        print(f"🛑 Stream cancelled for {source_file}")
                        return []  # Exit cleanly on cancellation
                    finally:
                        await response.aclose()  # Ensure connection closes gracefully

                    # === DEBUG: Print raw response if verbose ===
                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"🔍 RAW IMAGE RESPONSE FROM LLAMA ({len(content)} chars accumulated)")
                        print(f"   Total SSE lines received: {len(raw_lines_debug)}")
                        print(f"{'='*60}")
                        if not content:
                            print("⚠️ WARNING: No content was accumulated!")
                            print("First 10 raw SSE lines:")
                            for i, raw_line in enumerate(raw_lines_debug[:10]):
                                print(f"   [{i}] {raw_line}")
                        else:
                            print(f"Accumulated content ({len(content)} chars):")
                            print(content)
                        print(f"{'='*60}\n")

                    # Early return if cancelled mid-stream
                    if should_cancel_all or (should_cancel_current and processing_state.get("current_source") == source_file):
                        return []

                    # === CLEAN RESPONSE ===
                    # Strip thinking/reasoning tags FIRST
                    content_cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    content_cleaned = re.sub(r'<thinking>.*?</thinking>', '', content_cleaned, flags=re.DOTALL | re.IGNORECASE)
                    # Strip markdown code blocks
                    content_cleaned = re.sub(r'^```(?:jsonl|json)?\s*\n?', '', content_cleaned, flags=re.MULTILINE)
                    content_cleaned = re.sub(r'\n?```\s*$', '', content_cleaned, flags=re.MULTILINE)

                    if verbose:
                        print(f"🧹 Cleaned content ({len(content_cleaned)} chars):")
                        print(content_cleaned)
                        print()

                    # === DEFINE PATTERN BEFORE USING IT ===
                    json_pattern = re.compile(r'\{[^{}]*"instruction"[^{}]*"output"[^{}]*\}', re.DOTALL)

                    # === DEBUG: Show regex matches ===
                    if verbose:
                        matches = list(json_pattern.finditer(content_cleaned))
                        print(f"🔍 Regex found {len(matches)} potential JSON objects")
                        for i, m in enumerate(matches):
                            print(f"  Match {i} (first 150 chars): {m.group(0)[:150]}...")

                    qa_entries = []
                    for match in json_pattern.finditer(content_cleaned):
                        json_str = ' '.join(match.group(0).split())
                        try:
                            qa_data = json.loads(json_str)
                            if "instruction" not in qa_data or "output" not in qa_data:
                                continue
                            if "input" not in qa_data:
                                qa_data["input"] = ""
                            entry_id = f"{source_file.replace('/', '_').replace('.', '_')}_img_{image_index}_{len(qa_entries)}"
                            qa_entries.append(QAEntry(
                                id=entry_id,
                                instruction=qa_data.get("instruction", ""),
                                input=qa_data.get("input", ""),
                                output=qa_data.get("output", ""),
                                source_file=source_file,
                                chunk_index=-1,
                                enabled=True,
                                edited=False
                            ))
                        except json.JSONDecodeError as e:
                            if verbose:
                                print(f"   ⚠️ JSON parse error: {str(e)[:100]}")
                            continue

                    # Fallback: try line-by-line parsing if regex found nothing
                    if not qa_entries:
                        if verbose:
                            print("⚠️ Regex found no matches, trying line-by-line fallback...")
                        for line_num, line in enumerate(content_cleaned.strip().split('\n')):
                            line = re.sub(r'^[*`\s]*', '', line.strip()).strip()
                            line = re.sub(r'[*`]\s*$', '', line).strip()
                            if len(line) < 20:
                                continue
                            try:
                                qa_data = json.loads(line)
                                if "instruction" in qa_data and "output" in qa_data:
                                    if "input" not in qa_data:
                                        qa_data["input"] = ""
                                    entry_id = f"{source_file.replace('/', '_').replace('.', '_')}_img_{image_index}_{len(qa_entries)}"
                                    qa_entries.append(QAEntry(
                                        id=entry_id,
                                        instruction=qa_data.get("instruction", ""),
                                        input=qa_data.get("input", ""),
                                        output=qa_data.get("output", ""),
                                        source_file=source_file,
                                        chunk_index=-1,
                                        enabled=True,
                                        edited=False
                                    ))
                            except json.JSONDecodeError:
                                continue

                    if qa_entries:
                        print(f"✅ Successfully parsed {len(qa_entries)} Q&A entries from image")
                        return qa_entries

                    print(f"[WARN] No valid Q&A parsed from image. Retrying...")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2

        except httpx.ReadTimeout:
            print(f"Multimodal timeout on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
        except httpx.ConnectError as e:
            print(f"Multimodal connection error: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2

    return []



def get_files_to_skip_from_memory():
    """Determine which files are fully processed based on in-memory state"""
    files_to_skip = set()
    
    # Group entries by source file
    files_in_memory = {}
    for entry in processed_documents:
        src = entry.source_file
        if src not in files_in_memory:
            is_pdf = src.endswith('.pdf')
            file_path = os.path.join(PDF_DIR, src) if is_pdf else os.path.join(WEB_DIR, src)
            files_in_memory[src] = {
                'path': file_path,
                'count': 0,
                'exists': os.path.exists(file_path)
            }
        files_in_memory[src]['count'] += 1
    
    # Check each file against disk to see if it's unchanged
    for filename, info in files_in_memory.items():
        if not info['exists']:
            print(f"  Skipping {filename}: file no longer exists")
            files_to_skip.add(filename)
            continue
        
        # Simple heuristic: if we have entries for this file and it hasn't changed recently, skip
        # For better accuracy, you'd want to track chunk counts per file
        if info['count'] > 0:
            mtime = os.path.getmtime(info['path'])
            # If file was modified more than 1 minute ago, assume we've processed it
            if time.time() - mtime > 60:
                print(f"  Skipping {filename}: already processed (unchanged)")
                files_to_skip.add(filename)
            else:
                print(f"  Re-processing {filename}: recently modified")
    
    return files_to_skip



def safe_filename(name: str, fallback: str) -> str:
    name = "".join(c for c in name if c.isalnum() or c in "._-").strip("._")
    return name or fallback


def unique_path(directory: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(directory, filename)
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{base}_{counter}{ext}")
        counter += 1
    return candidate

def save_state():
    """Save current state to STATE_FILE for resuming later"""
    try:
        # Track processed files with their modification times
        processed_files = {}
        for entry in processed_documents:
            src = entry.source_file
            if src not in processed_files:
                file_path = os.path.join(PDF_DIR, src) if src.endswith('.pdf') else os.path.join(WEB_DIR, src)
                mtime = os.path.getmtime(file_path) if os.path.exists(file_path) else 0
                processed_files[src] = {
                    "modified_time": mtime,
                    "entry_count": sum(1 for e in processed_documents if e.source_file == src),
                    "last_chunk": max((e.chunk_index for e in processed_documents if e.source_file == src), default=-1)
                }

        state_data = {
            "timestamp": datetime.now().isoformat(),
            "processed_files": processed_files,
            "processed_documents": [entry.model_dump() if hasattr(entry, 'model_dump') else dict(entry) for entry in processed_documents]
        }
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)
        print(f"State saved to {STATE_FILE} ({len(processed_documents)} entries from {len(processed_files)} files)")
    except Exception as e:
        print(f"Error saving state: {e}")


def load_state():
    """Load previous state from STATE_FILE if it exists"""
    global processed_documents
    
    if not os.path.exists(STATE_FILE):
        print("No previous state found, starting fresh")
        return set()  # Return empty set of files to skip

    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
       
        loaded_entries = state_data.get("processed_documents", [])
        processed_documents = []
        for entry_data in loaded_entries:
            # Ensure reviewed field exists for backward compatibility
            if 'reviewed' not in entry_data:
                entry_data['reviewed'] = False
            processed_documents.append(QAEntry(**entry_data))
	 
        print(f"State loaded from {STATE_FILE} ({len(processed_documents)} entries)")

        # Determine which files are fully processed and unchanged
        files_to_skip = set()
        processed_files = state_data.get("processed_files", {})
        
        for filename, file_info in processed_files.items():
            file_path = os.path.join(PDF_DIR, filename) if filename.endswith('.pdf') else os.path.join(WEB_DIR, filename)
            
            if not os.path.exists(file_path):
                print(f"  Skipping {filename}: file no longer exists")
                files_to_skip.add(filename)
                continue
            
            current_mtime = os.path.getmtime(file_path)
            stored_mtime = file_info.get("modified_time", 0)
            
            # Count entries for this file in loaded state
            current_entry_count = sum(1 for e in processed_documents if e.source_file == filename)
            stored_entry_count = file_info.get("entry_count", 0)
            
            if current_mtime == stored_mtime and current_entry_count == stored_entry_count and current_entry_count > 0:
                print(f"  Skipping {filename}: already processed (unchanged)")
                files_to_skip.add(filename)
            else:
                print(f"  Re-processing {filename}: {'modified' if current_mtime != stored_mtime else 'incomplete'}")

        return files_to_skip

    except Exception as e:
        print(f"Error loading state, starting fresh: {e}")
        processed_documents = []
        return set()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text and tables from PDF file (advanced)"""
    text_chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_content = []

            # Extract tables
            tables = page.extract_tables()
            table_bboxes = []  # Track table positions to exclude from text extraction
            
            if tables:
                for table_idx, table in enumerate(tables):
                    if not table:
                        continue
                    
                    # Get table bounding box to exclude from text extraction
                    if hasattr(table, 'bbox'):
                        table_bboxes.append(table.bbox)
                    
                    # Convert table to readable format
                    table_lines = []
                    for row_idx, row in enumerate(table):
                        cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                        if any(cell for cell in cleaned_row):
                            table_lines.append("\t".join(cleaned_row))
                    
                    if table_lines:
                        table_text = "\n".join(table_lines)
                        page_content.append(f"[TABLE {table_idx + 1}]\n{table_text}\n[/TABLE]")

            # Extract text excluding table regions
            if table_bboxes:
                text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if text:
                    page_content.insert(0, text.strip())
            else:
                text = page.extract_text()
                if text:
                    page_content.append(text.strip())

            if page_content:
                combined = "\n\n".join(page_content)
                text_chunks.append(f"[PAGE {page_num}]\n{combined}")

    return "\n\n".join(text_chunks)


def extract_text_from_file(file_path: str) -> str:
    """Extract text from .txt file"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def detect_document_type(text: str, filename: str) -> str:
    """Auto-detect document type from structural patterns for prompt hinting."""
    page_markers = re.findall(r'\[PAGE \d+\]', text)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    if page_markers and paragraphs:
        avg_para_length = sum(len(p) for p in paragraphs) / len(paragraphs)
        if avg_para_length < 300 and len(page_markers) > 5:
            return """
  DOCUMENT TYPE HINT: This appears to be a slide deck or presentation export.
  - Each slide may contain brief bullet points or short statements
  - Group related slides together (e.g., 2-3 slides on the same subtopic)
  - Split when the visual topic clearly changes (new chart, new concept)"""

    qa_pattern = re.compile(r'(?:Q:|A:|\?|Answer:|Question:)', re.IGNORECASE)
    if len(qa_pattern.findall(text)) > 10:
        return """
  DOCUMENT TYPE HINT: This appears to be a transcript or Q&A format.
  - Topic shifts often occur between different questions or speakers
  - Group related exchanges together"""

    section_pattern = re.compile(r'^\d+[\.\)]\s+[A-Z]', re.MULTILINE)
    if len(section_pattern.findall(text)) > 3:
        return """
  DOCUMENT TYPE HINT: This appears to be a structured report or paper.
  - Section numbers indicate major topic boundaries
  - Split at section changes, keep subsections together"""

    header_pattern = re.compile(r'^#{1,6}\s+', re.MULTILINE)
    if len(header_pattern.findall(text)) > 2:
        return """
  DOCUMENT TYPE HINT: This appears to be a markdown-formatted article.
  - Headers indicate topic boundaries
  - Group content under the same header together"""

    return """
  DOCUMENT TYPE HINT: No clear document structure detected. Split based purely on semantic topic shifts."""

async def _detect_topic_boundaries(batch_text: str, start_index: int, global_count: int) -> List[int]:
    """Ask LLM to identify paragraph indices where topic shifts occur.
    Uses streaming pattern matching generate_qa_from_text."""
    verbose = config.get("processing", {}).get("debug_print", False)
    MAX_ANALYSIS_CHARS = 25000

    if len(batch_text) > MAX_ANALYSIS_CHARS:
        batch_text = batch_text[:MAX_ANALYSIS_CHARS]
        batch_text = batch_text.rsplit('\n\n', 1)[0] + "\n\n"

    numbered_paras = []
    for i, para in enumerate(batch_text.split('\n\n')):
        if para.strip():
            numbered_paras.append(f"[P{start_index + i}] {para.strip()}")

    numbered_text = "\n\n".join(numbered_paras)

    # Determine doc type hint
    force_type = config.get("processing", {}).get("force_doc_type", "")
    if force_type:
        type_map = {
            "slides": """
  DOCUMENT TYPE HINT: This is a slide deck. Group related slides, split on topic changes.""",
            "article": """
  DOCUMENT TYPE HINT: This is an article. Split at major topic shifts, keep examples together.""",
            "report": """
  DOCUMENT TYPE HINT: This is a structured report. Split at section boundaries.""",
            "transcript": """
  DOCUMENT TYPE HINT: This is a transcript. Group related Q&A exchanges together."""
        }
        doc_hint = type_map.get(force_type, "")
    else:
        doc_hint = detect_document_type(batch_text, "")

    prompt = SEMANTIC_CHUNK_TEMPLATE.safe_substitute(
        persona=PERSONA,
        audience=AUDIENCE,
        doc_type_hint=doc_hint,
        numbered_text=numbered_text
    )

    if verbose:
        print(f"\n🐛 [DEBUG] SEMANTIC CHUNK PROMPT (first 2000 chars):\n{prompt[:2000]}...\n")

    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        if should_cancel_all:
            return []

        try:
            async with httpx.AsyncClient(timeout=LLAMA_TIMEOUT * 2) as client:
                payload = {
                    "model": LLAMA_MODEL_NAME,
                    "prompt": prompt,
                    "max_tokens": 3000,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "stop": ["\n\n\n"],
                    "stream": True
                }

                print(f"Calling llama server for topic boundaries (attempt {attempt + 1}/{max_retries})...")
                async with client.stream("POST", f"{LLAMA_SERVER_URL}/v1/completions", json=payload) as response:
                    if response.status_code not in [200, 201]:
                        print(f"Error from llama server: {response.text[:500]}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                        continue

                    content = ""
                    raw_lines_debug = []
                    try:
                        async for line in response.aiter_lines():
                            if should_cancel_all:
                                print(f"🛑 Cancelled boundary detection")
                                break

                            if not line or line.startswith(":"):
                                continue

                            if verbose:
                                raw_lines_debug.append(line[:200])

                            if line.startswith("data: "):
                                data = line[6:].strip()
                                if data == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data)
                                    if "choices" in chunk:
                                        choice = chunk["choices"][0]
                                        delta = choice.get("delta", {})
                                        content += delta.get("content", "") or ""
                                        if "text" in choice:
                                            content += choice["text"] or ""
                                    elif "content" in chunk:
                                        content += chunk["content"] or ""
                                except json.JSONDecodeError as e:
                                    if verbose:
                                        print(f"   ⚠️ JSON parse error on line: {line[:150]}")
                                    continue

                    except asyncio.CancelledError:
                        print(f"🛑 Stream cancelled for boundary detection")
                        return []
                    finally:
                        await response.aclose()

                    # === DEBUG: Print raw response if verbose ===
                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"🔍 RAW BOUNDARY RESPONSE FROM LLAMA ({len(content)} chars accumulated)")
                        print(f"   Total SSE lines received: {len(raw_lines_debug)}")
                        print(f"{'='*60}")
                        if not content:
                            print("⚠️ WARNING: No content was accumulated!")
                            print("First 10 raw SSE lines:")
                            for i, raw_line in enumerate(raw_lines_debug[:10]):
                                print(f"   [{i}] {raw_line}")
                        else:
                            print(f"Accumulated content ({len(content)} chars):")
                            print(content)
                        print(f"{'='*60}\n")

                    # === CLEAN RESPONSE (exact same pattern as generate_qa_from_text) ===
                    content_cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    content_cleaned = re.sub(r'<thinking>.*?</thinking>', '', content_cleaned, flags=re.DOTALL | re.IGNORECASE)
                    content_cleaned = re.sub(r'^```(?:jsonl|json)?\s*\n?', '', content_cleaned, flags=re.MULTILINE)
                    content_cleaned = re.sub(r'\n?```\s*$', '', content_cleaned, flags=re.MULTILINE)

                    if verbose:
                        print(f"🧹 Cleaned content ({len(content_cleaned)} chars):")
                        print(content_cleaned)
                        print()

                    # === PARSE JSON ARRAY ===
                    json_match = re.search(r'\[\s*\d+(\s*,\s*\d+)*\s*\]', content_cleaned)
                    if json_match:
                        try:
                            boundaries = json.loads(json_match.group(0))
                            if verbose:
                                print(f"✅ Found boundaries at paragraphs: {boundaries}")
                            return boundaries
                        except json.JSONDecodeError as e:
                            if verbose:
                                print(f"   ⚠️ JSON parse error: {e}")

                    # Fallback: extract paragraph indices
                    numbers = re.findall(r'\b(\d+)\b', content_cleaned)
                    valid_boundaries = [int(n) for n in numbers if start_index < int(n) < global_count]
                    if valid_boundaries:
                        unique = list(dict.fromkeys(valid_boundaries))
                        if verbose:
                            print(f"   → Fallback boundaries: {unique}")
                        return unique

                    print(f"⚠️ No valid boundaries parsed. Retrying...")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2

        except httpx.ReadTimeout:
            print(f"Read timeout on boundary detection attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
        except httpx.ConnectError as e:
            print(f"Connection error on boundary detection: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2

    return []

async def semantic_chunk_text(text: str, filename: str, max_chunk_tokens: int = 16000) -> List[str]:
    """LLM-driven semantic chunking for unstructured text."""
    if not text or len(text.strip()) < 100:
        return []

    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        return [text.strip()]

    batch_size = config.get("processing", {}).get("semantic_batch_size", 60)
    overlap = config.get("processing", {}).get("semantic_overlap", 10)

    print(f"📝 Found {len(paragraphs)} paragraphs, running semantic chunking...")

    all_boundaries = [0]
    batch_start = 0

    while batch_start < len(paragraphs):
        batch_end = min(batch_start + batch_size, len(paragraphs))
        actual_start = max(0, batch_start - overlap) if batch_start > 0 else 0
        batch_paragraphs = paragraphs[actual_start:batch_end]
        batch_text = "\n\n".join(batch_paragraphs)

        if len(batch_text) < 200:
            batch_start = batch_end
            continue

        local_boundaries = await _detect_topic_boundaries(batch_text, actual_start, len(paragraphs))
        for b in local_boundaries:
            if b not in all_boundaries and 0 < b < len(paragraphs):
                all_boundaries.append(b)

        batch_start = batch_end

    all_boundaries.sort()

    valid_boundaries = sorted(set(
        b for b in all_boundaries 
        if 0 <= b < len(paragraphs)
    ))
    if not valid_boundaries:
        valid_boundaries = [0]
    all_boundaries = valid_boundaries
    # ==================================

    raw_chunks = []
    for i in range(len(all_boundaries) - 1):
        chunk_paragraphs = paragraphs[all_boundaries[i]:all_boundaries[i + 1]]
        raw_chunks.append("\n\n".join(chunk_paragraphs))

    raw_chunks = []
    for i in range(len(all_boundaries) - 1):
        chunk_paragraphs = paragraphs[all_boundaries[i]:all_boundaries[i + 1]]
        raw_chunks.append("\n\n".join(chunk_paragraphs))
    if all_boundaries[-1] < len(paragraphs):
        raw_chunks.append("\n\n".join(paragraphs[all_boundaries[-1]:]))

    OVERLAP_CHARS = 1500
    chunk_max_chars = int(max_chunk_tokens * 3.5)

    # Replace your final chunking loop with this:
    final_chunks = []
    current_chunk = ""
    
    for chunk_text in raw_chunks:
        # If the incoming chunk alone exceeds the limit, split it by paragraphs
        if len(chunk_text) > chunk_max_chars:
            if current_chunk.strip():
                final_chunks.append(current_chunk.strip())
            
            paras = chunk_text.split('\n\n')
            temp_chunk = ""
            for p in paras:
                if len(temp_chunk) + len(p) + 2 > chunk_max_chars:
                    final_chunks.append(temp_chunk.strip())
                    temp_chunk = p
                else:
                    temp_chunk += (p + "\n\n" if temp_chunk else p)
            if temp_chunk.strip():
                current_chunk = temp_chunk[-OVERLAP_CHARS:] if len(temp_chunk) > OVERLAP_CHARS else ""
            continue

        # Normal concatenation logic
        if len(current_chunk) + len(chunk_text) + 2 <= chunk_max_chars:
            current_chunk += "\n\n" + chunk_text
        else:
            final_chunks.append(current_chunk.strip())
            overlap_tail = current_chunk[-OVERLAP_CHARS:] if len(current_chunk) > OVERLAP_CHARS else ""
            current_chunk = overlap_tail + "\n\n" + chunk_text

    if current_chunk.strip():
        final_chunks.append(current_chunk.strip())

    print(f"✅ Semantic chunking complete: {len(final_chunks)} chunks from {len(paragraphs)} paragraphs")
    for i, c in enumerate(final_chunks):
        print(f"   Chunk {i+1}: ~{len(c)//4} tokens, {len(c)} chars")

    return final_chunks

def chunk_text(text: str, max_length: int = 2500) -> List[str]:
    """Split text into chunks for LLM processing"""
    paragraphs = text.split('\n\n')

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 <= max_length:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

async def generate_qa_from_text(text_chunk: str, source_file: str, chunk_index: int) -> List[QAEntry]:
    verbose = config.get("processing", {}).get("debug_print", False)
    
    prompt = TEXT_QA_TEMPLATE.safe_substitute(
        persona=PERSONA,
        text_chunk=text_chunk,
        qa_per_chunk=QA_PER_CHUNK,
        audience=AUDIENCE
    )
    if verbose:
        print(f"\n🐛 [DEBUG] RAW PROMPT SENT TO LLM (first 2000 chars):\n{prompt[:2000]}...\n")

    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        if should_cancel_all or (should_cancel_current and processing_state.get("current_source") == source_file):
            return []

        try:
            async with httpx.AsyncClient(timeout=LLAMA_TIMEOUT * 2) as client:
                payload = {
                    "model": "llm",
                    "prompt": prompt,
                    "max_tokens": LLAMA_MAX_TOKENS,
                    "temperature": LLAMA_TEMPERATURE,
                    "top_p": LLAMA_TOP_P,
                    "stop": ["\n\n\n"],
                    "stream": True
                }

                print(f"Calling llama server (attempt {attempt + 1}/{max_retries})...")
                async with client.stream("POST", f"{LLAMA_SERVER_URL}/v1/completions", json=payload) as response:
                    if response.status_code not in [200, 201]:
                        print(f"Error from llama server: {response.text[:500]}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                        continue

                    content = ""
                    raw_lines_debug = []  # Collect raw lines for debugging
                    try:
                        async for line in response.aiter_lines():
                            if should_cancel_all or (should_cancel_current and processing_state.get("current_source") == source_file):
                                print(f"🛑 Cancelled generation for {source_file}")
                                break

                            if not line or line.startswith(":"): 
                                continue
                            
                            if verbose:
                                raw_lines_debug.append(line[:200])  # Cap length for readability

                            if line.startswith("data: "):
                                data = line[6:].strip()
                                if data == "[DONE]": break
                                try:
                                    chunk = json.loads(data)
                                    # Handle both chat and completions streaming formats
                                    if "choices" in chunk:
                                        choice = chunk["choices"][0]
                                        # Chat completions format
                                        delta = choice.get("delta", {})
                                        content += delta.get("content", "") or ""
                                        # Completions format (non-chat)
                                        if "text" in choice:
                                            content += choice["text"] or ""
                                    # Some servers return at root level
                                    elif "content" in chunk:
                                        content += chunk["content"] or ""
                                except json.JSONDecodeError as e:
                                    if verbose:
                                        print(f"   ⚠️ JSON parse error on line: {line[:150]}")
                                    continue

                    except asyncio.CancelledError:
                        print(f"🛑 Stream cancelled for {source_file}")
                        return []  # Exit cleanly on cancellation
                    finally:
                        await response.aclose()  # Ensure connection closes gracefully

                    # === DEBUG: Print raw response if verbose ===
                    if verbose:
                        print(f"\n{'='*60}")
                        print(f"🔍 RAW RESPONSE FROM LLAMA ({len(content)} chars accumulated)")
                        print(f"   Total SSE lines received: {len(raw_lines_debug)}")
                        print(f"{'='*60}")
                        if not content:
                            print("⚠️ WARNING: No content was accumulated!")
                            print("First 10 raw SSE lines:")
                            for i, raw_line in enumerate(raw_lines_debug[:10]):
                                print(f"   [{i}] {raw_line}")
                        else:
                            print(f"Accumulated content ({len(content)} chars):")
                            print(content)
                        print(f"{'='*60}\n")

                    # Early return if cancelled mid-stream
                    if should_cancel_all or (should_cancel_current and processing_state.get("current_source") == source_file):
                        return []

                    # === CLEAN RESPONSE ===
                    # Strip thinking/reasoning tags FIRST
                    content_cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
                    content_cleaned = re.sub(r'<thinking>.*?</thinking>', '', content_cleaned, flags=re.DOTALL | re.IGNORECASE)
                    # Strip markdown code blocks
                    content_cleaned = re.sub(r'^```(?:jsonl|json)?\s*\n?', '', content_cleaned, flags=re.MULTILINE)
                    content_cleaned = re.sub(r'\n?```\s*$', '', content_cleaned, flags=re.MULTILINE)

                    if verbose:
                        print(f"🧹 Cleaned content ({len(content_cleaned)} chars):")
                        print(content_cleaned)
                        print()

                    # === DEFINE PATTERN BEFORE USING IT ===
                    json_pattern = re.compile(r'\{[^{}]*"instruction"[^{}]*"output"[^{}]*\}', re.DOTALL)

                    # === DEBUG: Show regex matches ===
                    if verbose:
                        matches = list(json_pattern.finditer(content_cleaned))
                        print(f"🔍 Regex found {len(matches)} potential JSON objects")
                        for i, m in enumerate(matches):
                            print(f"  Match {i} (first 150 chars): {m.group(0)[:150]}...")

                    qa_entries = []
                    for match in json_pattern.finditer(content_cleaned):
                        json_str = ' '.join(match.group(0).split())
                        try:
                            qa_data = json.loads(json_str)
                            if "instruction" not in qa_data or "output" not in qa_data:
                                continue
                            if "input" not in qa_data:
                                qa_data["input"] = ""
                            entry_id = f"{source_file.replace('/', '_').replace('.', '_')}_{chunk_index}_{len(qa_entries)}"
                            qa_entries.append(QAEntry(
                                id=entry_id,
                                instruction=qa_data.get("instruction", ""),
                                input=qa_data.get("input", ""),
                                output=qa_data.get("output", ""),
                                source_file=source_file,
                                chunk_index=chunk_index,
                                enabled=True,
                                edited=False
                            ))
                        except json.JSONDecodeError as e:
                            if verbose:
                                print(f"   ⚠️ JSON parse error: {str(e)[:100]}")
                            continue

                    # Fallback: try line-by-line parsing if regex found nothing
                    if not qa_entries:
                        if verbose:
                            print("⚠️ Regex found no matches, trying line-by-line fallback...")
                        for line_num, line in enumerate(content_cleaned.strip().split('\n')):
                            line = re.sub(r'^[*`\s]*', '', line.strip()).strip()
                            line = re.sub(r'[*`]\s*$', '', line).strip()
                            if len(line) < 20:
                                continue
                            try:
                                qa_data = json.loads(line)
                                if "instruction" in qa_data and "output" in qa_data:
                                    if "input" not in qa_data:
                                        qa_data["input"] = ""
                                    entry_id = f"{source_file.replace('/', '_').replace('.', '_')}_{chunk_index}_{len(qa_entries)}"
                                    qa_entries.append(QAEntry(
                                        id=entry_id,
                                        instruction=qa_data.get("instruction", ""),
                                        input=qa_data.get("input", ""),
                                        output=qa_data.get("output", ""),
                                        source_file=source_file,
                                        chunk_index=chunk_index,
                                        enabled=True,
                                        edited=False
                                    ))
                            except json.JSONDecodeError:
                                continue

                    if qa_entries:
                        print(f"✅ Successfully parsed {len(qa_entries)} Q&A entries")
                        return qa_entries
                    
                    print(f"[WARN] No valid Q&A parsed. Retrying...")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2

        except httpx.ReadTimeout:
            print(f"Read timeout on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
        except httpx.ConnectError as e:
            print(f"Connection error: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2

    return []

async def process_all_documents(selected_sources: Optional[List[str]] = None, images_only_sources: Optional[set] = None):
    global processed_documents, is_processing, processing_state, should_cancel_current, should_cancel_all

    if is_processing:
        print("Processing already in progress, skipping...")
        return

    is_processing = True
    enable_multimodal = config.get("processing", {}).get("enable_multimodal", True)

    sources_to_process = selected_sources or []
    processing_state["queue"] = list(sources_to_process)
    processing_state["completed"] = []
    processing_state["is_processing"] = True
    processing_state["total_units"] = 0      # Total chunks + images expected
    processing_state["completed_units"] = 0  # Successfully processed units

    try:
        files_to_skip = set() if selected_sources else get_files_to_skip_from_memory()

        # 1. PRE-CALCULATE TOTAL WORK UNITS per phase (fast, no LLM calls)
        processing_state["text_chunks_total"] = 0
        processing_state["images_total"] = 0
        processing_state["text_chunks_completed"] = 0
        processing_state["images_completed"] = 0
        processing_state["total_units"] = 0
        processing_state["completed_units"] = 0

        for filename in sources_to_process:
            is_pdf = filename.lower().endswith('.pdf')
            filepath = os.path.join(PDF_DIR, filename) if is_pdf else os.path.join(WEB_DIR, filename)
            if not os.path.exists(filepath):
                continue

            # Estimate text chunks
            if filename not in (images_only_sources or set()):
                text = extract_text_from_pdf(filepath) if is_pdf else extract_text_from_file(filepath)
                if len(text.strip()) >= 100:
                    processing_state["text_chunks_total"] += len(chunk_text(text, max_length=CHUNK_SIZE))

            # Count images
            if enable_multimodal and is_pdf:
                images = extract_images_from_pdf(filepath)
                processing_state["images_total"] += len(images)

        processing_state["total_units"] = (
            processing_state["text_chunks_total"] + processing_state["images_total"]
        )
        print(f"📊 Total work: {processing_state['text_chunks_total']} text chunks + {processing_state['images_total']} images = {processing_state['total_units']} units")

        # 2. PROCESS LOOP
        for filename in sources_to_process:
            if should_cancel_all:
                print("🛑 All processing cancelled by user")
                break
            if should_cancel_current and processing_state["current_source"] == filename:
                print(f"🛑 Cancelled processing for {filename}")
                should_cancel_current = False
                continue

            processing_state["current_source"] = filename
            
            if not selected_sources and filename in files_to_skip:
                print(f"Skipping already processed: {filename}")
                processing_state["completed"].append(filename)
                continue

            print(f"\n{'='*60}\nProcessing: {filename}\n{'='*60}")

            is_pdf = filename.lower().endswith('.pdf')
            filepath = os.path.join(PDF_DIR, filename) if is_pdf else os.path.join(WEB_DIR, filename)
            if not os.path.exists(filepath): continue

            try:
                # --- Text Processing ---
                if filename not in (images_only_sources or set()):
                    text = extract_text_from_pdf(filepath) if is_pdf else extract_text_from_file(filepath)
                    if len(text.strip()) >= 100:
                        use_semantic = config.get("processing", {}).get("semantic_chunking", True)
                        if use_semantic and len(text.strip()) > 500:
                            chunks = await semantic_chunk_text(
                                text,
                                filename=filename,
                                max_chunk_tokens=config.get("processing", {}).get("max_chunk_tokens", 16000)
                            )
                        else:
                            chunks = chunk_text(text, max_length=CHUNK_SIZE)
                        total_chunks = len(chunks)
                        print(f"  Text split into {total_chunks} chunks")

                        for i, chunk in enumerate(chunks):
                            if should_cancel_all or (should_cancel_current and processing_state["current_source"] == filename):
                                break

                            print(f"  Generating Q&A for text chunk {i+1}/{total_chunks}...")
                            qa_entries = await generate_qa_from_text(chunk, filename, i)

                            existing_ids = {e.id for e in processed_documents}
                            new_entries = [e for e in qa_entries if e.id not in existing_ids]
                            processed_documents.extend(new_entries)
                            print(f"    → Generated {len(new_entries)} NEW Q&A pairs")

                            processing_state["phase"] = "text"
                            processing_state["text_chunks_completed"] += 1
                            processing_state["completed_units"] = (
                                processing_state["text_chunks_completed"] + processing_state["images_completed"]
                            )
                            processing_state["progress_percent"] = int(
                                (processing_state["completed_units"] / max(processing_state["total_units"], 1)) * 100
                            )

                            if len(new_entries) > 0: save_state()
                            if BATCH_DELAY > 0: await asyncio.sleep(BATCH_DELAY)
                    else:
                        print(f"  Skipping text: too little content")
                else:
                    print(f"  Skipping text Q&A (Images Only mode)")

                if should_cancel_all or (should_cancel_current and processing_state.get("current_source") == filename):
                    print(f"🛑 Cancellation requested, skipping image processing for {filename}")
                    should_cancel_current = False
                    processing_state["completed"].append(filename)
                    continue

                # --- Image Processing ---
                if enable_multimodal and is_pdf:
                    print(f"\n  Extracting visual content...")
                    images = extract_images_from_pdf(filepath)
                    total_images = len(images)

                    for img_idx, img_info in enumerate(images):
                        if should_cancel_all or (should_cancel_current and processing_state["current_source"] == filename):
                            break

                        print(f"    Processing image {img_idx+1}/{total_images} (page {img_info.get('page', 0)})...")
                        
                        text_context = ""
                        if 'text' in locals() and text.strip():
                            text_context = get_image_context_text(text, img_info.get("page", 0))

                        qa_entries = await generate_qa_from_image(img_info, filename, img_idx, text_context)

                        existing_ids = {e.id for e in processed_documents}
                        new_entries = [e for e in qa_entries if e.id not in existing_ids]
                        processed_documents.extend(new_entries)
                        print(f"    → Generated {len(new_entries)} Q&A from image")

                        processing_state["phase"] = "images"
                        processing_state["images_completed"] += 1
                        processing_state["completed_units"] = (
                            processing_state["text_chunks_completed"] + processing_state["images_completed"]
                        )
                        progress = int(
                            (processing_state["completed_units"] / max(processing_state["total_units"], 1)) * 100
                        )
                        processing_state["progress_percent"] = min(progress, 100)

                        if len(new_entries) > 0: save_state()
                        if BATCH_DELAY > 0: await asyncio.sleep(BATCH_DELAY)

                # Mark complete
                processing_state["completed"].append(filename)
                print(f"✅ Completed: {filename}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()

        save_state()
        print(f"\n🎯 Total Q&A pairs: {len(processed_documents)}")

    except asyncio.CancelledError:
        print("🛑 process_all_documents cancelled")
        raise  # Re-raise so /api/process can catch it cleanly

    finally:
        # Always reset state on exit (normal or cancelled)
        is_processing = False
        processing_state["is_processing"] = False
        processing_state["current_source"] = None
        processing_state["phase"] = None
        should_cancel_current = False
        should_cancel_all = False
        save_state()  # Save partial progress before exiting


@app.get("/api/debug/state")
async def debug_state():
    """Debug endpoint to check state"""
    return {
        "processed_documents_count": len(processed_documents),
        "sample_entry": processed_documents[0].model_dump() if processed_documents else None,
        "has_reviewed_field": hasattr(processed_documents[0], 'reviewed') if processed_documents else False,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the review web interface"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/stats")
async def get_stats():
    """Get processing statistics"""
    pdf_count = len(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    txt_count = len(glob.glob(os.path.join(WEB_DIR, "*.txt")))

    enabled_entries = sum(1 for e in processed_documents if e.enabled)
    disabled_entries = sum(1 for e in processed_documents if not e.enabled)
    edited_entries = sum(1 for e in processed_documents if e.edited)

    return {
        "pdf_files": pdf_count,
        "text_files": txt_count,
        "total_documents": pdf_count + txt_count,
        "generated_qa_pairs": len(processed_documents),
        "enabled_entries": enabled_entries,
        "disabled_entries": disabled_entries,
        "edited_entries": edited_entries,
        "pending_review": len(processed_documents) - edited_entries
    }


@app.get("/api/sources")
async def get_sources():
    """Get list of source files with metadata for the sources table"""
    sources = {}
    
    for entry in processed_documents:
        src = entry.source_file
        if not src:
            continue
            
        if src not in sources:
            is_pdf = src.endswith('.pdf')
            file_path = os.path.join(PDF_DIR, src) if is_pdf else os.path.join(WEB_DIR, src)
            exists = os.path.exists(file_path)
            
            sources[src] = {
                "filename": src,
                "type": "PDF" if is_pdf else "Text",
                "exists": exists,
                "total_qa": 0,
                "reviewed_qa": 0,
                "enabled_qa": 0,
                "disabled_qa": 0,
                "edited_qa": 0,
                "ready_for_export": False,
            }
        
        sources[src]["total_qa"] += 1
        
        # Handle both old entries (no reviewed field) and new ones
        is_reviewed = getattr(entry, 'reviewed', False)
        if is_reviewed:
            sources[src]["reviewed_qa"] += 1
            
        if entry.enabled:
            sources[src]["enabled_qa"] += 1
        else:
            sources[src]["disabled_qa"] += 1
            
        if entry.edited:
            sources[src]["edited_qa"] += 1
    
    # Mark source as ready when ALL Q&A are reviewed
    for src, data in sources.items():
        data["ready_for_export"] = (data["reviewed_qa"] == data["total_qa"] and data["total_qa"] > 0)
    
    #print(f"[DEBUG] /api/sources: Found {len(sources)} sources, {len(processed_documents)} total entries")
    
    return {
        "sources": list(sources.values()),
        "total_sources": len(sources),
        "total_ready": sum(1 for s in sources.values() if s["ready_for_export"])
    }

@app.post("/api/process")
async def process_documents(req: Optional[ProcessRequest] = None):
    global is_processing, processed_documents, _active_processing_task

    if is_processing:
        raise HTTPException(status_code=409, detail="Processing already in progress.")

    req = req or ProcessRequest()
    selected_sources = req.sources or []
    images_only_sources = set(req.images_only or [])

    if selected_sources:
        original_count = len(processed_documents)
        processed_documents = [
            e for e in processed_documents
            if e.source_file not in selected_sources or
               (e.source_file in images_only_sources and e.chunk_index != -1)
        ]
        print(f"Removed {original_count - len(processed_documents)} existing entries")
        save_state()

    pdf_count = len(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    txt_count = len(glob.glob(os.path.join(WEB_DIR, "*.txt")))
    if pdf_count == 0 and txt_count == 0:
        raise HTTPException(status_code=400, detail="No files found in pdf/ or web/")

    _active_processing_task = asyncio.create_task(
        process_all_documents(selected_sources=selected_sources, images_only_sources=images_only_sources)
    )
    
    try:
        await _active_processing_task
    except asyncio.CancelledError:
        print("🛑 Processing task cancelled by signal")
        # Reset state cleanly
        is_processing = False
        processing_state["is_processing"] = False
        processing_state["current_source"] = None
        should_cancel_current = False
        should_cancel_all = False
        # Return a valid response instead of bubbling the error
        return JSONResponse(
            status_code=200,
            content={"status": "cancelled", "message": "Processing was cancelled by user."}
        )
    finally:
        _active_processing_task = None

    return {
        "status": "success",
        "message": f"Processed documents",
        "total_qa_pairs": len(processed_documents),
        "sources_processed": selected_sources if selected_sources else "all new"
    }

@app.get("/api/entries")
async def get_entries(page: int = 1, per_page: int = 20, source_file: Optional[str] = None):
    """Get paginated Q&A entries for review. Optional source_file filter."""
    
    # Filter by source if provided
    filtered_docs = processed_documents
    if source_file:
        filtered_docs = [e for e in processed_documents if e.source_file == source_file]
    
    start = (page - 1) * per_page
    end = start + per_page

    entries_data = []
    for entry in filtered_docs[start:end]:
        entries_data.append({
            "id": entry.id,
            "instruction": entry.instruction,
            "input": entry.input,
            "output": entry.output,
            "source_file": entry.source_file,
            "chunk_index": entry.chunk_index,
            "enabled": entry.enabled,
            "edited": entry.edited,
            "reviewed": entry.reviewed
        })

    return {
        "entries": entries_data,
        "total": len(filtered_docs),
        "page": page,
        "per_page": per_page,
        "total_pages": (len(filtered_docs) + per_page - 1) // per_page if filtered_docs else 0,
        "source_file": source_file
    }


@app.post("/api/entries/{entry_id}")
async def update_entry(entry_id: str, update: ReviewUpdate):
    """Update a single entry (edit content or enable/disable/review)"""
    global processed_documents

    entry = next((e for e in processed_documents if e.id == entry_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    if update.instruction is not None:
        entry.instruction = update.instruction
        entry.edited = True
        entry.reviewed = True  # Mark as reviewed when edited

    if update.input is not None:
        entry.input = update.input
        entry.edited = True

    if update.output is not None:
        entry.output = update.output
        entry.edited = True

    if update.enabled is not None:
        entry.enabled = update.enabled

    if update.reviewed is not None:
        entry.reviewed = update.reviewed

    # Save state after each update
    save_state()

    return {"status": "success", "entry": {
        "id": entry.id,
        "enabled": entry.enabled,
        "edited": entry.edited,
        "reviewed": entry.reviewed
    }}


@app.post("/api/export")
async def export_dataset(req: Optional[dict] = None):
    """Export reviewed entries to JSONL file. 
    Accepts optional JSON body with 'sources' list to filter which sources to export."""
    global processed_documents

    req = req or {}
    selected_sources = req.get("sources", [])  # List of source filenames to include
    
    if selected_sources:
        # Only export entries from selected (and ready) sources
        enabled_entries = [e for e in processed_documents 
                         if e.enabled and e.source_file in selected_sources]
    else:
        # Export all enabled entries from ready sources (backward compatible)
        ready_sources = set()
        for entry in processed_documents:
            src = entry.source_file
            all_reviewed = all(
                e.reviewed for e in processed_documents if e.source_file == src
            )
            if all_reviewed:
                ready_sources.add(src)
        
        enabled_entries = [e for e in processed_documents 
                         if e.enabled and e.source_file in ready_sources]

    if not enabled_entries:
        raise HTTPException(status_code=400, detail="No enabled entries to export")

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = req.get("dataset_name", "climate_dataset")
    filename = f"{dataset_name}_{timestamp}.jsonl"
    filepath = os.path.join(RESULT_DIR, filename)

    # Write JSONL
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in enabled_entries:
            json_line = json.dumps({
                "instruction": entry.instruction,
                "input": entry.input,
                "output": entry.output
            }, ensure_ascii=False)
            f.write(json_line + '\n')

    return {
        "status": "success",
        "filename": filename,
        "filepath": filepath,
        "entries_exported": len(enabled_entries),
        "message": f"Exported {len(enabled_entries)} entries to {RESULT_DIR}/{filename}"
    }


@app.post("/api/entries/mark-reviewed")
async def mark_entries_reviewed(req: dict):
    """Mark multiple entries as reviewed at once"""
    global processed_documents
    
    entry_ids = req.get("ids", [])
    reviewed = req.get("reviewed", True)
    
    updated = 0
    for entry in processed_documents:
        if entry.id in entry_ids:
            entry.reviewed = reviewed
            updated += 1
    
    save_state()
    return {"status": "success", "updated": updated}


@app.post("/api/sources/{source_file}/mark-all-reviewed")
async def mark_source_reviewed(source_file: str):
    """Mark all Q&A entries in a source as reviewed"""
    global processed_documents
    
    updated = 0
    for entry in processed_documents:
        if entry.source_file == source_file:
            entry.reviewed = True
            updated += 1
    
    save_state()
    return {"status": "success", "updated": updated}


@app.post("/api/sources/url")
async def add_url_source(req: URLSourceRequest):
    """Download a PDF from a URL and save it into pdf/"""
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    try:
        async with httpx.AsyncClient(
			timeout=60.0,
			follow_redirects=True,
			verify=certifi.where()
	) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=400, detail=f"Server returned {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download URL: {e}")

    content_type = resp.headers.get("content-type", "")
    if "application/pdf" not in content_type and not resp.content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="The URL did not return a PDF file")

    raw_name = os.path.basename(url.split("?")[0]) or "download.pdf"
    if not raw_name.lower().endswith(".pdf"):
        raw_name += ".pdf"
    filename = safe_filename(raw_name, "download.pdf")

    filepath = unique_path(PDF_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(resp.content)

    return {
        "status": "success",
        "filename": os.path.basename(filepath),
        "message": f"Saved to {PDF_DIR}/{os.path.basename(filepath)}"
    }


@app.post("/api/sources/text")
async def add_text_source(req: TextSourceRequest):
    """Save pasted article text into web/ with a timestamped filename"""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    if len(text) < 50:
        raise HTTPException(status_code=400, detail="Text is too short to be useful (min 50 characters)")

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    filepath = unique_path(WEB_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

    return {
        "status": "success",
        "filename": os.path.basename(filepath),
        "message": f"Saved to {WEB_DIR}/{os.path.basename(filepath)}"
    }


@app.delete("/api/sources/{source_file}")
async def remove_source(source_file: str):
    """Remove all Q&A entries for a given source file from state, resetting its processing status"""
    global processed_documents

    original_count = len(processed_documents)
    processed_documents = [e for e in processed_documents if e.source_file != source_file]
    removed_count = original_count - len(processed_documents)

    save_state()

    return {
        "status": "success",
        "removed": removed_count,
        "source_file": source_file,
        "message": f"Removed {removed_count} Q&A entries for {source_file}"
    }


@app.get("/api/llama-status")
async def check_llama_status():
    """Check if llama.cpp server is running"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Try the completion endpoint with a minimal request
            response = await client.post(
                f"{LLAMA_SERVER_URL}/v1/completions",
                json={"prompt": "test", "n_predict": 1, "model": LLAMA_MODEL_NAME},
                timeout=5.0
            )
            if response.status_code in [200, 201]:
                return {
                    "status": "connected",
                    "url": LLAMA_SERVER_URL,
                    "message": "Llama server is running"
                }
            else:
                return {
                    "status": "error",
                    "url": LLAMA_SERVER_URL,
                    "message": f"Llama server returned status {response.status_code}"
                }
    except httpx.ConnectError as e:
        return {
            "status": "disconnected",
            "url": LLAMA_SERVER_URL,
            "message": f"Cannot connect to llama server at {LLAMA_SERVER_URL}"
        }
    except Exception as e:
        return {
            "status": "error",
            "url": LLAMA_SERVER_URL,
            "message": f"Error: {str(e)}"
        }

@app.get("/api/config")
async def get_app_config():
    """Return app configuration for frontend"""
    return {"name": APP_NAME, "subtitle": APP_SUBTITLE}


if __name__ == "__main__":
    import asyncio
    import signal
    import uvicorn
    import sys

    host = config["server"].get("host", "0.0.0.0")
    port = config["server"].get("port", 8501)

    uvicorn_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(uvicorn_config)

    shutdown_event = asyncio.Event()

    async def cleanup():
        print("\n" + "=" * 60)
        print("🛑 Received shutdown signal. Cleaning up...")
        print("=" * 60)

        if processed_documents:
            try:
                save_state()
                print(f"✓ State saved ({len(processed_documents)} entries)")
            except Exception as e:
                print(f"✗ Error saving state on exit: {e}")

        # Cancel only *your* tasks if you can track them.
        # If you can't, at least don't nuke all tasks (Uvicorn uses many).
        # pending = [...]
        #
        # If you must cancel, do it carefully and exclude Uvicorn tasks.

        print("Stopping Uvicorn...")
        # No need to set should_exit here if you already did it in the signal handler.
        print("=" * 60)
        print("✓ Clean exit complete.")
        print("=" * 60)

    async def main():
        loop = asyncio.get_running_loop()

        def handle_signal(sig):
            sig_name = signal.Signals(sig).name
            print(f"\n  Received signal: {sig_name}")

            if not shutdown_event.is_set():
                shutdown_event.set()
                
                # 1. Tell your processing loop to stop immediately
                global should_cancel_all
                should_cancel_all = True
                print("🛑 Setting should_cancel_all = True")

                # 2. Force-cancel the active processing task if it exists
                global _active_processing_task
                if _active_processing_task and not _active_processing_task.done():
                    print("🛑 Forcibly cancelling active processing task...")
                    _active_processing_task.cancel()
                
                server.should_exit = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, handle_signal, sig)
            except NotImplementedError:
                pass

        try:
            await server.serve()
        except asyncio.CancelledError:
            print("🛑 Uvicorn cancelled")
        finally:
            if shutdown_event.is_set():
                await cleanup()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✓ Exiting...")
        sys.exit(0)


