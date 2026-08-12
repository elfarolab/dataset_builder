"""Minimal fake llama.cpp server for CI testing."""

import json
import re
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI(title="Mock Llama for CI")


def _count_paragraphs(prompt: str) -> int:
    """Count [PN] markers in the semantic chunking prompt."""
    return len(re.findall(r'\[P\d+\]', prompt))


def _smart_boundaries(prompt: str) -> str:
    """Return realistic boundary indices based on paragraph count.

    Rules:
      - 0-1 paragraphs → []  (nothing to split)
      - 2 paragraphs   → [1] (split in middle)
      - 3-5            → [2]
      - 6-10           → [3, 7]
      - 11+            → [n//4, n//2, 3*n//4]
    """
    n = _count_paragraphs(prompt)
    if n <= 1:
        return "[]"
    if n == 2:
        return "[1]"
    if n <= 5:
        return f"[{n // 2}]"
    if n <= 10:
        return f"[{n // 3}, {(2 * n) // 3}]"
    return f"[{n // 4}, {n // 2}, {(3 * n) // 4}]"


def _qa_lines():
    return [
        json.dumps({"instruction": "What are the main effects of climate change on farming?", "input": "", "output": "Climate change causes shifting growing seasons, increased drought frequency, and unpredictable rainfall patterns that affect crop yields."}),
        json.dumps({"instruction": "How can farmers adapt to changing weather patterns?", "input": "", "output": "Farmers can adopt drought-resistant crop varieties, implement precision irrigation systems, use cover cropping for moisture retention, and diversify rotations."}),
        json.dumps({"instruction": "What role does soil health play in climate resilience?", "input": "", "output": "Healthy soil with high organic matter retains moisture better, reduces erosion during extreme weather, and sequesters atmospheric carbon, making farms more resilient."}),
    ]


def _image_qa_lines():
    return [
        json.dumps({"instruction": "What does this diagram show about crop rotation?", "input": "", "output": "The diagram illustrates a four-year crop rotation cycle alternating legumes and cereals to maintain soil nitrogen levels."}),
        json.dumps({"instruction": "Describe the key elements visible in this agricultural chart.", "input": "", "output": "The chart compares yield data across conventional tillage, reduced tillage, and no-till methods, showing progressive improvement in soil carbon over time."}),
    ]


def _build_sse(text: str) -> str:
    payload = json.dumps({"choices": [{"delta": {"content": text}}]})
    return f"data: {payload}\n\ndata: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(req: dict):
    if not req.get("stream"):
        return JSONResponse({"choices": [{"text": "ok", "finish_reason": "stop"}]})

    prompt = req.get("prompt", "")

    async def stream():
        if "[P" in prompt:
            # Semantic chunking — paragraph-aware boundaries
            body = _build_sse(_smart_boundaries(prompt))
        else:
            # Text QA
            body = _build_sse("\n".join(_qa_lines()))
        await asyncio.sleep(0.01)
        yield body
        await asyncio.sleep(0.05)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(req: dict):
    if not req.get("stream"):
        return JSONResponse({"choices": [{"message": {"content": "ok"}}]})

    async def stream():
        body = _build_sse("\n".join(_image_qa_lines()))
        yield body
        await asyncio.sleep(0.05)

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    port = int(__import__("os").environ.get("MOCK_LLAMA_PORT", "18080"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

