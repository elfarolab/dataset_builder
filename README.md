# 📊 Universal Dataset Builder

A web-based tool for creating high-quality AI training datasets from documents (PDFs and text files) using a local llama.cpp backend. Extract text, generate Q&A pairs, review them in a browser interface, and export clean JSONL datasets ready for fine-tuning.

> 💡 **Historical Note:** This tool was originally developed to build instruction-following datasets for climate change AI assistants. The architecture is now fully generic and can be customized for any domain (medical, legal, technical, educational, etc.) by editing `config.yaml`.

## 🎯 Purpose

Build accurate, actionable training data for domain-specific AI assistants. This tool automates the extraction of structured information from technical documents and converts it into instruction-following format (`instruction`, `input`, `output`) suitable for models like Llama, Mistral, Qwen, or any transformer supporting Alpaca-style fine-tuning.

## ✨ Features

- **PDF & Text Parsing**: Extracts paragraphs and structured tables from PDFs using `pdfplumber`; reads `.txt` files natively
- **Multimodal Support**: Extracts and processes images/charts/diagrams from PDFs using vision-capable LLMs
- **Streaming LLM Calls**: Real-time token streaming with instant cancellation support
- **Real-Time Progress Tracking**: Cumulative progress bar showing exact chunks/images processed vs total expected work
- **Config-Driven UI & Prompts**: App name, subtitle, system persona, and target audience are fully customizable via `config.yaml`
- **Web Review Interface**: Browser-based UI to inspect, edit, enable/disable, and mark Q&A entries as reviewed before export
- **State Persistence**: Saves progress to `.review_state.json`; resume interrupted runs without reprocessing unchanged files
- **Smart Resumption**: Tracks file modification times; skips already-processed documents unless modified or explicitly selected
- **Chunked Processing**: Splits large documents into manageable chunks for reliable LLM context window usage
- **Selective Export**: Choose which sources to export; only reviewed & enabled entries are included in the final JSONL

## 📋 Requirements

- Python 3.10+
- A running llama.cpp server (or any OpenAI-compatible backend)
- Dependencies installed via `pip`

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/elfarolab/dataset_builder.git
   cd dataset_builder
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv env
   source env/bin/activate  # On Linux/macOS
   # or: env\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

4. **Start your llama.cpp server:**
   ```bash
   llama-server -m your-model.gguf --host 0.0.0.0 --port 8088
   ```
   Notes:
   - This command above is a generic example, customize the command with your llama-server custom options.

6. **Configure the application:**
   Edit `config.yaml` to match your setup:
   ```yaml
   app:
     name: "Dataset Builder"
     subtitle: "Review and refine AI training data from your documents"
     persona: "expert AI assistant"
     audience: "general public"

   llama_server:
     url: "http://<LAN IP>:8088"
     timeout: 120
     max_tokens: 5000
     temperature: 0.5
     top_p: 0.9

   paths:
     pdf_dir: "pdf"
     web_dir: "web"
     result_dir: "result"
     state_file: ".review_state.json"

   processing:
     chunk_size: 2500
     qa_per_chunk: 5
     batch_delay: 3.0
     image_qa_per_image: 3
     enable_multimodal: true
   ```
   Notes:
   - This sample above is generic, please use the provided config.yaml file as reference.

7. **Create input directories:**
   ```bash
   mkdir -p pdf web result
   ```
   Notes:
   - Directories are automatically created by the script but you can still do it to store your documents before first run.
   - Documents can also be added from the WEB UI.
   - Place PDF files in `pdf/`
   - Place text files in `web/`

## 🚀 Usage

1. **Start the web server:**
   ```bash
   python main.py
   ```
   The interface starts at `http://localhost:8501`.

2. **Add Sources (Optional):**
   - Click **"➕ Add Source"** to download a PDF via URL or paste article text directly into the web interface.

3. **Process Documents:**
   - Click **"📄 Process Documents"** → select which files to process
   - Toggle **"🖼️ Images Only"** for PDFs if you only want visual content processed
   - The processing panel shows real-time cumulative progress
   - Click **"⛔ Stop All"** or the source-specific `✕` button to instantly cancel streaming generation

4. **Review Generated Data:**
   - Click any source in the table to view its Q&A pairs
   - Edit `Instruction`, `Input`, or `Output` directly in the UI
   - Toggle checkboxes to enable/disable entries for export
   - Mark entries as reviewed individually or use **"✓ Mark All as Reviewed"**

5. **Export Dataset:**
   - Click **"💾 Export Dataset"** → select which sources to include
   - Only reviewed & enabled entries are exported
   - File is saved in `result/` with timestamp: `dataset_name_YYYYMMDD_HHMMSS.jsonl`

## 📊 How It Works

```
PDF/TXT files → Text + table extraction → Chunking → Streaming LLM Q&A generation → Web review → JSONL export
```

- **Text extraction**: Uses `pdfplumber` to read paragraphs and structured tables; tables are converted to tab-separated format with markers for structural awareness.
- **Image extraction**: Renders PDF pages containing raster/vector graphics into optimized PNGs (max 1024px) for vision LLMs.
- **Chunking**: Large documents are split by paragraphs into configurable chunks to fit model context windows.
- **Q&A generation**: Each chunk/image is sent via SSE streaming. Cancellation flags are checked per-token, allowing instant stop without waiting for timeouts.
- **Response parsing**: Handles both native llama.cpp and OpenAI-compatible formats; automatically strips reasoning/thinking tags (`<thinking>`, `<think>`, etc.) before JSONL extraction.
- **State management**: Tracks processed files, modification times, and review status; smart resumption skips unchanged content.

## 📁 Project Structure

```
dataset_builder/
├── main.py                # FastAPI backend + API routes
├── image_extractor.py     # PDF image rendering & context extraction
├── config.yaml            # App, server, and processing configuration
├── static/
│   └── index.html         # Web review interface (dynamic title/subtitle)
├── scripts/
│   └── sanitize-yaml.sh   # Script to sanitize private data before pushing to repo
├── pdf/                   # Input: place PDF files here
├── web/                   # Input: place .txt files here
├── result/                # Output: exported JSONL datasets
└── .review_state.json     # Auto-generated: progress & review tracking
```

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web review interface |
| `GET` | `/api/config` | Returns app name, subtitle, persona, audience |
| `GET` | `/api/stats` | Processing statistics & counts |
| `GET` | `/api/available-sources` | Lists all files on disk with processing status |
| `POST` | `/api/process` | Triggers document processing (accepts source selection) |
| `GET` | `/api/processing-status` | Real-time progress, queue, and cancellation state |
| `POST` | `/api/cancel-processing` | Instantly stops streaming generation |
| `GET` | `/api/entries?page=N&per_page=20` | Paginated Q&A entries (filter by `source_file`) |
| `POST` | `/api/entries/{id}` | Update content, toggle enable/review status |
| `POST` | `/api/export` | Export selected reviewed entries to JSONL |
| `GET` | `/api/llama-status` | Check backend connectivity |
| `POST` | `/api/sources/url` | Download PDF from URL into `pdf/` |
| `POST` | `/api/sources/text` | Save pasted text into `web/` |

## 📝 Best Practices: The `Input (Context)` Field

The dataset follows the standard **Alpaca instruction format**:
```json
{"instruction": "task/question", "input": "optional context/data", "output": "expected answer"}
```

### ✅ When to leave it empty (Recommended for most use cases)
If your questions are self-contained or you want the model to internalize knowledge, leave `"input": ""`. During fine-tuning, the model learns to answer based on the `instruction` alone, which matches real-world prompting behavior.

### 🔹 When to fill it in
Use the `Input` field when training for **context-dependent tasks**:
- **Summarization**: `Instruction`: "Summarize in 3 bullets." `Input`: "[Raw paragraph]"
- **Extraction**: `Instruction`: "Extract all dates and values." `Input`: "[Report excerpt]"
- **Translation**: `Instruction`: "Translate to French." `Input`: "[Source text]"
- **Few-shot/In-context learning**: Provide reference examples or schemas in `input`

### ⚠️ Fine-tuning note
Most training frameworks (Axolotl, Unsloth, HuggingFace `SFTTrainer`) automatically concatenate fields into prompt templates like:
```
Instruction: {instruction}
Input: {input}
Output: {output}
```
If you leave `input` empty, it's simply omitted or rendered as blank. This is perfectly valid and widely used in instruction-tuning pipelines.

## ⚙️ Configuration Tips

- **`app.persona` & `app.audience`**: Controls system prompt tone. Change to `"medical researcher"` / `"clinicians"` for healthcare, `"legal analyst"` / `"attorneys"` for law, etc.
- **`chunk_size`**: Reduce for smaller context windows; increase for longer documents. Default 2500 chars works well for 8K+ context models.
- **`qa_per_chunk`**: Number of pairs requested per chunk. Models may generate fewer if content is thin.
- **`batch_delay`**: Seconds between chunks to prevent server overload. Increase if you see `ReadTimeout` or OOM errors.
- **`enable_multimodal`**: Set to `false` if your model doesn't support vision tokens or to save VRAM/time.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [pdfplumber](https://github.com/jsvine/pdfplumber)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [llama.cpp](https://github.com/ggerganov/llama.cpp).
- Originally developed for climate research datasets; now fully open and domain-agnostic.
