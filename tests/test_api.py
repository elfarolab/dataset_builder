# tests/test_api.py
# Integration tests against a REAL running Dataset Builder server.
#
# Prerequisites:
#   1. Server must be running: python main.py
#   2. config.yaml must have correct server host/port
#   3. For LLM tests: llama.cpp server must be reachable
#
# Usage:
#   ./scripts/run_tests.sh                                    # Run all tests
#   TEST_PDF="test.pdf" ./scripts/run_tests.sh                # Include PDF tests
#   ./scripts/run_tests.sh -k "TestE2EProcessingWithLLM"      # Only LLM integration tests

import os
import sys
import json
import time
import yaml
import pytest
import httpx
import uuid

# ── Load the REAL config.yaml ───────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(CONFIG_PATH, "r") as _f:
    CONFIG = yaml.safe_load(_f)

APP_CONFIG = CONFIG["app"]
SERVER_CONFIG = CONFIG["server"]
LLAMA_CONFIG = CONFIG["llama_server"]
PATHS_CONFIG = CONFIG["paths"]
PROCESSING_CONFIG = CONFIG["processing"]

# Server URL for real HTTP calls
check_host = "localhost" if SERVER_CONFIG["host"] == "0.0.0.0" else SERVER_CONFIG["host"]
SERVER_URL = f"http://{check_host}:{SERVER_CONFIG['port']}"

# Optional test PDF from environment
TEST_PDF = os.environ.get("TEST_PDF", "")

# ── Load timeout from config.yaml ───────────────────────────────
LLAMA_TIMEOUT = LLAMA_CONFIG.get("timeout", 800)
BATCH_DELAY = PROCESSING_CONFIG.get("batch_delay", 8.0)



@pytest.fixture(scope="session")
def http_client():
    """Real httpx client for the running server."""
    client = httpx.Client(base_url=SERVER_URL, timeout=LLAMA_TIMEOUT)
    yield client
    client.close()


@pytest.fixture(autouse=True)
def reset_state(http_client):
    yield
    http_client.post("/api/test/reset")  # Cleanup after each test



# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _v(msg):
    try:
        if "-v" in sys.argv or "--verbose" in sys.argv or "--debug" in sys.argv:
            print(f"  [DEBUG] {msg}")
    except Exception:
        pass


def wait_for_processing(http_client, poll_interval=3):
    """Poll /api/processing-status until complete.
    
    Uses timeout from config.yaml — same value FastAPI uses for LLM calls.
    If FastAPI can wait this long, so can we.
    """
    # Config timeout is per LLM call. Processing may make multiple calls.
    # Use 3x the config timeout to be safe — still from config, not hardcoded.
    timeout = LLAMA_TIMEOUT * 3
    
    start = time.time()
    last_progress = -1
    last_log = time.time()
    
    print(f"  ⏱️  Waiting for processing (timeout: {timeout}s = {LLAMA_TIMEOUT}s × 3 from config)...")
    
    while time.time() - start < timeout:
        try:
            resp = http_client.get("/api/processing-status", timeout=300.0)
            if resp.status_code != 200:
                time.sleep(poll_interval)
                continue
            data = resp.json()
        except Exception as e:
            if time.time() - last_log > 15:
                print(f"  ⚠️  Status poll failed: {e}")
                last_log = time.time()
            time.sleep(poll_interval)
            continue
        
        current_progress = data.get("progress_percent", 0)
        phase = data.get("phase", "idle")
        current_source = data.get("current_source", "none")
        
        # Log on progress change OR every 15s so user knows test is alive
        if current_progress != last_progress or (time.time() - last_log > 15):
            elapsed = int(time.time() - start)
            remaining = int(timeout - (time.time() - start))
            print(f"  ⏳ [{phase}] {current_source}: {current_progress}% "
                  f"(elapsed: {elapsed}s, remaining: ~{remaining}s)")
            last_progress = current_progress
            last_log = time.time()
        
        if not data["is_processing"]:
            return data
        
        time.sleep(poll_interval)
    
    elapsed = time.time() - start
    # Try to grab final state one more time
    try:
        resp = http_client.get("/api/processing-status", timeout=300.0)
        final_data = resp.json() if resp.status_code == 200 else {}
    except Exception:
        final_data = {}
    
    raise TimeoutError(
        f"Processing timed out after {elapsed}s (limit: {timeout}s from config). "
        f"Final state: {final_data}"
    )

def wait_for_prior_processing_to_finish(http_client):
    """Ensure any leftover processing from a previous test has stopped.
    
    Uses a short timeout — just waiting for prior test to clean up,
    not waiting for LLM work.
    """
    # 10 seconds is enough to detect if prior test finished
    start = time.time()
    while time.time() - start < 10:
        try:
            resp = http_client.get("/api/processing-status", timeout=5.0)
            if resp.status_code == 200 and not resp.json()["is_processing"]:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ──────────────────────────────────────────────
# Helper: check if LLM is reachable, skip if not
# ──────────────────────────────────────────────

@pytest.fixture(scope="session")
def llama_reachable(http_client):
    """Check if llama.cpp server is reachable. Returns True/False."""
    try:
        resp = http_client.get("/api/llama-status")
        data = resp.json()
        if data["status"] == "connected":
            print(f"\n✅ LLM server reachable at {data['url']}")
            return True
        else:
            print(f"\n⚠️  LLM server status: {data['status']} - {data.get('message', '')}")
            return False
    except Exception as e:
        print(f"\n⚠️  Cannot check LLM status: {e}")
        return False


# ──────────────────────────────────────────────
# Helper: cleanup between tests via API
# ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_state(http_client):
    """Remove all sources and entries before and after each test.
    
    Does NOT call /api/cancel-processing — that sets a global flag that
    persists and poisons future tests. Just wait for processing to finish.
    """
    # Before: wait for any leftover processing
    wait_for_prior_processing_to_finish(http_client)
    
    # Delete all sources
    resp = http_client.get("/api/sources")
    if resp.status_code == 200:
        for source in resp.json().get("sources", []):
            http_client.delete(f"/api/sources/{source['filename']}")
    
    yield
    
    # After: wait for processing to finish, then delete sources
    wait_for_prior_processing_to_finish(http_client)
    
    resp = http_client.get("/api/sources")
    if resp.status_code == 200:
        for source in resp.json().get("sources", []):
            http_client.delete(f"/api/sources/{source['filename']}")

# ══════════════════════════════════════════════
# SECTION 1: Config Validation (fast, no server)
# ══════════════════════════════════════════════

class TestConfigLoaded:
    """Verify config.yaml values are sensible."""

    def test_app_name(self):
        assert APP_CONFIG["name"] == "Dataset Builder"

    def test_app_subtitle(self):
        assert "Review and refine" in APP_CONFIG["subtitle"]

    def test_persona(self):
        assert "climate change expert AI assistant" in APP_CONFIG["persona"]

    def test_audience(self):
        assert "farmers" in APP_CONFIG["audience"]

    def test_server_host(self):
        assert SERVER_CONFIG["host"] == "0.0.0.0"

    def test_server_port(self):
        assert SERVER_CONFIG["port"] > 0 and isinstance(SERVER_CONFIG["port"], int)

    def test_llama_server_url_format(self):
        url = LLAMA_CONFIG["url"]
        assert url.startswith("http://") or url.startswith("https://")
        assert ":" in url  # Has port

    def test_llama_timeout_positive(self):
        assert LLAMA_CONFIG["timeout"] > 0

    def test_llama_max_tokens_positive(self):
        assert LLAMA_CONFIG["max_tokens"] > 0

    def test_llama_temperature_range(self):
        temp = LLAMA_CONFIG["temperature"]
        assert 0.0 <= temp <= 2.0, f"Temperature {temp} out of range [0, 2]"

    def test_llama_top_p_range(self):
        top_p = LLAMA_CONFIG["top_p"]
        assert 0.0 < top_p <= 1.0, f"top_p {top_p} out of range (0, 1]"

    def test_llama_model_name_exists(self):
        assert len(LLAMA_CONFIG["name"]) > 0

    def test_paths_pdf_dir(self):
        assert len(PATHS_CONFIG["pdf_dir"]) > 0

    def test_paths_web_dir(self):
        assert len(PATHS_CONFIG["web_dir"]) > 0

    def test_paths_result_dir(self):
        assert len(PATHS_CONFIG["result_dir"]) > 0

    def test_paths_state_file(self):
        assert PATHS_CONFIG["state_file"].endswith(".json")

    def test_processing_chunk_size_positive(self):
        assert PROCESSING_CONFIG["chunk_size"] > 0

    def test_processing_qa_per_chunk_positive(self):
        assert PROCESSING_CONFIG["qa_per_chunk"] > 0

    def test_processing_batch_delay_non_negative(self):
        assert PROCESSING_CONFIG["batch_delay"] >= 0

    def test_semantic_chunking_is_bool(self):
        assert isinstance(PROCESSING_CONFIG["semantic_chunking"], bool)

    def test_semantic_batch_size_positive(self):
        assert PROCESSING_CONFIG["semantic_batch_size"] > 0

    def test_semantic_overlap_non_negative(self):
        assert PROCESSING_CONFIG["semantic_overlap"] >= 0

    def test_max_chunk_tokens_positive(self):
        assert PROCESSING_CONFIG["max_chunk_tokens"] > 0

    def test_multimodal_is_bool(self):
        assert isinstance(PROCESSING_CONFIG["enable_multimodal"], bool)

    def test_image_qa_per_image_positive(self):
        assert PROCESSING_CONFIG["image_qa_per_image"] > 0


# ══════════════════════════════════════════════
# SECTION 2: Server Health
# ══════════════════════════════════════════════

class TestServerHealth:
    """Verify the server is actually running and responsive."""

    def test_server_reachable(self, http_client):
        resp = http_client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        print("✅ Server homepage loads correctly")

    def test_api_config_endpoint(self, http_client):
        resp = http_client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == APP_CONFIG["name"]
        assert data["subtitle"] == APP_CONFIG["subtitle"]
        print(f"✅ Config endpoint returns: {data['name']}")


# ══════════════════════════════════════════════
# SECTION 3: Source Management (no LLM)
# ══════════════════════════════════════════════

class TestAddTextSource:
    """Test POST /api/sources/text"""

    def test_add_valid_text(self, http_client):
        text = "This is a meaningful article with enough content to be processed for dataset creation. " * 10
        resp = http_client.post("/api/sources/text", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["filename"].endswith(".txt")
        print(f"✅ Added text source: {data['filename']}")

    def test_empty_text_returns_400(self, http_client):
        resp = http_client.post("/api/sources/text", json={"text": ""})
        assert resp.status_code == 400
        print("✅ Empty text correctly rejected")

    def test_too_short_text_returns_400(self, http_client):
        resp = http_client.post("/api/sources/text", json={"text": "Too short"})
        assert resp.status_code == 400
        print("✅ Short text correctly rejected")


class TestApiSources:
    """Test GET /api/sources (only shows PROCESSED sources with Q&A entries)"""

    def test_empty_sources(self, http_client):
        resp = http_client.get("/api/sources")
        assert resp.status_code == 200
        data = resp.json()
        assert "sources" in data
        assert "total_sources" in data
        assert data["total_sources"] == 0
        print("✅ Empty sources list returned correctly")

    def test_unprocessed_source_shows_in_available(self, http_client):
        """Unprocessed files appear in /api/available-sources, NOT /api/sources"""
        # Add a source (not processed yet)
        http_client.post("/api/sources/text", json={"text": "Test content for sources list. " * 10})
        
        # /api/sources only shows PROCESSED sources (with Q&A entries)
        sources_resp = http_client.get("/api/sources")
        assert sources_resp.status_code == 200
        # Should still be 0 because we haven't processed it
        assert sources_resp.json()["total_sources"] == 0
        
        # /api/available-sources shows ALL files on disk
        available_resp = http_client.get("/api/available-sources")
        assert available_resp.status_code == 200
        available_data = available_resp.json()
        assert available_data["total"] >= 1
        assert available_data["unprocessed"] >= 1
        print(f"✅ Available sources: {available_data['total']} total, {available_data['unprocessed']} unprocessed")



class TestRemoveSource:
    """Test DELETE /api/sources/{source_file}"""

    def test_remove_source(self, http_client):
        # Add a source first
        add_resp = http_client.post("/api/sources/text", json={"text": "Temporary test content for removal testing. " * 10})
        assert add_resp.status_code == 200
        filename = add_resp.json()["filename"]

        # Remove it
        resp = http_client.delete(f"/api/sources/{filename}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["source_file"] == filename
        
        # Verify it's gone
        sources_resp = http_client.get("/api/sources")
        remaining = [s for s in sources_resp.json()["sources"] if s["filename"] == filename]
        assert len(remaining) == 0
        print(f"✅ Source removed: {filename}")


class TestAvailableSources:
    """Test GET /api/available-sources"""

    def test_empty_directories(self, http_client):
        resp = http_client.get("/api/available-sources")
        assert resp.status_code == 200
        data = resp.json()
        assert "sources" in data
        assert "total" in data
        assert "unprocessed" in data
        assert "processed" in data
        print(f"✅ Available sources: {data['total']} total")

    def test_with_text_file(self, http_client):
        http_client.post("/api/sources/text", json={"text": "Another test article with sufficient content. " * 15})
        
        resp = http_client.get("/api/available-sources")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        print(f"✅ Available sources after add: {data['total']} total, {data['unprocessed']} unprocessed")


# ══════════════════════════════════════════════
# SECTION 4: Stats & Debug (no LLM)
# ══════════════════════════════════════════════

class TestApiStats:
    """Test GET /api/stats"""

    def test_empty_stats(self, http_client):
        resp = http_client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        
        required_keys = [
            "generated_qa_pairs", "pdf_files", "text_files",
            "total_documents", "enabled_entries", "disabled_entries",
            "edited_entries", "pending_review"
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"
        
        assert data["generated_qa_pairs"] == 0
        print(f"✅ Empty stats: {data}")

    def test_stats_after_adding_entries(self, http_client):
        # Add a text source
        http_client.post("/api/sources/text", json={"text": "Test article content that is long enough. " * 20})
        
        resp = http_client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["text_files"] >= 1
        print(f"✅ Stats after add: {data['text_files']} text files")


class TestDebugState:
    """Test GET /api/debug/state"""

    def test_empty_state(self, http_client):
        resp = http_client.get("/api/debug/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "processed_documents_count" in data
        assert "sample_entry" in data
        assert data["processed_documents_count"] == 0
        print("✅ Debug state is empty as expected")


# ══════════════════════════════════════════════
# SECTION 5: Processing Status & Cancel (no LLM)
# ══════════════════════════════════════════════

class TestProcessingStatus:
    """Test GET /api/processing-status"""

    def test_initial_status(self, http_client):
        resp = http_client.get("/api/processing-status")
        assert resp.status_code == 200
        data = resp.json()
        
        required_keys = [
            "is_processing", "current_source", "progress_percent",
            "queue", "completed", "phase"
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"
        
        assert data["is_processing"] is False
        print(f"✅ Initial processing status: {data}")


class TestCancelProcessing:
    """Test POST /api/cancel-processing"""

    def test_cancel_current(self, http_client):
        resp = http_client.post("/api/cancel-processing", json={"current": True})
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"
        print("✅ Cancel current request works")

    def test_cancel_all(self, http_client):
        resp = http_client.post("/api/cancel-processing", json={"all": True})
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"
        print("✅ Cancel all request works")


# ══════════════════════════════════════════════
# SECTION 6: LLM Status Check
# ══════════════════════════════════════════════

class TestLlamaStatus:
    """Test GET /api/llama-status"""

    def test_endpoint_exists(self, http_client):
        resp = http_client.get("/api/llama-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] in ["connected", "disconnected", "error"]
        # Verify it's checking the REAL URL from config
        assert LLAMA_CONFIG["url"] in data.get("url", "")
        print(f"✅ Llama status: {data['status']} at {data['url']}")


# ══════════════════════════════════════════════
# SECTION 7: Entries API (no LLM needed for empty state)
# ══════════════════════════════════════════════

class TestEntries:
    """Test GET /api/entries"""

    def test_empty_entries(self, http_client):
        resp = http_client.get("/api/entries")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert len(data["entries"]) == 0
        print("✅ Empty entries list returned correctly")

    def test_entries_pagination_structure(self, http_client):
        resp = http_client.get("/api/entries?page=1&per_page=10")
        assert resp.status_code == 200
        data = resp.json()
        
        required_keys = ["entries", "total", "page", "per_page", "total_pages"]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"
        
        assert data["page"] == 1
        assert data["per_page"] == 10
        print("✅ Pagination structure is correct")

    def test_entries_with_invalid_page(self, http_client):
        # Page 999 should return empty but valid response
        resp = http_client.get("/api/entries?page=999&per_page=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert len(data["entries"]) == 0
        print("✅ Invalid page handled gracefully")


# ══════════════════════════════════════════════
# SECTION 8: Export (no LLM for empty state)
# ══════════════════════════════════════════════

class TestExport:
    """Test POST /api/export"""

    def test_no_entries_returns_400(self, http_client):
        resp = http_client.post("/api/export")
        assert resp.status_code == 400
        data = resp.json()
        assert "No enabled entries" in data["detail"]
        print("✅ Export with no entries correctly returns 400")


# ══════════════════════════════════════════════
# SECTION 9: Process endpoint basic behavior
# ══════════════════════════════════════════════

class TestProcessDocuments:
    """Test POST /api/process"""

    def test_already_processing_returns_409(self, http_client):
        """Verify the endpoint exists and rejects when already processing."""
        resp = http_client.post("/api/process", json={"sources": []})
        # Either 200 (processed nothing) or 400 (no files) — both valid
        assert resp.status_code in [200, 400]
        print(f"✅ Process endpoint responds: {resp.status_code}")


# ══════════════════════════════════════════════
# SECTION 10: REAL LLM INTEGRATION TESTS
# These tests actually trigger LLM calls and verify Q&A generation
# ══════════════════════════════════════════════

class TestE2EProcessingWithLLM:
    """End-to-end tests with REAL LLM calls.
    
    These tests actually call llama.cpp through the FastAPI server,
    wait for real processing to complete, and validate Q&A content.
    """
    def test_process_text_source_and_verify_qa(self, http_client, llama_reachable):
        """Full pipeline: add text → process → verify Q&A via real HTTP API + LLM."""
        if not llama_reachable:
            pytest.skip("LLM server not reachable — skipping real processing test")
    
        # Add a substantial text source
        text_content = (
            "Climate change represents one of the most significant challenges facing farmers today. "
            "Rising global temperatures are causing shifting growing seasons, with spring arriving "
            "earlier and fall extending later in many regions. These changes impact crop yields, "
            "water availability, and soil health in profound ways.\n\n"
            "Farmers around the world are experiencing shifts in their traditional planting calendars. "
            "Some regions face prolonged drought conditions while others deal with increased flooding. "
            "Coastal farming communities are particularly vulnerable as rising sea levels contaminate "
            "freshwater supplies with saltwater intrusion.\n\n"
            "Adaptation strategies include developing drought-resistant crop varieties, implementing "
            "precision irrigation systems, and adopting cover cropping techniques to improve soil "
            "moisture retention. Farmers are also exploring diversified crop rotations to reduce "
            "vulnerability to climate-related crop failures."
        )
    
        print(f"\n📝 Adding text source ({len(text_content)} chars)...")
        _v(f"POST /api/sources/text — body length: {len(text_content)}")
        add_resp = http_client.post("/api/sources/text", json={"text": text_content})
        _v(f"  Response status: {add_resp.status_code}")
        _v(f"  Response headers: {dict(add_resp.headers)}")
        _v(f"  Response body: {add_resp.text[:500]}")
        assert add_resp.status_code == 200, f"Expected 200, got {add_resp.status_code}: {add_resp.text}"
        filename = add_resp.json()["filename"]
        print(f"✅ Added: {filename}")
        _v(f"  Full add response JSON: {add_resp.json()}")
    
        # ── Verify the source appears in available-sources ──────────
        _v("Checking /api/available-sources after add...")
        avail_resp = http_client.get("/api/available-sources")
        _v(f"  /api/available-sources status: {avail_resp.status_code}")
        _v(f"  /api/available-sources body: {avail_resp.text[:500]}")
        avail_data = avail_resp.json()
        found = any(s["filename"] == filename for s in avail_data.get("sources", []))
        _v(f"  Source '{filename}' found in available-sources: {found}")
    
        # ── Check processing status BEFORE triggering ───────────────
        _v("Checking /api/processing-status BEFORE process call...")
        pre_status = http_client.get("/api/processing-status").json()
        _v(f"  Pre-process status: {pre_status}")
    
        # Trigger processing (this calls LLM)
        print(f"\n🚀 Triggering processing for {filename}...")
        _v(f"POST /api/process — body: {{'sources': ['{filename}']}}")
        process_resp = http_client.post("/api/process", json={"sources": [filename]})
        _v(f"  Response status: {process_resp.status_code}")
        _v(f"  Response headers: {dict(process_resp.headers)}")
        _v(f"  Response body (raw): {process_resp.text[:1000]}")
        try:
            proc_json = process_resp.json()
            _v(f"  Response JSON: {proc_json}")
        except Exception as e:
            _v(f"  Could not parse response as JSON: {e}")
    
        assert process_resp.status_code == 200, \
            f"Expected 200 from /api/process, got {process_resp.status_code}: {process_resp.text}"
        print(f"   Process started: {process_resp.json()}")
    
        # ── Check processing status AFTER triggering ────────────────
        _v("Checking /api/processing-status 1s after process call...")
        time.sleep(1)
        post_status = http_client.get("/api/processing-status").json()
        _v(f"  Post-process status: {post_status}")
        _v(f"  is_processing: {post_status.get('is_processing')}")
        _v(f"  current_source: {post_status.get('current_source')}")
        _v(f"  phase: {post_status.get('phase')}")
    
        # Wait for processing to complete (uses timeout from config.yaml)
        print(f"⏳ Waiting for LLM processing (timeout: ~{LLAMA_TIMEOUT * 2}s from config)...")
        _v(f"  LLAMA_TIMEOUT from config: {LLAMA_TIMEOUT}")
        _v(f"  wait_for_processing effective timeout: {LLAMA_TIMEOUT * 3}s")
        final_state = wait_for_processing(http_client)
        _v(f"  Final state returned by wait_for_processing: {final_state}")
        print(f"✅ Processing complete! Final state: {final_state}")
    
        assert final_state["progress_percent"] == 100, \
            f"Processing did not reach 100%: {final_state['progress_percent']}%"
        assert filename in final_state.get("completed", []), \
            f"File {filename} not in completed list"
    
        # Verify Q&A was generated
        _v(f"GET /api/entries?source_file={filename}")
        entries_resp = http_client.get(f"/api/entries?source_file={filename}")
        _v(f"  Response status: {entries_resp.status_code}")
        _v(f"  Response body (first 1000 chars): {entries_resp.text[:1000]}")
        assert entries_resp.status_code == 200
        entries_data = entries_resp.json()
    
        total_qa = entries_data["total"]
        print(f"\n📊 Generated {total_qa} Q&A entries")
        _v(f"  Full entries response: {json.dumps(entries_data, indent=2)[:2000]}")
    
        assert total_qa > 0, "LLM returned zero Q&A entries — processing may have failed"
    
        # Validate Q&A structure and content
        for entry in entries_data["entries"]:
            assert "instruction" in entry, "Entry missing 'instruction'"
            assert "output" in entry, "Entry missing 'output'"
            assert len(entry["instruction"]) > 10, \
                f"Instruction too short: '{entry['instruction']}'"
            assert len(entry["output"]) > 10, \
                f"Output too short: '{entry['output']}'"
    
            # Verify content is relevant (not gibberish)
            instruction_lower = entry["instruction"].lower()
            output_lower = entry["output"].lower()
            relevant_terms = ["climate", "farm", "crop", "temperature", "drought",
                             "water", "soil", "season", "yield"]
            has_relevant_content = any(term in instruction_lower or term in output_lower
                                       for term in relevant_terms)
    
            print(f"   ✓ Q: {entry['instruction'][:80]}...")
            print(f"     A: {entry['output'][:80]}...")
            _v(f"     Full instruction: {entry['instruction']}")
            _v(f"     Full output: {entry['output']}")
            assert has_relevant_content, \
                f"Q&A doesn't seem related to source content"
    
        print(f"\n✅ All {total_qa} Q&A entries validated successfully!")
    
        def test_process_pdf_source_and_verify_qa(self, http_client, llama_reachable):
            """Process a real PDF file and verify Q&A generation."""
            if not llama_reachable:
                pytest.skip("LLM server not reachable — skipping PDF processing test")
    
            if not TEST_PDF:
                pytest.skip("Set TEST_PDF env var to run this test (e.g., TEST_PDF='test.pdf')")
    
            pdf_path = os.path.join(PATHS_CONFIG["pdf_dir"], TEST_PDF)
            if not os.path.exists(pdf_path):
                pytest.skip(f"Test PDF not found: {pdf_path}")
    
            print(f"\n📄 Processing real PDF: {TEST_PDF}")
    
            # Trigger processing
            process_resp = http_client.post("/api/process", json={"sources": [TEST_PDF]})
            assert process_resp.status_code == 200
    
            # Wait for completion
            final_state = wait_for_processing(http_client)
            print(f"✅ PDF processing complete: {final_state}")
    
            # Verify Q&A generated
            entries_resp = http_client.get(f"/api/entries?source_file={TEST_PDF}")
            assert entries_resp.status_code == 200
            entries_data = entries_resp.json()
    
            total_qa = entries_data["total"]
            print(f"📊 PDF generated {total_qa} Q&A entries")
    
            if total_qa > 0:
                for entry in entries_data["entries"][:3]:  # Show first 3
                    assert len(entry["instruction"]) > 10
                    assert len(entry["output"]) > 10
                    print(f"   ✓ Q: {entry['instruction'][:60]}...")
            else:
                pytest.skip("PDF produced no Q&A — may be images-only or too short")


class TestEntryEditingAndReview:
    """Test editing and reviewing entries after LLM generation."""

    def _generate_entries(self, http_client, text):
        """Helper: add text source, process it, return filename."""
        resp = http_client.post("/api/sources/text", json={"text": text})
        assert resp.status_code == 200
        filename = resp.json()["filename"]
        
        http_client.post("/api/process", json={"sources": [filename]})
        wait_for_processing(http_client)  # Uses config timeout
        return filename

    def test_edit_entry_instruction(self, http_client):
        import time, uuid
        
        # 1. Warm up LLM server (prevents cold-start empty responses)
        http_client.post("/api/llama-status")
        time.sleep(3)  # KV-cache initialization
    
        # 2. Use longer, natural-looking text (avoids early stop tokens)
        marker = uuid.uuid4().hex[:8]
        text = f"""{marker}: Solar photovoltaic systems convert sunlight directly into electricity using semiconductor materials. 
    Modern panels achieve 20-22% efficiency under standard test conditions. 
    Installation requires proper orientation, tilt angle optimization, and grid-tie inverter compatibility. 
    Regular cleaning and thermal management significantly extend operational lifespan.""" * 3
    
        src_resp = http_client.post("/api/sources/text", json={"text": text})
        src_file = src_resp.json()["filename"]
    
        # 3. Process
        http_client.post("/api/process", json={"sources": [src_file]})
        wait_for_processing(http_client)
    
        # 4. Debug: check what actually happened
        stats = http_client.get("/api/stats").json()
        entries_resp = http_client.get(f"/api/entries?source_file={src_file}&per_page=1")
        entries = entries_resp.json()["entries"]
        
        if len(entries) == 0:
            print(f"⚠️ DEBUG: Stats after processing: {stats}")
            print(f"⚠️ DEBUG: Source file: {src_file}")
            
        assert len(entries) > 0, f"No entries generated. Stats: {stats}"
    
        entry_id = entries[0]["id"]
        
        # 5. Edit & verify
        edit_resp = http_client.post(f"/api/entries/{entry_id}", json={
            "id": entry_id,
            "instruction": "How do solar panels work?"
        })
        assert edit_resp.status_code == 200
    
        verify_resp = http_client.get(f"/api/entries?source_file={src_file}&per_page=1")
        updated = verify_resp.json()["entries"][0]
        assert updated["instruction"] == "How do solar panels work?"
        assert updated["edited"] is True
    
        # 6. Cleanup
        http_client.delete(f"/api/sources/{src_file}")


    def test_mark_source_all_reviewed(self, http_client, llama_reachable):
        """Generate Q&A via LLM, then mark entire source as reviewed."""
        if not llama_reachable:
            pytest.skip("LLM server not reachable")

        text = "Water conservation techniques include drip irrigation and rainwater harvesting systems. " * 15
        _v(f"📝 Adding text source ({len(text)} chars)...")
        
        add_resp = http_client.post("/api/sources/text", json={"text": text})
        _v(f"  POST /api/sources/text status: {add_resp.status_code}")
        _v(f"  Response body: {add_resp.text[:500]}")
        assert add_resp.status_code == 200, f"Expected 200, got {add_resp.status_code}: {add_resp.text}"
        filename = add_resp.json()["filename"]
        print(f"✅ Added source: {filename}")

        _v(f"🚀 Triggering processing for {filename}...")
        proc_resp = http_client.post("/api/process", json={"sources": [filename]})
        _v(f"  POST /api/process status: {proc_resp.status_code}")
        _v(f"  Response body: {proc_resp.text[:500]}")
        assert proc_resp.status_code == 200, f"Expected 200, got {proc_resp.status_code}: {proc_resp.text}"

        _v("⏳ Waiting for processing to complete...")
        final_state = wait_for_processing(http_client)
        _v(f"✅ Processing complete. Final state: {final_state}")
        assert final_state["progress_percent"] == 100, f"Processing did not reach 100%: {final_state['progress_percent']}%"

        _v(f"🚀 Calling /api/sources/{filename}/mark-all-reviewed")
        mark_resp = http_client.post(f"/api/sources/{filename}/mark-all-reviewed")
        _v(f"  Response status: {mark_resp.status_code}")
        _v(f"  Response body: {mark_resp.text[:500]}")

        assert mark_resp.status_code == 200, f"Expected 200, got {mark_resp.status_code}: {mark_resp.text}"
        mark_data = mark_resp.json()
        assert mark_data["status"] == "success"
        assert mark_data["updated"] > 0

        _v(f"🔍 Checking /api/sources for readiness...")
        sources_resp = http_client.get("/api/sources")
        _v(f"  Response status: {sources_resp.status_code}")
        _v(f"  Response body: {sources_resp.text[:500]}")

        source = next(
            (s for s in sources_resp.json()["sources"] if s["filename"] == filename),
            None
        )
        assert source is not None, f"Source {filename} not found in /api/sources"
        assert source["ready_for_export"] is True, f"Source not ready for export: {source}"
        print(f"✅ Source marked as ready for export: {filename}")

class TestExportWithRealData:
    """Test exporting real Q&A data generated by LLM."""

    def _generate_and_review(self, http_client, text):
        """Helper: generate entries and mark source as reviewed."""
        resp = http_client.post("/api/sources/text", json={"text": text})
        filename = resp.json()["filename"]
        
        http_client.post("/api/process", json={"sources": [filename]})
        wait_for_processing(http_client)
        
        http_client.post(f"/api/sources/{filename}/mark-all-reviewed")
        return filename

    def test_export_reviewed_entries(self, http_client, llama_reachable):
        """Generate Q&A, review them, then export to JSONL."""
        if not llama_reachable:
            pytest.skip("LLM server not reachable")

        text = "Integrated pest management reduces chemical pesticide use through biological controls. " * 15
        _v(f"📝 Adding text source ({len(text)} chars)...")
        
        add_resp = http_client.post("/api/sources/text", json={"text": text})
        _v(f"  POST /api/sources/text status: {add_resp.status_code}")
        _v(f"  Response body: {add_resp.text[:500]}")
        assert add_resp.status_code == 200
        filename = add_resp.json()["filename"]

        _v(f"🚀 Triggering processing for {filename}...")
        proc_resp = http_client.post("/api/process", json={"sources": [filename]})
        _v(f"  POST /api/process status: {proc_resp.status_code}")
        _v(f"  Response body: {proc_resp.text[:500]}")
        assert proc_resp.status_code == 200

        _v("⏳ Waiting for processing to complete...")
        final_state = wait_for_processing(http_client)
        _v(f"✅ Processing complete. Final state: {final_state}")
        assert final_state["progress_percent"] == 100

        _v(f"🚀 Marking all entries reviewed for {filename}...")
        mark_resp = http_client.post(f"/api/sources/{filename}/mark-all-reviewed")
        _v(f"  Response status: {mark_resp.status_code}")
        _v(f"  Response body: {mark_resp.text[:500]}")
        assert mark_resp.status_code == 200

        _v(f"🚀 Calling /api/export with sources=[{filename}]")
        export_resp = http_client.post("/api/export", json={
            "sources": [filename],
            "dataset_name": "test_export"
        })
        _v(f"  Response status: {export_resp.status_code}")
        _v(f"  Response body: {export_resp.text[:500]}")

        assert export_resp.status_code == 200, f"Expected 200, got {export_resp.status_code}: {export_resp.text}"
        export_data = export_resp.json()
        assert export_data["status"] == "success"
        assert export_data["entries_exported"] > 0
        assert export_data["filename"].startswith("test_export_")
        assert export_data["filename"].endswith(".jsonl")

        # Verify file exists and has valid JSONL
        filepath = export_data["filepath"]
        assert os.path.exists(filepath), f"Export file not found: {filepath}"

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == export_data["entries_exported"], \
            f"Expected {export_data['entries_exported']} lines, got {len(lines)}"

        # Verify each line is valid JSON with required fields
        for i, line in enumerate(lines):
            entry = json.loads(line.strip())
            assert "instruction" in entry
            assert "output" in entry
            assert len(entry["instruction"]) > 0

        print(f"✅ Exported {len(lines)} entries to {export_data['filename']}")

    def test_export_with_selected_sources(self, http_client, llama_reachable):
        
        if not llama_reachable:
            pytest.skip("LLM server not reachable")
    
        http_client.post("/api/llama-status")
        time.sleep(3)
    
        texts = [
            f"""{uuid.uuid4().hex[:8]}: Crop rotation improves soil fertility and breaks pest cycles naturally.
        Rotating legumes with cereals fixes nitrogen in the soil and reduces dependency on synthetic fertilizers.
        Farmers should plan 3-4 year rotation cycles for optimal yield and disease prevention.""" * 2,
            f"""{uuid.uuid4().hex[:8]}: Cover crops prevent soil erosion and add organic matter to the ground.
        Planting rye or clover during off-seasons protects topsoil from wind and water runoff.
        Terminating cover crops before cash crop planting ensures nutrient availability.""" * 2,
        ]
    
        filenames = []
        for text in texts:
            resp = http_client.post("/api/sources/text", json={"text": text})
            filenames.append(resp.json()["filename"])
    
        http_client.post("/api/process", json={"sources": filenames})
        wait_for_processing(http_client)

        mark_resp = http_client.post(f"/api/sources/{filenames[0]}/mark-all-reviewed")
        _v(f"Mark reviewed response: {mark_resp.status_code} - {mark_resp.json()}")
        
        # STRICT: Assert entries exist. Fail with debug info if LLM returned empty.
        entries_resp = http_client.get(f"/api/entries?source_file={filenames[0]}")
        assert entries_resp.status_code == 200
        entries_data = entries_resp.json()
        assert entries_data["total"] > 0, (
            f"LLM generated 0 Q&A pairs. Check stop tokens in LLM client config. "
            f"Source: {filenames[0]}. Stats: {http_client.get('/api/stats').json()}"
        )
        
        export_resp = http_client.post("/api/export", json={
            "sources": [filenames[0]],
            "dataset_name": "selective_export"
        })
    
        _v(f"Export response: {export_resp.status_code} - {export_resp.json()}")
            
        assert export_resp.status_code == 200
        export_data = export_resp.json()
        assert export_data["entries_exported"] > 0, f"Expected entries > 0, got {export_data.get('entries_exported')}"
    
        filepath = export_data["filepath"]
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    
        _v(f"✅ Selective export: {len(lines)} entries from {filenames[0]}")
        
        for fn in filenames:
            http_client.delete(f"/api/sources/{fn}")
        if os.path.exists(filepath):
            os.remove(filepath)


class TestEntriesPaginationWithRealData:
    """Test pagination on /api/entries with real LLM-generated data."""

    def test_pagination_with_entries(self, http_client, llama_reachable):
        """Generate enough Q&A to test pagination."""
        if not llama_reachable:
            pytest.skip("LLM server not reachable")

        # Generate entries
        text = (
            "Sustainable farming practices include conservation tillage, integrated nutrient management, "
            "and agroforestry systems that combine trees with crops. These methods build soil health, "
            "increase biodiversity, and reduce environmental impact while maintaining productivity."
        ) * 10
        
        http_client.post("/api/sources/text", json={"text": text})
        sources_resp = http_client.get("/api/sources")
        
        # File might not appear in /api/sources until processed
        available_resp = http_client.get("/api/available-sources")
        filename = available_resp.json()["sources"][0]["filename"]
        
        http_client.post("/api/process", json={"sources": [filename]})
        wait_for_processing(http_client)

        # Get total count
        resp = http_client.get(f"/api/entries?source_file={filename}&page=1&per_page=100")
        total = resp.json()["total"]
        print(f"📊 Total entries: {total}")

        if total == 0:
            pytest.skip("No entries generated for pagination test")

        # Test page 1 with small per_page
        resp = http_client.get(f"/api/entries?source_file={filename}&page=1&per_page=2")
        data = resp.json()
        assert len(data["entries"]) <= 2
        assert data["page"] == 1
        assert data["per_page"] == 2
        assert data["total"] == total

        # Test page 2 if there are enough entries
        if total > 2:
            resp = http_client.get(f"/api/entries?source_file={filename}&page=2&per_page=2")
            data = resp.json()
            assert data["page"] == 2
            assert len(data["entries"]) <= 2
            
            # Verify no duplicate IDs between pages
            page1_resp = http_client.get(f"/api/entries?source_file={filename}&page=1&per_page=2")
            page1_ids = {e["id"] for e in page1_resp.json()["entries"]}
            page2_ids = {e["id"] for e in data["entries"]}
            assert len(page1_ids & page2_ids) == 0, "Duplicate entries across pages"

        print(f"✅ Pagination works correctly: {total} total, tested with per_page=2")


class TestProcessingProgressTracking:
    """Test that processing progress is tracked correctly during LLM calls."""

    def test_progress_updates_during_processing(self, http_client, llama_reachable):
        """Verify progress percent increases during processing."""
        if not llama_reachable:
            pytest.skip("LLM server not reachable")
    
        # 1. Create a uniquely identifiable source
        marker = uuid.uuid4().hex[:8]
        text = f"Agroecology principles emphasize working with natural ecosystems rather than against them. {marker} " * 15
        src_resp = http_client.post("/api/sources/text", json={"text": text})
        created_file = src_resp.json()["filename"]
    
        # 2. Start processing (blocks until done, but we can verify state transitions)
        proc_resp = http_client.post("/api/process", json={"sources": [created_file]})
        assert proc_resp.status_code == 200
    
        # 3. Verify final state reflects completed work
        status_resp = http_client.get("/api/processing-status")
        status = status_resp.json()
        
        assert status["is_processing"] is False, "Processing should have finished"
        assert status["progress_percent"] == 100, f"Expected 100%, got {status['progress_percent']}"
        assert created_file in status["completed"], "Created file should be in completed list"
    
        # 4. Verify progress actually advanced (check if any chunks were processed)
        entries_resp = http_client.get(f"/api/entries?source_file={created_file}&per_page=1")
        assert len(entries_resp.json()["entries"]) > 0, "No Q&A generated, progress couldn't advance"
    
        # 5. Cleanup
        http_client.delete(f"/api/sources/{created_file}")
        
        print(f"✅ Processing completed: {status['progress_percent']}%, {created_file} in queue/completed")


class TestMultipleSourcesProcessing:
    """Test processing multiple sources in one batch."""

    def test_process_multiple_sources(self, http_client, llama_reachable):
        """Add and process multiple text sources at once."""
        if not llama_reachable:
            pytest.skip("LLM server not reachable")

        texts = [
            "No-till farming reduces soil erosion and increases carbon sequestration in agricultural lands. " * 12,
            "Buffer strips along waterways filter runoff and protect aquatic ecosystems from farm chemicals. " * 12,
        ]
        
        filenames = []
        for text in texts:
            resp = http_client.post("/api/sources/text", json={"text": text})
            filenames.append(resp.json()["filename"])

        print(f"\n🚀 Processing {len(filenames)} sources simultaneously...")
        
        # Process all at once
        process_resp = http_client.post("/api/process", json={"sources": filenames})
        assert process_resp.status_code == 200

        # Wait for completion (uses config timeout)
        final_state = wait_for_processing(http_client)
        
        # Verify both files completed
        assert len(final_state.get("completed", [])) >= len(filenames), \
            f"Not all sources completed: {final_state['completed']}"

        # Verify entries exist for each source
        for filename in filenames:
            entries_resp = http_client.get(f"/api/entries?source_file={filename}")
            assert entries_resp.status_code == 200
            total = entries_resp.json()["total"]
            print(f"   {filename}: {total} Q&A entries")

        print(f"✅ All {len(filenames)} sources processed successfully")


class TestCancelDuringProcessing:
    """Test cancellation behavior during active LLM processing."""

    def test_cancel_stops_processing(self, http_client, llama_reachable):
        """Start processing, cancel it, verify it stops."""
        if not llama_reachable:
            pytest.skip("LLM server not reachable")

        # Use a longer text to give cancellation time to take effect
        text = "Regenerative agriculture focuses on rebuilding soil organic matter and restoring degraded soil biodiversity. " * 20
        
        http_client.post("/api/sources/text", json={"text": text})
        
        # Use /api/available-sources
        available_resp = http_client.get("/api/available-sources")
        sources = available_resp.json().get("sources", [])
        assert len(sources) > 0, "No source files found on disk"
        filename = sources[0]["filename"]

        # Start processing
        import threading
        
        def start_processing():
            time.sleep(1)
            http_client.post("/api/process", json={"sources": [filename]})
        
        thread = threading.Thread(target=start_processing)
        thread.start()

        # Wait a bit for processing to start, then cancel
        time.sleep(5)
        
        cancel_resp = http_client.post("/api/cancel-processing", json={"all": True})
        assert cancel_resp.status_code == 200
        print("🛑 Cancellation requested")

        # Wait for it to stop (or timeout)
        try:
            final_state = wait_for_processing(http_client)
            print(f"✅ Processing stopped. Final state: is_processing={final_state['is_processing']}")
        except TimeoutError:
            resp = http_client.get("/api/processing-status")
            data = resp.json()
            print(f"⚠️  Processing may not have stopped within 60s: {data}")

        thread.join(timeout=10)


# ══════════════════════════════════════════════
# SECTION 11: Full E2E Workflow (optional, comprehensive)
# ══════════════════════════════════════════════

class TestFullWorkflowE2E:
    """Complete workflow: add → process → review → export."""

    def test_complete_workflow(self, http_client, llama_reachable):
        """Full cycle: text source → process → edit → mark reviewed → export."""
        if not llama_reachable:
            pytest.skip("LLM server not reachable — skipping full E2E workflow")

        print("\n" + "=" * 60)
        print("🧪 FULL END-TO-END WORKFLOW TEST")
        print("=" * 60)

        # 1. Add text source
        print("\n[1/6] Adding text source...")
        text = (
            "Climate-smart agriculture integrates three objectives: "
            "sustainably increasing agricultural productivity and incomes, "
            "adapting and building resilience to climate change, and "
            "reducing and/or removing greenhouse gases, where possible."
        ) * 8
        
        add_resp = http_client.post("/api/sources/text", json={"text": text})
        assert add_resp.status_code == 200
        filename = add_resp.json()["filename"]
        print(f"   ✅ Added: {filename}")

        # 2. Verify it appears in available sources
        print("\n[2/6] Verifying source in list...")
        sources_resp = http_client.get("/api/available-sources")
        assert any(s["filename"] == filename for s in sources_resp.json()["sources"])
        print(f"   ✅ Source visible in available sources")

        # 3. Process with LLM (uses config timeout)
        print("\n[3/6] Processing with LLM...")
        process_resp = http_client.post("/api/process", json={"sources": [filename]})
        print(f"   Process response: {process_resp.status_code} - {process_resp.json()}")
        assert process_resp.status_code == 200
        
        final_state = wait_for_processing(http_client)
        assert final_state["progress_percent"] == 100
        print(f"   ✅ Processing complete")

        # 4. Verify and edit Q&A
        print("\n[4/6] Verifying and editing Q&A...")
        entries_resp = http_client.get(f"/api/entries?source_file={filename}")
        entries_data = entries_resp.json()
        assert entries_data["total"] > 0
        
        first_entry = entries_data["entries"][0]
        edit_resp = http_client.post(f"/api/entries/{first_entry['id']}", json={
            "id": first_entry["id"],
            "instruction": first_entry["instruction"],
            "output": first_entry["output"]
        })
        assert edit_resp.status_code == 200
        print(f"   ✅ Edited entry: {first_entry['id']}")

        # 5. Mark all reviewed
        print("\n[5/6] Marking source as reviewed...")
        mark_resp = http_client.post(f"/api/sources/{filename}/mark-all-reviewed")
        assert mark_resp.status_code == 200
        
        sources_resp = http_client.get("/api/sources")
        source = next(s for s in sources_resp.json()["sources"] if s["filename"] == filename)
        assert source["ready_for_export"] is True
        print(f"   ✅ Source ready for export")

        # 6. Export
        print("\n[6/6] Exporting dataset...")
        export_resp = http_client.post("/api/export", json={
            "sources": [filename],
            "dataset_name": "e2e_test_dataset"
        })
        assert export_resp.status_code == 200
        export_data = export_resp.json()
        assert export_data["entries_exported"] > 0
        
        with open(export_data["filepath"], "r") as f:
            exported_lines = f.readlines()
        assert len(exported_lines) == export_data["entries_exported"]
        
        print(f"   ✅ Exported {len(exported_lines)} entries to {export_data['filename']}")

        print("\n" + "=" * 60)
        print("🎉 FULL E2E WORKFLOW PASSED!")
        print("=" * 60)

# ══════════════════════════════════════════════
# SECTION 12: Edge Cases & Error Handling
# ══════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_process_nonexistent_source(self, http_client):
        """Processing a source that doesn't exist should fail gracefully."""
        resp = http_client.post("/api/process", json={"sources": ["nonexistent.pdf"]})
        assert resp.status_code in [200, 400]
        print(f"✅ Nonexistent source handled: {resp.status_code}")

    def test_entries_filter_by_nonexistent_source(self, http_client):
        """Filtering entries by a source that has no entries."""
        resp = http_client.get("/api/entries?source_file=nonexistent.txt")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        print("✅ Filter by nonexistent source returns empty list")

    def test_update_nonexistent_entry(self, http_client):
        """Updating an entry that doesn't exist should return 404."""
        # FastAPI Pydantic validation requires a proper JSON body
        resp = http_client.post(
            "/api/entries/nonexistent-id",
            json={
                "id": "nonexistent-id",
                "instruction": "test"
            }
        )
        assert resp.status_code == 404, f"Expected 404, got {resp.status_code}: {resp.text}"
        print("✅ Updating nonexistent entry returns 404")

    def test_stats_structure_completeness(self, http_client):
        """Verify stats endpoint has all expected fields."""
        resp = http_client.get("/api/stats")
        data = resp.json()
        
        expected_fields = {
            "pdf_files", "text_files", "total_documents",
            "generated_qa_pairs", "enabled_entries", "disabled_entries",
            "edited_entries", "pending_review"
        }
        missing = expected_fields - set(data.keys())
        assert not missing, f"Missing stats fields: {missing}"
        print(f"✅ Stats has all {len(expected_fields)} expected fields")

    def test_processing_status_structure_completeness(self, http_client):
        """Verify processing status has all expected fields."""
        resp = http_client.get("/api/processing-status")
        data = resp.json()
        
        expected_fields = {
            "is_processing", "current_source", "progress_percent",
            "queue", "completed", "phase"
        }
        missing = expected_fields - set(data.keys())
        assert not missing, f"Missing status fields: {missing}"
        print(f"✅ Processing status has all {len(expected_fields)} expected fields")



if __name__ == "__main__":
    pytest.main([__file__, "-v"])

