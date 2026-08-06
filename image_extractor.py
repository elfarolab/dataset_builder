# image_extractor.py — COMPLETE FILE, OVERWRITE YOUR EXISTING ONE
"""Extract images from PDFs and process them with multimodal LLM."""

import base64
import io
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import fitz
from PIL import Image


def extract_images_from_pdf(pdf_path: str) -> List[Dict]:
    """Extract ALL visual content from a PDF by rendering pages with images."""
    images = []
    
    try:
        doc = fitz.open(pdf_path)
        print(f"  [INFO] PDF has {len(doc)} pages")

        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Check for raster images
            image_list = page.get_images(full=True)
            has_raster = len(image_list) > 0
            
            # Check for vector graphics
            has_vector = len(page.get_drawings()) > 0
            
            print(f"    Page {page_num + 1}: raster={has_raster}, vector={has_vector}")
           
            if has_raster or has_vector:
                zoom = 1.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)

                # Resize to max 1024x1024 (standard for vision LLMs)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                max_dim = 1024
                if img.width > max_dim or img.height > max_dim:
                    img.thumbnail((max_dim, max_dim))
                
                img_buf = io.BytesIO()
                img.save(img_buf, format="PNG")
                image_bytes = img_buf.getvalue()

                images.append({
                    "page": page_num + 1,
                    "image_bytes": image_bytes,
                    "format": "png",
                    "width": img.width,
                    "height": img.height,
                    "bbox": [0, 0, img.width, img.height],
                })
                print(f"      → Rendered page {page_num + 1} as {img.width}x{img.height} PNG")
 
            else:
                print(f"    Page {page_num + 1}: text only, skipping")

        doc.close()

    except Exception as e:
        print(f"[ERROR] Failed to extract images from PDF: {e}")
        import traceback
        traceback.print_exc()

    print(f"  [INFO] Extracted {len(images)} page images from {os.path.basename(pdf_path)}")
    return images


def extract_images_from_web_dir(web_dir: str) -> List[Dict]:
    """Extract standalone image files from web/ directory."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    images = []

    for filename in os.listdir(web_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            filepath = os.path.join(web_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    image_bytes = f.read()

                img = Image.open(io.BytesIO(image_bytes))
                width, height = img.size

                images.append({
                    "page": 0,
                    "image_bytes": image_bytes,
                    "format": ext.lstrip('.'),
                    "width": width,
                    "height": height,
                    "bbox": [0, 0, width, height],
                    "source_file": filename,
                })
            except Exception as e:
                print(f"  [WARN] Could not read image {filename}: {e}")

    print(f"  Found {len(images)} standalone images in {web_dir}")
    return images


def image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')


def get_image_context_text(full_text: str, page_num: int) -> str:
    """Extract the text from the same page as the image.
    
    For slide presentations, each page's text serves as the caption/context.
    Returns text between [PAGE N] and [PAGE N+1] markers.
    """
    page_pattern = r'\[PAGE (\d+)\]'
    matches = list(re.finditer(page_pattern, full_text))
    
    if not matches:
        return ""
    
    # Find the marker for our target page
    target_idx = None
    for i, match in enumerate(matches):
        if int(match.group(1)) == page_num:
            target_idx = i
            break
    
    if target_idx is None:
        return ""
    
    # Get text from this page marker to the next page marker (or end of doc)
    start_pos = matches[target_idx].start()
    end_pos = matches[target_idx + 1].start() if target_idx < len(matches) - 1 else len(full_text)
    
    section_text = full_text[start_pos:end_pos].strip()
    
    # Remove the [PAGE N] marker itself for cleaner context
    section_text = re.sub(r'\[PAGE \d+\]\s*', '', section_text, count=1).strip()
    
    # Limit to reasonable size for LLM context
    if len(section_text) > 2000:
        section_text = section_text[:2000] + "..."
    
    return section_text

