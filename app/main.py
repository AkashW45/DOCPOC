"""
LangChain + Azure Read OCR script

Place your Azure Computer Vision endpoint and key in environment variables:
  - AZURE_OCR_ENDPOINT
  - AZURE_OCR_KEY

Script will read images from the images folder, call Azure Read API,
create LangChain `Document` objects (one per page), and write a
markdown `output.md` at the workspace root with embedded images and OCR text.
"""
from pathlib import Path
import os
import time
import requests
from PIL import Image

try:
    from langchain.schema import Document
except Exception:
    try:
        from langchain.docstore.document import Document
    except Exception:
        from dataclasses import dataclass

        @dataclass
        class Document:
            page_content: str
            metadata: dict

# Configuration - prefer env vars
AZURE_ENDPOINT = os.environ.get("AZURE_OCR_ENDPOINT")
AZURE_KEY = os.environ.get("AZURE_OCR_KEY")

# Optional Tesseract path (env var) and import
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", r"C:\Users\user\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
except Exception:
    pytesseract = None

IMAGE_FOLDER = Path(r"C:\Users\user\Downloads\genAI-POC\images\ilovepdf_pages-to-jpg")
OUTPUT_MD = Path(__file__).resolve().parents[1] / "output.md"

# READ_API only used if Azure vars provided
READ_API = None
if AZURE_ENDPOINT and AZURE_KEY:
    READ_API = AZURE_ENDPOINT.rstrip("/") + "/vision/v3.2/read/analyze"

def azure_read_image(path: Path, timeout: int = 30) -> str:
    """Send image to Azure Read API and poll for results."""
    with open(path, "rb") as f:
        resp = requests.post(READ_API, headers={"Ocp-Apim-Subscription-Key": AZURE_KEY, "Content-Type": "application/octet-stream"}, data=f)

    if resp.status_code not in (202,):
        raise RuntimeError(f"Azure Read API error: {resp.status_code} {resp.text}")

    operation_url = resp.headers.get("Operation-Location")
    if not operation_url:
        raise RuntimeError("Missing Operation-Location header from Azure response")

    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(operation_url, headers={"Ocp-Apim-Subscription-Key": AZURE_KEY})
        rj = r.json()
        status = rj.get("status")
        if status == "succeeded":
            read_results = rj.get("analyzeResult", {}).get("readResults", [])
            lines = []
            for page in read_results:
                for line in page.get("lines", []):
                    lines.append(line.get("text", ""))
            return "\n".join(lines)
        elif status == "failed":
            raise RuntimeError(f"Azure Read failed: {rj}")
        time.sleep(0.5)

    raise TimeoutError("Timed out waiting for Azure Read result")


def main():
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}

    files = sorted([p for p in IMAGE_FOLDER.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    docs = []
    md_lines = ["# OCR Output\n"]

    for idx, path in enumerate(files, start=1):
        rel_path = os.path.relpath(path, OUTPUT_MD.parent).replace("\\", "/")
        md_lines.append(f"## Page {idx} — {path.name}\n")
        md_lines.append(f"![{path.name}]({rel_path})\n")

        text = ""
        # Prefer Azure Read if configured, otherwise use pytesseract.
        if READ_API:
            try:
                text = azure_read_image(path)
            except Exception as e_azure:
                # Fall back to local Tesseract if available
                if pytesseract:
                    try:
                        img = Image.open(path)
                        text = pytesseract.image_to_string(img, lang="eng")
                    except Exception as e_tess:
                        text = f"_Error during OCR: Azure error: {e_azure}; Tesseract error: {e_tess}_"
                else:
                    text = f"_Error during OCR: Azure error: {e_azure}_"
        else:
            if pytesseract:
                try:
                    img = Image.open(path)
                    text = pytesseract.image_to_string(img, lang="eng")
                except Exception as e:
                    text = f"_Tesseract error: {e}_"
            else:
                text = "_No OCR backend available. Set AZURE_OCR_* or install pytesseract and Tesseract executable._"

        docs.append(Document(page_content=text, metadata={"page": idx, "source": str(path)}))

        if not text.strip():
            md_lines.append("_No text recognized._\n")
        else:
            md_lines.append("```text")
            md_lines.append(text.rstrip())
            md_lines.append("```\n")

    OUTPUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {len(files)} pages to {OUTPUT_MD}")


if __name__ == "__main__":
    main()