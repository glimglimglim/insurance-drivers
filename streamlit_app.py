from __future__ import annotations
import base64
import io
import json
import mimetypes
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import List

import boto3
from langfuse.decorators import observe
from langfuse.openai import openai  # Langfuse-wrapped OpenAI client
import streamlit as st
from PIL import Image

# Optional PDF support
try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

# ──────────────────────────────────────────────────────────────────────────────
# Langfuse Configuration (via environment variables)
# ──────────────────────────────────────────────────────────────────────────────
LANGFUSE_SECRET_KEY = "sk-lf-884f8f3a-6fcb-41a0-831a-018b355a03b4"
LANGFUSE_PUBLIC_KEY = "pk-lf-9b6ba0a4-31cd-4347-ab73-17d0c35786c"
LANGFUSE_HOST = "https://langfuse.ai.wrs.dev"
os.environ.setdefault("LANGFUSE_SECRET_KEY", LANGFUSE_SECRET_KEY)
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", LANGFUSE_PUBLIC_KEY)
os.environ.setdefault("LANGFUSE_HOST", LANGFUSE_HOST)

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit App Configuration
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Document Data Extractor", layout="centered")

# Let the user select driver’s license vs. insurance card
doc_type = st.selectbox("Select Document Type", ["Driver's License", "Insurance Card"])

if doc_type == "Driver's License":
    st.title("🪪 ➜ 📋 Driver-License Data Extractor")
else:
    st.title("🩺 ➜ 📋 Insurance-Card Data Extractor")

# Fields for driver's license
DL_FIELDS = [
    "license_number", "class", "first_name", "middle_name", "last_name",
    "address", "city", "state", "zip", "date_of_birth", "issue_date",
    "expiration_date", "sex", "eye_color", "hair", "height", "organ_donor", "weight"
]

SYSTEM_PROMPT_DL = (
    "You are an identity-document data extractor. "
    "Extract the following fields from a U.S. driver's-license image and return *only* valid JSON "
    "with exactly these keys in this order: "
    + ", ".join(DL_FIELDS)
    + ". Use ISO-8601 dates (YYYY-MM-DD). If a field is missing, set its value to an empty string."
)

# Fields for insurance card (example set for a Medicare-style card)
IC_FIELDS = [
    "beneficiary_name",
    "member_id",
    "sex",
    "plan_name",
    "plan_type",
    "coverage_part_a",
    "coverage_part_a_effective_date",
    "coverage_part_b",
    "coverage_part_b_effective_date",
    "date_of_birth"
]

SYSTEM_PROMPT_IC = (
    "You are an identity-document data extractor. "
    "Extract the following fields from a U.S. insurance card (e.g. Medicare) and return *only* valid JSON "
    "with exactly these keys in this order: "
    + ", ".join(IC_FIELDS)
    + ". Use ISO-8601 dates (YYYY-MM-DD). If a field is missing, set its value to an empty string."
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — API Key & Client Initialization
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 API Keys & Clients")

    # OpenAI (Langfuse-wrapped)
    openai.api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    if not openai.api_key:
        k = st.text_input("OpenAI API key", type="password", placeholder="sk-...")
        if k:
            openai.api_key = k
    else:
        st.success("OpenAI key loaded.")

    # Gemini
    gemini_key = os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))
    if not gemini_key:
        gemini_key = st.text_input(
            "Gemini API key",
            type="password",
            placeholder="…",
            help="Required for Gemini extraction"
        )
        if gemini_key:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=gemini_key)
    else:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=gemini_key)
        st.success("Gemini client initialized.")

    # AWS Textract
    try:
        aws_cfg = st.secrets["aws"]
        textract = boto3.client(
            "textract",
            aws_access_key_id=aws_cfg["aws_access_key_id"],
            aws_secret_access_key=aws_cfg["aws_secret_access_key"],
            aws_session_token=aws_cfg.get("aws_session_token"),
            region_name=aws_cfg.get("region_name", "us-east-1"),
        )
        st.success("AWS Textract client initialized.")
    except Exception:
        st.error("Make sure you have an [aws] section in .streamlit/secrets.toml")

    st.markdown(
        "---\n"
        "⚠️ **Privacy reminder:** ensure you are authorized to process any personal data you upload."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────
def _file_to_images(path: Path) -> List[Image.Image]:
    mime, _ = mimetypes.guess_type(path)
    if mime == "application/pdf":
        if convert_from_path is None:
            raise RuntimeError("Install pdf2image and Poppler for PDF support.")
        return convert_from_path(path, dpi=300)
    if mime and mime.startswith("image/"):
        return [Image.open(path)]
    raise ValueError(f"Unsupported file type: {path}")

def _pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def file_to_base64_chunks(path: Path) -> List[str]:
    return [_pil_to_base64(im.convert("RGB")) for im in _file_to_images(path)]

def render_fields_grid(container, title: str, data: dict, fields: List[str], num_cols: int = 3):
    container.subheader(title)
    cols = container.columns(num_cols)
    for idx, field in enumerate(fields):
        col = cols[idx % num_cols]
        label = field.replace("_", " ").title()
        value = data.get(field, "") or ""
        col.markdown(f"""
        <div style="display:flex; flex-direction:column; gap:4px;">
            <div style="font-weight:bold;">{label}</div>
            <div style="
                background-color:#f8f9fa;
                padding:8px;
                border-radius:6px;
                color:#111;
                min-height:38px;
                font-family:monospace;
                font-size:0.95em;
                word-wrap:break-word;
                border: 1px solid #ddd;
            ">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Model Invocation Functions for DRIVER’S LICENSE
# ──────────────────────────────────────────────────────────────────────────────
@observe(as_type="generation", name="GPT-4.1-mini DL Extraction")
def gpt4_1_mini_dl_from_images(b64_images: List[str]) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_DL},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}
                }
                for b64 in b64_images
            ],
        },
    ]
    resp = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
        stream=False,
        max_tokens=4096,
    )
    return json.loads(resp.choices[0].message.content)

@observe(as_type="generation", name="GPT-4.1 DL Extraction")
def gpt4_1_dl_from_images(b64_images: List[str]) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_DL},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}
                }
                for b64 in b64_images
            ],
        },
    ]
    resp = openai.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
        stream=False,
        max_tokens=4096,
    )
    return json.loads(resp.choices[0].message.content)

@observe(as_type="generation", name="Gemini 2.0 Flash DL Extraction")
def gemini_dl_from_images(b64_images: List[str]) -> dict:
    from google.genai import types
    image_parts = [
        types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/png")
        for b64 in b64_images
    ]
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite-preview-02-05",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT_DL,
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=4096
        ),
        contents=image_parts
    )
    return json.loads(response.text)

@observe(as_type="generation", name="Gemini 2.5 Flash DL Extraction")
def gemini_2_5_dl_from_images(b64_images: List[str]) -> dict:
    from google.genai import types
    image_parts = [
        types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/png")
        for b64 in b64_images
    ]
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT_DL,
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=4096
        ),
        contents=image_parts
    )
    return json.loads(response.text)

@observe(as_type="custom", name="AWS Textract DL Extraction")
def textract_dl_from_images(path: Path) -> dict:
    """
    Simple Textract-based approach for Driver's License.
    """
    FIELD_KEYWORDS = {
        "license_number": ["license", "lic no", "dl number"],
        "class": ["class"],
        "first_name": ["first name", "given name"],
        "middle_name": ["middle name"],
        "last_name": ["last name", "surname"],
        "address": ["address"],
        "city": ["city"],
        "state": ["state"],
        "zip": ["zip", "postal code"],
        "date_of_birth": ["date of birth", "dob"],
        "issue_date": ["date of issue", "issue date"],
        "expiration_date": ["expiration date", "exp date", "exp"],
        "sex": ["sex", "gender"],
        "eye_color": ["eye color", "eyes"],
        "hair": ["hair"],
        "height": ["height"],
        "organ_donor": ["organ donor"],
        "weight": ["weight"]
    }

    results = {k: "" for k in DL_FIELDS}
    images = _file_to_images(path)

    for img in images:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        resp = textract.analyze_document(
            Document={'Bytes': buf.getvalue()},
            FeatureTypes=['FORMS']
        )
        blocks = resp.get('Blocks', [])
        block_map = {b['Id']: b for b in blocks}
        key_map = {b['Id']: b for b in blocks if b['BlockType'] == 'KEY_VALUE_SET' and 'KEY' in b.get('EntityTypes', [])}
        value_map = {b['Id']: b for b in blocks if b['BlockType'] == 'KEY_VALUE_SET' and 'VALUE' in b.get('EntityTypes', [])}

        def get_text(block):
            text = ""
            for rel in block.get('Relationships', []):
                if rel['Type'] == 'CHILD':
                    for cid in rel['Ids']:
                        word = block_map.get(cid)
                        if word and word['BlockType'] == 'WORD':
                            text += word['Text'] + ' '
            return text.strip()

        kvs: dict[str, str] = {}
        for key_id, key_block in key_map.items():
            key_text = get_text(key_block).lower()
            val_text = ""
            for rel in key_block.get('Relationships', []):
                if rel['Type'] == 'VALUE':
                    for vid in rel['Ids']:
                        val_block = value_map.get(vid)
                        if val_block:
                            val_text = get_text(val_block)
            kvs[key_text] = val_text

        # Match to our known fields
        for field, keywords in FIELD_KEYWORDS.items():
            for key_text, val_text in kvs.items():
                if any(keyword in key_text for keyword in keywords):
                    results[field] = val_text
                    break

    return results

# ──────────────────────────────────────────────────────────────────────────────
# Model Invocation Functions for INSURANCE CARD
# ──────────────────────────────────────────────────────────────────────────────

@observe(as_type="generation", name="GPT-4.1-mini Insurance Extraction")
def gpt4_1_mini_insurance_from_images(b64_images: List[str]) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_IC},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}
                }
                for b64 in b64_images
            ],
        },
    ]
    resp = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
        stream=False,
        max_tokens=4096,
    )
    return json.loads(resp.choices[0].message.content)

@observe(as_type="generation", name="GPT-4.1 Insurance Extraction")
def gpt4_1_insurance_from_images(b64_images: List[str]) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_IC},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}
                }
                for b64 in b64_images
            ],
        },
    ]
    resp = openai.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
        stream=False,
        max_tokens=4096,
    )
    return json.loads(resp.choices[0].message.content)

@observe(as_type="generation", name="Gemini 2.0 Flash Insurance Extraction")
def gemini_insurance_from_images(b64_images: List[str]) -> dict:
    from google.genai import types
    image_parts = [
        types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/png")
        for b64 in b64_images
    ]
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite-preview-02-05",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT_IC,
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=4096
        ),
        contents=image_parts
    )
    return json.loads(response.text)

@observe(as_type="generation", name="Gemini 2.5 Flash Insurance Extraction")
def gemini_2_5_insurance_from_images(b64_images: List[str]) -> dict:
    from google.genai import types
    image_parts = [
        types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/png")
        for b64 in b64_images
    ]
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT_IC,
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=4096
        ),
        contents=image_parts
    )
    return json.loads(response.text)

@observe(as_type="custom", name="AWS Textract Insurance Extraction")
def textract_ic_from_images(path: Path) -> dict:
    """
    Simple Textract-based approach for Insurance Card.
    """
    FIELD_KEYWORDS_IC = {
        "beneficiary_name": ["beneficiary", "name of beneficiary"],
        "member_id": ["claim number", "member id", "policy number"],
        "sex": ["sex", "gender"],
        "plan_name": ["plan name", "medical", "medicare", "hospital", "insurance"],
        "plan_type": ["type", "plan type"],
        "coverage_part_a": ["hospital (part a)", "part a coverage", "hospital coverage"],
        "coverage_part_a_effective_date": ["part a eff date", "hospital eff", "part a start"],
        "coverage_part_b": ["medical (part b)", "part b coverage", "medical coverage"],
        "coverage_part_b_effective_date": ["part b eff date", "medical eff", "part b start"],
        "date_of_birth": ["date of birth", "dob"]
    }

    results = {k: "" for k in IC_FIELDS}
    images = _file_to_images(path)

    for img in images:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        resp = textract.analyze_document(
            Document={'Bytes': buf.getvalue()},
            FeatureTypes=['FORMS']
        )
        blocks = resp.get('Blocks', [])
        block_map = {b['Id']: b for b in blocks}
        key_map = {
            b['Id']: b for b in blocks
            if b['BlockType'] == 'KEY_VALUE_SET' and 'KEY' in b.get('EntityTypes', [])
        }
        value_map = {
            b['Id']: b for b in blocks
            if b['BlockType'] == 'KEY_VALUE_SET' and 'VALUE' in b.get('EntityTypes', [])
        }

        def get_text(block):
            text = ""
            for rel in block.get('Relationships', []):
                if rel['Type'] == 'CHILD':
                    for cid in rel['Ids']:
                        word = block_map.get(cid)
                        if word and word['BlockType'] == 'WORD':
                            text += word['Text'] + ' '
            return text.strip()

        kvs: dict[str, str] = {}
        for key_id, key_block in key_map.items():
            key_text = get_text(key_block).lower()
            val_text = ""
            for rel in key_block.get('Relationships', []):
                if rel['Type'] == 'VALUE':
                    for vid in rel['Ids']:
                        val_block = value_map.get(vid)
                        if val_block:
                            val_text = get_text(val_block)
            kvs[key_text] = val_text

        # Match to our known fields for insurance
        for field, keywords in FIELD_KEYWORDS_IC.items():
            for key_text, val_text in kvs.items():
                if any(keyword in key_text for keyword in keywords):
                    results[field] = val_text
                    break

    return results

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
if doc_type == "Driver's License":
    upload_label = "Choose an image or PDF of a driver's license"
    fields_list = DL_FIELDS
else:
    upload_label = "Choose an image or PDF of an insurance card"
    fields_list = IC_FIELDS

uploaded_file = st.file_uploader(
    upload_label,
    type=["pdf", "png", "jpg", "jpeg", "tiff", "tif"],
)

if uploaded_file and openai.api_key and gemini_key:
    if st.button("🚀 Extract with GPT-4.1-mini, GPT-4.1, Gemini & AWS Textract", type="primary"):
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp.name)

        with st.spinner("Converting file …"):
            try:
                b64_chunks = file_to_base64_chunks(tmp_path)
                images = _file_to_images(tmp_path)
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.stop()

        # Depending on doc_type, call the appropriate extraction methods
        if doc_type == "Driver's License":
            # GPT-4.1-mini
            with st.spinner("Extracting with GPT-4.1-mini …"):
                try:
                    dl_openai_mini = gpt4_1_mini_dl_from_images(b64_chunks)
                except Exception as e:
                    st.error(f"OpenAI API error (mini): {e}")
                    dl_openai_mini = {k: "" for k in DL_FIELDS}

            # GPT-4.1
            with st.spinner("Extracting with GPT-4.1 …"):
                try:
                    dl_gpt4_1 = gpt4_1_dl_from_images(b64_chunks)
                except Exception as e:
                    st.error(f"OpenAI API error (full): {e}")
                    dl_gpt4_1 = {k: "" for k in DL_FIELDS}

            # Gemini 2.0
            with st.spinner("Extracting with Gemini 2.0 Flash …"):
                try:
                    dl_gemini = gemini_dl_from_images(b64_chunks)
                except Exception as e:
                    st.error(f"Gemini API error (2.0 Flash): {e}")
                    dl_gemini = {k: "" for k in DL_FIELDS}

            # Gemini 2.5
            with st.spinner("Extracting with Gemini 2.5 Flash …"):
                try:
                    dl_gemini_2_5 = gemini_2_5_dl_from_images(b64_chunks)
                except Exception as e:
                    st.error(f"Gemini API error (2.5 Flash): {e}")
                    dl_gemini_2_5 = {k: "" for k in DL_FIELDS}

            # Textract
            with st.spinner("Extracting with AWS Textract …"):
                try:
                    dl_textract = textract_dl_from_images(tmp_path)
                except Exception as e:
                    st.error(f"AWS Textract error: {e}")
                    dl_textract = {k: "" for k in DL_FIELDS}

            # Show results
            st.success("Extraction complete!")
            col_img, col_models = st.columns([1, 2], gap="large")
            with col_img:
                st.subheader("🖼️ Converted Image(s)")
                for idx, img in enumerate(images, start=1):
                    st.image(img, caption=f"Page {idx}", use_container_width=True)
            with col_models:
                tabs = st.tabs([
                    "🤖 GPT-4.1-mini Fields",
                    "🤖 GPT-4.1 Fields",
                    "🤖 Gemini 2.0 Flash Fields",
                    "🤖 Gemini 2.5 Flash Fields",
                    "🧾 Textract Fields"
                ])
                for tab, title, data in zip(
                    tabs,
                    [
                        "GPT-4.1-mini Fields",
                        "GPT-4.1 Fields",
                        "Gemini 2.0 Flash Fields",
                        "Gemini 2.5 Flash Fields",
                        "Textract Fields"
                    ],
                    [
                        dl_openai_mini,
                        dl_gpt4_1,
                        dl_gemini,
                        dl_gemini_2_5,
                        dl_textract
                    ],
                ):
                    with tab:
                        render_fields_grid(tab, title, data, DL_FIELDS)

        else:
            # Insurance Card extractions

            # GPT-4.1-mini
            with st.spinner("Extracting with GPT-4.1-mini …"):
                try:
                    ic_openai_mini = gpt4_1_mini_insurance_from_images(b64_chunks)
                except Exception as e:
                    st.error(f"OpenAI API error (mini): {e}")
                    ic_openai_mini = {k: "" for k in IC_FIELDS}

            # GPT-4.1
            with st.spinner("Extracting with GPT-4.1 …"):
                try:
                    ic_gpt4_1 = gpt4_1_insurance_from_images(b64_chunks)
                except Exception as e:
                    st.error(f"OpenAI API error (full): {e}")
                    ic_gpt4_1 = {k: "" for k in IC_FIELDS}

            # Gemini 2.0
            with st.spinner("Extracting with Gemini 2.0 Flash …"):
                try:
                    ic_gemini = gemini_insurance_from_images(b64_chunks)
                except Exception as e:
                    st.error(f"Gemini API error (2.0 Flash): {e}")
                    ic_gemini = {k: "" for k in IC_FIELDS}

            # Gemini 2.5
            with st.spinner("Extracting with Gemini 2.5 Flash …"):
                try:
                    ic_gemini_2_5 = gemini_2_5_insurance_from_images(b64_chunks)
                except Exception as e:
                    st.error(f"Gemini API error (2.5 Flash): {e}")
                    ic_gemini_2_5 = {k: "" for k in IC_FIELDS}

            # Textract
            with st.spinner("Extracting with AWS Textract …"):
                try:
                    ic_textract = textract_ic_from_images(tmp_path)
                except Exception as e:
                    st.error(f"AWS Textract error: {e}")
                    ic_textract = {k: "" for k in IC_FIELDS}

            # Show results
            st.success("Extraction complete!")
            col_img, col_models = st.columns([1, 2], gap="large")
            with col_img:
                st.subheader("🖼️ Converted Image(s)")
                for idx, img in enumerate(images, start=1):
                    st.image(img, caption=f"Page {idx}", use_container_width=True)
            with col_models:
                tabs = st.tabs([
                    "🤖 GPT-4.1-mini Fields",
                    "🤖 GPT-4.1 Fields",
                    "🤖 Gemini 2.0 Flash Fields",
                    "🤖 Gemini 2.5 Flash Fields",
                    "🧾 Textract Fields"
                ])
                for tab, title, data in zip(
                    tabs,
                    [
                        "GPT-4.1-mini Fields",
                        "GPT-4.1 Fields",
                        "Gemini 2.0 Flash Fields",
                        "Gemini 2.5 Flash Fields",
                        "Textract Fields"
                    ],
                    [
                        ic_openai_mini,
                        ic_gpt4_1,
                        ic_gemini,
                        ic_gemini_2_5,
                        ic_textract
                    ],
                ):
                    with tab:
                        render_fields_grid(tab, title, data, IC_FIELDS)

elif uploaded_file:
    st.info("Please provide OpenAI, Gemini, and AWS credentials to proceed.")
else:
    st.write("👈 Upload a file and provide API keys to get started.")
