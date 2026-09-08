
import json

import pytesseract
import streamlit as st
import torch
from PIL import Image
from transformers import AutoTokenizer

st.set_page_config(page_title="Indian Receipt AI", page_icon="🇮🇳", layout="wide")


@st.cache_resource
def load_model(model_path: str, is_adapter: bool):
    if is_adapter:
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.bfloat16
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.bfloat16
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def extract_json(model, tokenizer, receipt_text: str) -> dict | None:
    instruction = (
        "Extract the following fields from this Indian receipt as JSON: "
        "vendor, location, items (name, qty, unit_price, price), subtotal, gst, "
        "cgst, sgst, total, payment_method, date (YYYY-MM-DD), currency.\n\n"
        f"Receipt:\n{receipt_text}"
    )
    messages = [
        {"role": "system", "content": "You extract structured data from Indian receipts and respond with JSON only."},
        {"role": "user", "content": instruction},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=400, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    text = gen.strip()
    if "```" in text:
        text = text.split("```")[1].replace("json", "", 1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def main():
    st.title("🇮🇳 Indian Financial Document AI")
    st.caption("Upload a receipt / invoice / bill, or paste its text, to extract structured data.")

    with st.sidebar:
        st.header("Model")
        model_path = st.text_input("Model path or HF id", value="models/qlora-finetuned")
        is_adapter = st.checkbox("Is a LoRA adapter dir", value=True)
        load_btn = st.button("Load model")

    if load_btn or "model" in st.session_state:
        if load_btn:
            with st.spinner("Loading model..."):
                st.session_state["model"], st.session_state["tokenizer"] = load_model(model_path, is_adapter)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")
        uploaded = st.file_uploader("Upload receipt image", type=["png", "jpg", "jpeg"])
        receipt_text = ""
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded receipt", use_container_width=True)
            with st.spinner("Running OCR..."):
                receipt_text = pytesseract.image_to_string(img)
        receipt_text = st.text_area("Receipt text (edit OCR output or paste directly)", value=receipt_text, height=300)

    with col2:
        st.subheader("Extraction")
        if st.button("Extract structured data", type="primary"):
            if "model" not in st.session_state:
                st.error("Load a model first (sidebar).")
            elif not receipt_text.strip():
                st.error("Provide receipt text or upload an image.")
            else:
                with st.spinner("Extracting..."):
                    result = extract_json(st.session_state["model"], st.session_state["tokenizer"], receipt_text)
                if result is None:
                    st.error("Model did not return valid JSON. Try again or edit the input text.")
                else:
                    st.json(result)
                    st.download_button(
                        "Download JSON",
                        data=json.dumps(result, indent=2, ensure_ascii=False),
                        file_name="extracted_receipt.json",
                        mime="application/json",
                    )


if __name__ == "__main__":
    main()
