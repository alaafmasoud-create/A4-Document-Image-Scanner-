import streamlit as st
import cv2
import numpy as np
import pytesseract
from scanner import scan_document_from_array

st.set_page_config(page_title="Smart Document Scanner + OCR", layout="wide")

st.title("📄 Smart Document Scanner + OCR")
st.write("Upload a document photo, straighten it, enhance it, and extract text.")

# إذا كنت تشغّل محليًا على ويندوز فقط، أزل التعليق عن السطر التالي وعدّل المسار إذا لزم:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

ocr_language = st.text_input("OCR language(s)", value="eng", help="Examples: eng , spa , ara , eng+spa")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    result = scan_document_from_array(img)

    if result["message"]:
        if result["mode"] == "detected":
            st.success(result["message"])
        else:
            st.warning(result["message"])

    st.subheader("Processing overview")
    col1, col2 = st.columns(2)

    with col1:
        st.image(
            cv2.cvtColor(result["original"], cv2.COLOR_BGR2RGB),
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            result["debug_edges"],
            caption="Detected Edges",
            use_container_width=True
        )

    col3, col4 = st.columns(2)

    with col3:
        st.image(
            cv2.cvtColor(result["debug_contour"], cv2.COLOR_BGR2RGB),
            caption="Chosen Contour / Fallback Box",
            use_container_width=True
        )

    with col4:
        st.image(
            cv2.cvtColor(result["warped"], cv2.COLOR_BGR2RGB),
            caption="Straightened / Cropped Document",
            use_container_width=True
        )

    st.subheader("Final result")
    col5, col6 = st.columns(2)

    with col5:
        st.image(
            result["enhanced"],
            caption="Enhanced Grayscale",
            use_container_width=True
        )

    with col6:
        st.image(
            result["scanned"],
            caption="Scanned Output",
            use_container_width=True
        )

    # OCR على الصورة النهائية
    st.subheader("Extracted text (OCR)")
    try:
        custom_config = r"--oem 3 --psm 6"
        extracted_text = pytesseract.image_to_string(
            result["scanned"],
            lang=ocr_language,
            config=custom_config
        )

        st.text_area("Recognized text", extracted_text, height=300)

        # تنزيل النص
        st.download_button(
            label="Download extracted text",
            data=extracted_text,
            file_name="recognized.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"OCR failed: {e}")

    # تنزيل الصورة النهائية
    success, buffer = cv2.imencode(".png", result["scanned"])
    if success:
        st.download_button(
            label="Download scanned image",
            data=buffer.tobytes(),
            file_name="scanned_document.png",
            mime="image/png"
        )
