import streamlit as st
import cv2
import numpy as np
import pytesseract
from scanner import scan_document_from_array

st.set_page_config(page_title="Smart Document Scanner + OCR", layout="wide")

st.title("📄 Smart Document Scanner + OCR")
st.write("Upload a document photo, straighten it, enhance it, and extract text.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
ocr_language = st.text_input("OCR language(s)", value="eng", help="Examples: eng , spa , ara , eng+spa")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    result = scan_document_from_array(img)

    message = result.get("message")
    if message:
        if result.get("mode") == "detected":
            st.success(message)
        else:
            st.warning(message)

    original_img = result.get("original", img)
    warped_img = result.get("warped", img)
    enhanced_img = result.get("enhanced")
    scanned_img = result.get("scanned")
    debug_edges = result.get("debug_edges")
    debug_contour = result.get("debug_contour")

    st.subheader("Processing overview")

    # الصف الأول
    col1, col2 = st.columns(2)

    with col1:
        st.image(
            cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        if debug_edges is not None:
            st.image(
                debug_edges,
                caption="Detected Edges",
                use_container_width=True
            )
        else:
            st.info("No edge-debug image available in the current scanner.py version.")

    # الصف الثاني
    col3, col4 = st.columns(2)

    with col3:
        if debug_contour is not None:
            st.image(
                cv2.cvtColor(debug_contour, cv2.COLOR_BGR2RGB),
                caption="Chosen Contour / Fallback Box",
                use_container_width=True
            )
        else:
            st.info("No contour-debug image available in the current scanner.py version.")

    with col4:
        if warped_img is not None:
            if len(warped_img.shape) == 3:
                st.image(
                    cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB),
                    caption="Straightened / Cropped Document",
                    use_container_width=True
                )
            else:
                st.image(
                    warped_img,
                    caption="Straightened / Cropped Document",
                    use_container_width=True
                )

    st.subheader("Final result")
    col5, col6 = st.columns(2)

    with col5:
        if enhanced_img is not None:
            st.image(
                enhanced_img,
                caption="Enhanced Grayscale",
                use_container_width=True
            )
        else:
            st.info("No enhanced image available.")

    with col6:
        if scanned_img is not None:
            st.image(
                scanned_img,
                caption="Scanned Output",
                use_container_width=True
            )
        else:
            st.error("No scanned output was returned by scanner.py")

    # OCR
    if scanned_img is not None:
        st.subheader("Extracted text (OCR)")
        try:
            custom_config = r"--oem 3 --psm 6"
            extracted_text = pytesseract.image_to_string(
                scanned_img,
                lang=ocr_language,
                config=custom_config
            )

            st.text_area("Recognized text", extracted_text, height=300)

            st.download_button(
                label="Download extracted text",
                data=extracted_text,
                file_name="recognized.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"OCR failed: {e}")

        success, buffer = cv2.imencode(".png", scanned_img)
        if success:
            st.download_button(
                label="Download scanned image",
                data=buffer.tobytes(),
                file_name="scanned_document.png",
                mime="image/png"
            )
