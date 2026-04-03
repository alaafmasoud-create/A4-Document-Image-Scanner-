import streamlit as st
import cv2
import numpy as np


def process_document(file_bytes):
    # Convert uploaded file to OpenCV image
    file_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode uploaded image.")

    hh, ww = img.shape[:2]

    # Edge detection
    canny = cv2.Canny(img, 50, 200)

    # Find contours
    contours = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Filter small contours
    cimg = np.zeros_like(canny)
    for cntr in contours:
        area = cv2.contourArea(cntr)
        if area > 20:
            cv2.drawContours(cimg, [cntr], 0, 255, 1)

    # Collect points for convex hull
    points = np.column_stack(np.where(cimg.transpose() > 0))
    if len(points) == 0:
        raise ValueError("No valid document detected in the image.")

    hull = cv2.convexHull(points)

    # Create filled mask from hull
    mask = np.zeros_like(cimg, dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 255)

    # Mask the document only
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # Get rotated rectangle
    rotrect = cv2.minAreaRect(hull)
    (center), (width, height), angle = rotrect

    # Correct angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        if width > height:
            angle = -(90 + angle)
        else:
            angle = -angle

    neg_angle = -angle

    # Rotate image and mask with the same matrix
    M = cv2.getRotationMatrix2D(center, neg_angle, scale=1.0)

    rotated_img = cv2.warpAffine(
        masked_img,
        M,
        (ww, hh),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    rotated_mask = cv2.warpAffine(
        mask,
        M,
        (ww, hh),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Crop exactly around the rotated document
    coords = cv2.findNonZero(rotated_mask)
    if coords is None:
        raise ValueError("Could not crop the document after rotation.")

    x, y, w, h = cv2.boundingRect(coords)
    cropped = rotated_img[y:y+h, x:x+w]

    # Optional extra trim to remove tiny black borders
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    coords2 = cv2.findNonZero(thresh)

    if coords2 is not None:
        x2, y2, w2, h2 = cv2.boundingRect(coords2)
        cropped = cropped[y2:y2+h2, x2:x2+w2]

    return cropped


st.set_page_config(page_title="Document Straightener", layout="centered")

st.title("Document Straightener")
st.write("Upload a document image and get only the straightened document.")

uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.read()
        result = process_document(file_bytes)

        st.subheader("Final Result")
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
