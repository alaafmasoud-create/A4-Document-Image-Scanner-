import streamlit as st
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

# =========================================
# Page config
# =========================================
st.set_page_config(page_title="Smart Document Scanner", page_icon="📄", layout="centered")

# =========================================
# UI styling
# =========================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 900px;
}
h1, h2, h3 {
    color: #1f2937;
}
.stRadio > div {
    flex-direction: row;
    gap: 1rem;
}
.stDownloadButton button {
    width: 100%;
    border-radius: 10px;
    font-weight: 600;
}
.stFileUploader label div[data-testid="stMarkdownContainer"] p {
    font-weight: 600;
}
.small-footer {
    text-align: center;
    font-size: 12px;
    color: #6b7280;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# Constants
# =========================================
MAX_PREVIEW_WIDTH = 1100
JPEG_QUALITY = 95

# =========================================
# Session state
# =========================================
if "manual_points" not in st.session_state:
    st.session_state.manual_points = []

if "last_click_time" not in st.session_state:
    st.session_state.last_click_time = None

if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None


# =========================================
# Geometry helpers
# =========================================
def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    max_width = max(max_width, 1)
    max_height = max(max_height, 1)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


# =========================================
# Cached helpers
# =========================================
@st.cache_data(show_spinner=False)
def decode_image(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return image


@st.cache_data(show_spinner=False)
def make_preview(image, max_width=MAX_PREVIEW_WIDTH):
    h, w = image.shape[:2]
    if w <= max_width:
        return image.copy(), 1.0

    scale = max_width / w
    preview = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return preview, scale


@st.cache_data(show_spinner=False)
def auto_detect_document(preview_image):
    debug = {}

    gray = cv2.cvtColor(preview_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    screen_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx.reshape(4, 2)
            break

    debug["gray"] = gray
    debug["edges"] = edges
    preview_marked = preview_image.copy()

    if screen_cnt is not None:
        cv2.polylines(preview_marked, [screen_cnt.astype(np.int32)], True, (0, 255, 0), 3)

    debug["preview_marked"] = preview_marked
    return screen_cnt, debug


def to_original_points(points_preview, scale):
    pts = np.array(points_preview, dtype=np.float32)
    return pts / scale


def draw_points(image, points):
    img = image.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (int(x), int(y)), 8, (0, 0, 255), -1)
        cv2.putText(
            img,
            str(i + 1),
            (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )
    if len(points) == 4:
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 0), 3)
    return img


def encode_jpg(image, quality=JPEG_QUALITY):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    success, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return None
    return buffer.tobytes()


def reset_manual_points():
    st.session_state.manual_points = []
    st.session_state.last_click_time = None


# =========================================
# Header
# =========================================
st.title("Smart Document Scanner")
st.write("Upload an image of an A4 document. Use the default automatic mode, or manually adjust the corners for a more accurate crop.")

# =========================================
# Upload
# =========================================
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])

mode = st.radio("Scanner Settings", ["Automatic", "Manual"], horizontal=True)
show_debug = st.checkbox("Show debug images", value=False)

if uploaded_file is not None:
    # Reset points when a different file is uploaded
    if st.session_state.last_file_name != uploaded_file.name:
        reset_manual_points()
        st.session_state.last_file_name = uploaded_file.name

    file_bytes = uploaded_file.read()
    original = decode_image(file_bytes)

    if original is None:
        st.error("Could not read the uploaded image.")
        st.stop()

    preview, scale = make_preview(original)
    result_image = None

    if mode == "Automatic":
        with st.spinner("Detecting document..."):
            detected_pts, debug = auto_detect_document(preview)

        st.subheader("Preview")
        st.image(cv2.cvtColor(debug["preview_marked"], cv2.COLOR_BGR2RGB), use_container_width=True)

        if detected_pts is not None:
            pts_original = to_original_points(detected_pts, scale)
            result_image = four_point_transform(original, pts_original)
            st.success("Document detected successfully.")
        else:
            st.warning("Automatic detection could not find 4 document corners. Please switch to Manual mode.")

        if show_debug:
            st.subheader("Debug Images")
            st.image(debug["gray"], caption="Gray", use_container_width=True)
            st.image(debug["edges"], caption="Edges", use_container_width=True)

    else:
        st.subheader("Click on the 4 corners")

        display_image = draw_points(preview, st.session_state.manual_points)
        clicked = streamlit_image_coordinates(
            cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB),
            key="manual_clicks"
        )

        if clicked is not None:
            click_time = clicked.get("time")
            x = clicked.get("x")
            y = clicked.get("y")

            if (
                x is not None and y is not None and
                click_time is not None and
                click_time != st.session_state.last_click_time
            ):
                if len(st.session_state.manual_points) < 4:
                    st.session_state.manual_points.append((x, y))
                    st.session_state.last_click_time = click_time
                    st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset points", use_container_width=True):
                reset_manual_points()
                st.rerun()

        with col2:
            if st.button("Remove last point", use_container_width=True):
                if st.session_state.manual_points:
                    st.session_state.manual_points.pop()
                st.rerun()

        st.caption(f"Selected points: {len(st.session_state.manual_points)}/4")

        if len(st.session_state.manual_points) == 4:
            pts_original = to_original_points(st.session_state.manual_points, scale)
            result_image = four_point_transform(original, pts_original)
            st.success("Manual selection completed.")

    # Final result
    if result_image is not None:
        st.subheader("Final Result")
        st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_container_width=True)

        jpg_bytes = encode_jpg(result_image, quality=JPEG_QUALITY)
        if jpg_bytes is not None:
            st.download_button(
                label="Download Final Result",
                data=jpg_bytes,
                file_name="scanned_document.jpg",
                mime="image/jpeg",
                use_container_width=True
            )

st.markdown('<div class="small-footer">By Alan Masoud</div>', unsafe_allow_html=True)
