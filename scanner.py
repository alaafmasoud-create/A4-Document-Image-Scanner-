import cv2
import numpy as np
import os

def reorder_points(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)

    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]

    return new_points

def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest

def scan_document(image_path, output_path="scanned_output.jpg", width=600, height=800):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    img = cv2.resize(img, (width, height))
    img_copy = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blur, 150, 200)

    kernel = np.ones((5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    threshold = cv2.erode(dilated, kernel, iterations=1)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = biggest_contour(contours)

    if biggest.size != 0:
        biggest = reorder_points(biggest)

        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(img_copy, matrix, (width, height))

        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        scanned = cv2.adaptiveThreshold(
            warped_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        cv2.imwrite(output_path, scanned)
        print(f"Scanned image saved as: {output_path}")
    else:
        print("No document-like contour detected.")

if __name__ == "__main__":
    input_image = "images/sample1.jpg"
    output_image = "images/output_sample1.jpg"

    if not os.path.exists("images"):
        os.makedirs("images")

    scan_document(input_image, output_image)
