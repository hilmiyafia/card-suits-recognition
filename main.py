
import os
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_PLAIN
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
RESOLUTION = 100
SAMPLES = np.arange(RESOLUTION) / RESOLUTION

def extract_contours(image):
    gray = image[:, :, 0]
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def extract_feature(contour, area):
    ps = contour[:, 0, 0] + contour[:, 0, 1] * 1j
    # Normalize position
    ps -= np.mean(ps)
    # Normalize area
    ps *= (100 / area) ** 0.5
    # Normalize perimeter
    ps = np.interp(SAMPLES, np.arange(len(ps)) / len(ps), ps)
    # Normaluze rotation
    ps = np.abs(np.fft.fft(ps))
    return ps

def score(a, b):
    return np.absolute(a - b).sum()
    
templates = []
for file in os.listdir("templates"):
    name = file.split(".")[0]
    image = cv2.imread(f"templates/{file}")
    contour = extract_contours(image)[0]
    area = cv2.contourArea(contour)
    templates.append((name, extract_feature(contour, area)))

video = cv2.VideoCapture("video.mp4")
while True:
    retval, image = video.read()
    if not retval: break
    for contour in extract_contours(image):
        area = cv2.contourArea(contour)
        if area < 300: continue
        feature = extract_feature(contour, area)
        scores = [score(feature, template) for _, template in templates]
        index = np.argmin(scores)
        if scores[index] > 350: continue
        x, y, w, h = cv2.boundingRect(contour)
        name = templates[index][0]
        point_a = (x - 6, y - 6)
        point_b = (x + w + 6, y + h + 6)
        image = cv2.rectangle(image, point_a, point_b, BLUE, 3)
        image = cv2.putText(image, name, point_a, FONT, 1.5, CYAN, 2, 1)
    cv2.imshow("Video", image)
    cv2.waitKey(20)
cv2.waitKey()
video.release()
