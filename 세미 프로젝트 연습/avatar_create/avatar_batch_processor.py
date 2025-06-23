import cv2
import numpy as np
from PIL import Image
from avatar_generator import generate_avatar_image, download_image_from_url

def extract_faces_with_positions(image_pil):
    image_np = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    face_regions = []
    for (x, y, w, h) in faces:
        cropped = image_np[y:y+h, x:x+w]
        face_pil = Image.fromarray(cropped)
        face_regions.append(((x, y, w, h), face_pil))
    return face_regions

def generate_enhanced_faces(face_regions, prompt):
    enhanced_faces = []
    for (box, face_pil) in face_regions:
        urls = generate_avatar_image(prompt=prompt, n_images=1)
        if urls:
            enhanced_img = download_image_from_url(urls[0])
            enhanced_faces.append((box, enhanced_img))
    return enhanced_faces

def replace_faces_on_image(original_pil, enhanced_faces):
    base_np = np.array(original_pil.convert("RGB"))
    for (x, y, w, h), new_face in enhanced_faces:
        resized = new_face.resize((w, h))
        base_np[y:y+h, x:x+w] = np.array(resized)
    return Image.fromarray(base_np)
