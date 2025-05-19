import os
import argparse
from PIL import Image, ImageFile
import numpy as np
from retinaface import RetinaFace

ImageFile.LOAD_TRUNCATED_IMAGES = True

def align_and_extract_faces(image_path, output_dir, counter):
    try:
        image = Image.open(image_path)
        basename = os.path.splitext(os.path.basename(image_path))[0]
        faces = RetinaFace.extract_faces(img_path=image_path, align=True)
        for i, face in enumerate(faces):
            face_image = Image.fromarray(face)
            face_image.save(os.path.join(output_dir, f"{basename}_{counter}.jpg"))
            counter += 1
        return counter
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return counter

def process_images(input_dir, output_dir):
    counter = 0
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.ppm', '.pgm', '.pbm', '.pnm')):
                image_path = os.path.join(root, file)
                counter = align_and_extract_faces(image_path, output_dir, counter)

def main():
    parser = argparse.ArgumentParser(description="Process images to extract and align faces.")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save output images")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    process_images(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()