import os
import argparse
import sys
from pdf2image import convert_from_path
import cv2
import numpy as np
from PIL import Image


def remove_watermark_and_convert(input_pdf, output_pdf, dpi=300):
    print(f"Processing {input_pdf} -> {output_pdf}")

    # 1. Convert PDF to List of Images
    try:
        images = convert_from_path(input_pdf, dpi=dpi)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return False

    cleaned_images = []

    for i, img in enumerate(images):
        # Convert PIL to OpenCV (BGR)
        open_cv_image = np.array(img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        # 2. Convert to Grayscale
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

        # 3. Apply Binary Thresholding (Otsu's binarization)
        # This forces pixels to be black or white. Light watermarks usually turn white.
        # We assume text is dark and watermark is lighter.
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Optional: Dilation/Erosion if text is too thin (Skipped for now, Otsu is usually good)

        # Convert back to PIL
        cleaned_pil = Image.fromarray(thresh)
        cleaned_images.append(cleaned_pil)

    # 4. Save back to PDF
    if cleaned_images:
        cleaned_images[0].save(
            output_pdf,
            "PDF",
            resolution=100.0,
            save_all=True,
            append_images=cleaned_images[1:],
        )
        print("Watermark removal complete.")
        return True
    else:
        print("No images processed.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean PDF Watermarks for OCR")
    parser.add_argument("input_pdf", help="Path to input PDF")
    parser.add_argument("output_pdf", help="Path to output cleaned PDF")

    args = parser.parse_args()

    # Install dependencies check
    # We need 'pdf2image' (wraps poppler) and 'opencv-python'
    # pip install pdf2image opencv-python-headless

    remove_watermark_and_convert(args.input_pdf, args.output_pdf)
