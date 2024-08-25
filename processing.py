# processing.py
import io
import zipfile
from datetime import datetime
from PIL import Image
from yolo_utils import detect_objects as yolo_detect_objects

def detect_objects(image, confidence_threshold):
    return yolo_detect_objects(image, confidence_threshold)

def save_results_as_zip(results, zip_filename):
    today = datetime.now().strftime("%Y-%m-%d")
    final_zip_filename = f"{zip_filename}_{today}.zip"
    zip_path = f"/tmp/{final_zip_filename}"

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for idx, (annotated_image, _) in enumerate(results):
            img_byte_arr = io.BytesIO()
            annotated_image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            file_name = f"Defect_{idx + 1}_{today}.png"
            zipf.writestr(file_name, img_byte_arr.getvalue())
    
    return zip_path, final_zip_filename
