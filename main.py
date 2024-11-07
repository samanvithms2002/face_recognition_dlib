import os
import io
import cv2
import numpy as np
import json
import tempfile
import imutils
import uvicorn
import logging
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from deepface import DeepFace
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from dataset_creation import VideoFaceAugmenter
from multiprocessing import Pool, cpu_count

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

class FaceRecognitionSystem:
    def __init__(self):
        self.reference_folder = "reference_images"
        self.metadata_file = "reference_metadata.json"
        self.known_faces = {}

        if not os.path.exists(self.reference_folder):
            os.makedirs(self.reference_folder)
            logging.info(f"Created reference folder at {self.reference_folder}")

        self.load_reference_images()

    def load_reference_images(self):
        """Load and encode all reference images including augmented ones using multiprocessing."""
        logging.info("Loading reference images...")

        # Load metadata if available
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}
            logging.warning(
                f"Metadata file {self.metadata_file} not found. Proceeding without metadata."
            )

        # Prepare all images for parallel encoding
        image_paths = []
        for person_name in os.listdir(self.reference_folder):
            person_path = os.path.join(self.reference_folder, person_name)
            if os.path.isdir(person_path):
                for img_file in os.listdir(person_path):
                    if img_file.endswith((".jpg", ".jpeg", ".png")):
                        image_paths.append((person_name, os.path.join(person_path, img_file)))

                # Include augmented images
                augmented_path = os.path.join(person_path, "augmented")
                if os.path.exists(augmented_path):
                    for aug_file in os.listdir(augmented_path):
                        if aug_file.endswith((".jpg", ".jpeg", ".png")):
                            image_paths.append((person_name, os.path.join(augmented_path, aug_file)))

        # Encode images in parallel
        with Pool(cpu_count()) as pool:
            results = pool.map(self.encode_image, image_paths)

        # Organize encoded images by person
        for result in results:
            if result:
                name, encoding = result
                self.known_faces.setdefault(name, []).append(encoding)

        logging.info("Loaded all reference images.")

    def encode_image(self, args):
        """Helper function to encode a single image for multiprocessing."""
        person_name, path = args
        try:
            encoding = DeepFace.represent(img_path=path, model_name="Facenet", enforce_detection=False)
            if encoding:
                return person_name, encoding[0]["embedding"]
        except Exception as e:
            logging.error(f"Error encoding image {path} for {person_name}: {e}")
        return None

    async def process_image(self, image_data: bytes, person_folder: str):
        """Process and save a reference image with augmentation."""
        try:
            image = Image.open(io.BytesIO(image_data))
            image = image.convert("RGB")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}.jpg"
            path = os.path.join(person_folder, filename)
            image.save(path)
            logging.info(f"Saved reference image {filename} for {person_folder}")

            # Get encoding for original image
            encoding = DeepFace.represent(path, model_name="Facenet")
            logging.info(f"Generated encoding for {filename}")

            # Generate augmented images using augmentation class
            augmented_folder = os.path.join(person_folder, "augmented")
            os.makedirs(augmented_folder, exist_ok=True)
            augmenter = VideoFaceAugmenter()
            augmenter.augment_single_image(
                cv2.imread(path),
                augmenter.create_augmentation_pipelines(),
                filename,
                augmented_folder,
            )
            logging.info(f"Generated augmented images for {filename}")

            return {"status": "success", "encoding": encoding}
        except Exception as e:
            logging.error(f"Error processing image for {person_folder}: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def add_reference_images(self, files: List[UploadFile], name: str):
        person_folder = os.path.join(self.reference_folder, name)
        os.makedirs(person_folder, exist_ok=True)
        logging.info(f"Adding reference images for {name}")

        results = []
        for file in files:
            contents = await file.read()
            result = await self.process_image(contents, person_folder)
            if result["status"] == "success":
                self.known_faces.setdefault(name, []).append(result["encoding"])
            results.append({"filename": file.filename, "status": result["status"]})

        # Reload all faces to include augmented images
        self.load_reference_images()
        return {
            "message": f"Processed {len(files)} images for {name}",
            "results": results,
        }

    def batch_cosine_similarity(self, input_encoding, reference_encodings, threshold=0.3):
        """Calculate similarity for a batch of encodings."""
        similarities = cosine_similarity([input_encoding], reference_encodings)
        matches = np.where(similarities > threshold)[1]
        return matches

    def identify_face(self, image_data):
        """Identify faces using stored encodings with batch processing."""
        logging.info("Identifying face in image...")

        image = Image.open(io.BytesIO(image_data))
        image = image.convert("RGB")
        image_np = np.array(image)

        detected_face = DeepFace.represent(
            img_path=image_np, model_name="Facenet", enforce_detection=False
        )
        if not detected_face:
            logging.warning("No face detected in image.")
            return [{"name": "Unknown", "confidence": 0}]

        input_encoding = detected_face[0]["embedding"]
        results = []

        for name, encodings in self.known_faces.items():
            ref_encodings = np.array(encodings)
            matches = self.batch_cosine_similarity(input_encoding, ref_encodings)
            if len(matches) > 0:
                similarity = cosine_similarity([input_encoding], [ref_encodings[matches[0]]])[0][0]
                logging.info(f"Match found for {name} with similarity {similarity}")
                results.append({"name": name, "confidence": similarity})
                break

        return results or [{"name": "Unknown", "confidence": 0}]

    def identify_face_in_video(self, video_data: bytes):
        """Identify faces in a video using batch processing."""
        logging.info("Processing video for face identification...")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_file.write(video_data)
            tmp_video_path = tmp_file.name
        logging.info(f"Saved temporary video file at {tmp_video_path}")

        cap = cv2.VideoCapture(tmp_video_path)
        frame_list = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = imutils.resize(frame, width=400)
            frame_list.append(frame)

        cap.release()
        os.remove(tmp_video_path)

        # Parallelize frame processing
        with Pool(cpu_count()) as pool:
            results = pool.map(self.process_frame_for_video, frame_list)

        recognized_persons = [result for result in results if result]
        if recognized_persons:
            recognized_person = max(set(recognized_persons), key=recognized_persons.count)
            return recognized_person

        return "Unknown"

    def process_frame_for_video(self, frame):
        """Process a single frame for face recognition."""
        try:
            detected_face = DeepFace.represent(
                frame, model_name="Facenet", enforce_detection=False
            )
            if detected_face:
                input_encoding = detected_face[0]["embedding"]
                for name, encodings in self.known_faces.items():
                    ref_encodings = np.array(encodings)
                    matches = self.batch_cosine_similarity(input_encoding, ref_encodings)
                    if len(matches) > 0:
                        return name
        except Exception as e:
            logging.error(f"Error processing frame for video: {str(e)}")
        return None

    def get_statistics(self):
        """Return stored image statistics including augmented images."""
        logging.info("Generating statistics for reference images...")
        stats = {"total_people": len(self.known_faces), "people": {}}
        for name, encodings in self.known_faces.items():
            original_count = len(encodings)
            stats["people"][name] = {
                "original_images": original_count,
                "augmented_images": len(self.known_faces[name]),
                "total_images": original_count
            }
        return stats

# Initialize face recognition system
face_system = FaceRecognitionSystem()

# API endpoints remain the same as in the previous code


@app.post("/add_reference")
async def add_reference(files: List[UploadFile] = File(...), name: str = Form(...)):
    if not name:
        logging.error("Name not provided for reference images.")
        raise HTTPException(status_code=400, detail="Name is required")
    if not files:
        logging.error("No files provided for reference images.")
        raise HTTPException(status_code=400, detail="No files provided")

    logging.info(f"Adding reference images for {name}")
    return await face_system.add_reference_images(files, name)


@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    contents = await file.read()
    results = face_system.identify_face(contents)
    logging.info(f"Face identification results: {results}")
    return JSONResponse(results)


@app.post("/identify_in_video")
async def identify_in_video(file: UploadFile = File(...)):
    contents = await file.read()
    recognized_person = face_system.identify_face_in_video(contents)
    logging.info(f"Person identified in video: {recognized_person}")
    return {"name": recognized_person or "Unknown"}


@app.get("/statistics")
async def statistics():
    stats = face_system.get_statistics()
    logging.info("Retrieved reference image statistics")
    return stats


# if __name__ == "__main__":
# uvicorn.run(app, host="0.0.0.0", port=8100)
