import os
import io
import cv2  # OpenCV for video processing
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from deepface import DeepFace
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from typing import List
import json
import tempfile
import imutils
import uvicorn


app = FastAPI()

class FaceRecognitionSystem:
    def __init__(self):
        self.reference_folder = "reference_images"
        self.metadata_file = "reference_metadata.json"
        self.known_faces = {}

        if not os.path.exists(self.reference_folder):
            os.makedirs(self.reference_folder)
        
        self.load_reference_images()

    def load_reference_images(self):
        """Load and encode all reference images."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        for person_name in os.listdir(self.reference_folder):
            person_path = os.path.join(self.reference_folder, person_name)
            if os.path.isdir(person_path):
                self.known_faces[person_name] = []

                for img_file in os.listdir(person_path):
                    if img_file.endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(person_path, img_file)
                        encoding = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)
                        self.known_faces[person_name].append(encoding)

    async def process_image(self, image_data: bytes, person_folder: str):
        """Process and save a reference image."""
        try:
            image = Image.open(io.BytesIO(image_data))
            image = image.convert('RGB')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}.jpg"
            path = os.path.join(person_folder, filename)
            image.save(path)

            encoding = DeepFace.represent(path, model_name="Facenet")
            return {"status": "success", "encoding": encoding}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def add_reference_images(self, files: List[UploadFile], name: str):
        person_folder = os.path.join(self.reference_folder, name)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)

        results = []
        for file in files:
            contents = await file.read()
            result = await self.process_image(contents, person_folder)
            if result["status"] == "success":
                self.known_faces.setdefault(name, []).append(result["encoding"])
            results.append({"filename": file.filename, "status": result["status"]})

        return {
            "message": f"Processed {len(files)} images for {name}",
            "results": results
        }

    def identify_face(self, image_data):
        """Identify faces using the stored encodings."""
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')
        image_np = np.array(image)
        
        # Get the embedding of the input image
        detected_face = DeepFace.represent(img_path=image_np, model_name="Facenet", enforce_detection=False)
        if not detected_face:
            return [{"name": "Unknown", "confidence": 0}]
        
        input_encoding = detected_face[0]["embedding"]
        results = []
        
        # Compare input encoding with known face encodings
        for name, encodings in self.known_faces.items():
            for ref_encoding in encodings:
                # Calculate cosine similarity
                similarity = cosine_similarity([input_encoding], [ref_encoding])[0][0]
                if similarity > 0.3:  # Adjust this threshold as needed for your use case
                    results.append({"name": name, "confidence": similarity})
                    break
        
        return results or [{"name": "Unknown", "confidence": 0}]
    
    def identify_face_in_video(self, video_data: bytes):
        """Identify faces in the video."""
        
        # Create a temporary file in memory
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(video_data)
            tmp_video_path = tmp_file.name

        # Open the video with OpenCV
        cap = cv2.VideoCapture(tmp_video_path)
        recognized_person = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to make it easier to process (optional)
            frame = imutils.resize(frame, width=400)

            # Directly use the video frame (a NumPy array) for face recognition
            try:
                detected_face = DeepFace.represent(frame, model_name="Facenet", enforce_detection=False)
                if detected_face:
                    input_encoding = detected_face[0]["embedding"]
                    results = []
                    # Compare input encoding with known face encodings
                    for name, encodings in self.known_faces.items():
                        for ref_encoding in encodings:
                            ref_encoding = ref_encoding[0]["embedding"]
                            similarity = cosine_similarity([input_encoding], [ref_encoding])[0][0]
                            if similarity > 0.3:  # Adjust threshold as needed
                                recognized_person = name
                                break
                        if recognized_person:
                                    # Release OpenCV VideoCapture
                            cap.release()

                            # Delete the temporary file after processing
                            os.remove(tmp_video_path)

                            return recognized_person
                            break
            except Exception as e:
                print(e)
                # In case no face is detected, just continue to the next frame
                pass
        
        # Release OpenCV VideoCapture
        cap.release()

        # Delete the temporary file after processing
        os.remove(tmp_video_path)

        return recognized_person

    def get_statistics(self):
        """Return stored image statistics."""
        return {"total_people": len(self.known_faces), "people": {name: len(encodings) for name, encodings in self.known_faces.items()}}

# Initialize the face recognition system
face_system = FaceRecognitionSystem()

@app.post("/add_reference")
async def add_reference(files: List[UploadFile] = File(...), name: str = Form(...)):
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    return await face_system.add_reference_images(files, name)

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    contents = await file.read()
    results = face_system.identify_face(contents)
    return JSONResponse(content={"results": results})

@app.post("/identify_video")
async def identify_video(file: UploadFile = File(...)):
    contents = await file.read()
    # Identify person in the video directly from the uploaded data
    person_name = face_system.identify_face_in_video(contents)
    if person_name is None:
        person_name = "Unknown"

    return JSONResponse(content={"data": {"person_name": person_name}})

@app.get("/statistics")
async def get_statistics():
    return face_system.get_statistics()

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8100)