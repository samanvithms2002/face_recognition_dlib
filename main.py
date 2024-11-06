import face_recognition
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import os
from PIL import Image
from io import BytesIO
import json
from typing import Dict, List
from datetime import datetime
from typing import List as TypeList

app = FastAPI()

class FaceRecognitionSystem:
    def __init__(self):
        self.reference_folder = "reference_images"
        self.metadata_file = "reference_metadata.json"
        self.known_face_encodings: Dict[str, List[np.ndarray]] = {}
        
        if not os.path.exists(self.reference_folder):
            os.makedirs(self.reference_folder)
        
        self.load_reference_images()

    def load_reference_images(self):
        """Load and encode all reference images."""
        self.known_face_encodings = {}
        
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        for person_name in os.listdir(self.reference_folder):
            person_path = os.path.join(self.reference_folder, person_name)
            if os.path.isdir(person_path):
                self.known_face_encodings[person_name] = []
                
                for img_file in os.listdir(person_path):
                    if img_file.endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(person_path, img_file)
                        
                        image = face_recognition.load_image_file(img_path)
                        face_encodings = face_recognition.face_encodings(image)
                        
                        if len(face_encodings) > 0:
                            self.known_face_encodings[person_name].append(face_encodings[0])

    async def process_image(self, image_data: bytes, person_folder: str) -> dict:
        """Process a single image and return result."""
        try:
            image = Image.open(BytesIO(image_data))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}.jpg"
            path = os.path.join(person_folder, filename)
            image.save(path)
            
            face_image = face_recognition.load_image_file(path)
            face_encodings = face_recognition.face_encodings(face_image)
            
            if len(face_encodings) == 0:
                os.remove(path)
                return {
                    "status": "error",
                    "message": "No face detected in image",
                    "filename": filename
                }
            
            return {
                "status": "success",
                "message": "Face detected and encoded successfully",
                "filename": filename,
                "encoding": face_encodings[0]
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing image: {str(e)}",
                "filename": "unknown"
            }

    async def add_reference_images(self, files: TypeList[UploadFile], name: str):
        """Add multiple reference images for a person."""
        # Create person folder if it doesn't exist
        person_folder = os.path.join(self.reference_folder, name)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)
        
        results = []
        successful_encodings = []

        # Process each image
        for file in files:
            contents = await file.read()
            result = await self.process_image(contents, person_folder)
            
            if result["status"] == "success":
                successful_encodings.append(result["encoding"])
            
            results.append({
                "filename": file.filename,
                "status": result["status"],
                "message": result["message"]
            })

        # Update encodings for the person
        if name not in self.known_face_encodings:
            self.known_face_encodings[name] = []
        
        self.known_face_encodings[name].extend(successful_encodings)

        return {
            "message": f"Processed {len(files)} images for {name}",
            "successful_uploads": len(successful_encodings),
            "total_reference_images": len(self.known_face_encodings[name]),
            "results": results
        }

    def identify_face(self, image_data):
        """Identify faces in the given image with improved matching logic."""
        face_locations = face_recognition.face_locations(image_data)
        face_encodings = face_recognition.face_encodings(image_data, face_locations)
        
        results = []
        for face_encoding in face_encodings:
            best_match = {"name": "Unknown", "confidence": 0.0}
            
            for person_name, person_encodings in self.known_face_encodings.items():
                face_distances = face_recognition.face_distance(person_encodings, face_encoding)
                
                top_matches = sorted(face_distances)[:3]
                avg_distance = np.mean(top_matches)
                confidence = 1 - avg_distance
                
                if confidence > best_match["confidence"] and confidence > 0.6:
                    best_match = {
                        "name": person_name,
                        "confidence": float(confidence),
                        "reference_images_count": len(person_encodings)
                    }
            
            results.append(best_match)
        
        return results

    def get_statistics(self):
        """Get statistics about stored reference images."""
        stats = {
            "total_people": len(self.known_face_encodings),
            "people": {
                name: len(encodings) 
                for name, encodings in self.known_face_encodings.items()
            }
        }
        return stats

# Initialize the face recognition system
face_system = FaceRecognitionSystem()

@app.post("/add_reference")
async def add_reference(files: TypeList[UploadFile] = File(...), name: str = Form(...)):
    """Add multiple reference images with an associated name."""
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    return await face_system.add_reference_images(files, name)

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    """Identify faces in the uploaded image."""
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_np = np.array(image)
    results = face_system.identify_face(image_np)
    return JSONResponse(content={"results": results})

@app.get("/statistics")
async def get_statistics():
    """Get statistics about stored reference images."""
    return face_system.get_statistics()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)