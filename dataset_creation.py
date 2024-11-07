import os
import cv2
import json
import numpy as np
from PIL import Image
import albumentations as A
from pathlib import Path
from deepface import DeepFace
from tqdm import tqdm

class VideoFaceAugmenter:
    def __init__(self):
        self.reference_folder = "reference_images"
        self.metadata_file = "reference_metadata.json"
        self.augmented_metadata = {}
        
        if not os.path.exists(self.reference_folder):
            os.makedirs(self.reference_folder)
            
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.augmented_metadata = json.load(f)

    def create_augmentation_pipelines(self):
        """Create multiple augmentation pipelines for different scenarios"""
        
        # Pipeline 1: Lighting variations
        lighting_pipeline = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.RandomGamma(gamma_limit=(70, 130), p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
            A.RandomShadow(p=0.3),
            A.CLAHE(p=0.3),
        ])
        
        # Pipeline 2: Motion and blur simulation
        motion_pipeline = A.Compose([
            A.OneOf([
                A.MotionBlur(blur_limit=(3, 7), p=0.8),
                A.GaussianBlur(blur_limit=(3, 7), p=0.8),
                A.MedianBlur(blur_limit=5, p=0.5),
            ], p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.4),
        ])
        
        # Pipeline 3: Geometric transformations
        geometric_pipeline = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=20,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.7
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-20, 20),
                shear={"x": (-10, 10), "y": (-10, 10)},
                p=0.3
            ),
        ])
        
        return [lighting_pipeline, motion_pipeline, geometric_pipeline]

    def augment_single_image(self, image, pipelines, base_filename, output_folder):
        """Apply multiple augmentation passes to a single image"""
        augmented_files = []
        
        # First pass: Apply each pipeline individually
        for idx, pipeline in enumerate(pipelines):
            augmented = pipeline(image=image)['image']
            filename = f"aug_p{idx}_{base_filename}"
            output_path = os.path.join(output_folder, filename)
            Image.fromarray(augmented).save(output_path)
            augmented_files.append(output_path)
            
        # Second pass: Combine pipelines
        for i in range(len(pipelines)):
            for j in range(i + 1, len(pipelines)):
                augmented = pipelines[i](image=image)['image']
                augmented = pipelines[j](image=augmented)['image']
                filename = f"aug_p{i}{j}_{base_filename}"
                output_path = os.path.join(output_folder, filename)
                Image.fromarray(augmented).save(output_path)
                augmented_files.append(output_path)
        
        return augmented_files

    def augment_reference_images(self, num_variations=25):
        """
        Create augmented dataset optimized for video recognition
        
        Args:
            num_variations: Number of variations per pipeline combination
        """
        pipelines = self.create_augmentation_pipelines()
        
        for person_name in os.listdir(self.reference_folder):
            person_path = os.path.join(self.reference_folder, person_name)
            if not os.path.isdir(person_path):
                continue
                
            print(f"\nProcessing images for {person_name}")
            augmented_folder = os.path.join(person_path, "augmented")
            if not os.path.exists(augmented_folder):
                os.makedirs(augmented_folder)
            
            # Process each original image
            original_images = [f for f in os.listdir(person_path) 
                             if f.endswith((".jpg", ".jpeg", ".png")) and not f.startswith("aug_")]
            
            for img_file in tqdm(original_images, desc="Processing images"):
                img_path = os.path.join(person_path, img_file)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Create variations
                for variation in range(num_variations):
                    try:
                        augmented_files = self.augment_single_image(
                            image, 
                            pipelines,
                            f"var{variation}_{img_file}",
                            augmented_folder
                        )
                        
                        # Verify face detection for each augmented image
                        for aug_file in augmented_files:
                            try:
                                _ = DeepFace.represent(aug_file, model_name="Facenet", enforce_detection=True)
                            except Exception as e:
                                if os.path.exists(aug_file):
                                    os.remove(aug_file)
                                continue
                            
                            # Update metadata
                            if person_name not in self.augmented_metadata:
                                self.augmented_metadata[person_name] = []
                            self.augmented_metadata[person_name].append(os.path.basename(aug_file))
                                
                    except Exception as e:
                        print(f"Error processing variation {variation} of {img_file}: {str(e)}")
                        continue
                
            # Save metadata after processing each person
            with open(self.metadata_file, 'w') as f:
                json.dump(self.augmented_metadata, f, indent=4)

def main():
    augmenter = VideoFaceAugmenter()
    augmenter.augment_reference_images(num_variations=25)
    
    # Print summary
    total_augmented = sum(len(files) for person in os.listdir(augmenter.reference_folder)
                         for _, _, files in os.walk(os.path.join(augmenter.reference_folder, person, "augmented")))
    print(f"\nAugmentation complete! Total augmented images created: {total_augmented}")

if __name__ == "__main__":
    main()