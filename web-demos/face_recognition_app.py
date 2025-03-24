import gradio as gr
import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from PIL import Image
import os
import pickle
import json
from datetime import datetime

# Initialize the face analysis model
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Create a directory to store registered faces
REGISTERED_FACES_DIR = "registered_faces"
if not os.path.exists(REGISTERED_FACES_DIR):
    os.makedirs(REGISTERED_FACES_DIR)

# Load registered faces database
def load_database():
    db_path = os.path.join(REGISTERED_FACES_DIR, "face_database.pkl")
    if os.path.exists(db_path):
        with open(db_path, 'rb') as f:
            return pickle.load(f)
    return {}

# Save registered faces database
def save_database(database):
    db_path = os.path.join(REGISTERED_FACES_DIR, "face_database.pkl")
    with open(db_path, 'wb') as f:
        pickle.dump(database, f)

# Initialize database
face_database = load_database()

def list_registered_faces():
    if not face_database:
        return "No faces registered yet"
    
    result = "Registered Faces:\n"
    for name, data in face_database.items():
        result += f"- {name} (registered on {data['timestamp']})\n"
    return result

def register_face(image, name):
    if image is None:
        return "Error: No image provided", list_registered_faces()
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Process the image
    faces = app.get(image)
    
    if len(faces) == 0:
        return "Error: No face detected in the image", list_registered_faces()
    
    if len(faces) > 1:
        return "Error: Multiple faces detected. Please provide an image with a single face.", list_registered_faces()
    
    # Get face embedding
    face = faces[0]
    embedding = face.embedding
    
    # Store in database
    face_database[name] = {
        'embedding': embedding,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save database
    save_database(face_database)
    
    return f"Successfully registered {name}", list_registered_faces()

def process_image(image, threshold):
    if image is None:
        return None
        
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Process the image
    faces = app.get(image)
    
    # Draw bounding boxes and landmarks
    for face in faces:
        bbox = face.bbox.astype(int)
        
        # Get face embedding
        embedding = face.embedding
        
        # Compare with registered faces
        is_authorized = False
        matched_name = None
        min_distance = float('inf')
        
        if not face_database:
            label = "No registered faces"
            color = (0, 0, 255)  # Red
        else:
            for name, data in face_database.items():
                # Calculate cosine similarity between embeddings
                similarity = np.dot(embedding, data['embedding']) / (np.linalg.norm(embedding) * np.linalg.norm(data['embedding']))
                distance = 1 - similarity  # Convert similarity to distance
                
                if distance < min_distance:
                    min_distance = distance
                    if distance < threshold:
                        is_authorized = True
                        matched_name = name
            
            # Add label with distance score
            if is_authorized:
                label = f"{matched_name} ({min_distance:.3f})"
                color = (0, 255, 0)  # Green
            else:
                label = f"Unauthorized ({min_distance:.3f})"
                color = (0, 0, 255)  # Red
        
        # Draw bounding box
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Add label
        cv2.putText(image, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw landmarks
        landmarks = face.kps.astype(int)
        for landmark in landmarks:
            cv2.circle(image, (landmark[0], landmark[1]), 2, (0, 0, 255), -1)
    
    # Convert back to PIL Image
    return Image.fromarray(image)

# Create the Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Face Recognition System")
    
    with gr.Tab("Register New Face"):
        with gr.Row():
            register_image = gr.Image(type="pil", label="Upload face image")
            register_name = gr.Textbox(label="Enter name")
        register_button = gr.Button("Register Face")
        register_output = gr.Textbox(label="Registration Status")
        registered_faces = gr.Textbox(label="Registered Faces", value=list_registered_faces())
    
    with gr.Tab("Face Recognition"):
        with gr.Row():
            input_image = gr.Image(type="pil", label="Upload image")
            output_image = gr.Image(type="pil", label="Result")
        with gr.Row():
            threshold = gr.Slider(
                minimum=0.1,
                maximum=0.5,  # Reduced maximum threshold
                value=0.3,    # Lower default threshold
                step=0.05,
                label="Face Matching Threshold (lower = stricter matching)",
                info="Lower values mean stricter matching (fewer false positives), higher values mean more lenient matching (fewer false negatives)"
            )
        process_button = gr.Button("Process Image")
    
    # Set up event handlers
    register_button.click(
        fn=register_face,
        inputs=[register_image, register_name],
        outputs=[register_output, registered_faces]
    )
    
    process_button.click(
        fn=process_image,
        inputs=[input_image, threshold],
        outputs=output_image
    )

if __name__ == "__main__":
    iface.launch(share=True) 