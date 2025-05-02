import numpy as np
from pathlib import Path
from visualize_hands_wrist_markers import visualize_frame

# Paths
MESH_FOLDER = "hand_meshes"
OBJECT_MESH_PATH = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/PREP/hand_meshes/spam_pla.obj"
ARUCO_DATA_PATH = "../DATA/wrist_aruco_toy_example/wrist_aruco_toy_example_objet_tracking.pkl"
WRIST_ARUCO_DATA_PATH = "../DATA/wrist_aruco_toy_example/wrist_aruco_toy_example_wrist_tracking.pkl"

# Define the fine-tuning transformation for the object
# This is an example - you might need to adjust these values
fine_tune_T = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

# Define the fine-tuning transformation for the hand (if needed)
hand_fine_tune_T = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

def main():
    # Visualize frame 1205 (you can change this to any frame number)
    frame_number = 1205
    
    visualize_frame(
        mesh_folder=MESH_FOLDER,
        frame_number=frame_number,
        object_mesh_path=OBJECT_MESH_PATH,
        aruco_data_path=ARUCO_DATA_PATH,
        wrist_aruco_data_path=WRIST_ARUCO_DATA_PATH,
        fine_tune_T=fine_tune_T,
        hand_fine_tune_T=hand_fine_tune_T,
        hand_id=0,  # Change to 1 for the other hand
        marker_id=1,  # ArUco marker ID for the object
        wrist_marker_id=0  # ArUco marker ID for the wrist
    )

if __name__ == "__main__":
    main() 