import cv2
import numpy as np
import h5py
import argparse
import os
import pickle

def detect_and_save_aruco(h5_file_path, output_pickle_path, output_video_path=None):
    # --- Camera Calibration ---
    fx = 1513.7213134765625
    fy = 1513.7213134765625
    cx = 969.6605834960938
    cy = 528.1881713867188
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1)) 

    # --- ArUco Marker Parameters ---
    marker_length = 0.02  # Marker size in meters

    # Define the 3D coordinates of the marker corners in its own coordinate system.
    object_points = np.array([
        [-marker_length / 2,  marker_length / 2, 0],
        [ marker_length / 2,  marker_length / 2, 0],
        [ marker_length / 2, -marker_length / 2, 0],
        [-marker_length / 2, -marker_length / 2, 0]
    ], dtype=np.float32)

    # --- Setup the ArUco Detector ---
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    parameters = cv2.aruco.DetectorParameters()
    # Parameter tuning
    parameters.polygonalApproxAccuracyRate = 0.01
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 8
    parameters.adaptiveThreshWinSizeMin = 5
    parameters.adaptiveThreshWinSizeMax = 99
    parameters.adaptiveThreshWinSizeStep = 5

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # --- Open the HDF5 file and read RGB frames ---
    with h5py.File(h5_file_path, 'r') as f:
        rgb_dataset = f['rgb']
        n_frames = rgb_dataset.shape[0]
        height, width, _ = rgb_dataset[0].shape

        if output_video_path is not None:
            fps = 30 
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        else:
            video_writer = None

        results = []

        for frame_idx in range(n_frames):
            frame_result = {"frame": frame_idx, "id": None, "transformation": None}

            # Get the current frame (as uint8).
            frame = np.array(rgb_dataset[frame_idx])
            # Preprocess the frame for marker detection.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Detect markers.
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None and len(ids) > 0:
                # Check for marker with ID == 1; if not found, use the first detected marker.
                selected_index = 0

                marker_corners = corners[selected_index]
                marker_id = int(ids[selected_index][0])
                image_points = marker_corners.reshape((4, 2))
                success, rvec, tvec = cv2.solvePnP(object_points, image_points,
                                                    camera_matrix, dist_coeffs)
                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = tvec.flatten()
                    frame_result["id"] = marker_id
                    frame_result["transformation"] = T.tolist()
                    cv2.aruco.drawDetectedMarkers(frame, [marker_corners], np.array([[marker_id]]))
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

            results.append(frame_result)

            if video_writer is not None:
                video_writer.write(frame)

            # Visualization
            # cv2.imshow("Frame", frame)
            # if cv2.waitKey(1) == 27:
            #     break

        if video_writer is not None:
            video_writer.release()

    # cv2.destroyAllWindows()

    # Save the results using pickle.
    with open(output_pickle_path, 'wb') as outfile:
        pickle.dump(results, outfile)

    print(f"Detection complete. Results saved to {output_pickle_path}")
    if output_video_path is not None:
        print(f"Video saved as {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect ArUco markers in H5 file frames and log marker id and 4x4 transformation matrix."
    )
    parser.add_argument("--h5_path", type=str, required=True,
                        help="Path to input .h5 file containing the 'rgb' dataset")
    parser.add_argument("--output_pickle", type=str, default="aruco_transforms.pkl",
                        help="Path to output pickle file (default: aruco_transforms.pkl)")
    parser.add_argument("--output_video", type=str, default=None,
                        help="(Optional) Path to output video file with overlay (e.g., output.avi)")
    args = parser.parse_args()

    if not os.path.exists(args.h5_path):
        print(f"H5 file not found: {args.h5_path}")
        exit(1)

    detect_and_save_aruco(args.h5_path, args.output_pickle, args.output_video)