import cv2
import numpy as np
import h5py
import argparse
import os
import pickle
import math


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
    marker_length = 0.02  # meters
    object_points = np.array([
        [-marker_length/2,  marker_length/2, 0],
        [ marker_length/2,  marker_length/2, 0],
        [ marker_length/2, -marker_length/2, 0],
        [-marker_length/2, -marker_length/2, 0]
    ], dtype=np.float32)

    # --- ArUco Detector Setup ---
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.polygonalApproxAccuracyRate = 0.01
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 8
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 99
    params.adaptiveThreshWinSizeStep = 5
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    # --- Open HDF5 ---
    with h5py.File(h5_file_path, 'r') as f:
        rgb_dataset = f['rgb']
        n_frames = rgb_dataset.shape[0]
        h, w, _ = rgb_dataset[0].shape

        # VideoWriter
        video_writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h))

        results = []
        last_T1 = None
        last_detected_frame = -1
        max_angle = math.pi/2  # 90°

        for i in range(n_frames):
            frame = np.array(rgb_dataset[i])
            frame_result = {"frame": i, "id": None, "transformation": None}

            # preprocessing
            # lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            # l,a,b = cv2.split(lab)
            # clahe = cv2.createCLAHE(2.0,(8,8))
            # lab = cv2.merge((clahe.apply(l),a,b))
            # frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            # invG = 1/0.7
            # table = ((np.arange(256)/255.0)**invG*255).astype(np.uint8)
            # frame = cv2.LUT(frame, table)
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # p5,p95 = np.percentile(gray,(5,95))
            # gray = np.clip((gray-p5)*(255.0/(p95-p5)),0,255).astype(np.uint8)
            # gray = cv2.bilateralFilter(gray,9,75,75)


            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None and any(int(x[0]) == 1 for x in ids):
                idx = next(j for j,m in enumerate(ids) if int(m[0]) == 1)
                pts = corners[idx].reshape((4,2))
                ok, rvec, tvec = cv2.solvePnP(object_points, pts, camera_matrix, dist_coeffs)
                if ok:
                    R, _ = cv2.Rodrigues(rvec)
                    T = np.eye(4)
                    T[:3,:3] = R
                    T[:3,3] = tvec.flatten()

                    # decide validity
                    if last_T1 is None:
                        valid = True
                    else:
                        dt = i - last_detected_frame
                        if dt == 1:
                            z_prev = last_T1[:3, 2]
                            z_cur = R[:, 2]
                            angle = math.acos(np.clip(np.dot(z_prev, z_cur), -1.0, 1.0))
                            valid = angle <= max_angle
                        else:
                            valid = True

                    if valid:
                        # store as numpy array for indexing
                        last_T1 = T
                        last_detected_frame = i
                        frame_result["id"] = 1
                        frame_result["transformation"] = T.tolist()
                        cv2.aruco.drawDetectedMarkers(frame, [corners[idx]], np.array([[1]]))
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

            results.append(frame_result)
            if video_writer:
                video_writer.write(frame)
            # cv2.imshow('Frame', frame)
            # if cv2.waitKey(1) == 27:
            #     break

        if video_writer:
            video_writer.release()
    # cv2.destroyAllWindows()

    with open(output_pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved {output_pickle_path}")
    if output_video_path:
        print(f"Video saved: {output_video_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--h5_path', required=True)
    p.add_argument('--output_pickle', default='aruco.pkl')
    p.add_argument('--output_video', default=None)
    args = p.parse_args()
    if not os.path.exists(args.h5_path):
        print('Missing H5'); exit(1)
    detect_and_save_aruco(args.h5_path, args.output_pickle, args.output_video)
