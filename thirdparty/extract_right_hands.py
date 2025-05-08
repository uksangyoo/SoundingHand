import os
import h5py
import numpy as np
import cv2
import torch
from pathlib import Path
from hamer.models import load_hamer
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import hamer
import trimesh

# Constants
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
DEFAULT_CHECKPOINT = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/thirdparty/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
CACHE_DIR_HAMER = "../thirdparty/hamer/hamer_cache"
MANO_DATA_DIR = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/thirdparty/hamer/_DATA/data"
os.environ['MANO_DATA_DIR'] = MANO_DATA_DIR
def process_h5_file(h5_path, output_folder, checkpoint=DEFAULT_CHECKPOINT, rescale_factor=2.0, batch_size=10):
    """
    Process an H5 file to extract hand meshes.
    
    Args:
        h5_path: Path to the H5 file
        output_folder: Directory where output files will be saved
        checkpoint: Path to the HaMeR checkpoint
        rescale_factor: Factor to rescale bounding boxes
        batch_size: Number of frames to process at once
    """
    # Setup output folder
    os.makedirs(output_folder, exist_ok=True)
    
    
    # Load HaMeR model
    model, model_cfg = load_hamer(checkpoint)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load keypoint detector
    cpm = ViTPoseModel(device)

    # Load body detector
    cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # Setup renderer
    # renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Open HDF5 file
    with h5py.File(h5_path, 'r') as h5_file:
        total_frames = h5_file['rgb'].shape[0]
        print(f"Total frames in dataset: {total_frames}")
        
        frames_with_detections = 0
        total_hands_detected = 0
        
        # Process frames in batches
        for start_idx in range(0, total_frames, batch_size):
            end_idx = min(start_idx + batch_size, total_frames)
            print(f"Processing frames {start_idx} to {end_idx-1}")
            
            # Load batch of frames
            rgb_frames = h5_file['rgb'][start_idx:end_idx]
            depth_frames = h5_file['depth'][start_idx:end_idx]
            
            # Process each frame in the batch
            for batch_idx, (img_cv2, depth_frame) in enumerate(zip(rgb_frames, depth_frames)):
                frame_idx = start_idx + batch_idx
                
                # Convert to BGR for detector which expects BGR
                img_cv2_bgr = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

                # Detect humans in the frame
                det_out = detector(img_cv2)
                img = img_cv2.copy()  # Keep in RGB for the rest of the pipeline

                det_instances = det_out['instances']
                valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)  # Lowered threshold
                pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                pred_scores = det_instances.scores[valid_idx].cpu().numpy()

                if len(pred_bboxes) == 0:
                    print(f"Frame {frame_idx}: No person detections found")
                    continue

                # Detect human keypoints for each person (expects RGB)
                vitposes_out = cpm.predict_pose(
                    img,
                    [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
                )

                bboxes = []
                is_right = []

                # Use hands based on hand keypoint detections
                for vitposes in vitposes_out:
                    left_hand_keyp = vitposes['keypoints'][-42:-21]
                    right_hand_keyp = vitposes['keypoints'][-21:]

                    # Rejecting not confident detections - more lenient threshold
                    keyp = left_hand_keyp
                    valid = keyp[:, 2] > 0.5  # Lowered threshold
                    if sum(valid) > 2:  # Lowered required keypoints
                        bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                        bboxes.append(bbox)
                        is_right.append(0)
                    keyp = right_hand_keyp
                    valid = keyp[:, 2] > 0.5  # Lowered threshold
                    if sum(valid) > 2:  # Lowered required keypoints
                        bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                        bboxes.append(bbox)
                        is_right.append(1)

                if len(bboxes) == 0:
                    print(f"Frame {frame_idx}: No hand detections found")
                    continue

                frames_with_detections += 1
                total_hands_detected += len(bboxes)
                print(f"Frame {frame_idx}: Found {len(bboxes)} hands")

                boxes = np.stack(bboxes)
                right = np.stack(is_right)

                # Run reconstruction on all detected hands
                dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=rescale_factor)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1)

                for batch in dataloader:
                    batch = recursive_to(batch, device)
                    with torch.no_grad():
                        out = model(batch)

                    # Get camera parameters
                    multiplier = (2*batch['right']-1)
                    pred_cam = out['pred_cam']
                    pred_cam[:,1] = multiplier*pred_cam[:,1]
                    box_center = batch["box_center"].float()
                    box_size = batch["box_size"].float()
                    img_size = batch["img_size"].float()
                    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

                    # Process each hand in the batch
                    for n in range(batch['img'].shape[0]):
                        # Get vertices and camera parameters
                        verts = out['pred_vertices'][n].detach().cpu().numpy()
                        is_right_hand = batch['right'][n].cpu().numpy()
                        cam_t = pred_cam_t_full[n]

                        # Transform vertices to camera frame
                        # Flip x-coordinate for right hands
                        verts[:,0] = (2*is_right_hand-1)*verts[:,0]
                        
                        # Create mesh
                        mesh = trimesh.Trimesh(
                            vertices=verts,
                            faces=model.mano.faces,
                            vertex_colors=np.tile(LIGHT_BLUE, (verts.shape[0], 1))
                        )
                        
                        # Save mesh
                        if n==1:
                            mesh_path = os.path.join(output_folder, f'frame_{frame_idx}_hand_{n}.obj')
                            mesh.export(mesh_path)
                        
                            # Save camera parameters
                            np.save(os.path.join(output_folder, f'frame_{frame_idx}_hand_{n}_cam_t.npy'), cam_t)

                            # Save all model outputs for this hand
                            outputs_to_save = {}
                            for key, value in out.items():
                                if isinstance(value, dict):
                                    # Save nested dicts (e.g., pred_mano_params)
                                    for subkey, subvalue in value.items():
                                        outputs_to_save[f'{key}_{subkey}'] = subvalue[n].detach().cpu().numpy() if hasattr(subvalue[n], 'detach') else subvalue[n]
                                elif isinstance(value, torch.Tensor):
                                    outputs_to_save[key] = value[n].detach().cpu().numpy()
                                elif isinstance(value, np.ndarray):
                                    outputs_to_save[key] = value[n]
                                else:
                                    # Skip non-array/tensor outputs
                                    pass
                            np.savez(os.path.join(output_folder, f'frame_{frame_idx}_hand_{n}_outputs.npz'), **outputs_to_save)

                        # # Render the result
                        # regression_img = renderer(
                        #     verts,
                        #     cam_t,
                        #     batch['img'][n],
                        #     mesh_base_color=LIGHT_BLUE,
                        #     scene_bg_color=(1, 1, 1),
                        # )

                        # # Save rendered image
                        # output_path = os.path.join(output_folder, f'frame_{frame_idx}_hand_{n}.png')
                        #cv2.imwrite(output_path, 255 * regression_img[:, :, ::-1])

if __name__ == "__main__":
    # Set up your directories here
    DATA_ROOT = "/media/frida/Extreme SSD/sounding_hand/yuemin"
    SAVE_ROOT = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/yuemin"
    #object_directories = ["campbell_pla", "spam_pla"]
    # object_directories = ["campbell_pla","campbell_real", "cheese", "cheezit", "clamp", "drill",
    #                       "juice", "knife", "marker", "mug", "mustard", "ranch", "screwdriver", "spam_pla",
    #                         "spam_real", "wrench"]
    # object_directories = [ "drill","juice", "knife", "marker", "mug", "mustard", "ranch", "screwdriver", "spam_pla",
    #                     "spam_real", "wrench"]
    #List directories in DATA_ROOT
    object_directories = os.listdir(DATA_ROOT)
    print(object_directories)
    for object_directory in object_directories:
        save_directory = os.path.join(SAVE_ROOT, object_directory)
        directory = os.path.join(DATA_ROOT, object_directory, "h5")

        for h5_file in os.listdir(directory):
            h5_path = os.path.join(directory, h5_file)
            
            # Process output0 (camera 0)
            output0_dir = os.path.join(save_directory, h5_file, "output0", "hands")
            #if not os.path.exists(output0_dir):
            if True:
                os.makedirs(output0_dir, exist_ok=True)
                print(f"Processing {output0_dir}")
                h5_files = sorted([f for f in os.listdir(h5_path) if f.endswith('.h5')])
                process_h5_file(os.path.join(h5_path, h5_files[0]), output0_dir)
            else:
                print(f"Skipping {output0_dir} - already exists")

            # Process output1 (camera 1)
            output1_dir = os.path.join(save_directory, h5_file, "output1", "hands")
            if not os.path.exists(output1_dir):
                os.makedirs(output1_dir, exist_ok=True)
                print(f"Processing {output1_dir}")
                h5_files = sorted([f for f in os.listdir(h5_path) if f.endswith('.h5')])
                process_h5_file(os.path.join(h5_path, h5_files[1]), output1_dir)
            else:
                print(f"Skipping {output1_dir} - already exists")