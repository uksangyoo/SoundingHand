import os
import h5py
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d
import gc

# Import SAM2 for video segmentation
from sam2.build_sam import build_sam2_video_predictor

# Constants
SAM2_CHECKPOINT = "/home/frida/packages/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Setup device
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Use mixed precision with bfloat16
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # Enable TF32 for better performance on Ampere GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

def show_mask(mask, ax, obj_id=None, random_color=False):
    """Visualize a mask"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    """Visualize points (positive in green, negative in red)"""
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
              s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
              s=marker_size, edgecolor='white', linewidth=1.25)

def get_interactive_clicks(image):
    """
    Open an OpenCV window to get interactive clicks from the user.
    """
    points = []
    labels = []
    current_label = 1  # 1 for positive, 0 for negative
    
    # Create a copy of the image to draw on
    display_img = image.copy()
    
    # Window name
    window_name = "Select Points (p=positive click, n=negative click, r=reset, q=done)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal display_img
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point
            points.append([x, y])
            labels.append(current_label)
            
            # Draw point on image
            color = (0, 255, 0) if current_label == 1 else (0, 0, 255)
            cv2.circle(display_img, (x, y), 5, color, -1)
            cv2.circle(display_img, (x, y), 8, color, 2)
            
            # Update display
            cv2.imshow(window_name, display_img)
            print(f"Added {'positive' if current_label == 1 else 'negative'} click at ({x}, {y})")
    
    # Set mouse callback
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Display instructions
    instr_img = display_img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    instructions = [
        "Instructions:",
        "- Left click: Add a point",
        "- p: Switch to positive clicks (green)",
        "- n: Switch to negative clicks (red)",
        "- r: Reset all points",
        "- q: Done selecting points"
    ]
    
    y_pos = 30
    for line in instructions:
        cv2.putText(instr_img, line, (10, y_pos), font, 0.7, (255, 255, 255), 2)
        y_pos += 30
    
    # Initial display
    cv2.imshow(window_name, instr_img)
    cv2.waitKey(2000)  # Show instructions for 2 seconds
    cv2.imshow(window_name, display_img)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('p'):
            current_label = 1
            print("Mode: Positive clicks (green)")
        elif key == ord('n'):
            current_label = 0
            print("Mode: Negative clicks (red)")
        elif key == ord('r'):
            # Reset
            points.clear()
            labels.clear()
            display_img = image.copy()
            cv2.imshow(window_name, display_img)
            print("Reset all points")
        elif key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
    if not points:
        print("No points selected. Exiting.")
        return None, None
        
    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)

def depth_to_point_cloud(depth_image, fx, fy, cx, cy, depth_scale=1.0, max_depth=10.0, mask=None):
    """
    Convert a depth image to a point cloud, with optional mask to filter points.
    Fixed version that ensures mask dimensions are properly handled.
    """
    height, width = depth_image.shape
    
    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    
    # Calculate depth values
    z = depth_image * depth_scale
    
    # Determine valid points (non-zero depth, less than max_depth)
    valid_points = (z > 0) & (z < max_depth)
    
    # Apply mask if provided
    if mask is not None:
        # Handle different mask dimensions
        if len(mask.shape) == 3 and mask.shape[0] == 1:
            # If mask is (1, height, width)
            mask = mask[0]  # Remove the first dimension to get (height, width)
        
        # Ensure mask has the same shape as depth
        if mask.shape != depth_image.shape:
            print(f"Warning: Mask shape {mask.shape} != depth shape {depth_image.shape}.")
            # Only resize if dimensions don't match and dimensions are not empty
            if mask.shape[0] > 0 and mask.shape[1] > 0:
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
            else:
                print("Invalid mask dimensions. Using depth only.")
                mask = None
        
        if mask is not None:
            valid_points = valid_points & mask
    
    # Get coordinates of valid points
    y_coords = y_grid[valid_points]
    x_coords = x_grid[valid_points]
    z_values = z[valid_points]
    
    # Calculate 3D coordinates
    x_world = (x_coords - cx) * z_values / fx
    y_world = (y_coords - cy) * z_values / fy
    
    # Stack coordinates into points
    points = np.column_stack((x_world, y_world, z_values))
    
    # Stack image coordinates for RGB extraction
    image_coords = np.column_stack((y_coords, x_coords))
    
    return points, image_coords

def save_point_cloud(points, rgb=None, output_path="point_cloud.ply"):
    """
    Save point cloud to a PLY file.
    """
    # Check if there are any points
    if points is None or len(points) == 0:
        print(f"Warning: No valid points to save for {output_path}")
        return
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if rgb is not None and len(rgb) > 0:
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    o3d.io.write_point_cloud(output_path, pcd)
    
    # Clean up
    pcd = None
    gc.collect()

def process_h5_video_frames(h5_path, video_dir, start_frame, num_frames):
    """
    Extract frames from H5 file and save as images.
    Use pure integer filenames to comply with SAM2 requirements.
    Returns the start_frame to maintain correct frame numbering in the output.
    """
    os.makedirs(video_dir, exist_ok=True)
    
    with h5py.File(h5_path, 'r') as h5_file:
        total_frames_in_file = h5_file['rgb'].shape[0]
        end_frame = min(start_frame + num_frames, total_frames_in_file)
        batch_frames = end_frame - start_frame
        
        # Save first frame of this batch for interactive selection
        first_frame = h5_file['rgb'][start_frame][...].copy()
        preview_path = os.path.join(os.path.dirname(video_dir), f"preview_batch_{start_frame}.jpg")
        cv2.imwrite(preview_path, first_frame)
        
        # Let user select points with interactive clicks
        print(f"Please select points for batch starting at frame {start_frame}. Use 'p' for positive clicks, 'n' for negative clicks, 'r' to reset, and 'q' when done.")
        click_points, click_labels = get_interactive_clicks(first_frame)
        
        if click_points is None or len(click_points) == 0:
            print("No points selected. Skipping this batch.")
            os.remove(preview_path)
            return None, None, end_frame
        
        # Remove preview image after selection
        os.remove(preview_path)
        
        # Save frames with sequential integer filenames for SAM2 compatibility
        # but keep track of the original frame numbers
        print(f"Extracting {batch_frames} frames (from {start_frame} to {end_frame-1}) to {video_dir}")
        for i, original_idx in enumerate(range(start_frame, end_frame)):
            frame = h5_file['rgb'][original_idx][...].copy()
            frame_path = os.path.join(video_dir, f"{i}.jpg")  # Use sequential integers for SAM2
            cv2.imwrite(frame_path, frame)
            
            if i % 10 == 0:
                print(f"Processed temporary frame {i}/{batch_frames} (original frame {original_idx})")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    return click_points, click_labels, end_frame

def segment_video_and_extract_point_clouds(video_dir, h5_path, output_folder, click_points, click_labels, start_frame):
    """
    Segment video and extract point clouds.
    Start_frame specifies the original frame number in the H5 file.
    """
    # Setup output folders
    pcd_output_folder = os.path.join(output_folder, 'pointclouds')
    os.makedirs(pcd_output_folder, exist_ok=True)
    mask_output_folder = os.path.join(output_folder, 'masks')
    os.makedirs(mask_output_folder, exist_ok=True)
    
    # ZED camera 0 parameters
    # fx = 1513.7213134765625
    # fy = 1513.7213134765625
    # cx = 969.6605834960938
    # cy = 528.1881713867188
    # ZED camera 1 parameters
    fx = 1506.400634765625
    fy = 1506.400634765625
    cx = 962.9034423828125
    cy = 532.4006958007812
    depth_scale = 0.001  # convert mm to meters
    max_depth = 10.0
    
    # Get frame names
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    # Sort by frame index - SAM2 expects pure integer filenames
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    if not frame_names:
        print("No frames found in directory")
        return
    
    # Initialize SAM2 predictor
    print("Initializing SAM2 video predictor...")
    predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    
    # Initialize inference state
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)
    
    # Set object ID
    obj_id = 1
    ann_frame_idx = 0  # Always use the first frame in the batch for annotation
    
    # Add clicks to the first frame
    print("Adding clicks to first frame...")
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=obj_id,
        points=click_points,
        labels=click_labels,
    )
    
    # Save visualization of initial segmentation
    plt.figure(figsize=(9, 6))
    plt.title(f"Initial frame {start_frame} with clicks")
    first_frame = cv2.imread(os.path.join(video_dir, frame_names[ann_frame_idx]))
    plt.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    show_points(click_points, click_labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    plt.savefig(os.path.join(output_folder, f'first_frame_{start_frame:05d}_with_clicks.png'))
    plt.close()
    
    # Run propagation and save masks
    print("Propagating segmentation through video...")
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()
            
            # Calculate the original frame number
            original_frame_idx = start_frame + out_frame_idx
            
            # Save mask with original frame number
            mask_file = os.path.join(mask_output_folder, f'mask_{original_frame_idx:05d}.png')
            cv2.imwrite(mask_file, (mask * 255).astype(np.uint8))
            
            if out_frame_idx not in video_segments:
                video_segments[out_frame_idx] = {}
            video_segments[out_frame_idx][out_obj_id] = mask
        
        if out_frame_idx % 10 == 0:
            print(f"Processed segmentation for batch frame {out_frame_idx} (original frame {start_frame + out_frame_idx})")
            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Free memory after segmentation
    del predictor, inference_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Process depth data and create point clouds
    print("Generating point clouds from depth data...")
    with h5py.File(h5_path, 'r') as h5_file:
        for batch_frame_idx in range(len(frame_names)):
            try:
                # Skip frames without segmentation
                if batch_frame_idx not in video_segments or obj_id not in video_segments[batch_frame_idx]:
                    print(f"No segmentation for batch frame {batch_frame_idx}, skipping...")
                    continue
                
                # Calculate the original frame number
                original_frame_idx = start_frame + batch_frame_idx
                
                # Get mask and ensure it's properly shaped
                mask = video_segments[batch_frame_idx][obj_id]
                
                # Get depth and RGB data using the original frame index
                depth_frame = h5_file['depth'][original_frame_idx][...].copy()
                rgb_frame = h5_file['rgb'][original_frame_idx][...].copy()
                
                # Check data dimensions
                print(f"Original frame {original_frame_idx}: RGB shape: {rgb_frame.shape}, Depth shape: {depth_frame.shape}, Mask shape: {mask.shape}")
                
                # Process point cloud with mask
                try:
                    # Full point cloud (without mask)
                    full_points, full_coords = depth_to_point_cloud(
                        depth_frame, fx, fy, cx, cy, depth_scale, max_depth
                    )
                    
                    # Masked point cloud
                    masked_points, masked_coords = depth_to_point_cloud(
                        depth_frame, fx, fy, cx, cy, depth_scale, max_depth, mask
                    )
                    
                    # Extract RGB values safely for masked points
                    if len(masked_coords) > 0:
                        # Ensure coordinates are within bounds
                        valid_y = np.clip(masked_coords[:, 0], 0, rgb_frame.shape[0] - 1)
                        valid_x = np.clip(masked_coords[:, 1], 0, rgb_frame.shape[1] - 1)
                        
                        # Extract RGB values
                        masked_rgb = rgb_frame[valid_y, valid_x]
                        
                        # Save masked point cloud with original frame number
                        masked_pcd_path = os.path.join(pcd_output_folder, f'frame_{original_frame_idx:05d}_masked.ply')
                        save_point_cloud(masked_points, masked_rgb, masked_pcd_path)
                    
                    # Extract RGB values safely for full point cloud
                    if len(full_coords) > 0:
                        # Ensure coordinates are within bounds
                        valid_y = np.clip(full_coords[:, 0], 0, rgb_frame.shape[0] - 1)
                        valid_x = np.clip(full_coords[:, 1], 0, rgb_frame.shape[1] - 1)
                        
                        # Extract RGB values
                        full_rgb = rgb_frame[valid_y, valid_x]
                        
                        # Save full point cloud with original frame number
                        full_pcd_path = os.path.join(pcd_output_folder, f'frame_{original_frame_idx:05d}_full.ply')
                        save_point_cloud(full_points, full_rgb, full_pcd_path)
                    
                    # Create visualization 
                    vis_img = rgb_frame.copy()
                    overlay = np.zeros_like(vis_img)
                    
                    # Handle 3D mask properly
                    if len(mask.shape) == 3 and mask.shape[0] == 1:
                        mask_vis = mask[0]  # Use the first channel if mask is 3D
                    else:
                        mask_vis = mask
                        
                    # Apply mask to overlay
                    overlay[mask_vis] = [0, 255, 0]  # Green mask
                    vis_img = cv2.addWeighted(vis_img, 1, overlay, 0.5, 0)
                    
                    # Save visualization with original frame number
                    vis_path = os.path.join(output_folder, f'visualization_{original_frame_idx:05d}.png')
                    cv2.imwrite(vis_path, vis_img)
                
                except Exception as e:
                    print(f"Error generating point cloud for frame {original_frame_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                
                if batch_frame_idx % 10 == 0:
                    print(f"Generated point cloud for frame {original_frame_idx}")
                    # Clean up
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing frame {start_frame + batch_frame_idx}: {e}")
                import traceback
                traceback.print_exc()
            
            # Clean up variables
            if 'depth_frame' in locals(): del depth_frame
            if 'rgb_frame' in locals(): del rgb_frame
            if 'mask' in locals(): del mask
            if 'masked_points' in locals(): del masked_points
            if 'masked_coords' in locals(): del masked_coords
            if 'masked_rgb' in locals(): del masked_rgb
            if 'full_points' in locals(): del full_points
            if 'full_coords' in locals(): del full_coords
            if 'full_rgb' in locals(): del full_rgb
            if 'vis_img' in locals(): del vis_img
            if 'overlay' in locals(): del overlay
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def process_in_batches(h5_path, base_output_folder, batch_size=300):
    """
    Process the entire H5 file in batches, allowing point selection for each batch.
    """
    os.makedirs(base_output_folder, exist_ok=True)
    
    # Get total number of frames in H5 file
    with h5py.File(h5_path, 'r') as h5_file:
        total_frames = h5_file['rgb'].shape[0]
    
    print(f"Total frames in H5 file: {total_frames}")
    
    # Process in batches
    current_frame = 0
    batch_number = 1
    
    while current_frame < total_frames:
        print(f"\n=== Processing Batch {batch_number} (starting at frame {current_frame}) ===\n")
        
        # Create temporary directory for this batch's frames
        batch_video_dir = os.path.join(base_output_folder, f"temp_frames_batch_{batch_number}")
        os.makedirs(batch_video_dir, exist_ok=True)
        
        try:
            # Step 1: Extract frames for this batch
            click_points, click_labels, next_frame = process_h5_video_frames(
                h5_path, batch_video_dir, current_frame, batch_size
            )
            
            if click_points is not None:
                # Step 2: Run segmentation and point cloud extraction for this batch
                segment_video_and_extract_point_clouds(
                    batch_video_dir, h5_path, base_output_folder, click_points, click_labels, current_frame
                )
            
            # Update frame counter and batch number
            current_frame = next_frame
            batch_number += 1
        
        except Exception as e:
            print(f"Error processing batch {batch_number}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean up temporary files for this batch
            print(f"Cleaning up temporary files for batch {batch_number-1}...")
            import shutil
            if os.path.exists(batch_video_dir):
                shutil.rmtree(batch_video_dir)
            
            # Give the system a moment to release resources
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Waiting 5 seconds before starting the next batch...")
            import time
            time.sleep(5)

if __name__ == "__main__":
    h5_path = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/0419_dual_cam_toy_example/h5/t1_cam1.h5"
    base_output_folder = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/0419_dual_cam_toy_example/output1"
    try:
        # Process the entire file in batches of 300 frames
        process_in_batches(h5_path, base_output_folder, batch_size=1000)
        
        print("All batches processed successfully!")
    
    except Exception as e:
        print(f"Error during batch processing: {e}")
        import traceback
        traceback.print_exc()