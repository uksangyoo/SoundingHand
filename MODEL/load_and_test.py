import h5py
import sys

def print_hdf5_structure(group, indent=""):
    """Recursively prints the structure of an HDF5 group."""
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            print(f"{indent}Group: {key}")
            print_hdf5_structure(item, indent + "  ")
        elif isinstance(item, h5py.Dataset):
            print(f"{indent}Dataset: {key}, shape: {item.shape}, dtype: {item.dtype}")
        else:
            print(f"{indent}Unknown item: {key}")

if __name__ == "__main__":
    # Use the provided absolute path
    file_path = "/media/frida/3376a50a-001d-45d9-89a7-589977ec1b04/SoundingHand/DATA/wrist_aruco_toy_example/wrist_aruco_toy_example.h5"

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Inspecting file: {file_path}")
            print("-" * 30)
            print_hdf5_structure(f['/'])
            print("-" * 30)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
