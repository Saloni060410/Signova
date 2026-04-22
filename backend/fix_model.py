import zipfile
import tempfile
import shutil
import os

zip_path = r"C:\Users\Kailash Sharma\Downloads\MPR_SEMI\signlang_project\MPR_STATIC_M\asl_resnet50.pth.zip"
out_path = r"C:\Users\Kailash Sharma\Downloads\MPR_SEMI\signlang_project\MPR_STATIC_M\asl_resnet50.pth"

with tempfile.TemporaryDirectory() as d:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(d)
    
    # Check what is inside
    files = list(os.walk(d))
    
    # We want to find data.zip and extract it to data/
    for root, dirs, filenames in os.walk(d):
        if 'data.zip' in filenames:
            data_zip_path = os.path.join(root, 'data.zip')
            # Extract data.zip into 'data' folder next to it
            data_dir = os.path.join(root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            with zipfile.ZipFile(data_zip_path, 'r') as dzf:
                dzf.extractall(data_dir)
            
            # Remove data.zip
            os.remove(data_zip_path)

    # Now we want to repack it properly.
    # PyTorch expects a structure like `archive_name/data.pkl` etc. 
    # Let's find the deepest directory that has data.pkl
    base_dir = None
    for root, dirs, filenames in os.walk(d):
        if 'data.pkl' in filenames:
            base_dir = root
            break
            
    if base_dir:
        # Repack base_dir contents natively into out_path
        # PyTorch infers the top-level directory so we can just use "archive"
        with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_STORED) as zf_out:
            for root, dirs, filenames in os.walk(base_dir):
                for f in filenames:
                    abs_path = os.path.join(root, f)
                    rel_path = os.path.relpath(abs_path, base_dir)
                    zf_out.write(abs_path, os.path.join("archive", rel_path))

        print(f"Successfully repaired and saved to {out_path}")
    else:
        print("data.pkl not found!")