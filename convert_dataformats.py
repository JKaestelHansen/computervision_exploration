# %%
import os
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from natsort import natsorted


def convert_tiffs_to_yolo_segmentation(image_dir, mask_dir, output_dir, img_format="jpeg", class_id=0):
    """
    Convert TIFF images and mask TIFFs into YOLOv8 segmentation format.
    
    Args:
        image_dir (str or Path): Folder with image TIFFs
        mask_dir (str or Path): Folder with corresponding mask TIFFs
        output_dir (str or Path): YOLO dataset output folder
        img_format (str): 'jpg' or 'png'
        class_id (int): Default class id for objects
    """
    image_dir, mask_dir, output_dir = Path(image_dir), Path(mask_dir), Path(output_dir)
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for img_path in image_dir.glob("*.tif*"):
        mask_path = mask_dir / f"mask{img_path.stem[1:]}.tif"
        if not mask_path.exists():
            print(f"⚠️ No mask for {img_path.name}, skipping")
            continue
        
        # --- Save image ---
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        out_img_path = images_out / f"{img_path.stem}.{img_format}"
        img.save(out_img_path, format=img_format.upper())
        
        # --- Process mask ---
        mask = np.array(Image.open(mask_path))
        label_lines = []
        
        # Each unique object id in mask (ignoring background 0)
        for obj_id in np.unique(mask):
            if obj_id == 0:
                continue
            obj_mask = (mask == obj_id).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) < 3:  # too small
                    continue
                # Flatten & normalize
                poly = contour.reshape(-1, 2)
                poly_norm = []
                for x, y in poly:
                    poly_norm.append(x / W)
                    poly_norm.append(y / H)
                
                line = f"{class_id} " + " ".join(f"{p:.6f}" for p in poly_norm)
                label_lines.append(line)
        
        # --- Save label file ---
        label_path = labels_out / f"{img_path.stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))
            
    print(f"\n✅ Done! Images in {images_out}, labels in {labels_out}")


data_path = '/Users/bzx569/Documents/PostDoc/Work/cell_tracking_segmentation/cell_data/'
exp_path = 'BF-C2DL-MuSC/02'
GT_path = 'BF-C2DL-MuSC/02_GT/TRA'
ST_path = 'BF-C2DL-MuSC/02_ST'
ERR_SEG_path = 'BF-C2DL-MuSC/02_ERR_SEG'
 
exp_files = os.listdir(os.path.join(data_path, exp_path))
exp_files = natsorted(exp_files)

GT_files = os.listdir(os.path.join(data_path, GT_path))
GT_files = natsorted(GT_files)

ERR_SEG_files = os.listdir(os.path.join(data_path, ERR_SEG_path))
ERR_SEG_files = natsorted(ERR_SEG_files)


convert_tiffs_to_yolo_segmentation(
    image_dir=os.path.join(data_path, exp_path),
    mask_dir=os.path.join(data_path, ERR_SEG_path),
    output_dir=os.path.join(data_path, exp_path+'_yolo_dataset'),
    img_format="jpeg",
    class_id=0
)