import numpy as np
from PIL import Image

img_path = '/home/qilin/Documents/GS4Building/data/building1/undistorted/images/DJI_20241217101307_0006_D.JPG'
original_img = Image.open(img_path)
img_width, img_height = original_img.size

depth_path = '/home/qilin/Documents/GS4Building/data/depth_normal/raw_depth/DJI_20241217101307_0006_D.npy'
depth_map = np.load(depth_path)

if depth_map.shape[0] == img_height and depth_map.shape[1] == img_width:
    print("尺寸匹配！")
else:
    print(f"尺寸不匹配: 图像 {img_width}x{img_height} vs 深度图 {depth_map.shape[1]}x{depth_map.shape[0]}")