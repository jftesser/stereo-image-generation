import os
import torch
import cv2
import argparse
import numpy as np

from tqdm import tqdm
from torch.backends import cudnn
from torchvision.transforms import Compose

from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet

parser = argparse.ArgumentParser(description='MiDaS')
parser.add_argument('--input', default='./example', type=str, help='Input filename or folder.')
args = parser.parse_args()

IPD = 6.5
MONITOR_W = 38.5


def write_depth(depth, bits=1, reverse=True):
    depth_min = depth.min()
    depth_max = depth.max()
    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0
    if not reverse:
        out = max_val - out

    if bits == 2:
        depth_map = out.astype("uint16")
    else:
        depth_map = out.astype("uint8")

    return depth_map


def generate_stereo(left_img, depth):
    h, w, c = left_img.shape

    depth_min = depth.min()
    depth_max = depth.max()
    depth = (depth - depth_min) / (depth_max - depth_min)

    right = np.zeros_like(left_img)

    deviation_cm = IPD * 0.12
    deviation = deviation_cm * MONITOR_W * (w / 1920)

    print("\ndeviation:", deviation)

    for row in range(h):
        for col in range(w):
            col_r = col - int((1 - depth[row][col] ** 2) * deviation)
            # col_r = col - int((1 - depth[row][col]) * deviation)
            if col_r >= 0:
                right[row][col_r] = left_img[row][col]

    right_fix = np.array(right)
    gray = cv2.cvtColor(right_fix, cv2.COLOR_BGR2GRAY)
    rows, cols = np.where(gray == 0)
    for row, col in zip(rows, cols):
        for offset in range(1, int(deviation)):
            r_offset = col + offset
            l_offset = col - offset
            if r_offset < w and not np.all(right_fix[row][r_offset] == 0):
                right_fix[row][col] = right_fix[row][r_offset]
                break
            if l_offset >= 0 and not np.all(right_fix[row][l_offset] == 0):
                right_fix[row][col] = right_fix[row][l_offset]
                break

    return right_fix


def overlap(im1, im2):
    width1 = im1.shape[1]
    height1 = im1.shape[0]
    width2 = im2.shape[1]
    height2 = im2.shape[0]

    # final image
    composite = np.zeros((height2, width2, 3), np.uint8)

    # iterate through "left" image, filling in red values of final image
    for i in range(height1):
        for j in range(width1):
            try:
                composite[i, j, 2] = im1[i, j, 2]
            except IndexError:
                pass

    # iterate through "right" image, filling in blue/green values of final image
    for i in range(height2):
        for j in range(width2):
            try:
                composite[i, j, 1] = im2[i, j, 1]
                composite[i, j, 0] = im2[i, j, 0]
            except IndexError:
                pass

    return composite


def run(model_path):
    """
    Run MonoDepthNN to compute depth maps.
    """
    # Input images
    img_list = os.listdir(args.input)
    img_list.sort()

    # output dir
    output_dir = './stereo'
    os.makedirs(output_dir, exist_ok=True)

    # set torch options
    torch.cuda.empty_cache()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # select device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device: %s" % device)

    # load network
    model = MidasNet(model_path, non_negative=True)

    transform = Compose(
        [
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    model.to(device)
    model.eval()

    for idx in tqdm(range(len(img_list))):
        sample = img_list[idx]
        print('sample:', sample)
        left_img = cv2.imread(os.path.join(args.input, sample))
        print('left_img:', left_img)
        if left_img is not None:
            img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB) / 255.0

            #  Apply transforms
            image = transform({"image": img})["image"]

            #  Predict and resize to original resolution
            with torch.no_grad():
                image = torch.from_numpy(image).to(device).unsqueeze(0)
                depth = model.forward(image)

                depth = (
                    torch.nn.functional.interpolate(
                        depth.unsqueeze(1),
                        size=left_img.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )

            depth = cv2.blur(depth, (3, 3))

            depth_map = write_depth(depth, bits=2, reverse=False)
            right_img = generate_stereo(left_img, depth_map)
            stereo = np.hstack([left_img, right_img])
            anaglyph = overlap(left_img, right_img)

            # cv2.imshow('depth map', depth_map)
            # cv2.imshow('side by side', stereo)
            # cv2.imshow("anaglyph", anaglyph)
            # cv2.waitKey(0)

            cv2.imwrite(os.path.join(output_dir, 'MiDaS_{}.png'.format(sample.split('.')[0])), stereo)
            cv2.imwrite(os.path.join(output_dir, 'MiDaS_3d_{}.png'.format(sample.split('.')[0])), anaglyph)


if __name__ == "__main__":
    MODEL_PATH = "model-f46da743.pt"

    # compute depth maps
    run(MODEL_PATH)









