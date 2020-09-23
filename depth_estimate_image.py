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


def run(model_path):
    """
    Run MonoDepthNN to compute depth maps.
    """
    # Input images
    img_list = os.listdir(args.input)
    img_list.sort()

    # output dir
    output_dir = './depth'
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
        raw_image = cv2.imread(os.path.join(args.input, sample))
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        #  Apply transforms
        image = transform({"image": raw_image})["image"]

        #  Predict and resize to original resolution
        with torch.no_grad():
            image = torch.from_numpy(image).to(device).unsqueeze(0)
            prediction = model.forward(image)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=raw_image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        depth_map = write_depth(prediction, bits=2, reverse=False)

        cv2.imwrite(os.path.join(output_dir, 'MiDaS_{}.png'.format(sample.split('.')[0])), depth_map)


if __name__ == "__main__":
    MODEL_PATH = "model-f46da743.pt"

    # compute depth maps
    run(MODEL_PATH)









