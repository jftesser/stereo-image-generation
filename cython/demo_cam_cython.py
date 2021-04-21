import time
import torch
import cv2
import numpy as np

import shifting_pixel

from torch.backends import cudnn
from torchvision.transforms import Compose

from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet


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


def generate_stereo(left_image, depth):
    h, w, c = left_image.shape
    depth_min = depth.min()
    depth_max = depth.max()
    depth = (depth - depth_min) / (depth_max - depth_min)

    right_image = np.zeros_like(left_image)

    deviation_cm = IPD * 0.12
    deviation = deviation_cm * MONITOR_W * (w / 1920)

    # t_shifting = time.time()
    right_image = np.array(shifting_pixel.shift(left_image, right_image, depth, deviation))
    # print('\rImage Shifting: %f sec ' % (time.time() - t_shifting), end='')

    # t_inpainting = time.time()
    gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    rows, cols = np.where(gray == 0)
    mask = np.zeros((gray.shape[:2]), dtype=np.uint8)
    for row, col in zip(rows, cols):
        mask[row, col] = 255
    right_fix = cv2.inpaint(right_image, mask, 2, cv2.INPAINT_NS)
    # print('\rImage Inpainting: %f sec ' % (time.time() - t_inpainting), end='')



    return right_fix


def run(model_path):
    """
    Run MonoDepthNN to compute depth maps.
    """
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

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cam.set(cv2.CAP_PROP_FPS, 30)

    while True:
        t_total = time.time()
        _, left_img = cam.read()

        # t_depth_est = time.time()
        image = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB) / 255.0

        #  Apply transforms
        image = transform({"image": image})["image"]

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

        depth_map = write_depth(depth, bits=2, reverse=False)
        # print('\rDepth Estimation: %f sec ' %(time.time() - t_depth_est), end='')

        right_img = generate_stereo(left_img, depth_map)
        # stereo = np.hstack([left_img, right_img])
        # cv2.imshow("stereo", stereo)



        composite = np.zeros([left_img.shape[0], left_img.shape[1], 3], dtype=np.uint8)
        anaglyph = np.array(shifting_pixel.overlap(left_img, right_img, composite))
        # anaglyph = overlap(left_img, right_img)
        # cv2.imshow("anaglyph", anaglyph)


        demo = np.hstack([left_img, right_img, anaglyph])
        cv2.imshow("demo", demo)

        fps = 1. / (time.time() - t_total)
        print('\rframerate: %f fps' % fps, end='')
        cv2.waitKey(1)


if __name__ == "__main__":
    MODEL_PATH = "model-f46da743.pt"

    # compute depth maps
    run(MODEL_PATH)

