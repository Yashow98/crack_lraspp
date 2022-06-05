import argparse
import os
import time
import json

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import lraspp_mobilenetv3_large


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(args):
    classes = 1
    weights_path = "./save_weights/best_model.pth"
    img_path = args.test_path
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    # create model
    model = lraspp_mobilenetv3_large(num_classes=classes + 1)

    # load weights
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    model.load_state_dict(weights_dict)
    model.to(device)

    # load image
    original_img = Image.open(img_path)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.Resize(520),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print(f"inference+NMS time: {t_end - t_start}")

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255
        # 将不敢兴趣的区域像素设置成0(黑色)
        prediction[prediction == 0] = 0
        mask = Image.fromarray(prediction)
        mask.save("test_result.png")


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch lr-aspp predicting")
    # parser.add_argument('--model-flag', default='unet', type=str, help='model class flag')
    parser.add_argument('--test-path', default='./DRIVE/test/images/007.jpg', type=str, help='test image path')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
