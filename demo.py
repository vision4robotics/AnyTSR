import argparse
import os
import math
from PIL import Image
from torchvision.transforms import InterpolationMode
import torch
from torchvision import transforms

import models
from utils import make_coord

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/GT')
    parser.add_argument('--model',default="weights/checkpoints.pth")
    parser.add_argument('--scale',default=6)
    parser.add_argument('--output', default='output')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_spec = torch.load(args.model)
    model = models.make(model_spec, load_sd=True).cuda()

    input_files = [f for f in os.listdir(args.input) if f.lower().endswith(('jpeg', 'jpg', 'png'))]

    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    for img_file in input_files:
        file = os.path.join(args.input, img_file)
        img = transforms.ToTensor()(
                Image.open(file).convert('L'))

        h_lr = math.floor(img.shape[-2] / args.scale + 1e-9)
        w_lr = math.floor(img.shape[-1] / args.scale + 1e-9)
        h_hr = round(h_lr * args.scale)
        w_hr = round(w_lr * args.scale)
        img = img[:, :h_hr, :w_hr]
        img_down = resize_fn(img, (h_lr, w_lr))
        crop_lr, crop_hr = img_down, img

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr

        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        batch = {'scale':torch.tensor([h_hr/h_lr]),
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb}

        t = {"sub": [0.5], "div": [0.5]}
        inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
        inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
        t = {"sub": [0.5], "div": [0.5]}
        gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
        gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        inp = (batch['inp'] - inp_sub) / inp_div
        coord = batch['coord'].unsqueeze(0)
        cell = batch['cell'].unsqueeze(0)
        sr_scale = batch['scale'].unsqueeze(0)

        model.eval()

        with torch.no_grad():
            pred = model(inp, coord, cell, sr_scale).squeeze(0)
            pred = pred * gt_div + gt_sub
            pred.clamp_(0, 1)

            base_name, ext = os.path.splitext(img_file)

            output_file = os.path.join(output_dir, f"AnyTSR_{base_name}_x{args.scale}{ext}")  

            outputimg = transforms.ToPILImage()(pred)
            transforms.ToPILImage()(pred).save(output_file)
