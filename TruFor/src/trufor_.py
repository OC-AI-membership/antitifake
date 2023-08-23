# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

"""
Created in September 2022
@author: fabrizio.guillaro
"""

import io
import argparse
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .config import update_config
from .config import _C as config
from .models.cmx.builder_np_conf import myEncoderDecoder as confcmx
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Test TruFor')
parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
parser.add_argument('-in', '--input', type=str, default='../images',
                    help='can be a single file, a directory or a glob statement')
parser.add_argument('-out', '--output', type=str, default='./output', help='output folder')
parser.add_argument('-save_np', '--save_np', action='store_true', help='whether to save the Noiseprint++ or not')
parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
update_config(config, args)

save_np = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(formatter={'float': '{: 7.3f}'.format})

if device != 'cpu':
    # cudnn setting
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

def trufor(img):
    save_np = True

    img_RGB = np.array(Image.open(img).convert("RGB"))
    rgb = torch.tensor(img_RGB.transpose(2, 0, 1), dtype=torch.float) / 255.0
    rgb = rgb.unsqueeze(0)

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        raise ValueError("Model file is not specified.")

    print('=> loading model from {}'.format(model_state_file))
    checkpoint = torch.load(model_state_file, map_location=torch.device(device))

    if config.MODEL.NAME == 'detconfcmx':
        
        model = confcmx(cfg=config)
    else:
        raise NotImplementedError('Model not implemented')

    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    with torch.no_grad():
        try:
            rgb = rgb.to(device)
            model.eval()

            det = None
            conf = None

            pred, conf, det, npp = model(rgb)

            if conf is not None:
                conf = torch.squeeze(conf, 0)
                conf = torch.sigmoid(conf)[0]
                conf = conf.cpu().numpy()

            if npp is not None:
                npp = torch.squeeze(npp, 0)[0]
                npp = npp.cpu().numpy()

            if det is not None:
                det_sig = torch.sigmoid(det).item()

            pred = torch.squeeze(pred, 0)
            pred = F.softmax(pred, dim=0)[1]
            pred = pred.cpu().numpy()

            out_dict = dict()
            out_dict['map'] = pred
            out_dict['imgsize'] = tuple(rgb.shape[2:])
            if det is not None:
                out_dict['score'] = det_sig
                det_sig_rd = round(det_sig,3)
            if conf is not None:
                out_dict['conf'] = conf
            if save_np:
                out_dict['np++'] = npp
        except:
            import traceback

            traceback.print_exc()
            pass

        
        plt.imshow(pred, cmap='RdBu_r', clim=[0,1])
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image_pil = Image.open(buf)

    return det_sig_rd, image_pil
