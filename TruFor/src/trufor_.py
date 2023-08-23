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

import sys, os
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob

from PIL import Image
import torch
from torch.nn import functional as F
# from TruFor2.src.config import update_config
# from TruFor2.src.config import _C as config
from .config import update_config
from .config import _C as config
# from data_core import myDataset
from .models.cmx.builder_np_conf import myEncoderDecoder as confcmx
parser = argparse.ArgumentParser(description='Test TruFor')
parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
parser.add_argument('-in', '--input', type=str, default='../images',
                    help='can be a single file, a directory or a glob statement')
parser.add_argument('-out', '--output', type=str, default='./output', help='output folder')
parser.add_argument('-save_np', '--save_np', action='store_true', help='whether to save the Noiseprint++ or not')
parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
update_config(config, args)

# output = args.output

save_np = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(formatter={'float': '{: 7.3f}'.format})

if device != 'cpu':
    # cudnn setting
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

# if '*' in input:
#     list_img = glob(input, recursive=True)
#     list_img = [img for img in list_img if not os.path.isdir(img)]
# elif os.path.isfile(input):
#     list_img = [input]
# elif os.path.isdir(input):
#     list_img = glob(os.path.join(input, '**/*'), recursive=True)
#     list_img = [img for img in list_img if not os.path.isdir(img)]
# else:
#     raise ValueError("input is neither a file or a folder")



def trufor(img):
    save_np = True

    img_RGB = np.array(Image.open(img).convert("RGB"))
    rgb = torch.tensor(img_RGB.transpose(2, 0, 1), dtype=torch.float)
    rgb = rgb.unsqueeze(0)
    # test_dataset = myDataset(list_img=img)

    # testloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=1)  # 1 to allow arbitrary input sizes

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
                print(det_sig)
            if conf is not None:
                out_dict['conf'] = conf
            if save_np:
                out_dict['np++'] = npp
        except:
            import traceback

            traceback.print_exc()
            pass
    return det_sig
