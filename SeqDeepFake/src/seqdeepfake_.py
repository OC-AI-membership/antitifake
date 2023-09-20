import torch
from PIL import Image
from SeqDeepFake.models import SeqFakeFormer
from torchvision import transforms
from SeqDeepFake.models.configuration import Config

import warnings
warnings.filterwarnings(action='ignore')

model_checkpoint = './SeqDeepFake/results/best_model_adaptive.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_config = Config('./SeqDeepFake/configs/r50.json')

model = SeqFakeFormer.build_model(model_config)
model.load_state_dict(torch.load(model_checkpoint, map_location=device)['best_state_dict_adaptive'])
model.to(device)
model.eval()


def create_caption_and_mask(cfg):
    caption_template = cfg.PAD_token_id * torch.ones((1, cfg.max_position_embeddings), dtype=torch.long).cuda()
    mask_template = torch.ones((1, cfg.max_position_embeddings), dtype=torch.bool).cuda()

    caption_template[:, 0] = cfg.SOS_token_id
    mask_template[:, 0] = False

    return caption_template, mask_template


mapping = {
    0: 'None',
    1: 'bangs',
    2: 'eyeglasses',
    3: 'beard',
    4: 'smiling',
    5: 'young',
    6: 'nose',
    7: 'eye',
    8: 'eyebrow',
    9: 'lip',
    10: 'hair'
}

mapping_results = {
    'eyeglasses': 'eyes',
    'smiling': 'mouth',
    'young': 'face',
    'lib': 'mouth',
}


def model_predict(image_path):
    pil_image = Image.open(image_path)
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = image_transform(pil_image).unsqueeze(0).to(device)

    # Generate a caption using the model
    caption, cap_mask = create_caption_and_mask(model_config)
    for i in range(model_config.max_position_embeddings - 1):
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == model_config.EOS_token_id:
            caption = caption[:, 1:]
            zero = torch.zeros_like(caption)
            caption = torch.where(caption == model_config.PAD_token_id, zero, caption)
            break

        caption[:, i + 1] = predicted_id[0]
        cap_mask[:, i + 1] = False

    if caption.shape[1] == 6:
        caption = caption[:, 1:]

    results = caption[0].tolist()

    true_false = 'False' if all(val == 0 for val in results) else 'True'
    mapped = [mapping[val] for val in results if val != 0]
    mapped_results = [mapping_results[result] if result in mapping_results else result for result in mapped]
    result_string = '. '.join(mapped_results) + '.'

    return true_false, result_string, mapped
