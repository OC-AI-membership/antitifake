import torch
from PIL import Image
from SeqDeepFake.models import SeqFakeFormer
from torchvision import transforms
from SeqDeepFake.models.configuration import Config

import warnings
warnings.filterwarnings(action='ignore')

attribute_checkpoint = './SeqDeepFake/results/resnet50/facial_attributes/log20230731_053559/snapshots/best_model_adaptive.pt'
component_checkpoint = './SeqDeepFake/results/resnet50/facial_components/log20230801_005345/snapshots/best_model_adaptive.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_config = Config('./SeqDeepFake/configs/r50.json')

attribute_model = SeqFakeFormer.build_model(model_config)
attribute_model.load_state_dict(torch.load(attribute_checkpoint, map_location=device)['best_state_dict_adaptive'])
attribute_model.to(device)
attribute_model.eval()

component_model = SeqFakeFormer.build_model(model_config)
component_model.load_state_dict(torch.load(component_checkpoint, map_location=device)['best_state_dict_adaptive'])
component_model.to(device)
component_model.eval()


def create_caption_and_mask(cfg):
    caption_template = cfg.PAD_token_id * torch.ones((1, cfg.max_position_embeddings), dtype=torch.long).cuda()
    mask_template = torch.ones((1, cfg.max_position_embeddings), dtype=torch.bool).cuda()

    caption_template[:, 0] = cfg.SOS_token_id
    mask_template[:, 0] = False

    return caption_template, mask_template


attributes_mapping = {
    0: 'NA',
    1: 'Bangs',
    2: 'Eyeglasses',
    3: 'Beard',
    4: 'Smiling',
    5: 'Young'
}
attributes_results = {
    'Eyeglasses': 'Eyes',
    'Smiling': 'Mouth',
    'Young': 'Face'
}

components_mapping = {
    0: 'NA',
    1: 'nose',
    2: 'eye',
    3: 'eyebrow',
    4: 'lib',
    5: 'hair'
}


def attribute_predict(image_path):
    pil_image = Image.open(image_path)
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = image_transform(pil_image).unsqueeze(0).to(device)

    # Generate a caption using the model
    caption, cap_mask = create_caption_and_mask(model_config)
    for i in range(model_config.max_position_embeddings - 1):
        predictions = attribute_model(image, caption, cap_mask)
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

    true_or_fake = 'False' if all(val == 0 for val in results) else 'True'
    mapped = [attributes_mapping[val] for val in results if val != 0]
    mapped_results = [attributes_results[result] if result in attributes_results else result for result in mapped]

    # Mapping num - attribute
    result_string = '. '.join(mapped_results) + '.'

    return true_or_fake, result_string, mapped


def component_predict(image_path):
    pil_image = Image.open(image_path)
    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = image_transform(pil_image).unsqueeze(0).to(device)

    # Generate a caption using the model
    caption, cap_mask = create_caption_and_mask(model_config)
    for i in range(model_config.max_position_embeddings - 1):
        predictions = component_model(image, caption, cap_mask)
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

    true_or_fake = 'False' if all(val == 0 for val in results) else 'True'
    mapped_results = [components_mapping[val] for val in results if val != 0]

    # Mapping num - component
    result_string = '. '.join(mapped_results) + '.'

    return true_or_fake, result_string, mapped_results
