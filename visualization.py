import os
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

from options import VisualizeOptions


def visualize():
    opt = VisualizeOptions().parse()
    assert sorted(os.listdir(opt.image_dir)) == ['no_stairs', 'stairs']
    
    random.seed(opt.seed)
    model = torch.load(opt.pretrained_model)
    model.to(opt.device)
    model.eval()
    
    transform = transforms.Compose([
            transforms.Resize(size=opt.image_size),
            transforms.CenterCrop(size=opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    if opt.layer:
        cam_extractor = SmoothGradCAMpp(model, opt.layer, input_shape=(3, opt.image_size, opt.image_size))
    else:
        cam_extractor = SmoothGradCAMpp(model, input_shape=(3, opt.image_size, opt.image_size))
    
    for i, label in enumerate(os.listdir(opt.image_dir)):
        image_set = os.listdir(os.path.join(opt.image_dir, label))
        original_img = random.choice(image_set)
        original_img = Image.open(os.path.join(opt.image_dir, label, original_img))
        
        img = transform(original_img)
        img = img.to(opt.device).unsqueeze(0)
        output = model(img)
        label_pred = torch.argmax(output, dim=1)
        activation_map = cam_extractor(torch.argmax(output, dim=1).item(), label)
        label_pred = label_pred.detach().cpu().numpy()
        label_pred = 'no_stairs' if label_pred[0] == 0 else 'stairs'
        result = overlay_mask(original_img, to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
        
        plt.subplot(2, 2, i+1)
        plt.axis('off')
        plt.title(f'pred: {label_pred}\n real: {label}', fontsize='small')
        plt.imshow(original_img)
        plt.subplot(2, 2, i+3)
        plt.axis('off')
        plt.title(f'Heatmap')
        plt.imshow(result)

    plt.tight_layout()
    plt.savefig(os.path.join(opt.save_dir, opt.save_name))


if __name__ == '__main__':
    visualize()
