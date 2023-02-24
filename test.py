import sys
from PIL import Image
import torch
from run import optimize_prompt
#ww
def test_optimize_prompt():
    # Load test image
    test_image = Image.new('RGB', (256, 256), color='white')

    # Define test args
    args = argparse.Namespace()
    args.iter = 10
    args.clip_model = 'ViT-B/32'
    args.clip_pretrain = True
    args.print_step = 5
    args.learning_rate = 0.1

    # Load CLIP model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrain, device=DEVICE)

    # Optimize prompt
    learned_prompt = optimize_prompt(
        model,
        preprocess,
        args,
        DEVICE,
        target_images=[test_image])

    # Assert that the output is a torch.Tensor
    assert isinstance(learned_prompt, torch.Tensor)

    
