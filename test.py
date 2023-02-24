import unittest
import torch
from PIL import Image
from optim_utils import read_json, optimize_prompt
import open_clip

class TestOptimizePrompt(unittest.TestCase):
    def setUp(self):
        self.image_paths = ['test_images/image1.jpg', 'test_images/image2.jpg']
        self.images = [Image.open(image_path) for image_path in self.image_paths]
        self.args = argparse.Namespace()
        self.args.__dict__.update(read_json("sample_config.json"))
        self.args.print_new_best = True
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.args.clip_model, pretrained=self.args.clip_pretrain, device=self.DEVICE)

    def test_optimize_prompt(self):
        prompt = optimize_prompt(self.model, self.preprocess, self.args, self.DEVICE, target_images=self.images)
        self.assertIsNotNone(prompt)
        self.assertIsInstance(prompt, str)

if __name__ == '__main__':
    unittest.main()
