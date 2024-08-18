import torch
from transformers import VisionEncoderDecoderModel, NougatTokenizerFast, NougatImageProcessor
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', type=str, help='The folder to deepspeed zero012 checkpoints')
parser.add_argument('--output_dir', type=str, help='The folder to the huggingface checkpoint')
parser.add_argument('--image_width', type=int, default=1344, help='The width resolution of rendered images')
parser.add_argument('--image_height', type=int, default=224, help='The height resolution of the rendered images.')

args = parser.parse_args()

model = VisionEncoderDecoderModel.from_pretrained('facebook/nougat-base')
model.load_state_dict(
    torch.load(f'{args.ckpt_dir}/mp_rank_00_model_states.pt', map_location='cpu')['module']
)
model.save_pretrained(args.output_dir)

tokenizer = NougatTokenizerFast.from_pretrained('facebook/nougat-base')
image_processor = NougatImageProcessor.from_pretrained('facebook/nougat-base')
image_processor.size = {'height': args.image_height, 'width': args.image_width}

tokenizer.save_pretrained(args.output_dir)
image_processor.save_pretrained(args.output_dir)
