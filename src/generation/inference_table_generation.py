import torch
import json
from tqdm import tqdm
from PIL import Image
import argparse
from transformers import VisionEncoderDecoderModel
from transformers import NougatTokenizerFast, NougatImageProcessor


def inference_generation(model_path, input_path, image_dir, output_path):
    model = VisionEncoderDecoderModel.from_pretrained(model_path).cuda()
    tokenizer = NougatTokenizerFast.from_pretrained(model_path)
    processor = NougatImageProcessor.from_pretrained(model_path)

    model.eval()
    result = {}
    with open(input_path, 'r') as fp:
        L = fp.readlines()
        for i, l in tqdm(enumerate(L)):
            item = json.loads(l)
            image = Image.open(f"{image_dir}/{item['image']}").convert('RGB')
            pixel_values = processor(image, return_tensors="pt").pixel_values.cuda()
            decoder_input_ids = tokenizer(tokenizer.bos_token, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
            
            with torch.no_grad():
                outputs = model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=2048,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=2,
                    do_sample=False,
                    num_return_sequences=1
                )
            output = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            result[item['id']] = {
                'image': item['image'], 'latex': item['latex'], 'latte_1': output
            }

        json.dump(result, open(output_path, 'w'), indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='The folder to the generation model')
    parser.add_argument('--input_path', type=str, help='The path to the test.jsonl file')
    parser.add_argument('--image_dir', type=str, help='The folder of the rendered ground-truth images')
    parser.add_argument('--output_path', type=str, help='The file to store the generation result')

    args = parser.parse_args()

    inference_generation(
        model_path=args.model_path, input_path=args.input_path, image_dir=args.input_dir, output_path=args.output_path
    )
