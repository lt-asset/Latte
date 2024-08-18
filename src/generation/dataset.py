import json
import os
import copy
import torch
from PIL import Image
import random

random.seed(7)

class GenerationDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, image_processor, data_file, image_folder, max_len=512):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        with open(data_file, 'r') as fp:
            self.data = [json.loads(l) for l in fp.readlines()]
            
        self.image_folder = image_folder
        self.max_len = max_len
        
    def __getitem__(self, idx):
        latex = self.data[idx]['latex']
        image_file = str(self.data[idx]['image'])
        
        input_ids = self.tokenizer.encode(latex)
        labels = copy.deepcopy(input_ids)
        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]
            labels = labels[: self.max_len]
        else:
            input_ids += [1] * (self.max_len - len(input_ids))
            labels += [-100] * (self.max_len - len(labels))
        
        input_ids = torch.LongTensor(input_ids[:-1])
        labels = torch.LongTensor(labels[1:])
        
        attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.masked_fill(input_ids.eq(1), 0.0)

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].squeeze(0).to(dtype=torch.bfloat16)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pixel_values': image_tensor
        }
    
    def __len__(self):
        return len(self.data)


class RepairDiffImageBugLocDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, image_processor, data_file, image_folder, max_len=1024):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_folder = image_folder
        self.max_len = max_len
        
        content = json.load(open(data_file, 'r'))
        self.data = []
        for k in content:
            for item in content[k]:
                self.data.append({'image': item['diff_image'], 'latex': item['repair_target'], 'wrong_latex': item['output'], 'loc': item['loc']})
        random.shuffle(self.data)
        
    def __getitem__(self, idx):
        wrong_latex = self.data[idx]['wrong_latex']
        latex = self.data[idx]['latex']
        loc = self.data[idx]['loc']
        image_file = str(self.data[idx]['image'])
        
        wrong_latex = self.tokenizer.encode(wrong_latex)[:-1]
        latex = self.tokenizer.encode(latex)
        
        prompts = wrong_latex[loc: ] + latex[: loc]
        labels = latex[loc: ]

        input_ids = prompts + labels
        labels = [-100] * len(prompts) + labels
        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]
            labels = labels[: self.max_len]
        else:
            input_ids += [1] * (self.max_len - len(input_ids))
            labels += [-100] * (self.max_len - len(labels))

        input_ids = torch.LongTensor(input_ids[:-1])
        labels = torch.LongTensor(labels[1:])
        
        attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.masked_fill(input_ids.eq(1), 0.0)

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].squeeze(0).to(dtype=torch.bfloat16)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pixel_values': image_tensor
        }
    
    def __len__(self):
        return len(self.data)
