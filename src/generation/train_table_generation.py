import torch
import os
import time
import json
import deepspeed
from transformers import VisionEncoderDecoderModel
from transformers import NougatTokenizerFast, NougatImageProcessor
from dataset import GenerationDataset
from transformers import get_cosine_schedule_with_warmup
from transformers.deepspeed import HfDeepSpeedConfig

os.environ["TOKENIZERS_PARALLELISM"] = 'false'
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

ds_config = json.load(open('ds_config.json', 'r'))
dschf = HfDeepSpeedConfig(ds_config)
torch.manual_seed(7)

epochs = 2
batch_size = 32

model = VisionEncoderDecoderModel.from_pretrained('facebook/nougat-base')
model.gradient_checkpointing_enable()
tokenizer = NougatTokenizerFast.from_pretrained('facebook/nougat-base')
image_processor = NougatImageProcessor.from_pretrained('facebook/nougat-base')
image_processor.size = {'height': 672, 'width': 1344}
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print('Model Loaded, Start Loading Data, Parameters: {}'.format(sum(p.numel() for p in model.parameters())))

train_dataset = GenerationDataset(
    tokenizer, image_processor, '../../tab2latex/train.jsonl', 
    '../../tab2latex/train/', max_len=1024
)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=1000, num_training_steps=int(epochs * len(train_dataset) / batch_size)
)

engine, _, train_dataloader, _ = deepspeed.initialize(
    model=model, training_data=train_dataset, config_params=ds_config, optimizer=optimizer, lr_scheduler=scheduler)
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print('Train Dataset: {}, Train Dataloader: {}'.format(len(train_dataset), len(train_dataloader)))

engine.module.train()
for epoch in range(epochs):
    start_time = time.time()
    train_loss = []
    for i, data in enumerate(train_dataloader):
        input_ids = data['input_ids'].to(engine.device)
        attention_mask = data['attention_mask'].to(engine.device)
        labels = data['labels'].to(engine.device)
        pixel_values = data['pixel_values'].to(engine.device)
        
        loss = engine.module(pixel_values=pixel_values, decoder_input_ids=input_ids, decoder_attention_mask=attention_mask, labels=labels).loss
        engine.backward(loss.mean())
        engine.step()
        train_loss.append(loss.mean().item())
        
        if i % 1000 == 0:
            torch.cuda.empty_cache()
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print('epoch: {}, step: {}/{}, loss: {}, lr: {}, time: {}s'.format(
                    epoch + 1, i, len(train_dataloader), round(sum(train_loss) / len(train_loss), 4), 
                    round(engine.optimizer.param_groups[0]['lr'], 8), int(time.time() - start_time)
                ))
            start_time = time.time()
            train_loss = []

    os.makedirs('../../models/ds_ckpts/', exist_ok=True)
    engine.save_checkpoint('../../models/ds_ckpts/')
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print('checkpoint saved')
