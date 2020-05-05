#gpt_dwnld_HF.py
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

#device = xm.xla_device()
device=torch.device("cuda")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
print('saving models and tokenizer')
tokenizer.save_pretrained('/spell/GPT2Model')
model.save_pretrained('/spell/GPT2Model')
print('model and tokenizer saved')