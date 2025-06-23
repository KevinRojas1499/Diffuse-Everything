import torch
import torch.nn as nn
import abc
from transformers import CLIPTokenizer, CLIPTextModel, T5Tokenizer, T5Model, GPT2TokenizerFast

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class Tokenizer(abc.ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def encode(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def decode(self, *args, **kwargs):
        pass

class FrozenCLIP(Tokenizer):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version,clean_up_tokenization_spaces=False)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.max_length = max_length
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self._device = 'cpu'
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def tokenize(self, text, max_length=None, repeat=False):
        max_length = max_length if max_length is not None else self.max_length
        if repeat:
            if isinstance(text, str):
                text = (text + ' ') * 10
            else:
                text = [ (text + ' ') * 10 for text in text]
        batch_encoding = self.tokenizer(text, truncation=True, max_length=max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        return batch_encoding["input_ids"]

    def forward(self, text):
        if isinstance(text, torch.Tensor):
            tokens = text
        else:
            tokens = self.tokenize(text).to(self._device)

        attention_mask = tokens != self.pad_token_id
        outputs = self.transformer(input_ids=tokens, attention_mask=attention_mask)

        z = outputs.last_hidden_state
        return tokens, z

    def encode(self, text):
        return self(text)[1]
    
    def decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
    
    def to(self, device):
        self.transformer = self.transformer.to(device)
        self._device = device
        return self

class FrozenT5(Tokenizer):
    """Uses the T5 transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="t5-large", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5Model.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.masked_token_id = self.tokenizer.unk_token_id
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def tokenize(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        return batch_encoding["input_ids"]

    def forward(self, text):
        if isinstance(text, torch.Tensor):
            tokens = text
        else:
            tokens = self.tokenize(text)
        attention_mask = tokens != self.pad_token_id 

        outputs = self.transformer.to(tokens.device).encoder(input_ids=tokens, attention_mask=attention_mask)

        z = outputs.last_hidden_state
        return tokens, z

    def encode(self, text):
        return self.tokenize(text)
    
    def decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

class FrozenGPT2(Tokenizer):
    """Uses the GPT2 fast tokenizer for text (from Hugging Face)"""
    def __init__(self, version='gpt2-large', device='cuda', max_length=20):
        super().__init__()
        self.tokenizer = GPT2TokenizerFast.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.vocab_size = self.tokenizer.vocab_size
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def tokenize(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        return batch_encoding["input_ids"].to(self.device)

    def encode(self, text):
        return None

    def decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)



class CharacterTokenizer(Tokenizer):
    def __init__(self, full_text):
        super().__init__()
        chars = sorted(list(set(full_text)))
        self.int_to_string = { i:ch for i,ch in enumerate(chars) }
        self.string_to_int = { ch:i for i,ch in enumerate(chars) }

    def _encode_one(self, text):
        return [self.stoi[c] for c in text]
    
    def encode(self, texts, return_tensor=True):
        encoded = [self._encode_one(text) for text in texts]
        return torch.tensor(encoded) if return_tensor else encoded
        
    def _decode_one(self, tokens):
        return ''.join([self.itos[i] for i in tokens]) 
       
    def decode(self, tokens_list):
        return [''.join([self.itos[i] for i in tokens]) for tokens in tokens_list]
    

def get_tokenizer(tokenizer_name, full_text=None, block_size=77):
    if tokenizer_name == 'clip':
        return FrozenCLIP(max_length=block_size)
    elif tokenizer_name == 't5':
        return FrozenT5(max_length=block_size)
    elif tokenizer_name == 'character':
        return CharacterTokenizer(full_text)
    elif tokenizer_name == 'gpt2':
        return FrozenGPT2(max_length=block_size)