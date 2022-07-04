import torch
import numpy as np
import pandas as pd
from .base_connector import *
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoConfig


class MaskedLM(Base_Connector):

    def __init__(self, args):
        super().__init__()

        self.model_name = args.model_name
        self.tokenization = TOKENIZATION[self.model_name]
        self.model_type = LM_TYPE[self.model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = AutoConfig.from_pretrained(self.model_name)
        self._init_vocab() # Compatibility with existing code
        
        if self.model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.MASK = "<extra_id_0>" # for t5 only for now 
            self.tokenizer.mask_token = self.MASK
        elif self.model_type == "masked":
            self.MASK = self.tokenizer.mask_token
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.eval() # EVAL ONLY ?

        self.EOS = self.tokenizer.eos_token
        self.CLS = self.tokenizer.cls_token
        self.SEP = self.tokenizer.sep_token
        self.UNK = self.tokenizer.unk_token
        self.pad_id = self.tokenizer.pad_token_id
        self.unk_index = self.tokenizer.unk_token_id

        # used to output top-k predictions
        self.k = args.k

    def _cuda(self):
        self.model.cuda()
    
    def get_id(self, string):
        if "bpe" in self.tokenization:
            string = " " + string  
        tokenized_text = self.tokenizer.tokenize(string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return indexed_string

    def get_batch_generation(self, sentences_list, logger= None, try_cuda=True):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        # Compatibility with existing code
        sentences_list = [item for sublist in sentences_list for item in sublist]
        input = self.tokenizer(sentences_list, padding=True, return_tensors="pt")
        masked_indices_list = np.argwhere(input.input_ids.numpy() == self.tokenizer.mask_token_id)[:,1]
        masked_indices_list = [[i] for i in masked_indices_list]
        if 't5' in self.model_name:
            input['labels'] = input['input_ids']
        with torch.no_grad():
            scores = self.model(**input.to(self._model_device)).logits
            log_probs = F.log_softmax(scores, dim=-1).cpu()
        # second returned value is off for seq2seq
        return log_probs, list(input.input_ids.cpu().numpy()), masked_indices_list
    
    def get_input_tensors_batch_train(self, sentences_list, samples_list):
        if not sentences_list:
            return None

        # Compatibility with existing code
        sentences_list = [item for sublist in sentences_list for item in sublist]
        input = self.tokenizer(sentences_list, padding=True, return_tensors="pt")
        masked_indices_list = np.argwhere(input.input_ids.numpy() == self.tokenizer.mask_token_id)[:,1]
        masked_indices_list = [[i] for i in masked_indices_list]
        if 't5' in self.model_name:
            input['labels'] = input['input_ids']

        #Optiprompt specific
        sample = pd.DataFrame(samples_list)
        sample["obj_label"] = " " + sample["obj_label"]
        sample["token_id"] = sample["obj_label"].apply(lambda x: self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))[0])
        labels = sample["token_id"].values

        labels_tensor = torch.full_like(input.attention_mask, -100)
        predict_mask = input.input_ids.eq(self.tokenizer.mask_token_id)
        labels_tensor[predict_mask] = torch.tensor(labels)

        return input ,masked_indices_list, labels_tensor, labels.tolist(), predict_mask

