import torch
import numpy as np
import pandas as pd
from base_connector import *
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoConfig
import random


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
        output = self.tokenizer(sentences_list, padding=True, return_tensors="pt")

        if self.model_type == "masked":
            masked_indices_list = np.argwhere(output.input_ids.numpy() == self.tokenizer.mask_token_id)[:,1]
        elif self.model_type == "seq2seq":
            masked_indices_list = [1] * len(sentences_list) # second generated token is always mask

        masked_indices_list = [[i] for i in masked_indices_list]

        with torch.no_grad():
            if self.model_type == "seq2seq":
                scores = self.model.generate(output.input_ids.to(self._model_device), 
                                                    max_new_tokens=2, output_scores=True, return_dict_in_generate=True).scores
                scores = torch.stack(scores, dim=1)
            else:
                scores = self.model(**output.to(self._model_device)).logits
            log_probs = F.log_softmax(scores, dim=-1).cpu()
        # second returned value is off for seq2seq
        return log_probs, list(output.input_ids.cpu().numpy()), masked_indices_list
    
    def get_input_tensors_batch_train(self, sentences_list, samples_list):
        if not sentences_list:
            return None

        # Compatibility with existing code
        sentences_list = [item for sublist in sentences_list for item in sublist]
        output = self.tokenizer(sentences_list, padding=True, return_tensors="pt")

        if self.model_type == "masked":
            masked_indices_list = np.argwhere(output.input_ids.numpy() == self.tokenizer.mask_token_id)[:,1]
        elif self.model_type == "seq2seq":
            masked_indices_list = [1] * len(sentences_list) # second generated token is always mask

        masked_indices_list = [[i] for i in masked_indices_list]

        #Optiprompt specific
        sample = pd.DataFrame(samples_list)
        sample["obj_label"] = " " + sample["obj_label"]
        sample["token_id"] = sample["obj_label"].apply(lambda x: self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))[0])
        labels = sample["token_id"].values

        labels_tensor = torch.full_like(output.attention_mask, -100)
        predict_mask = output.input_ids.eq(self.tokenizer.mask_token_id)
        labels_tensor[predict_mask] = torch.tensor(labels)

        return output ,masked_indices_list, labels_tensor, labels.tolist()
    
    def run_batch(self, sentences_list, samples_list, try_cuda=True, training=True, filter_indices=None, index_list=None, vocab_to_common_vocab=None):
        if try_cuda and torch.cuda.device_count() > 0:
            self.try_cuda()

        input, masked_indices_list, labels_tensor, mlm_label_ids = self.get_input_tensors_batch_train(sentences_list, samples_list)

        if training:
            self.model.train()
            output = self.model(**input.to(self._model_device), labels=labels_tensor.to(self._model_device))
            loss = output[0]
        else:
            self.model.eval()
            with torch.no_grad():
                output = self.model(**input.to(self._model_device), labels=labels_tensor.to(self._model_device))
                loss = output.loss
                logits = output.logits
            log_probs = F.log_softmax(logits, dim=-1).cpu()

        if training:
            return loss
        else:
            # During testing, return accuracy and top-k predictions
            tot = log_probs.shape[0]
            cor = 0
            preds = []
            topk = []
            common_vocab_loss = []

            for i in range(log_probs.shape[0]):
                masked_index = masked_indices_list[i][0]
                log_prob = log_probs[i][masked_index]
                mlm_label = mlm_label_ids[i]
                if filter_indices is not None:
                    log_prob = log_prob.index_select(dim=0, index=filter_indices)
                    pred_common_vocab = torch.argmax(log_prob)
                    pred = index_list[pred_common_vocab]

                    # get top-k predictions
                    topk_preds = []
                    topk_log_prob, topk_ids = torch.topk(log_prob, self.k)
                    for log_prob_i, idx in zip(topk_log_prob, topk_ids):
                        ori_idx = index_list[idx]
                        token = self.vocab[ori_idx]
                        topk_preds.append({'token': token, 'log_prob': log_prob_i.item()})
                    topk.append(topk_preds)

                    # compute entropy on common vocab
                    common_logits = logits[i][masked_index].cpu().index_select(dim=0, index=filter_indices)
                    common_log_prob = -F.log_softmax(common_logits, dim=-1)
                    common_label_id = vocab_to_common_vocab[mlm_label]
                    common_vocab_loss.append(common_log_prob[common_label_id].item())
                else:
                    pred = torch.argmax(log_prob)
                    topk.append([])
                if pred == labels_tensor[i][masked_index]:
                    cor += 1
                    preds.append(1)
                else:
                    preds.append(0)
                            
            return log_probs, cor, tot, preds, topk, loss, common_vocab_loss 

