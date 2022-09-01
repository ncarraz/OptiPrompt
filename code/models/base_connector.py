import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

TOKENIZATION = {
    "roberta-base":"bpe",
    "roberta-large":"bpe",
    "allenai/longformer-base-4096":"bpe",
    "allenai/longformer-large-4096":"bpe",
    "distilroberta-base":"bpe",
    "bert-base-cased":"wordpiece",
    "bert-large-cased":"wordpiece",
    "distilbert-base-cased":"wordpiece",
    "facebook/bart-base":"bpe",
    "facebook/bart-large":"bpe",
    "t5-small":"sentencepiece",
    "t5-base":"sentencepiece",
    "t5-large":"sentencepiece",
    "gpt2":"bpe",
    "gpt2-medium":"bpe",
    "gpt2-large":"bpe",
    "gpt2-xl":"bpe",
    "xlnet-base-cased":"sentencepiece",
    "xlnet-large-cased":"sentencepiece",
    "transfo-xl-wt103":"word",
    "google/t5-v1_1-base":"sentencepiece"
}

LM_TYPE = {
     "roberta-base":"masked",
     "roberta-large":"masked",
     "allenai/longformer-base-4096":"masked",
     "allenai/longformer-large-4096":"masked",
     "distilroberta-base":"masked",
     "bert-base-cased":"masked",
     "bert-large-cased":"masked",
     "distilbert-base-cased":"masked",
     "gpt2":"causal",
     "gpt2-medium":"causal",
     "gpt2-large":"causal",
     "gpt2-xl":"causal",
     "xlnet-base-cased":"causal",
     "xlnet-large-cased":"causal",
     "facebook/bart-base":"masked",
     "facebook/bart-large":"masked",
     "t5-small":"seq2seq",
     "t5-base":"seq2seq",
     "t5-large":"seq2seq",
     "google/t5-v1_1-base":"seq2seq"
 }


class Base_Connector():

    def __init__(self):

        # these variables should be initialized
        self.vocab = None
        # This defines where the device where the model is. Changed by try_cuda.
        self._model_device = 'cpu'

    def optimize_top_layer(self, vocab_subset):
        """
        optimization for some LM
        """
        pass

    def update_embeddings(self):
        """Returns the wordpiece embedding module."""
        if self.config.model_type == "bart":
            embeddings = self.model.model.encoder.embed_tokens
        elif self.config.model_type == "gpt2":
            embeddings = self.model.transformer.wte
        elif self.config.model_type == "t5":
            embeddings = self.model.encoder.embed_tokens
        else:
            base_model = getattr(self.model, self.config.model_type)
            embeddings = base_model.embeddings.word_embeddings
        self.embeddings = embeddings

    def _init_inverse_vocab(self):
        self.inverse_vocab = {w: i for i, w in enumerate(self.vocab)}
    
    def _init_vocab(self):
        if self.tokenization in ["bpe", "sentencepiece"]: 
            # Convert vocabulary to BERT
            special_tokens = [self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.unk_token,
                            self.tokenizer.sep_token, self.tokenizer.pad_token, self.tokenizer.cls_token,
                            self.tokenizer.mask_token]
            separator_tokens = {"bpe":"Ġ", "sentencepiece":"▁"}
            sep_token = separator_tokens[self.tokenization]
            converted_vocab = {}
            for w, i in self.tokenizer.vocab.items():
                value = w
                if value[0] == sep_token:  # if the token starts with a whitespace
                    value = value[1:]
                elif value not in special_tokens:
                    # this is subword information
                    value = "_{}_".format(value)

                if value in converted_vocab:
                    # print("WARNING: token '{}' is already in the vocab".format(value))
                    value = "{}_{}".format(value, i)
                converted_vocab[value] = i
        else:
            converted_vocab = self.tokenizer.vocab

        # Compatibility with existing code
        self.vocab = list(dict(sorted(converted_vocab.items(), key=lambda item: item[1])).keys())
        self.inverse_vocab = converted_vocab

    def try_cuda(self):
        """Move model to GPU if one is available."""
        if torch.cuda.is_available():
            if self._model_device != 'cuda':
                print('Moving model to CUDA')
                self._cuda()
                self._model_device = 'cuda'
        else:
            print('No CUDA found')

    def _cuda(self):
        """Move model to GPU."""
        raise NotImplementedError

    def init_indices_for_filter_logprobs(self, vocab_subset, logger=None):
        index_list = []
        new_vocab_subset = []
        for word in vocab_subset:
            if word in self.inverse_vocab:
                inverse_id = self.inverse_vocab[word]
                index_list.append(inverse_id)
                new_vocab_subset.append(word)
            else:
                msg = "word {} from vocab_subset not in model vocabulary!".format(word)
                if logger is not None:
                    logger.warning(msg)
                else:
                    print("WARNING: {}".format(msg))
        indices = torch.as_tensor(index_list)
        return indices, index_list

    def get_id(self, string):
        raise NotImplementedError()

    def get_batch_generation(self, sentences_list, logger= None, try_cuda=True):
        raise NotImplementedError()
    
    def get_loss(self, predict_logits, label_ids):
        predict_logp = F.log_softmax(predict_logits, dim=-1)
        target_logp = predict_logp.gather(-1, label_ids)
        target_logp = target_logp - 1e32 * label_ids.eq(0)  # Apply mask
        target_logp = torch.logsumexp(target_logp, dim=-1)
        return -target_logp
    
    def run_batch(self, sentences_list, samples_list, try_cuda=True, training=True, filter_indices=None, index_list=None, vocab_to_common_vocab=None):
        if try_cuda and torch.cuda.device_count() > 0:
            self.try_cuda()

        input, masked_indices_list, labels_tensor, mlm_label_ids, predict_mask = self.get_input_tensors_batch_train(sentences_list, samples_list)

        if training:
            self.model.train()
            output = self.model(**input.to(self._model_device))
            logits = output.logits
            predict_logits = logits.masked_select(predict_mask.to(self._model_device).unsqueeze(-1)).view(logits.size(0), -1)
            loss = self.get_loss(predict_logits, torch.tensor(mlm_label_ids).unsqueeze(-1).to(self._model_device)).mean()
        else:
            self.model.eval()
            with torch.no_grad():
                output = self.model(**input.to(self._model_device))
                logits = output.logits
                predict_logits = logits.masked_select(predict_mask.to(self._model_device).unsqueeze(-1)).view(logits.size(0), -1)
                loss = self.get_loss(predict_logits, torch.tensor(mlm_label_ids).unsqueeze(-1).to(self._model_device)).mean()
                log_probs = F.log_softmax(logits, dim=-1).cpu()
                pred_log_probs = F.log_softmax(predict_logits, dim=-1).cpu()

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
                #log_prob = log_probs[i][masked_index]
                log_prob = pred_log_probs[i]
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