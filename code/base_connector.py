import re
import torch

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
    "xlnet-base-cased":"sentencepiece",
    "xlnet-large-cased":"sentencepiece",
    "transfo-xl-wt103":"word"
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
     "xlnet-base-cased":"causal",
     "xlnet-large-cased":"causal",
     "facebook/bart-base":"masked",
     "facebook/bart-large":"masked",
     "t5-small":"seq2seq",
     "t5-base":"seq2seq",
     "t5-large":"seq2seq"
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