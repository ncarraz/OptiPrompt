import json
import argparse
import os
import random
import sys
import logging
from tqdm import tqdm
import torch
import numpy as np

from models import build_model_by_name
from utils import load_vocab, load_data, batchify, save_model, evaluate, get_relation_meta
from run_optiprompt import prepare_for_dense_prompt, init_template


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def load_optiprompt(model, output_dir, original_vocab_size):
    prepare_for_dense_prompt(model)
    logger.info("Loading OptiPrompt's [V]s..")
    with open(os.path.join(output_dir, 'prompt_vecs.npy'), 'rb') as f:
        vs = np.load(f)
    
    # copy fine-tuned new_tokens to the pre-trained model
    with torch.no_grad():
        model.embeddings.weight[original_vocab_size:] = torch.Tensor(vs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base-cased', help='the huggingface model name')
    parser.add_argument('--source_dir', type=str, default=None, help='the directory containing the saved promps')
    parser.add_argument('--target_dir', type=str, default='results', help='the output directory to store the prediction results')
    parser.add_argument('--common_vocab_filename', type=str, default='common_vocabs/common_vocab_cased.txt', help='common vocabulary of models (used to filter triples)')
    parser.add_argument('--relation_profile', type=str, default='relation_metainfo/LAMA_relations.jsonl', help='meta infomation of 41 relations, containing the pre-defined templates')

    parser.add_argument('--test_data_dir', type=str, default="data/filtered_LAMA")
    parser.add_argument('--eval_batch_size', type=int, default=8)

    parser.add_argument('--seed', type=int, default=6)

    parser.add_argument('--random_init', type=str, default='none', choices=['none', 'embedding', 'all'], help='none: use pre-trained model; embedding: random initialize the embedding layer of the model; all: random initialize the whole model')

    parser.add_argument('--num_vectors', type=int, default=5, help='how many dense vectors are used in OptiPrompt')
    parser.add_argument('--init_manual_template', action='store_true', help='whether to use manual template to initialize the dense vectors')
    parser.add_argument('--k', type=int, default=5, help='how many predictions will be outputted')

    args = parser.parse_args()

    logger.info(args)
    n_gpu = torch.cuda.device_count()
    logger.info('# GPUs: %d'%n_gpu)
    if n_gpu == 0:
        logger.warning('No GPU found! exit!')

    logger.info('Model: %s'%args.model_name)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(args.seed)
    
    model = build_model_by_name(args)
    original_vocab_size = len(list(model.tokenizer.get_vocab()))
    logger.info('Original vocab size: %d'%original_vocab_size)
    prepare_for_dense_prompt(model)
    
    if args.common_vocab_filename is not None:
        vocab_subset = load_vocab(args.common_vocab_filename)
        logger.info('Common vocab: %s, size: %d'%(args.common_vocab_filename, len(vocab_subset)))
        filter_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)
    else:
        filter_indices = None
        index_list = None
    
    if n_gpu > 1:
            model.model = torch.nn.DataParallel(model.model)
    
    for relation in os.listdir(args.test_data_dir):
        relation = relation.split(".")[0]
        print("RELATION {}".format(relation))

        target_dir = os.path.join(args.target_dir, args.model_name.replace("/","_"), relation)
        os.makedirs(target_dir, exist_ok=True)
        logger.addHandler(logging.FileHandler(os.path.join(target_dir, "eval.log"), 'w'))
        source_dir = os.path.join(args.source_dir, relation)

        template = init_template(args, model, relation)
        logger.info('Template: %s'%template)
        
        load_optiprompt(model, source_dir, original_vocab_size)

        test_data = os.path.join(args.test_data_dir, relation + ".jsonl")
        eval_samples = load_data(test_data, template, vocab_subset=vocab_subset, mask_token=model.MASK)
        eval_samples_batches, eval_sentences_batches = batchify(eval_samples, args.eval_batch_size * n_gpu)
        evaluate(model, eval_samples_batches, eval_sentences_batches, filter_indices, index_list, output_topk=target_dir)
