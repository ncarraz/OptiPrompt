from sklearn.neighbors import NearestNeighbors
from models import build_model_by_name
import argparse
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import json


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='bert-base-cased', help='the huggingface model name')
parser.add_argument('--input_dir', type=str, default=None, help='the directory containing the saved vectors. macthes the model name')
parser.add_argument('--k', type=int, default=5, help='back compatibility')

if __name__ == "__main__":
    args = parser.parse_args()
    model = build_model_by_name(args)
    model.update_embeddings()
    emb = model.embeddings.weight.detach().numpy()
    for f in os.listdir(args.input_dir):
        filepath = os.path.join(args.input_dir, f, "prompt_vecs.npy")
        with open(filepath, "rb") as f:
            prompt_vec = np.load(f)
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(emb)
        discrete_tokens = neigh.kneighbors(prompt_vec, 1, return_distance=False)

        discrete_tokens = np.squeeze(discrete_tokens)[:5]
        prompt_string = model.tokenizer.decode(discrete_tokens)
        prompt_string = "[X]{} [Y].".format(prompt_string)
        prompt_tokens = model.tokenizer.convert_ids_to_tokens(discrete_tokens)
        out = {
            'relation': f,
            'template': prompt_string,
            'tokens': prompt_tokens
        }
        out_json = json.dumps(out)
        print(out_json)


    