from sklearn.neighbors import NearestNeighbors
from models import build_model_by_name
import argparse
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import json
from run_optiprompt import init_template


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='bert-base-cased', help='the huggingface model name')
parser.add_argument('--input_dir', type=str, default=None, help='the directory containing the saved vectors. macthes the model name')
parser.add_argument('--k', type=int, default=5, help='back compatibility')
parser.add_argument('--num_vectors', type=int, default=5, help='back compatibility')
parser.add_argument('--init_manual_template', action="store_true", help='back compatibility')
parser.add_argument('--init_random', action="store_true", help='back compatibility')
parser.add_argument('--relation_profile', type=str, default="relation_metainfo/LAMA_relations.jsonl", help='meta infomation of 41 relations, containing the pre-defined templates')

if __name__ == "__main__":
    args = parser.parse_args()
    model = build_model_by_name(args)
    model.update_embeddings()
    model.tokenizer.add_tokens(['[X]', " [Y]"], special_tokens=True)
    emb = model.embeddings.weight.detach().numpy()
    output_file = open("{}-hardened-optiprompt.jsonl".format(args.model_name),"w")
    for relation in os.listdir(args.input_dir):
        filepath = os.path.join(args.input_dir, relation, "prompt_vecs.npy")
        with open(filepath, "rb") as f:
            prompt_vec = np.load(f)
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(emb)
        discrete_tokens = neigh.kneighbors(prompt_vec, 1, return_distance=False)

        discrete_tokens = np.squeeze(discrete_tokens)[:5]
        prompt_tokens = model.tokenizer.convert_ids_to_tokens(discrete_tokens)
        prompt = init_template(args, model, relation)
        prompt = ["[X]"] + prompt_tokens + [" [Y]"] + ["."]
        #print(prompt.strip().replace("  "," ").split(" "), prompt_tokens)
        
        #for i in range(len(prompt_tokens)):    
        #    prompt = prompt.replace("[V{}]".format(i+1), prompt_tokens[i])
        #print(prompt.strip().replace("  "," ").split(" "))
        prompt = model.tokenizer.convert_tokens_to_string(prompt)
        out = {
            'relation': relation,
            'template': prompt,
            'tokens': prompt_tokens
        }
        out_json = json.dumps(out)
        output_file.write(out_json + "\n")
    output_file.close()


    
