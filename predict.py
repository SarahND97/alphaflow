import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='splits/transporters_only.csv')
parser.add_argument('--templates_dir', type=str, default=None)
parser.add_argument('--msa_dir', type=str, default='./alignment_dir')
parser.add_argument('--mode', choices=['alphafold', 'esmfold'], default='alphafold')
parser.add_argument('--samples', type=int, default=1)
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--outpdb', type=str, default='./outpdb/default')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--original_weights', action='store_true')
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--subsample', type=int, default=None)
parser.add_argument('--resample', action='store_true', default=False) # change default to false
parser.add_argument('--tmax', type=float, default=1.0)
parser.add_argument('--no_diffusion', action='store_true', default=False)
parser.add_argument('--self_cond', action='store_true', default=False)
parser.add_argument('--noisy_first', action='store_true', default=False)
parser.add_argument('--runtime_json', type=str, default=None)
parser.add_argument('--no_overwrite', action='store_true', default=False)
parser.add_argument('--folddock', action='store_true', default=False)
parser.add_argument('--output_structure', action='store_true', default=False)

args = parser.parse_args()

import torch, tqdm, os, wandb, json, time
import pandas as pd
import pytorch_lightning as pl
import numpy as np
from collections import defaultdict
from alphaflow.data.data_modules import collate_fn
from alphaflow.model.wrapper import AlphaFoldWrapper, ESMFoldWrapper
from alphaflow.utils.tensor_utils import tensor_tree_map
import alphaflow.utils.protein as protein
from alphaflow.data.inference import AlphaFoldCSVDataset, CSVDataset, FoldDockCSVDataset
from collections import defaultdict
from openfold.utils.import_weights import import_jax_weights_
from alphaflow.config import model_config

from alphaflow.utils.logging import get_logger
logger = get_logger(__name__)
torch.set_float32_matmul_precision("high")

config = model_config(
    'initial_training',
    train=True, 
    low_prec=True
) 

schedule = np.linspace(args.tmax, 0, args.steps+1)
if args.tmax != 1.0:
    schedule = np.array([1.0] + list(schedule))
loss_cfg = config.loss
data_cfg = config.data
data_cfg.common.use_templates = False
data_cfg.common.max_recycling_iters = 0

if args.subsample: # https://elifesciences.org/articles/75751#s3
    data_cfg.predict.max_msa_clusters = args.subsample // 2
    data_cfg.predict.max_extra_msa = args.subsample

@torch.no_grad()
def main():

    valset = {
        'alphafold': AlphaFoldCSVDataset,
        'esmfold': CSVDataset,
    }[args.mode](
        data_cfg,
        args.input_csv,
        msa_dir=args.msa_dir,
        templates_dir=args.templates_dir,
    )

    if args.folddock:
        valset = {
            'multimer': FoldDockCSVDataset,
        }['multimer'](
            data_cfg,
            args.input_csv,
        )

    # valset[0]
    logger.info("Loading the model")
    model_class = {'alphafold': AlphaFoldWrapper, 'esmfold': ESMFoldWrapper}[args.mode]

    if args.weights:
        ckpt = torch.load(args.weights, map_location='cpu')
        # Add arguments to work with newer version of OpenFold
        ckpt['hyper_parameters']["config"]["model"]["extra_msa"]["extra_msa_stack"]["opm_first"] = False
        ckpt['hyper_parameters']["config"]["model"]["extra_msa"]["extra_msa_stack"]["fuse_projection_weights"] = False
        ckpt['hyper_parameters']["config"]["model"]["evoformer_stack"]["no_column_attention"] = False
        ckpt['hyper_parameters']["config"]["model"]["evoformer_stack"]["opm_first"] = False
        ckpt['hyper_parameters']["config"]["model"]["evoformer_stack"]["fuse_projection_weights"] = False
        
        model = model_class(**ckpt['hyper_parameters'])
        model.model.load_state_dict(ckpt['params'], strict=False)
        model = model.cuda()
        
    
    elif args.original_weights:
        model = model_class(config, None, training=False)
        if args.mode == 'esmfold':
            path = "esmfold_3B_v1.pt"
            model_data = torch.load(path, map_location='cpu')
            model_state = model_data["model"]
            model.model.load_state_dict(model_state, strict=False)
            model = model.to(torch.float).cuda()
            
        elif args.mode == 'alphafold':
            import_jax_weights_(model.model, 'params_model_1.npz', version='model_3')
            model = model.cuda()
        
    else:
        model = model_class.load_from_checkpoint(args.ckpt, map_location='cpu')
        model.load_ema_weights()
        model = model.cuda()

    model.eval()
    
    logger.info("Model has been loaded")
    
    results = defaultdict(list)
    os.makedirs(args.outpdb, exist_ok=True)
    runtime = defaultdict(list)
    iptm_scores = []
    ptm_scores = []

    for i, item in enumerate(valset):
        if args.pdb_id and item['name'] not in args.pdb_id:
            continue
        if args.no_overwrite and os.path.exists(f'{args.outpdb}/{item["name"]}.pdb'):
            continue

        for j in tqdm.trange(args.samples): 
            if args.subsample or args.resample:
                item = valset[i] # resample MSA
            device = torch.cuda.current_device()

            print("pdbid: ", item["name"])
            print("seqres_len: ", len(item["seqres"]))
            print("row: ", i+1)
            batch = collate_fn([item])
            batch = tensor_tree_map(lambda x: x.cuda(), batch)  
            start = time.time()
            prots = model.inference(batch, as_protein=False, noisy_first=args.noisy_first,
                        no_diffusion=args.no_diffusion, schedule=[1.0,0.5], self_cond=args.self_cond)
            runtime[item['name']].append(time.time() - start)
            
            # Save pTM/ipTM scores
            iptm_scores.append(round(float(prots[-1]["iptm_score"]), 2))
            print("iptm score: ", round(float(prots[-1]["iptm_score"]), 2))

            del batch
            del prots 
            torch.cuda.empty_cache()

    # Add pTM/ipTM scores to the input_csv
    df = pd.read_csv(args.input_csv)
    print(args.input_csv)
    df["iptm"] = iptm_scores
    #df["ptm"] = ptm_scores
    df.to_csv(path_or_buf=args.input_csv, header=True, index=False)

    # if args.output_structure:  
    #     with open(f'{args.outpdb}/{item["name"]}.pdb', 'w') as f:
    #        f.write(protein.prots_to_pdb(result))

    if args.runtime_json:
        with open(args.runtime_json, 'w') as f:
            f.write(json.dumps(dict(runtime)))
if __name__ == "__main__":
    main()