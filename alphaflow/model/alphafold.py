# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn


from openfold.model.embedders import (
   InputEmbedder,
   RecyclingEmbedder,
   ExtraMSAEmbedder,
)
from openfold.model.evoformer import EvoformerStack, ExtraMSAStack
from openfold.model.heads import AuxiliaryHeads
from openfold.model.structure_module import StructureModule
  
import openfold.np.residue_constants as residue_constants
from openfold.utils.feats import (
   pseudo_beta_fn,
   build_extra_msa_feat,
   atom14_to_atom37,
)
from openfold.utils.tensor_utils import add
from .input_stack import InputPairStack
from .layers import GaussianFourierProjection
from openfold.model.primitives import Linear


from openfold.utils.loss import compute_tm, compute_predicted_aligned_error


from alphaflow.utils.logging import get_logger
logger = get_logger(__name__)


class AlphaFold(nn.Module):
   """
   Alphafold 2.


   Implements Algorithm 2 (but with training).
   """


   def __init__(self, config, extra_input=False):
       """
       Args:
           config:
               A dict-like config object (like the one in config.py)
       """
       super(AlphaFold, self).__init__()
       #print("when does this happen?")


       self.globals = config.globals
       self.config = config.model
       self.template_config = self.config.template
       self.extra_msa_config = self.config.extra_msa


       # Main trunk + structure module
       self.input_embedder = InputEmbedder(
           **self.config["input_embedder"],
       )
       self.recycling_embedder = RecyclingEmbedder(
           **self.config["recycling_embedder"],
       )
      


       if(self.extra_msa_config.enabled):
           self.extra_msa_embedder = ExtraMSAEmbedder(
               **self.extra_msa_config["extra_msa_embedder"],
           )
           # self.extra_msa_config["extra_msa_stack"]["opm_first"] = False
           # self.extra_msa_config["extra_msa_stack"]["fuse_projection_weights"] = False
           self.extra_msa_stack = ExtraMSAStack(
               **self.extra_msa_config["extra_msa_stack"],
           )
      
       self.evoformer = EvoformerStack(
           **self.config["evoformer_stack"],
       )
       self.structure_module = StructureModule(
           **self.config["structure_module"],
       )
       self.config["heads"]["tm"]["enabled"] = True
       self.aux_heads = AuxiliaryHeads(
           self.config["heads"],
       )


       ################
       self.input_pair_embedding = Linear(
           self.config.input_pair_embedder.no_bins,
           self.config.evoformer_stack.c_z,
           init="final",
       )
       self.input_time_projection = GaussianFourierProjection(
           embedding_size=self.config.input_pair_embedder.time_emb_dim
       )
       self.input_time_embedding = Linear(
           self.config.input_pair_embedder.time_emb_dim,
           self.config.evoformer_stack.c_z,
           init="final",
       )
       self.input_pair_stack = InputPairStack(**self.config.input_pair_stack)
       self.extra_input = extra_input
       if extra_input:
           self.extra_input_pair_embedding = Linear(
               self.config.input_pair_embedder.no_bins,
               self.config.evoformer_stack.c_z,
               init="final",
           )  
           self.extra_input_pair_stack = InputPairStack(**self.config.input_pair_stack)
      
       ################


   def _get_input_pair_embeddings(self, dists, mask, use_lma, use_deepspeed_evo_attention):
       #logger.info("1")
       mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
      
       lower = torch.linspace(
           self.config.input_pair_embedder.min_bin,
           self.config.input_pair_embedder.max_bin,
           self.config.input_pair_embedder.no_bins,
       device=dists.device)
       dists = dists.unsqueeze(-1)
       inf = self.config.input_pair_embedder.inf
       upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
       dgram = ((dists > lower) * (dists < upper)).type(dists.dtype)


       #logger.info("2")
       #print("dgam: ", dgram)
       inp_z = self.input_pair_embedding(dgram * mask.unsqueeze(-1))
       #print("inp_z: ", inp_z) # same for everything, maybe it should be the same for everything? 
       #if not long_sequence_inference:
       #    inp_z = self.input_pair_stack(inp_z, mask, chunk_size=None)
       #else:
       #logger.info("3")
       inp_z = self.input_pair_stack(inp_z, mask, chunk_size=None, use_lma=use_lma, use_deepspeed_evo_attention=use_deepspeed_evo_attention) 
       # t: torch.tensor,
       # mask: torch.tensor,
       # chunk_size: int,
       # use_lma: bool = False,
       # inplace_safe: bool = False,
       # _mask_trans: bool = True,
       #logger.info("4")
       return inp_z


   def _get_extra_input_pair_embeddings(self, dists, mask, use_lma, use_deepspeed_evo_attention):


       mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
      
       lower = torch.linspace(
           self.config.input_pair_embedder.min_bin,
           self.config.input_pair_embedder.max_bin,
           self.config.input_pair_embedder.no_bins,
       device=dists.device)
       dists = dists.unsqueeze(-1)
       inf = self.config.input_pair_embedder.inf
       upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
       dgram = ((dists > lower) * (dists < upper)).type(dists.dtype)


       inp_z = self.extra_input_pair_embedding(dgram * mask.unsqueeze(-1))
       #if not long_sequence_inference:
       #inp_z = self.input_pair_stack(inp_z, mask, chunk_size=None)
       #else:
       inp_z = self.input_pair_stack(inp_z, mask, chunk_size=None, use_lma=use_lma, use_deepspeed_evo_attention=use_deepspeed_evo_attention) 
       return inp_z
  
   def forward(self, batch, prev_outputs=None):
       device = torch.cuda.current_device()
       #logger.info("beginning of forward")
       #print(f"Allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
       #print(f"Reserved memory: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")
       feats = batch
       # Primary output dictionary
       outputs = {}
       outputs["asym_id"] = feats["asym_id"]


       # # This needs to be done manually for DeepSpeed's sake
       # dtype = next(self.parameters()).dtype
       # for k in feats:
       #     if(feats[k].dtype == torch.float32):
       #         feats[k] = feats[k].to(dtype=dtype)


       # Grab some data about the input
       batch_dims = feats["target_feat"].shape[:-2]
       no_batch_dims = len(batch_dims)
       n = feats["target_feat"].shape[-2]
       n_seq = feats["msa_feat"].shape[-3]
       device = feats["target_feat"].device
      
       #logger.info("before n>800: self.globals.offload_inference:")
       #print("before n>800: self.globals.offload_inference: ", self.globals.offload_inference)
       # If sequence is longer than 800 amino acids activate long sequence inference
       long_sequence_inference = False
       self.globals.use_deepspeed_evo_attention = True
       self.globals.use_lma = False
       if n > 800:
       #logger.info("long sequence inference")
            long_sequence_inference = True
            self.globals.offload_inference = True
            self.globals.use_lma = False
            self.globals.use_flash = False
            self.template_config.offload_inference = True
            self.template_config.template_pair_stack.tune_chunk_size = False
            self.config.extra_msa.extra_msa_stack.tune_chunk_size = False
            self.config.evoformer_stack.tune_chunk_size = False # /proj/berzelius-2021-29/users/x_sarna/.cache/torch_extensions/py39_cu116/evoformer_attn
       #print("after n>800: self.globals.offload_inference: ", self.globals.offload_inference)
      
       # Controls whether the model uses in-place operations throughout /proj/berzelius-2021-29/users/x_sarna/.conda/bin/nvcc
       # The dual condition accounts for activation checkpoints
       inplace_safe = not (self.training or torch.is_grad_enabled())


       # Prep some features
       seq_mask = feats["seq_mask"]
       pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
       msa_mask = feats["msa_mask"]
      
       ## Initialize the MSA and pair representations


       # m: [*, S_c, N, C_m]
       # z: [*, N, N, C_z]
       m, z = self.input_embedder(
           feats["target_feat"],
           feats["residue_index"],
           feats["msa_feat"],
           inplace_safe=inplace_safe,
       )
       if prev_outputs is None:
           m_1_prev = m.new_zeros((*batch_dims, n, self.config.input_embedder.c_m), requires_grad=False)
           # [*, N, N, C_z]
           z_prev = z.new_zeros((*batch_dims, n, n, self.config.input_embedder.c_z), requires_grad=False)
           # [*, N, 3]
           x_prev = z.new_zeros((*batch_dims, n, residue_constants.atom_type_num, 3), requires_grad=False)


       else:
           m_1_prev, z_prev, x_prev = prev_outputs['m_1_prev'], prev_outputs['z_prev'], prev_outputs['x_prev']


       x_prev = pseudo_beta_fn(
           feats["aatype"], x_prev, None
       ).to(dtype=z.dtype)


       # m_1_prev_emb: [*, N, C_m]
       # z_prev_emb: [*, N, N, C_z]
       m_1_prev_emb, z_prev_emb = self.recycling_embedder(
           m_1_prev,
           z_prev,
           x_prev,
           inplace_safe=inplace_safe,
       )


       # [*, S_c, N, C_m]
       m[..., 0, :, :] += m_1_prev_emb


       # [*, N, N, C_z]
       z = add(z, z_prev_emb, inplace=inplace_safe)


       # delete previous information
       del m_1_prev_emb, z_prev_emb, x_prev, m_1_prev, z_prev


       #logger.info("before noised_pseudo_beta_dists")
       #print(f"Allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
       #print(f"Reserved memory: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")
       #
       #######################
       if 'noised_pseudo_beta_dists' in batch:
           #logger.info("inside noised_pseudo_beta_dists")
           inp_z = self._get_input_pair_embeddings(
               batch['noised_pseudo_beta_dists'],
               batch['pseudo_beta_mask'],
               self.globals.use_lma,
               self.globals.use_deepspeed_evo_attention
           )
           inp_z = inp_z + self.input_time_embedding(self.input_time_projection(batch['t']))[:,None,None]
          
       else: # otherwise DDP complains
           B, L = batch['aatype'].shape
           #logger.info("ELSE")
           #print("is this where I get my memory error?")
           torch.cuda.empty_cache()
           #print(f"Allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
           #print(f"Reserved memory: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")
           #print(B)
           #print(L)
           inp_z = self._get_input_pair_embeddings(
               z.new_zeros(B, L, L),
               z.new_zeros(B, L),
               self.globals.use_lma,
               self.globals.use_deepspeed_evo_attention
           )
           #logger.info("after get_input_pair_embeddings")
           #print("it sure is")
           inp_z = inp_z + self.input_time_embedding(self.input_time_projection(z.new_zeros(B)))[:,None,None]


       z = add(z, inp_z, inplace=inplace_safe)
       del inp_z


       #logger.info("before extra input")
       #print(f"Allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
       #print(f"Reserved memory: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")
       torch.cuda.empty_cache()
       #print("empty cache")
       #print(f"Allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
       #print(f"Reserved memory: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")


       #############################
       if self.extra_input:
           if 'extra_all_atom_positions' in batch:
               extra_pseudo_beta = pseudo_beta_fn(batch['aatype'], batch['extra_all_atom_positions'], None)
               extra_pseudo_beta_dists = torch.sum((extra_pseudo_beta.unsqueeze(-2) - extra_pseudo_beta.unsqueeze(-3)) ** 2, dim=-1)**0.5
               extra_inp_z = self._get_extra_input_pair_embeddings(
                   extra_pseudo_beta_dists,
                   batch['pseudo_beta_mask'],
                   self.globals.use_lma,
                   self.globals.use_deepspeed_evo_attention
               )
               del extra_pseudo_beta, extra_pseudo_beta_dists
           else: # otherwise DDP complains
               B, L = batch['aatype'].shape
               extra_inp_z = self._get_extra_input_pair_embeddings(
                   z.new_zeros(B, L, L),
                   z.new_zeros(B, L),
                   self.globals.use_lma,
                   self.globals.use_deepspeed_evo_attention
               ) * 0.0
  
           z = add(z, extra_inp_z, inplace=inplace_safe)
           del extra_inp_z
       ########################


       # Embed extra MSA features + merge with pairwise embeddings
       if self.config.extra_msa.enabled:
           # [*, S_e, N, C_e]
           a = self.extra_msa_embedder(build_extra_msa_feat(feats))


           if(self.globals.offload_inference):
               # To allow the extra MSA stack (and later the evoformer) to
               # offload its inputs, we remove all references to them here
               input_tensors = [a, z]
               del a, z
  
               # [*, N, N, C_z]
               z = self.extra_msa_stack._forward_offload(
                   input_tensors,
                   msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                   chunk_size=self.globals.chunk_size,
                   use_lma=self.globals.use_lma,
                   use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                   pair_mask=pair_mask.to(dtype=m.dtype),
                   _mask_trans=self.config._mask_trans,
               )
  
               del input_tensors
           else:
               # [*, N, N, C_z]


               z = self.extra_msa_stack(
                   a, z,
                   msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                   chunk_size=self.globals.chunk_size,
                   use_lma=self.globals.use_lma,
                   pair_mask=pair_mask.to(dtype=m.dtype),
                   inplace_safe=inplace_safe,
                   _mask_trans=self.config._mask_trans,
               )
               del a


       # Run MSA + pair embeddings through the trunk of the network
       # m: [*, S, N, C_m]
       # z: [*, N, N, C_z]
       # s: [*, N, C_s]         
       if(self.globals.offload_inference):
           #logger.info("offload inference")
           input_tensors = [m, z]
           del m, z
           m, z, s = self.evoformer._forward_offload(
               input_tensors,
               msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
               pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
               chunk_size=self.globals.chunk_size,
               use_lma=self.globals.use_lma,
               use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
               _mask_trans=self.config._mask_trans,
           )
  
           del input_tensors
       else:
           #print("z before evoformer: ")
           #print("z.shape", z.shape)
           #print(z)
           m, z, s = self.evoformer(
               m,
               z,
               msa_mask=msa_mask.to(dtype=m.dtype),
               pair_mask=pair_mask.to(dtype=z.dtype),
               chunk_size=self.globals.chunk_size,
               use_lma=self.globals.use_lma,
               use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
               use_flash=self.globals.use_flash,
               inplace_safe=inplace_safe,
               _mask_trans=self.config._mask_trans,
           )
           #print("after evoformer")
           #print("z.shape: ", z.shape) 
           #print(z) # z is the exact same before and after the evoformer


       outputs["msa"] = m[..., :n_seq, :, :]
       outputs["pair"] = z
       outputs["single"] = s


       del z
      
       # Predict 3D structure
       outputs["sm"] = self.structure_module(
           outputs,
           feats["aatype"],
           mask=feats["seq_mask"].to(dtype=s.dtype),
           inplace_safe=inplace_safe,
           _offload_inference=self.globals.offload_inference,
       )
       outputs["final_atom_positions"] = atom14_to_atom37(
           outputs["sm"]["positions"][-1], feats
       )
       outputs["final_atom_mask"] = feats["atom37_atom_exists"]
       outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]


      
       outputs.update(self.aux_heads(outputs))




       # [*, N, C_m]
       outputs['m_1_prev'] = m[..., 0, :, :]


       # [*, N, N, C_z]
       outputs['z_prev'] = outputs["pair"]


       # [*, N, 3]
       outputs['x_prev'] = outputs["final_atom_positions"]

       # Get the iptm score
       # Maybe the logits are the same everytime
       print("outputs.keys(): ", outputs.keys())
       print("ptm_score: ", outputs["ptm_score"])
       print("iptm_score: ", outputs["iptm_score"])
       #print("calculate iptm")
       #outputs['iptm_score'] = compute_tm(
       #            outputs["tm_logits"], asym_id=feats["asym_id"], interface=True, **self.config["heads"]["tm"]
       #        )
       # print("calculate pae")        
       print(outputs["max_predicted_aligned_error"])
       #pae = compute_predicted_aligned_error(outputs["tm_logits"], **self.config["heads"]["tm"])
       #print("pae: ", round(float(pae["max_predicted_aligned_error"]),2))
       #print(max(pae["aligned_confidence_probs"]))
       # print(outputs["tm_logits"]) # these are 0????? why???? Now it makes sense that these are relative to the size of the proteins
    #    ptm = compute_tm(
    #                outputs["tm_logits"], asym_id=feats["asym_id"], interface=False, **self.config["heads"]["tm"]
    #            )
    #    print("ptm: ", round(float(ptm),2))     
       del outputs["tm_logits"]
       # memory increased before this
       # print("iptm score calculations")
       # print(f"Allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
       # print(f"Reserved memory: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")
       # del outputs["tm_logits"]
       # print("after tm_logits delete")
    #    print(f"Allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
    #    print(f"Reserved memory: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")
    #    logger.info("end of forward")
    #    print(f"Allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
    #    print(f"Reserved memory: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")
       torch.cuda.empty_cache()
       return outputs


   # def forward(self, batch):
   #     """
   #     Args:
   #         batch:
   #             Dictionary of arguments outlined in Algorithm 2. Keys must
   #             include the official names of the features in the
   #             supplement subsection 1.2.9.


   #             The final dimension of each input must have length equal to
   #             the number of recycling iterations.


   #             Features (without the recycling dimension):


   #                 "aatype" ([*, N_res]):
   #                     Contrary to the supplement, this tensor of residue
   #                     indices is not one-hot.
   #                 "target_feat" ([*, N_res, C_tf])
   #                     One-hot encoding of the target sequence. C_tf is
   #                     config.model.input_embedder.tf_dim.
   #                 "residue_index" ([*, N_res])
   #                     Tensor whose final dimension consists of
   #                     consecutive indices from 0 to N_res.
   #                 "msa_feat" ([*, N_seq, N_res, C_msa])
   #                     MSA features, constructed as in the supplement.
   #                     C_msa is config.model.input_embedder.msa_dim.
   #                 "seq_mask" ([*, N_res])
   #                     1-D sequence mask
   #                 "msa_mask" ([*, N_seq, N_res])
   #                     MSA mask
   #                 "pair_mask" ([*, N_res, N_res])
   #                     2-D pair mask
   #                 "extra_msa_mask" ([*, N_extra, N_res])
   #                     Extra MSA mask
   #                 "template_mask" ([*, N_templ])
   #                     Template mask (on the level of templates, not
   #                     residues)
   #                 "template_aatype" ([*, N_templ, N_res])
   #                     Tensor of template residue indices (indices greater
   #                     than 19 are clamped to 20 (Unknown))
   #                 "template_all_atom_positions"
   #                     ([*, N_templ, N_res, 37, 3])
   #                     Template atom coordinates in atom37 format
   #                 "template_all_atom_mask" ([*, N_templ, N_res, 37])
   #                     Template atom coordinate mask
   #                 "template_pseudo_beta" ([*, N_templ, N_res, 3])
   #                     Positions of template carbon "pseudo-beta" atoms
   #                     (i.e. C_beta for all residues but glycine, for
   #                     for which C_alpha is used instead)
   #                 "template_pseudo_beta_mask" ([*, N_templ, N_res])
   #                     Pseudo-beta mask
   #     """
   #     # Initialize recycling embeddings
  
   #     m_1_prev, z_prev, x_prev = None, None, None
   #     prevs = [m_1_prev, z_prev, x_prev]


   #     is_grad_enabled = torch.is_grad_enabled()


   #     # Main recycling loop
   #     num_iters = batch["aatype"].shape[-1]
   #     for cycle_no in range(num_iters):
   #         # Select the features for the current recycling cycle
   #         fetch_cur_batch = lambda t: t[..., cycle_no]
   #         feats = tensor_tree_map(fetch_cur_batch, batch)


   #         # Enable grad iff we're training and it's the final recycling layer
   #         is_final_iter = cycle_no == (num_iters - 1)
   #         with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
   #             if is_final_iter:
   #                 # Sidestep AMP bug (PyTorch issue #65766)
   #                 if torch.is_autocast_enabled():
   #                     torch.clear_autocast_cache()


   #             # Run the next iteration of the model
   #             outputs, m_1_prev, z_prev, x_prev = self.iteration(
   #                 feats,
   #                 prevs,
   #                 _recycle=(num_iters > 1)
   #             )


   #             if(not is_final_iter):
   #                 del outputs
   #                 prevs = [m_1_prev, z_prev, x_prev]
   #                 del m_1_prev, z_prev, x_prev


   #     # Run auxiliary heads
   #     outputs.update(self.aux_heads(outputs))


   #     return outputs

