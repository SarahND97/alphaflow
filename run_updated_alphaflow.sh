##### AF2 CONFIGURATION #####
COMMON="/proj/berzelius-2021-29/users/x_sarna"
SINGULARITY=$COMMON"/singularity-images/esmfold.sif" 	# Path of singularity image.

SECONDS=0
MSA_DIR=$COMMON/msas/2.3/neg_homomers/
PDB=6dxo_B_6dxo_B
FOLDOCK=/proj/berzelius-2021-29/users/x_sarna/programs/FoldDock
FOLDOCK_MSAS=/proj/berzelius-2021-29/users/x_sarna/msas/folddock/negative_homomers/${PDB}
PAIREDMSA=$FOLDOCK_MSAS/${PDB}_paired.a3m
FUSEDMSA=$FOLDOCK_MSAS/${PDB}_fused.a3m
#MSAS=${PAIREDMSA} ${FUSEDMSA} 

singularity exec --nv --bind $COMMON:$COMMON $SINGULARITY \
    python3 predict.py \
    --mode alphafold \
    --input_csv splits/test_multimer.csv \
    --folddock \
    --msas ${PAIREDMSA} ${FUSEDMSA} \
    --msa_dir $MSA_DIR/ \
    --weights alphaflow/alphaflow_12l_md_templates_base_202406.pt \
    --samples 1 \
    --outpdb results/test/
    # \
duration=$SECONDS
echo "Elapsed Time: $((SECONDS/3600))h $(((SECONDS/60)%60))m $((SECONDS%60))s"

# Things I need to add in order for FoldDock trick to work for AlphaFlow
# Merged_MSAS
# chain_break_list
# data_pipeline = foldonly.FoldDataPipeline()

# for n in {1..5}; do
#     cp $MERGED_FASTA ${MERGED_ID}_${n}.fasta

#     ### Run Alphafold2 to fold provided chains
#     singularity exec --nv --bind $BASEDIR:$BASEDIR $SINGULARITY \
#         python3 $AFHOME/run_alphafold.py \
#             --fasta_paths=${MERGED_ID}_${n}.fasta \
# 	    --output_dir=$OUTFOLDER \
# 	    --model_names=$MODEL_NAME \
#             --max_recycles=$MAX_RECYCLES \
# 	    --data_dir=$PARAM \
# 	    --preset=$PRESET \
#             --fold_only \
#             --msas=$MSAS \
#             --chain_break_list=$L1 \

#     rm ${MERGED_ID}_${n}.fasta
