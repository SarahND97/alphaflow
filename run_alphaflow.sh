##### AF2 CONFIGURATION #####
COMMON="/proj/berzelius-2021-29/users/x_sarna"
SINGULARITY=$COMMON"/singularity-images/esmfold.sif" 	# Path of singularity image.

SECONDS=0
MSA_DIR=$COMMON/msas/2.3/neg_homomers/
PDB=6dxo_B_6dxo_B

singularity exec --nv --bind $COMMON:$COMMON $SINGULARITY \
    python3 predict.py \
    --mode alphafold \
    --input_csv splits/test.csv \
    --msa_dir $MSA_DIR/\
    --weights alphaflow/alphaflow_12l_md_templates_base_202406.pt \
    --samples 1 \
    --outpdb results/test/
duration=$SECONDS
echo "Elapsed Time: $((SECONDS/3600))h $(((SECONDS/60)%60))m $((SECONDS%60))s"