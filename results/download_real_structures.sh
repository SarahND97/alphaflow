#!/bin/bash

input_file="pdb_ids.txt"
output_directory="real_structures"

# Create the output directory if it doesn't exist
mkdir -p $output_directory

# Read through each line of the input file and download the corresponding PDB file
while IFS= read -r pdb_id; do
    # Construct the download URL
    url="https://files.rcsb.org/download/$pdb_id"
    
    # Download the file using wget or curl
    echo "Downloading $pdb_id..."
    curl -o "$output_directory/$pdb_id" $url
    
done < "$input_file"

echo "Download complete."