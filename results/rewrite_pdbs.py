import os


def rewrite_pdb_file(input_filename, output_filename):
    with open(input_filename, "r") as infile, open(output_filename, "w") as outfile:
        lines = infile.readlines()

        # Skip the first line (MODEL 0 or MODEL 1)
        lines = lines[2:]
        # Skip last 2 lines
        lines = lines[:-2]

        lines.append("END")
        # Write the modified lines to the output file
        outfile.writelines(lines)


def process_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(".pdb"):
            input_filepath = os.path.join(input_directory, filename)
            output_filepath = os.path.join(output_directory, filename)

            print(f"Processing file: {filename}")
            rewrite_pdb_file(input_filepath, output_filepath)

    print(f"Processing complete. Files saved in {output_directory}")


# Example usage:
input_directory = "positive_samples"  # Directory containing the input PDB files
output_directory = (
    "rewritten_positive_samples"  # Directory where modified files will be saved
)

process_directory(input_directory, output_directory)
