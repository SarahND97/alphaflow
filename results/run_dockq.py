import os
import subprocess
import csv


def run_dockq(pdb_id, rewritten_dir, real_dir):
    # Construct the file paths
    rewritten_pdb = os.path.join(rewritten_dir, f"{pdb_id}.pdb")
    real_pdb = os.path.join(real_dir, f"{pdb_id}.pdb")

    # Run the DockQ command with subprocess and capture the output
    command = ["DockQ", rewritten_pdb, real_pdb, "--short"]

    try:
        result = subprocess.run(command, capture_output=True, text=True)

        # Extract DockQ score from the output
        output = result.stdout.splitlines()
        dockq_score = None

        # Find the line with the DockQ score
        for line in output:
            if line.startswith("DockQ:") or line.startswith("DockQ"):
                # Extract the DockQ score from the line
                dockq_score = float(line.split()[1])
                break

        if dockq_score is not None:
            return dockq_score
        else:
            print(f"Failed to retrieve DockQ score for {pdb_id}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Error running DockQ for {pdb_id}: {e}")
        return None


def process_pdb_ids(pdb_list_file, rewritten_dir, real_dir, output_csv):
    with open(pdb_list_file, "r") as f, open(output_csv, "w", newline="") as csvfile:
        pdb_ids = [line.strip().split(".")[0] for line in f if line.strip()]
        # csv_writer = csv.writer(csvfile)

        # # Write the CSV header
        # csv_writer.writerow(["pdbid", "dockq"])

        for pdb_id in pdb_ids:
            dockq_score = run_dockq(pdb_id, rewritten_dir, real_dir)

            if dockq_score is not None:
                # Write the pdb_id and DockQ score to the CSV file
                # csv_writer.writerow([pdb_id, dockq_score])
                print(f"Processed {pdb_id}, DockQ: {dockq_score}")
            else:
                print(f"Skipping {pdb_id} due to errors.")


# Example usage
pdb_list_file = "pdb_ids.txt"  # File containing PDB IDs
rewritten_dir = "rewritten_positive_samples"  # Directory containing rewritten pdbs
real_dir = "real_structures"  # Directory containing real pdbs
output_csv = "dockq_scores.csv"  # Output CSV file

process_pdb_ids(pdb_list_file, rewritten_dir, real_dir, output_csv)
