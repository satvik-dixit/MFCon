import os
import csv

def get_wav_paths(parent_directory):
    # List to store the paths of wav files
    wav_paths = []

    # Traverse the directory tree
    for root, _, files in os.walk(parent_directory):
        for file in files:
            if file.endswith('.wav'):
                # Construct the full path of the wav file
                full_path = os.path.join(root, file)
                wav_paths.append(full_path)

    return wav_paths

def write_paths_to_csv(wav_paths, output_csv):
    # Write the paths to the CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['utt_paths'])
        # Write the paths
        for path in wav_paths:
            writer.writerow([path])

def main():
    # Define the parent directory and output CSV file path
    parent_directory = '/ocean/projects/cis220031p/sdixit1/mfa_conformer/musan'  # Change this to your directory path
    output_csv = '/ocean/projects/cis220031p/sdixit1/mfa_conformer/data/noise.csv'  # The output CSV file

    # Get the list of wav file paths
    wav_paths = get_wav_paths(parent_directory)
    
    # Write the paths to the CSV file
    write_paths_to_csv(wav_paths, output_csv)
    print(f"CSV file created at: {output_csv}")

if __name__ == "__main__":
    main()
