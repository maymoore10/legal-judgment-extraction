# scp -r  morm@192.168.90.212:/home/morm/LegalVerdict_FinalProject/src/rulebased ~/Downloads/final_project_backup/src/
import regex as re
import os
from utils.utils import *
import csv
# Compile a regex pattern to match any of the keywords
# pattern = re.compile('|'.join(keywords.keys()), re.IGNORECASE)

# Define keywords that might indicate a verdict (add more as needed)

top_sentences_df = []

files_dir = test_dir
for filename in os.listdir(files_dir):
    file_path = os.path.join(files_dir, filename)
    # Check if it is a file
    if os.path.isfile(file_path):
        print(f'Processing file: {file_path}')
        # You can open and read the file, or do other processing here
        # For example, to read and print each line of the file:
        with open(file_path, 'r') as file:
            text = file.read().rstrip()
            top_sentences, sorted_sentences, max_score, avg_score = rate_sentences(text)
            top_sentences_df.append({'filename': filename, 'result': top_sentences, 'max_score': max_score, 'avg_score': avg_score,
                                     "size": len(sorted_sentences), 'alternatives': sorted_sentences})



save_csv_file(os.path.join(results_path, f"{today_format}/test/rule_based_sentences.csv"), top_sentences_df)

top_sentences_df = pd.DataFrame(top_sentences_df)
not_none = top_sentences_df[top_sentences_df['result'] != 'none'].shape[0]
count = top_sentences_df.shape[0]
print(f'extracted {not_none} out of {count} files')
