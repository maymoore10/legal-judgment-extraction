import pandas as pd
from utils.utils import *
import json

#for generating non sentencing data
# path = '/home/morm/legal-IR/results/20240909/mixed/summaries_sentence-transformers-alephbert.csv'
# data = pd.read_csv(path, sep='\t', encoding="utf8")
# df = []
# for idx, row in data.iterrows():
#     file_name = row['filename']
#     text = row['result']
#     if contains_hebrew(text):
#         criminal = text.split(". ")
#         num_sentences = min(len(criminal), 2)
#         filtered_sentences = criminal[:num_sentences]
#         for sentence in filtered_sentences:
#             df.append({"filename": file_name, "label": sentence})
#
# save_csv_file(os.path.join(resources_path, 'none_sentence.csv'), df)


# for mapping data to json
# data = load_data(annotated_sentences_mixed, mixed_dir)
# json_df = {'data': []}
# paragraphs = []
# for index, row in data.iterrows():
#     title = row['fileName']
#     context = row['context']
#     question = 'מה ההחלטה המשפטית?'
#     answer_text = row['label']
#     answer_start = context.find(answer_text.split(". ")[0])
#
#     qas = {
#         "question": question,
#         "id": f"unique-id-{index}",
#         "answers": [
#             {
#                 "text": answer_text,
#                 "answer_start": answer_start
#             }
#         ],
#         "is_impossible": False  # assuming the answer always exists
#     }
#
#     # Add to the paragraphs list
#     paragraphs.append({
#         "context": context,
#         "qas": [qas]
#     })
#
#     # Add to the data structure
#     json_df['data'].append({
#         "title": title,
#         "paragraphs": paragraphs
#     })
#
# # Save to a JSON file
# json_file = os.path.join(resources_path, "annotated_json.json")
# with open(json_file, "w", encoding='utf-8') as f:
#     json.dump(json_df, f, ensure_ascii=False, indent=4)
#
# print(f"JSON file saved to {json_file}")
#
#


# merging files under a folder
# results = []
# # path = os.path.join(results_path, '2024091/test')
# path = '/results/20240913/summary_2last'
# merged_df = None
# for dirpath, dirnames, filenames in os.walk(path):
#     for filename in filenames:
#         if filename.endswith('csv'):
#             file_path = os.path.join(dirpath, filename)
#             print(f'Processing file: {file_path}')
#             with open(file_path, 'r') as file:
#                 df = pd.read_csv(file_path, sep='\t', encoding="utf8")
#                 fn = file_path.replace(path, '')
#                 df = df.rename(columns={'result': fn})
#
#                 # if df['fileName']:
#                 df = df.rename(columns={'fileName': 'filename'})
#                 print('processing file:', filename)
#                 if merged_df is None:
#                     # Initialize the merged DataFrame with the first file's data
#                     merged_df = df
#                 else:
#                     merged_df = pd.merge(merged_df, df, on='filename', how='outer')
#
#
#
# # save_csv_file(os.path.join(results_path, f"evaluations_{today_format}.csv"), merged_df)
#
# merged_df.to_csv('merged_file_summary_2last.csv', index=False, sep='\t')


# # for renaming files
# import os
# import shutil
#
# folder_path = mixed_dir
#
# # Iterate over all files in the folder
# for filename in os.listdir(folder_path):
#     if filename.endswith('.txt.txt'):
#         # Create the new filename by removing the extra .txt
#         new_filename = filename.replace('.txt.txt', '.txt')
#         old_file_path = os.path.join(folder_path, filename)
#         new_file_path = os.path.join(folder_path, new_filename)
#
#         # Check if the new file already exists
#         if os.path.exists(new_file_path):
#             # If it exists, replace it
#             os.remove(new_file_path)
#
#         # Rename the file
#         os.rename(old_file_path, new_file_path)
#         print(f'Renamed: {filename} -> {new_filename}')
#     elif filename.endswith('.txt'):
#         continue
#     else:
#         # Create the new filename by removing the extra .txt
#         new_filename = filename + '.txt'
#         old_file_path = os.path.join(folder_path, filename)
#         new_file_path = os.path.join(folder_path, new_filename)
#
#         # Check if the new file already exists
#         if os.path.exists(new_file_path):
#             # If it exists, replace it
#             os.remove(new_file_path)
#
#         # Rename the file
#         os.rename(old_file_path, new_file_path)
#         print(f'Renamed: {filename} -> {new_filename}')
