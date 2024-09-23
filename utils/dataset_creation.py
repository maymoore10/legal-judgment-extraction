import os.path

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
import datasets
from langdetect import detect
import re
from utils.utils import *

# Initialize Spark session
spark = SparkSession.builder \
    .appName("DatasetProcessing") \
    .getOrCreate()


def is_hebrew(text):
    try:
        return detect(text) == 'he'
    except:
        return False


def preprocess(text):
    try:
        sentences = text.split('. ')
        hebrew_sentences = [sentence for sentence in sentences if is_hebrew(sentence)]
    except:
        return "none"

    if not hebrew_sentences:
        return "none"
    else:
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text).strip()
        return text


def get_label(text):
    top_sentences, sorted_sentences, max_score, avg_score = rate_sentences(text)
    return top_sentences


def save_file(row):
    file_name = row['FileName']
    text = row['clean_text']
    if not file_name.endswith('.txt'):
        file_name += '.txt'

    # Write each file (ensure file paths are appropriate for your environment)
    with open(os.path.join(test_dir, file_name), 'w') as f:
        f.write(text)


# Load dataset
dataset = datasets.load_dataset('LevMuchnik/SupremeCourtOfIsrael')
df = pd.DataFrame(dataset['train'])
filtered_df = df[df['TypeCode'] == 2]
# Select only the 'fileName' and 'text' columns
subset_df = filtered_df[['FileName', 'text']][30000:35000]
spark_df = spark.createDataFrame(subset_df)

preprocess_udf = udf(preprocess, StringType())
rate_sentences_udf = udf(get_label, StringType())

# Select and preprocess columns
spark_df = spark_df.withColumn('clean_text', preprocess_udf(col('text')))
spark_df = spark_df.withColumn('label', rate_sentences_udf(col('clean_text')))
spark_df = spark_df.filter(spark_df.label != 'none')
spark_df.foreach(lambda row: save_file(row.asDict()))

selected_df = spark_df.select('FileName', 'label')


# Convert label column to string type
save_path = os.path.join(resources_path, 'test_5K_mixed_rulebased.csv')
#
# # Save to CSV with tab delimiter
selected_df.write.option("delimiter", "\t").option("header", "true").csv(save_path)
spark.stop()