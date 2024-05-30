import argparse
from ..utils import log_data_to_mysql
import zipfile
import pandas as pd
from gradio_client import Client
import tempfile
import os
import shutil
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument("--result_file", default=None, type=str)
parser.add_argument("--report_to_mysql", type=bool, default=False)
parser.add_argument("--sql_host", type=str, default=None)
parser.add_argument("--sql_port", type=int, default=3306)
parser.add_argument("--sql_user", type=str, default=None)
parser.add_argument("--sql_password", type=str, default=None)
parser.add_argument("--sql_db", type=str, default=None)
parser.add_argument("--sql_table", type=str, default="mmvet")
parser.add_argument("--sql_tag", type=str, default=None)

args = parser.parse_args()
if not args.report_to_mysql:
    exit(0)
client = Client("https://whyu-mm-vet-evaluator.hf.space/")

print("Calculating scores...")
scores_file = client.predict(args.result_file, fn_index=0)
zip_file = zipfile.ZipFile(scores_file)
temp_dir = tempfile.mkdtemp()
zip_file.extractall(temp_dir)

score_csv = [f for f in os.listdir(temp_dir) if f.endswith("cap-score-1runs.csv")][0]

df = pd.read_csv(os.path.join(temp_dir, score_csv))

print(df)

data_dict = {
    "tag": args.sql_tag,
    "rec": df.at[0, "rec"],
    "ocr": df.at[0, "ocr"],
    "know": df.at[0, "know"],
    "gen": df.at[0, "gen"],
    "spat": df.at[0, "spat"],
    "math": df.at[0, "math"],
    "total": df.at[0, "total"],
}
shutil.rmtree(temp_dir)
if args.sql_host is not None:
    try:
        log_data_to_mysql(
            args.sql_host, args.sql_port, args.sql_user, args.sql_password, args.sql_db, args.sql_table, data_dict
        )
    except Exception as e:
        logger.exception(e)
