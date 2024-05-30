import argparse
from ..utils import log_data_to_mysql
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--result_file", default=None, type=str)
parser.add_argument("--report_to_mysql", type=bool, default=False)
parser.add_argument("--sql_host", type=str, default=None)
parser.add_argument("--sql_port", type=int, default=3306)
parser.add_argument("--sql_user", type=str, default=None)
parser.add_argument("--sql_password", type=str, default=None)
parser.add_argument("--sql_db", type=str, default=None)
parser.add_argument("--sql_table", type=str, default="mathvista")
parser.add_argument("--sql_tag", type=str, default=None)

args = parser.parse_args()

if not args.report_to_mysql:
    exit(0)

workdir = os.path.dirname(args.result_file)
acc_file = [f for f in os.listdir(workdir) if f.endswith("csv")][0]
df = pd.read_csv(os.path.join(workdir, acc_file))
data_dict = {}
data_dict["tag"] = args.sql_tag
for idx, row in df.iterrows():
    data_dict[row["Task&Skill"].replace(" ", "_")] = row["acc"]
log_data_to_mysql(
    args.sql_host, args.sql_port, args.sql_user, args.sql_password, args.sql_db, args.sql_table, data_dict
)
