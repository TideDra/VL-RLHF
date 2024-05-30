# Evaluation Guide

All the evaluation scripts can be found in `scripts/eval`.

## Set-up Environment
VL-RLHF uses [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) to evaluate most of the benchmarks.

To avoid package conflicts, please install VLMEvalKit in another conda environment:
```bash
conda create -n vlmeval python=3.10
conda activate vlmeval
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

To assist the evaluation, e.g. extract choice from VLM response for some benchmarks, we use [lmdeploy](https://github.com/InternLM/lmdeploy) to deploy a local judge model:
```bash
conda activate vlmeval
pip install lmdeploy
```

Then set some environment variables in `scripts/eval/config.sh`:
```bash
export conda_path="~/miniconda3/bin/activate" #path to the conda activate file
export vlrlhf_env_name="vlrlhf" #name of environment that installed vlrlhf
export vlmeval_env_name="vlmeval" #name of environment that installed vlmevalkit
export judger_path="ckpts/Qwen1.5-14B-Chat" #path to the local judger checkpoint
export judger_port=23333 #service port
export judger_tp=2 #tensor parallelism size
```

You should also set some environment variables for VLMEvalKit, please refer to their [guide](https://github.com/open-compass/VLMEvalKit/blob/main/Quickstart.md#deploy-a-local-language-model-as-the-judge--choice-extractor)

## Prepare Dataset
Please put all benchmark files in `VL-RLHF/data_dir`.
```bash
mkdir data_dir
```
### MM-Vet
```bash
cd data_dir
wget https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip
unzip mm-vet.zip
```

### SEEDBench-Image
```bash
cd data_dir
mkdir SEED-Bench
de SEED-Bench
wget https://huggingface.co/datasets/AILab-CVC/SEED-Bench/resolve/main/SEED-Bench.json?download=true -O SEED-Bench.json
wget https://huggingface.co/datasets/AILab-CVC/SEED-Bench/resolve/main/SEED-Bench-image.zip?download=true -O SEED-Bench-image.zip
unzip SEED-Bench-image.zip
```

### MMBench
```bash
cd data_dir
wget https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv
```

### MathVista
```bash
cd data_dir
wget https://opencompass.openxlab.space/utils/VLMEval/MathVista_MINI.tsv
```

### MMMU
```bash
cd data_dir
wget https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv
```

### MME
```bash
cd data_dir
wget https://opencompass.openxlab.space/utils/VLMEval/MME.tsv
```

### POPE
```bash
cd data_dir
git clone https://github.com/AoiDragon/POPE.git
mkdir coco2014
cd coco2014
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip

```

## Report to MySQL
VL-RLHF supports to report the evaluation results to MySQL database. To do this, you need to firstly create a table for each benchmark and set the columns properly.

We have already created a database [vlrlhf.sql](./vlrlhf.sql) for you to easily use. Just import it using:
```bash
mysql -u username -p VLRLHF< vlrlhf.sql #the database VLRLHf must already exist.
```

Then, set some environment variables in the [config](./config.sh):
```bash
export SQL_HOST="localhost" #host name of your mysql server
export SQL_PORT=3306 # port of your mysql server
export SQL_USER="root" #username
export SQL_PASSWORD="Your Passward"
export SQL_DB="VLRLHF" # name of the database
export report_to_mysql=True #turn on reporting to MySQL
```

# Evaluate

Firstly, Set the path of checkpoint and the name of experiment as environment variables:
```bash
export CKPT=ckpts/QwenVL
export TAG="QwenVL"
```
Then run any evaluation script in the `scripts/eval`, like
```bash
bash scripts/eval/mmvet.sh
```
If you report the results to MySQL, then the value of `TAG` will be parsed into column names of the table `exps` in `vlrlhf.sql`

For example, the table `exps` has these columns by default:`tag`,`method`,`epoch` and so on. If `TAG` is set as:
```bash
export TAG="tag:qwenvl_dpo,method:dpo,epoch=1"
```
Then the new record in `exps` will be like:
```
  tag     | method | epoch |
---------------------------
qwenvl_dpo|   dpo  |   1   |
```
The rules for writing `TAG` includes
- Use `,` as the seperator between items, without space.
- Use `:` to assign value to column whose type is string, and use `=` to assign value to column whose type is int or float.

The column `tag` is set as primary key in each table of `vlrlhf.sql`. So you can use it to identify each experiment.

You can also evaluate multiple checkpoints on multiple benchmarks with one script:
```bash
export CKPT_PRE=ckpts/QwenVL-dpo/checkpoint-
export TAG_PRE="tag:qwenvl_dpo,step="
export SUFFIX=100,200,300
export BENCHMARKS=mmvet,mmbench,seedbench_gen
bash scripts/eval/eval_all.sh
```
Values in `SUFFIX` seperated by `,` will be joined with `CKPT_PRE` and `TAG_PRE`. The outputs will be saved in `VL-RLHF/eval/all`.
