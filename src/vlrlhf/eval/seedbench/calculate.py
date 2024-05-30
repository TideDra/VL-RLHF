import json
import argparse
from ..utils import log_data_to_mysql
from tqdm import tqdm


def is_integer_string(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def filter_questions(data, task="all"):
    if task == "image":
        return [q for q in data if 1 <= q["question_type_id"] <= 9]
    elif task == "video":
        return [q for q in data if 10 <= q["question_type_id"] <= 12]
    elif task == "all":
        return data
    elif is_integer_string(task):
        return [q for q in data if q["question_type_id"] == int(task)]
    else:
        raise ValueError(f"Invalid task: {task}")


def run_inference(responses, qa_anno):
    answer_list = []
    step = 0
    for qa_item in tqdm(qa_anno):

        question_id = qa_item["question_id"]
        pred_id = responses[question_id]["choice"]
        gt = qa_item["answer"]
        answer_record = {
            "question_id": qa_item["question_id"],
            "prediction": pred_id,
            "gt": gt,
            "q_type_id": qa_item["question_type_id"],
        }
        answer_list.append(answer_record)
        # output prediction record for each question
        step += 1

    print("evaluation finished! Calculating accuracy...")
    type_counts = {}
    correct_counts = {}

    for item in answer_list:
        pred, gt, data_type = item["prediction"], item["gt"], item["q_type_id"]

        type_counts[data_type] = type_counts.get(data_type, 0) + 1
        if pred == gt:
            correct_counts[data_type] = correct_counts.get(data_type, 0) + 1

    print("Accuracy for each data type:")
    total_count = 0
    total_correct = 0
    data_dict = {}

    for data_type in type_counts.keys():
        accuracy = correct_counts[data_type] / type_counts[data_type] * 100
        print(f"Data type {data_type}: {accuracy:.2f}%")
        data_dict[data_type_id2name[data_type].replace(" ", "")] = round(accuracy, 2)
        total_count += type_counts[data_type]
        total_correct += correct_counts[data_type]

    total_accuracy = total_correct / total_count * 100
    print(f"Total accuracy: {total_accuracy:.2f}%")
    data_dict["Total"] = round(total_accuracy, 2)
    return data_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arg Parser")
    parser.add_argument("--result_file", type=str, default="instruct_blip.json")
    parser.add_argument("--anno_path", type=str, default="SEED-Bench.json")
    parser.add_argument("--task", type=str, default="image")
    parser.add_argument("--report_to_mysql", type=bool, default=False)
    parser.add_argument("--sql_host", type=str, default=None)
    parser.add_argument("--sql_port", type=int, default=3306)
    parser.add_argument("--sql_user", type=str, default=None)
    parser.add_argument("--sql_password", type=str, default=None)
    parser.add_argument("--sql_db", type=str, default=None)
    parser.add_argument("--sql_table", type=str, default="seedbench")
    parser.add_argument("--sql_tag", type=str, default=None)
    args = parser.parse_args()

    qa_anno = json.load(open(args.anno_path, "rb"))
    data_type_id2name = {v: k for k, v in qa_anno["question_type"].items()}
    if "questions" in qa_anno.keys():
        qa_anno = qa_anno["questions"]
    qa_anno = filter_questions(qa_anno, args.task)

    print(f"evaluating.. {args.result_file}")
    # The interface for testing MLLMs
    with open(args.result_file, "r") as f:
        responses = json.load(f)
    data_dict = run_inference(responses, qa_anno)
    if args.report_to_mysql:
        try:
            data_dict["tag"] = args.sql_tag
            log_data_to_mysql(
                args.sql_host, args.sql_port, args.sql_user, args.sql_password, args.sql_db, args.sql_table, data_dict
            )
        except Exception as e:
            print(e)
