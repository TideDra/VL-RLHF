from lmdeploy import pipeline, GenerationConfig
from argparse import ArgumentParser
import json

tmpl = (
    "You are an AI assistant who will help me to match "
    "an answer with several options of a single-choice question. "
    "You are provided with a question, several options, and an answer, "
    "and you need to find which option is most similar to the answer. "
    "If the meaning of all options are significantly different from the answer, output Z. "
    "Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n"
    "Example 1: \n"
    "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
    "Answer: a cute teddy bear\nYour output: A\n"
    "Example 2: \n"
    "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
    "Answer: Spider\nYour output: Z\n"
    "Example 3: \n"
    "Question: {}?\nOptions: {}\nAnswer: {}\nYour output: "
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--result_file", type=str, default="result.json")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen1.5-14B-Chat")
    parser.add_argument("--output_file", type=str, default="final_result.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_path = args.result_file
    model_path = args.model_path
    with open(result_path, "r") as f:
        results = json.load(f)
    final_answers = {}
    unclear_results = []
    for r in results:
        question_id = r["question_id"]
        response = r["response"]
        if response[0] in ["A", "B", "C", "D"]:
            choice = response[0]
            final_answers[question_id] = {"choice": choice, "response": response}
        else:
            unclear_results.append(r)
    if len(unclear_results) > 0:
        pipe = pipeline(model_path)
        prompts = []
        for r in unclear_results:
            question = r["question"]
            options = f"A. {r['choice_a']} B. {r['choice_b']} C. {r['choice_c']} D. {r['choice_d']}"
            answer = r["response"]
            prompts.append(tmpl.format(question, options, answer))
        gen_config = GenerationConfig(top_k=1, temperature=0.0)
        preds = pipe(prompts, gen_config=gen_config)
        for r, pred in zip(unclear_results, preds):
            question_id = r["question_id"]
            response = pred.text[0]
            if response[0] in ["A", "B", "C", "D"]:
                choice = response[0]
                final_answers[question_id] = {"choice": choice, "response": response}
            else:
                final_answers[question_id] = {"choice": "Z", "response": "Z"}
    with open(args.output_file, "w") as f:
        json.dump(final_answers, f)
