import json

def convert_file(input_path, output_path):
    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            record = json.loads(line)
            gemini_record = {
                "contents": [
                    {"role": "user", "parts": [{"text": record["input_text"]}]},
                    {"role": "model", "parts": [{"text": record["output_text"]}]}
                ]
            }
            f_out.write(json.dumps(gemini_record) + "\n")
    print(f"Converted {input_path} -> {output_path}")

files = [
    "data/iris_v1_train.jsonl",
    "data/iris_v1_eval.jsonl",
    "data/iris_v2_train.jsonl",
    "data/iris_v2_eval.jsonl",
]

for path in files:
    output_path = path.replace(".jsonl", "_gemini.jsonl")
    convert_file(path, output_path)
