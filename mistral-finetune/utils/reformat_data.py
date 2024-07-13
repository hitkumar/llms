import argparse
import json
import os
import random
import string

def reformat_jsonl(input_file):
    """
    TODO: Only including functionality needed for ultrachat dataset, include logic related to function calling
    """
    output_file = input_file + '.tmp'
    content_keys = ["content", "text"]

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for idx, line in enumerate(infile):
            data = json.loads(line)
            skip_sample = False
            if "messages" in data:
                for i, msg in enumerate(data["messages"]):
                    if all(msg.get(key) in ["", None] for key in content_keys):
                        skip_sample = True
                
                # last message should be from assistant
                while len(data["messages"]) > 0 and data["messages"][-1]["role"] != "assistant":
                    data["messages"].pop()
                
                if len(data["messages"]) == 0:
                    skip_sample = True

            if not skip_sample:
                outfile.write(json.dumps(data) + "\n")
            else:
                print(f"skipping {idx} sample")
        
    os.rename(output_file, input_file)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reformat a json file")
    parser.add_argument("file", type=str, help="file to reformat")
    args = parser.parse_args()
    reformat_jsonl(args.file)
