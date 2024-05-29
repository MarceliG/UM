import glob
import json
import os
import re


class JSONLProcessor:
    def __init__(self, datas_location):
        self.datas_location = datas_location
        self.output_suffix = "Output"

    def validate_json(self, json_object, seen_texts):
        if "text" not in json_object:
            return False

        if len(json_object["text"]) <= 5:
            return False

        if json_object["text"] in seen_texts:
            return False

        return True

    def normalize_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove non-alphanumeric characters except spaces
        text = re.sub(r"[^a-z0-9\s]", "", text)
        # Trim leading and trailing spaces
        text = text.strip()
        return text

    def process_json_object(self, json_object):
        # Remove "images" field if it exists
        json_object.pop("images", None)

        # Normalize "text" field if it exists
        if "text" in json_object:
            json_object["text"] = self.normalize_text(json_object["text"])
        return json_object

    def process_files(self):
        files = glob.glob(os.path.join(self.datas_location, "*.jsonl"))

        if not files:
            print("No files to prepare")
            return

        for file in files:
            base_name, file_ext = os.path.splitext(file)
            output_file_path = f"{base_name}_{self.output_suffix}{file_ext}"

            #skip processed files
            if self.output_suffix in file:
                continue
            if os.path.exists(output_file_path):
                continue

            print(f"Processing {file}...")

            seen_texts = set()

            with open(file, "r") as infile, open(output_file_path, "w") as outfile:
                for line in infile:
                    json_object = json.loads(line)
                    updated_json_object = self.process_json_object(json_object)

                    if self.validate_json(updated_json_object, seen_texts):
                        seen_texts.add(updated_json_object["text"])
                        outfile.write(json.dumps(updated_json_object) + "\n")

            print(f"Processed {file} and saved to {output_file_path}")


processor = JSONLProcessor("./../datas/")
processor.process_files()
