import glob
import json
import os


class SVMpreprocesing:
    def __init__(self, datas_location):
        self.datas_location = datas_location
        self.output_suffix = "SVM"

    def delete_unnecessary_columns(self, json_object):
        json_object.pop("title", None)
        json_object.pop("asin", None)
        json_object.pop("parent_asin", None)
        json_object.pop("user_id", None)
        json_object.pop("timestamp", None)
        json_object.pop("helpful_vote", None)
        json_object.pop("verified_purchase", None)

        return json_object

    def convert_rate(self, rating):
        if rating <= 2.5:
            rating = 0
        else:
            rating = 1

        return rating

    def process_json_object(self, json_object):
        # Delete unnecessary data because svm only accept 2
        json_object = self.delete_unnecessary_columns(json_object)

        # Convert rating
        json_object["rating"] = self.convert_rate(json_object["rating"])

        return json_object

    def preprocesing(self):
        files = glob.glob(os.path.join(self.datas_location, "*.jsonl"))

        # directory for svm exists
        directory = os.path.join(self.datas_location, "..", "SVM")
        if not os.path.exists(directory):
            os.makedirs(directory)

        all_texts = []
        all_labels = []

        output_file_path = os.path.join(directory, f"{self.output_suffix}.jsonl")

        for file in files:
            if self.output_suffix in file:
                continue

            print(f"Processing {file}...")

            with open(file, "r") as infile:
                for line in infile:
                    json_object = json.loads(line)
                    updated_json_object = self.process_json_object(json_object)

                    all_texts.append(updated_json_object["text"])
                    all_labels.append(updated_json_object["rating"])

            output_data = {
                "text": all_texts,
                "label": all_labels,
            }

            with open(output_file_path, "w") as outfile:
                json.dump(output_data, outfile)

            print(f"Processed {file} and saved to {output_file_path}")


data_directory = os.path.join(os.path.dirname(__file__), "..", "..", "datas", "Output")
processor = SVMpreprocesing(data_directory)
processor.preprocesing()
