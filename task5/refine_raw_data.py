import pandas as pd
import json
import re


filename = "./data/mrbench_v3_devset_train_data.json"

with open(filename, "r") as f:
    train_data = json.load(f)

df = pd.json_normalize(train_data, sep='.')
print("shape", df.shape)

label_columns = dict()
for col in df.columns:
    match = re.match(r"^tutor_responses\.([^.]+)\.response$", col)
    if match:
        label_columns[col] = match.group(1)

def create_finetuning_dataset():
    labeled_dataset = []
    for i, row in df.iterrows():
        for col in label_columns.keys():
            if row[col]:
                labeled_dataset.append({
                    "conversation_id": row["conversation_id"],
                    "conversation_history": row["conversation_history"],
                    "response": row[col],
                    "label": label_columns[col]
                })
    return pd.DataFrame(labeled_dataset)
print("shape final", create_finetuning_dataset().shape)
create_finetuning_dataset().to_csv(filename.replace(".json", ".csv"))