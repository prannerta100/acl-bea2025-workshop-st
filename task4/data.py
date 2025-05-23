import numpy as np
import pandas as pd

import json


def create_dataframe(json_path: str, final_file_name: str) -> None:
    data = json.load(open(json_path, "r"))

    convos = []
    actions = []
    locations = []
    tutor = []
    mis_id = []
    guidances = []

    for n in range(len(data)):
        for k in data[n]["tutor_responses"]:
            hist = data[n]["conversation_history"]
            hist += "\nTutor: " + data[n]["tutor_responses"][k]["response"]

            action = data[n]["tutor_responses"][k]["annotation"]["Actionability"]
            loc = data[n]["tutor_responses"][k]["annotation"]["Mistake_Location"]
            mis = data[n]["tutor_responses"][k]["annotation"]["Mistake_Identification"]
            guide = data[n]["tutor_responses"][k]["annotation"]["Providing_Guidance"]

            convos.append(hist)
            actions.append(action)
            tutor.append(k)
            locations.append(loc)
            mis_id.append(mis)
            guidances.append(guide)


    df = pd.DataFrame()
    df["conversation"] = convos
    df["actionability"] = actions
    df["mis_location"] = locations
    df["tutor"] = tutor
    df["mis_id"] = mis_id
    df["guidance"] = guidances

    df = df.iloc[np.random.permutation(df.shape[0])]
    df.to_csv(final_file_name, index=None)


def prep_test(test_json_path: str, final_file_name: str) -> None:
    test = json.load(open(test_json_path, "r"))

    df = pd.DataFrame()

    conversation_id = []
    conversation_history = []
    tutor = []
    response = []

    for convo in test:
        for tut, resp in convo["tutor_responses"].items():
            conversation_id.append(convo["conversation_id"])
            conversation_history.append(convo["conversation_history"])
            tutor.append(tut)
            response.append(resp["response"])

    df["conversation_id"] = conversation_id
    df["conversation_history"] = conversation_history
    df["tutor"] = tutor
    df["response"] = response
    df["conversation"] = df["conversation_history"] + "\nTutor: " + df["response"]
    df.to_csv(final_file_name, index=False)


if __name__ == "__main__":

    np.random.seed(42)
    data = json.load(open("data/mrbench_v3_devset.json", "r"))
    test_indices = np.random.choice(range(300), 50, replace=False)
    train_data = [data[int(i)] for i in range(300) if i not in test_indices]
    test_data = [data[int(i)] for i in test_indices]

    json.dump(train_data, open("data/mrbench_v3_devset_train_data.json", "w"))
    json.dump(test_data, open("data/mrbench_v3_devset_test_data.json", "w"))

    create_dataframe(
        "data/mrbench_v3_devset_train_data.json",
        "data/train_all_bea.csv"
    )

    create_dataframe(
        "data/mrbench_v3_devset_test_data.json",
        "data/valid_all_bea.csv"
    )

    prep_test(
        "data/mrbench_v3_testset.json",
        "data/test.csv"
    )

