import os
import pandas as pd
import numpy as np

def sum_up(label_name, type="last"):
    path = os.path.join("data","log", label_name)
    list_results = os.listdir(path)[-5:]
    print(list_results)
    ner_f1_list, re_f1_list = [], []
    for sub_name in list_results:
        result_path = os.path.join(path, sub_name, "eval_valid.csv")
        result_data = pd.read_csv(result_path, delimiter=";")
        if len(result_data) < 1: 
            continue

        ner_f1_micro_results = result_data['ner_f1_micro'].to_numpy()
        rel_f1_micro_results = result_data['rel_f1_micro'].to_numpy()
        if type=="last":
            ner_f1_micro = ner_f1_micro_results[-1]
            rel_f1_micro = rel_f1_micro_results[-1]
        elif type=="best":
            rel_f1_micro = np.max(rel_f1_micro_results)
            idx = list(rel_f1_micro_results).index(rel_f1_micro)
            ner_f1_micro = ner_f1_micro_results[idx]

        ner_f1_list.append(ner_f1_micro)
        re_f1_list.append(rel_f1_micro)
            
        # break

    print("Average micro NER F1 score: {}".format(np.mean(ner_f1_list)))
    print("Average micro RE F1 score: {}".format(np.mean(re_f1_list)))


sum_up("inverse_loss", "last")