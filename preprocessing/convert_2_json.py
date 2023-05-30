import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
import spacy
import pandas as pd
from tqdm import tqdm
from spacy.tokens import Doc
import json
from convert_json2brat import ConvertJson2Brat
visualizator = ConvertJson2Brat()

visualized_path =  "/Users/dongpham/Documents/NII_internship/brat-1.3p1/data/examples/test_processed_semeval"
if os.path.exists(visualized_path) is False:
    os.mkdir(visualized_path)

nlp = spacy.load("en_core_sci_sm")
# nlp = spacy.load("en_core_web_sm")
def custom_tokenizer(text):
    text = " ".join(text.split())
    tokens = text.split(" ")
    return Doc(nlp.vocab, tokens)
    #global tokens_dict
    #if text in tokens_dict:
    #   return Doc(nlp.vocab, tokens_dict[text])
    #else:
    #   VaueError("No tokenization for input text: ", text)

# nlp.tokenizer = custom_tokenizer

def get_punctuation_count(string):
    return len(re.findall(r'[\[\(\]\)]', string))


RELATION_MAPPING = {"COMPARE": "Compare", "USAGE":"Used-for", "PART_WHOLE":"Part-of", "MODEL-FEATURE":"Feature-of","RESULT":"Evaluate-for", "TOPIC":"Topic"}

def load_relation(path):
    with open(path, "r") as f:
        data = f.readlines()

    # print(data)
    relation_df = pd.DataFrame(columns=["doc_id", "entity_1", "entity_2", "relation_type", "reverse"])
    for idx, rel in enumerate(data):
        rel = rel.replace("\n","")
        matched = re.search(r"\(.*\)", rel)
        if len(matched[0][1:-1].split(","))==3:
            id_1, id_2, _ = matched[0][1:-1].split(",")
            reverse = True
        else:
            id_1, id_2 = matched[0][1:-1].split(",")
            reverse = False
        doc_id = id_1.split(".")[0]
        rel_type = rel.replace(matched[0], "")

        # print(doc_id, id_1, id_2, rel_type, reverse)

        relation_df.loc[idx] = [doc_id, id_1, id_2, rel_type, reverse]

        # break
    return relation_df

def matching_entity(tokens, entities, id_sample):
    matched_tokens = []
    entity_list = []
    start_p_list = []
    end_p_list = []

    df = pd.DataFrame(columns=['text', 'entity_type', 'start', 'end'])
    entity_dict = {}
    count = 0
    count_entity = 0
    for idx, token in enumerate(tokens):
        if token[:4]=="ENT-":
            # try:
            entity_text = entities[id_sample+"."+str(token[4:])]
            # except:
            #     print(tokens)
            matched_tokens.append(entity_text)
            entity_list.append(id_sample+"."+str(token[4:]))
            start_p_list.append(count)
            end_p_list.append(count+len(entity_text.split()))

            entity_dict[id_sample+"."+str(token[4:])] = [entity_text, count, count+len(entity_text.split()), count_entity]
            count_entity += 1
            count = count + len(entity_text.split())
        else:
            matched_tokens.append(token)
            entity_list.append("NO-ENTITY")
            start_p_list.append(count)
            end_p_list.append(count+len(token.split()))
            # print(count, token)

            count = count + len(token.split())




    # print(matched_toke
    # print(matched_tokens, entity_list, start_p_list, end_p_list)
    df['text'] = matched_tokens
    df['entity_type'] = entity_list
    df['start'] = start_p_list
    df['end'] = end_p_list

    return df, entity_dict


def find_entity(text):
    text = text.replace("<abstract>", "")
    text = text.replace("</abstract>", "")
    cmd = r"(?<=<entity id=)(.*?)(?=</entity>)"
    match = re.findall(cmd, text)
    for i, en in enumerate(match):
        entity_match = "<entity id=" + en + "</entity>"
        # print(entity_match)

        entity_id = re.findall(r"\"(.*?)\"", en)[0].split(".")[1]
        text = text.replace(entity_match, "ENT-{}".format(entity_id))

    return text.strip()

def preprocessing_text(text):
    text = text.replace("(", " -LRB- ")
    text = text.replace(")", " -RRB- ")
    text = text.replace("[", " -LSB- ")
    text = text.replace("]", " -RSB- ")
    text = text.replace("%", " % ")
    text = text.replace("culture-", "culture -")
    # text = text.replace("~1.25", "~ 1.25")
    text = text.replace("character-", "character -")
    text = text.replace("he/she/man/woman", "he/she/man / woman")
    text = text.replace("roughly.01", "roughly .01")
    text = text.replace("-hyponym", "- hyponym")
    text = text.replace("-speaking", "- speaking")

    text = text.replace("&lt;", "<")
    # text = text.replace("-hyponym", "hyponym ")
    # text = text.replace("-hyponym", "hyponym ")

    return " ".join(text.split())

def correct_spacy(tokens):
    text =  " ".join(tokens)
    text = text.replace("~1.25", "~ 1.25")
    text = text = text.replace("22 - 38", "22-38")
    text = text = text.replace("10 K sentences", "10K sentences")
    text = text.replace("113 K Chinese", "113K Chinese")
    text = text.replace("120 K English", "120K English")
    text = text.replace("~0.25", "~ 0.25")
    text = text.replace("MSR- closed", "MSR - closed")
    text = text.replace("character -based", "character-based")
    text = text.replace("-based text", "- based text")
    text = text.replace("acts , etc .", "acts , etc.")
    text = text.replace("the DoD.", "the DoD .")
    text = text.replace("standard 360 K floppy", "standard 360K floppy")
    text = text.replace("Models 1 - 2", "Models 1-2")
    text = text.replace("and P#P = P", "and P #P = P")
    text = text.replace("with NP- and", "with NP - and")
    text = text.replace("\" convenient \"", "`` convenient ''")
    text = text.replace("most suitabledata", "most suitable data")
    text = text.replace("than 400 GB", "than 400GB")
    text = text.replace("than 4 GB", "than 4GB")
    text = text.replace("Biber,1993 ; Nagao,1993 ; Smadja,1993", "Biber ,1993 ; Nagao ,1993 ; Smadja ,1993")

    text = text.replace("distance , viz . the", "distance , viz the")
    text = text.replace("several voting- and", "several voting - and")
    text = text.replace("Prolog , cf .", "Prolog , cf.")

    # text = text.replace("Establishing a \" best \" correspondence between the \" UNL-tree+L0 \" and the \" MS-L0 structure\" ,", "Establishing a `` best '' correspondence between the '' UNL-tree + L0 '' and the '' MS-L0 structure '' ,")
    text = text.replace("Windows ' 95 platforms", "Windows '95 platforms")
    text = text.replace("Establishing a \" best \"", "Establishing a `` best ''")
    text = text.replace("\" UNL-tree+L0 \"", "'' UNL-tree + L0 ''")
    text = text.replace("\" MS-L0 structure \"", "'' MS-L0 structure ''")






   
    # -based

    return text.split(" ")




def convert_xml_to_json(path, relation_df, save_path=None):
    with open(path, 'r') as f:
        data = f.read()

    Bs_data = BeautifulSoup(data, "xml")
    texts = Bs_data.find_all("text")
    documents = []
    for sample in tqdm(texts):
        doc_id = sample.get('id')
        # if True:
        # print(doc_id)
        doc_rel_df = relation_df[relation_df["doc_id"]==doc_id]
        abstract = sample.find('abstract')
        abstract_text = abstract.text.strip()
        entities = sample.find_all("entity")
        # print("a")
        # if get_punctuation_count(abstract_text)>0 and doc_id in ["J05-4003"]:
        # if get_punctuation_count(abstract_text)>0:
        if True:
            entities_dic = {}
            for entity in entities:
                id_entity = entity.get("id")
                text_entity = entity.text
                text_entity = " ".join([e.text for e in nlp(preprocessing_text(text_entity))])
                # print(text_entity)
                entities_dic[id_entity] = text_entity

            unseen_abstract = find_entity(preprocessing_text(str(abstract)))
            # print("unseen_abstract", unseen_abstract)
            unseen_tokens = nlp(unseen_abstract)
            original_tokens = nlp(preprocessing_text(abstract_text))
            # print("abstract_text", abstract_text)
            unseen_tokens = [token.text for token in unseen_tokens]
            original_tokens = [token.text for token in original_tokens]
            unseen_tokens = correct_spacy(unseen_tokens)
            original_tokens = correct_spacy(original_tokens)

            # print(original_tokens)
            # break
            # print("*")
            # print(unseen_tokens)
            # print(unseen_abstract)
            # print(original_tokens)

            sample_entity_df, sample_entity_dic = matching_entity(unseen_tokens, entities_dic, doc_id)

            # print(sample_entity_dic)
            # for i in range(len(sample_entity_dic)):
                # if 

            sample_dic = {}
            sample_dic["tokens"] = original_tokens
            entity_list = []
            for key, value in sample_entity_dic.items():
                entity_list.append(dict(type="OtherScientificTerm", start=value[1], end=value[2]))

            relation_list = []
            for i in range(len(doc_rel_df)):
                entity_id_1 = doc_rel_df.iloc[i]['entity_1']
                entity_id_2 = doc_rel_df.iloc[i]['entity_2']
                rel_type = doc_rel_df.iloc[i]['relation_type']
                reverse = doc_rel_df.iloc[i]['reverse']

                if reverse:
                    start = sample_entity_dic[entity_id_2][3]
                    # try:
                    end = sample_entity_dic[entity_id_1][3]
                    # except:
                        # print(sample_entity_dic)
                else:
                    # print(sample_entity_dic)
                    start = sample_entity_dic[entity_id_1][3]
                    end = sample_entity_dic[entity_id_2][3]

                relation_list.append(dict(type=RELATION_MAPPING[rel_type], head=start, tail=end))

            sample_dic["entities"] = entity_list
            sample_dic["relations"] = relation_list
            sample_dic['orig_id'] = doc_id

            # print(sample_dic)
            # precessed_sample = process_sample(sample_dic)

            documents.append(sample_dic)
            visualizator.convert_each_sample(sample_dic, visualized_path)
            # break

            # break
        # break
    # print(root)

    if save_path:
        with open(save_path, "w") as file:
            json.dump(documents, file)
        
            




if __name__ == "__main__":
    xml_path = "test/2.test.text.xml"
    relation_path = "test/keys.test.2.txt"
    save_path = "test/semeval-2018_test.json"
    relation_df = load_relation(relation_path)
    convert_xml_to_json(xml_path, relation_df, save_path)