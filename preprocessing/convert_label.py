import os
import json
from sklearn.model_selection import train_test_split
import numpy as np
# from convert_2_json import ConvertJson2Brat
# visualizator = ConvertJson2Brat()


FIX_RELATION = ["Compare", "Used-for", "Part-of", "Feature-of", "Evaluate-for"]

def correct_pos_relations(rel, old_enitities, new_entities):
    old_head = rel['head']
    old_tail = rel['tail']

    entity_head = old_enitities[old_head]
    entity_tail = old_enitities[old_tail]

    new_head = 0
    new_tail = 0
    for idx, new_entity in enumerate(new_entities):
        if check_overlaped_entity(new_entity, entity_head):
            new_head = idx
        if check_overlaped_entity(new_entity, entity_tail):
            new_tail = idx

    return new_head, new_tail

def create_separate_data(data):
    new_data = []
    for sample in data:
        entities = sample['entities']
        relations = sample['relations']

        new_sample = sample.copy()
        new_sample['sci_entities'] = entities
        new_sample['sem_entities'] = entities
        new_sample['sci_relations'] = relations
        new_sample['sem_relations'] = relations

        new_data.append(new_sample)

    return new_data

def separate_data_label(data):
    new_data = []
    for sample in data:
        text = sample['tokens']
        if len(text)>500:
            continue
        entities = sample['entities']
        relations = sample['relations']

        sci_entities, sem_entities = [], []
        for entity in entities:
            entity_type_split = entity['type'].split("_")
            if len(entity_type_split) > 1:
                new_entity_type = entity['type']
                if entity_type_split[0]=='sci':
                    sci_entities.append(dict(type=new_entity_type, start=entity['start'], end=entity['end']))
                elif entity_type_split[0]=="sem":
                    sem_entities.append(dict(type=new_entity_type, start=entity['start'], end=entity['end']))

        sci_relations, sem_relations = [], []
        for rel in relations:
            rel_type_split = rel['type'].split("_")
            if len(rel_type_split) > 1:
                new_rel_type = rel['type']
                if rel_type_split[0]=='sci':
                    head, tail = correct_pos_relations(rel, entities, sci_entities)
                    sci_relations.append(dict(type=new_rel_type, head=head, tail=tail))
                elif rel_type_split[0]=="sem":
                    head, tail = correct_pos_relations(rel, entities, sem_entities)
                    sem_relations.append(dict(type=new_rel_type, head=head, tail=tail))

        new_sample = sample.copy()
        new_sample['sci_entities'] = sci_entities
        new_sample['sem_entities'] = sem_entities
        new_sample['sci_relations'] = sci_relations
        new_sample['sem_relations'] = sem_relations

        new_data.append(new_sample)

    return new_data

def split_data(data_path):
    data = load_json(data_path)
    range_ = range(0, len(data))
    print(len(range_))
    X_train, X_test = train_test_split(range_, test_size=0.1, random_state=42)

    train, test = [], []
    for idx in X_train:
        train.append(data[idx])

    for idx in X_test:
        test.append(data[idx])

    print("train", len(train))
    print("test", len(test))

    return train, test

def check_duplicated_relation_pairs(rel_1, rel_2):
    if rel_1['type']==rel_2['type'] and rel_1['head']==rel_2['head'] and rel_1['tail']==rel_2['tail']:
        return True
    elif rel_1['type']==rel_2['type'] and rel_1['head']==rel_2['tail'] and rel_1['head']==rel_2['tail']:
        return True
    else:
        return False

def check_conflicting_relation(rel_1, rel_2):
    if rel_1['type']!=rel_2['type'] and rel_1['head']==rel_2['head'] and rel_1['tail']==rel_2['tail']:
        return True
    elif rel_1['type']!=rel_2['type'] and rel_1['head']==rel_2['tail'] and rel_1['head']==rel_2['tail']:
        return True
    else:
        return False

def check_conflicting_entity(rel_1, entities):
    ent_1_head = entities[rel_1['head']]
    ent_1_tail = entities[rel_1['tail']]


    if ent_1_head==ent_1_tail:
        return True
    else:
        return False

def check_overlaped_entity(ent_1, ent_2):
    if ent_1['type']==ent_2['type'] and ent_1['start']==ent_2['start'] and ent_1['end']==ent_2['end']:
        return True
    else:
        return False


def remove_duplicated_priotity(data, priority="Sci"):
    new_data = []
    for sample in data:
        
        relations = sample['relations']
        if len(relations)==0:
            new_data.append(sample)
            continue
        sorted_relations = sorted(relations, key=lambda x: x['head'])

        if priority=="Sci":
            new_relations = [sorted_relations[0]]
            for i in range(1, len(sorted_relations)):
                if check_conflicting_relation(sorted_relations[i-1], sorted_relations[i]):
                    continue
                else:
                    new_relations.append(sorted_relations[i])
        elif print_types=="Sem":
            new_relations = [sorted_relations[-1]]
            for i in range(0, len(sorted_relations)-1):
                if check_conflicting_relation(sorted_relations[i], sorted_relations[i+1]):
                    # new_relations.append(sorted_relations[i+1])
                    continue
                else:
                    new_relations.append(sorted_relations[i])

        else:
            new_relations = [sorted_relations[0]]
            for i in range(1, len(sorted_relations)):
                if check_duplicated_relation_pairs(sorted_relations[i-1], sorted_relations[i]):
                    continue
                else:
                    new_relations.append(sorted_relations[i])


        # solving conflicting relations

        for idx, rel in enumerate(new_relations):
            if check_conflicting_entity(rel, sample['entities']):
                new_relations.pop(idx)
        if sample['orig_id']=="from_0_to_1_SemEval-X96-1059_and_SciERC-0_0_MIXED":
            print(relations)
            print()
            print(new_relations)
            print()
            print(sample['entities'])

        new_data.append(dict(tokens=sample['tokens'], entities=sample['entities'], relations=new_relations, orig_id=sample['orig_id']))
        
    return new_data

def check(texts, orig_id):
    for text in texts:
        if text in orig_id:
            return True
    return False

def convert_entity_label(data, name="entities"):
    new_data = []
    for sample in data:
        new_labels = []
        for a  in sample[name]:
            new_type = a['type']
            # new_type = "OtherScientificTerm"
            new_label = dict(type=new_type, start=a['start'], end=a['end'])
            new_labels.append(new_label)
        new_sample = sample.copy()
        new_sample[name] = new_labels
        new_data.append(new_sample)

    return new_data

def fix_relation_label(data):
    new_data = []
    for sample in data:
        new_labels = []
        for a in sample["relations"]:
                if len(a['type'].split("_")) > 1 and a['type'].split("_")[1] in FIX_RELATION:
                    new_type = a['type']
                elif a['type'] in FIX_RELATION:
                    new_type = a['type'] 
                else: continue
                new_label = dict(type=new_type, head=a['head'], tail=a['tail'])
                new_labels.append(new_label)
        # print(new_labels)
        new_sample = sample.copy()
        new_sample["relations"] = new_labels
        new_data.append(new_sample)

    return new_data

def convert_relation_label(data, name="relations"):
    new_data = []
    for sample in data:
        new_labels = []
        for a in sample[name]:
            # new_type = "_".join(a['type'].split("_")[1])
            new_type = a['type'].split("_")[1]
            # new_type = "OtherScientificTerm"
            new_label = dict(type=new_type, head=a['head'], tail=a['tail'])
            new_labels.append(new_label)
        new_sample = sample.copy()
        new_sample[name] = new_labels
        new_data.append(new_sample)

    return new_data

def print_types(data):
    entity_types = []
    relation_types = []
    for sample in data:
        s_entity_type = []
        s_relation_type = []
        s_entity_type = [ent['type'] for ent in sample['entities']]
        s_relation_type = [rel['type'] for rel in sample['relations']]

        entity_types.extend(s_entity_type)
        relation_types.extend(s_relation_type)

    print("Entity types:", np.unique(entity_types))
    print("Relation types:", np.unique(relation_types))
    print("*"*100)

def removing_rel_types(data, rel_type):
    for sample in data:
        relations = sample['sem_relations']
        new_rels = []
        for rel in relations:
            if rel['type']==rel_type:
                continue
            new_rels.append(rel)

        sample['sem_relations']=new_rels

    return data

    
def saving_data(data, saved_path):
    with open(saved_path, "w") as file:
        json.dump(data, file)

def load_json(path):
    with open(path, "r") as file:
        data = json.load(file)

    return data

if __name__ == "__main__":
    train_sem_path = "experimental_ovp_dataset_v2/sem_set/train_sem_sentence_1_padding_0.json"
    test_sem_path = "experimental_ovp_dataset_v2/sem_set/test_sem_sentence_1_padding_0.json"

    train_sci_path = "experimental_ovp_dataset_v2/sci_set/train_sci_sentence_1_padding_0.json"
    test_sci_path = "experimental_ovp_dataset_v2/sci_set/test_sci_sentence_1_padding_0.json"

    mixed_path = "experimental_ovp_dataset_v2/mixed_set/mixed_set_sentence_1_padding_0.json"
    matched_path = "experimental_ovp_dataset_v2/matched_set/matched_set_sentence_1_padding_0.json"
    half_path = "experimental_ovp_dataset_v2/half_set/half_set_sentence_1_padding_0.json"
    cross_path = "experimental_ovp_dataset_v2/crossRE_set/ai-test.json"


    # # transfer all entities to only one type + get only mapping relation types
    # for path in [train_sem_path, test_sem_path, train_sci_path, test_sci_path, half_path]:
    #     data = load_json(path)
    #     new_data = convert_entity_label(data)
    #     new_data = fix_relation_label(new_data)
    #     new_data = convert_relation_label(new_data)
    #     print_types(new_data)
    #     saved_name = path.split("/")[-1].split(".")[0] + "(no_entity_type+no_mapping)" + ".json"
    #     saved_path = os.path.join("/".join(path.split("/")[:-1]), saved_name)
    #     saving_data(new_data, saved_path)



    # visualized_path =  "/Users/dongpham/Documents/NII_internship/brat-1.3p1/data/examples/check_processing_mixed_set"
    # if os.path.exists(visualized_path) is False:
    #     os.mkdir(visualized_path)
    # processing mixed set: transfer all entities + only mapping relation types + remove duplicated relations and conflicting
    # for path in [mixed_path]:
    #     data = load_json(path)
    #     print("original len", len(data))
    #     new_data = convert_entity_label(data)
    #     new_data = fix_relation_label(new_data)
    #     new_data = convert_relation_label(new_data)
    #     new_data = remove_duplicated_priotity(new_data, priority="Sem")
    #     print("processing len", len(new_data))
    #     print_types(new_data)

    #     saved_name = path.split("/")[-1].split(".")[0] + "(no_entity_type+no_mapping+Sem)" + ".json"
    #     saved_path = os.path.join("/".join(path.split("/")[:-1]), saved_name)
    #     saving_data(new_data, saved_path)


    # transfer all entities to only one type
    # for path in [train_sem_path, test_sem_path, train_sci_path, test_sci_path, half_path, matched_path, mixed_path]:
    #     data = load_json(path)
    #     new_data = convert_entity_label(data)
    #     new_data = fix_relation_label(new_data)
    #     # new_data = convert_relation_label(new_data)
    #     print_types(new_data)
    #     saved_name = path.split("/")[-1].split(".")[0] + "(no_entity_type)" + ".json"
    #     saved_path = os.path.join("/".join(path.split("/")[:-1]), saved_name)
    #     saving_data(new_data, saved_path)


    # separate data label in match
    # for path in [matched_path, test_sci_path, test_sem_path]:
    # for path in [matched_path]:
    #     data = load_json(path)
    #     new_data = fix_relation_label(data)
    #     new_data = separate_data_label(new_data)
    #     new_data = convert_entity_label(new_data, name="sci_entities")
    #     # new_data = convert_relation_label(new_data, name="sci_relations")
    #     new_data = convert_entity_label(new_data, name="sem_entities")
    #     # new_data = convert_relation_label(new_data, name="sem_relations")
    #     print(new_data[0])
    #     print_types(new_data)
        # break
        # saved_name = path.split("/")[-1].split(".")[0] + "(separate_data_label)" + ".json"
        # saved_path = os.path.join("/".join(path.split("/")[:-1]), saved_name)
        # saving_data(new_data, saved_path)


    # create separate data from cross data
    # for path in [cross_path]:
    #     data = load_json(path)
    #     new_data = create_separate_data(data)
    #     print(new_data[10])
    #     print_types(new_data)
    #     # break
    #     saved_name = path.split("/")[-1].split(".")[0] + "(separate_data_label)" + ".json"
    #     saved_path = os.path.join("/".join(path.split("/")[:-1]), saved_name)
    #     saving_data(new_data, saved_path)


    # separate data label in Sci dataset with standard separtating train and test
    train_sci_path = "experimental_ovp_dataset_v4/sci_set/train_sci_sentence_1_padding_0.json"
    # train_sci_path_ov = "experimental_ovp_dataset_v3/sci_set/train_sci(overlapped)_sentence_1_padding_0.json"
    test_sci_path = "experimental_ovp_dataset_v4/sci_set/test_sci_sentence_1_padding_0.json"
    # for path in [train_sci_path, test_sci_path]:
    # # for path in [test_sciREX_path]:
    #     data = load_json(path)
    #     # new_data = fix_relation_label(data)
    #     new_data = separate_data_label(data)
    #     new_data = convert_entity_label(new_data, name="sci_entities")
    #     new_data = convert_relation_label(new_data, name="sci_relations")
    #     new_data = convert_entity_label(new_data, name="sem_entities")
    #     new_data = convert_relation_label(new_data, name="sem_relations")
    #     print(new_data[0])
    #     print_types(new_data)
    #     # break
    #     saved_name = path.split("/")[-1].split(".")[0] + "(separate_data_label+have_entity_types+no_fix_label)" + ".json"
    #     saved_path = os.path.join("/".join(path.split("/")[:-1]), saved_name)
    #     saving_data(new_data, saved_path)


    # test_sciREX_path = "experimental_ovp_dataset_v4/sciREX/all_sciREX.json"
    # for path in [test_sciREX_path]:
    #     data = load_json(path)
    #     # new_data = fix_relation_label(data)
    #     new_data = separate_data_label(data)
    #     new_data = convert_entity_label(new_data, name="sci_entities")
    #     new_data = convert_relation_label(new_data, name="sci_relations")
    #     new_data = convert_entity_label(new_data, name="sem_entities")
    #     new_data = convert_relation_label(new_data, name="sem_relations")
    #     print(new_data[0])
    #     print_types(new_data)
        # break
        # saved_name = path.split("/")[-1].split(".")[0] + "(separate_data_label+have_entity_types+no_fix_label)" + ".json"
        # saved_path = os.path.join("/".join(path.split("/")[:-1]), saved_name)
        # saving_data(new_data, saved_path)


    # test_sci_path = "experimental_ovp_dataset_v4/sci_set/test_sci_sentence_1_padding_0(separate_data_label+have_entity_types+no_fix_label).json"
    # train_sci_path = "experimental_ovp_dataset_v4/sci_set/train_sci_sentence_1_padding_0(separate_data_label+have_entity_types+no_fix_label).json"
    # for path in [test_sci_path, train_sci_path]:
    #     data = load_json(path)
    #     new_data = removing_rel_types(data, rel_type="Evaluate-for")
    #     new_data = removing_rel_types(data, rel_type="Topic")
    #     print_types(new_data)

    #     saved_name = path.split("/")[-1].split(".")[0] + "+no_sem_evaluate" + ".json"
    #     saved_path = os.path.join("/".join(path.split("/")[:-1]), saved_name)
    #     saving_data(new_data, saved_path)


    # test_sem_path = "standard_sem_dataset/test_sem_sentence_1_padding_0.json"
    # train_sem_path = "standard_sem_dataset/train_sem_sentence_1_padding_0.json"
    # for path in [train_sem_path, test_sem_path]:
    #     data = load_json(path)
    #     # new_data = fix_relation_label(data)
    #     new_data = separate_data_label(data)
    #     new_data = convert_entity_label(new_data, name="sci_entities")
    #     new_data = convert_relation_label(new_data, name="sci_relations")
    #     new_data = convert_entity_label(new_data, name="sem_entities")
    #     new_data = convert_relation_label(new_data, name="sem_relations")
    #     print(new_data[0])
    #     print_types(new_data)
    #     # break
    #     saved_name = path.split("/")[-1].split(".")[0] + "(separate_data_label+have_entity_types+no_fix_label)" + ".json"
    #     saved_path = os.path.join("/".join(path.split("/")[:-1]), saved_name)
    #     saving_data(new_data, saved_path)

    # train_sci_path = "experimental_ovp_dataset_v4/sci_set/train_sci_sentence_1_padding_0(separate_data_label+have_entity_types+no_fix_label).json"
    # for path in [train_sci_path]:
    #     new_data = []
    #     data = load_json(path)
    #     for sample in data:
    #         if len(sample['sem_entities'])==0 and len(sample['sem_relations'])==0:
    #             continue

    #         new_data.append(sample)

    #     print("lenght", len(new_data))
    #     print_types(new_data)

    #     saved_name = path.split("/")[-1].split(".")[0] + "+only_ov" + ".json"
    #     saved_path = os.path.join("/".join(path.split("/")[:-1]), saved_name)
    #     saving_data(new_data, saved_path)

    train_matched = "experimental_ovp_dataset_v2/matched_set/matched_set_sentence_1_padding_0(separate_data_label+no_mapping).json"
    for path in [train_matched]:
        data = load_json(path)
        print(len(data))
        for idx in range(1,11):
            print(idx)
            new_data = data[:-int(idx*100)]
            print(len(new_data))

            saved_path = "experimental_ovp_dataset_v4/reduced_data/train_reduced_{}.json".format(idx)
            saving_data(new_data, saved_path)
    
        
            






    

    



    
