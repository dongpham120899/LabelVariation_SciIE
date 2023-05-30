import os
import json
from difflib import SequenceMatcher
from visualize_html import Visualize_HTML
from convert_json2brat import ConvertJson2Brat
from analysis_json import AnalysisData
import numpy as np

FIX_RELATION = ["Compare", "Used-for", "Part-of", "Feature-of", "Evaluate-for"]


def saving_data(data, saved_path):
    with open(saved_path, "w") as file:
        json.dump(data, file)


class CreateMixedDataset():
    def __init__(self):
        self.visualizator = ConvertJson2Brat()

    def _create_mixed_(self, sem_path, sci_path, visualized_path):
        sem_data = self.load_json(sem_path)
        sci_data = self.load_json(sci_path)

        count_overlapping = 0

        mixed_data, matched_data = [], []
        overlapped_sci_list = []
        overlapped_sem_list = []
        overlapped_sci_data = []
        overlapped_sem_data = []
        half_set = []
        for sample_1 in sci_data:
            for sample_2 in sem_data:
                abs_1 = " ".join(sample_1["tokens"])
                abs_2 = " ".join(sample_2["tokens"])

                sub_name = "SemEval-"+sample_2['orig_id'] + "_and_" + "SciERC-"+sample_1['orig_id']
                # if sub_name!="SemEval-C04-1096_and_SciERC-1_0_MATCHED":
                #     continue
                # calculate similarity score (matcher char)
                sim_core = self.check_sequence_matcher(abs_2, abs_1)
                # print("len 1", len(sample_1["tokens"]))
                # print("len 2", len(sample_2["tokens"]))

                # if sim_core > 0.6 and len(sample_1["tokens"])==len(sample_2['tokens']): # if two sample is overlapped
                if sim_core > 0.6: # if two sample is overlapped
                    if len(sample_1["tokens"])!=len(sample_2['tokens']):
                        print("sub_name", sub_name)
                        print()
                        print(abs_1)
                        print()
                        print(abs_2)
                        print("*"*100)
                    count_overlapping += 1

                    
                    # run mixed 2 dataset (transfer entities from Sci to Sem, mix relations and keep relation-levels)
                    mixed_entities = self.check_overlapped_entity_sample(sample_1, sample_2) # mixed enities 
                    mixed_relations, new_mixed_entities = self.get_mixed_relations(sample_1, sample_2, mixed_entities)
                    mixed_sample = dict(tokens=sample_2['tokens'], entities=new_mixed_entities, relations=mixed_relations, orig_id=sub_name+"_MIXED")
                    mixed_data.append(mixed_sample)

                    overlapped_sci_list.append(sample_1['orig_id'])
                    overlapped_sem_list.append(sample_2['orig_id'])

                    relabel_sample_1 = self.add_prefix_relation(sample_1, "sci")
                    relabel_sample_2 = self.add_prefix_relation(sample_2, "sem")
                    # relabel_sample_2 = self.add_prefix_entity(relabel_sample_2, "2")
                    overlapped_sci_data.append(relabel_sample_1)
                    overlapped_sem_data.append(relabel_sample_2)
                    if count_overlapping%2==0:
                        half_set.append(relabel_sample_1)
                    else: 
                        half_set.append(relabel_sample_2)

                    # macth 2 samples
                    match_sample = self.match_samples(sample_1, sample_2, sub_name)
                    matched_data.append(match_sample)

                    
                    


                    # ****************************************************************
                    # visualize with brat
                    sub_path = os.path.join(visualized_path, sub_name)
                    if os.path.exists(sub_path) is False:
                        os.mkdir(sub_path)
                    self.visualizator.convert_each_sample(sample_1, sub_path)
                    self.visualizator.convert_each_sample(sample_2, sub_path)
                    self.visualizator.convert_each_sample(mixed_sample, sub_path)
                    self.visualizator.convert_each_sample(match_sample, sub_path)
                    # ****************************************************************
                    break


        un_overlapped_sci = []
        for sample_1 in sci_data:
            if sample_1['orig_id'] not in overlapped_sci_list:
                un_overlapped_sci.append(self.add_prefix_relation((self.add_prefix_entity(sample_1, "sci")), "sci"))


        print("number of Overlapped sample:", count_overlapping)

        return mixed_data, un_overlapped_sci, overlapped_sci_data, overlapped_sem_data, half_set, matched_data
        
    def load_json(self, path):
        with open(path, "r") as file:
            data = json.load(file)
        return data
    
    def add_prefix_relation(self, sample, prefix):
        sample_copy = sample.copy()
        new_relations = []
        for rel in sample['relations']:
            new_type = prefix+"_"+rel["type"]
            rel['type'] = new_type
            new_relations.append(rel.copy())

        # sample_copy = dict(tokens=sample['token'], entities=sample['entities'], entities=new_relations, orig_id=sample['orig_id'])
        sample_copy['relations'] = new_relations

        return sample_copy
    
    def add_prefix_entity(self, sample, prefix):
        sample_copy = sample.copy()
        new_entities = []
        for ent in sample['entities']:
            new_type = prefix + "_" + ent["type"] 
            ent['type'] = new_type
            new_entities.append(ent.copy())

        sample_copy['entities'] = new_entities

        return sample_copy

    
    def check_sequence_matcher(self, sentence1, sentence2):
        return SequenceMatcher(None, sentence1, sentence2).ratio()
    
    def entity_overlapped(self, entity_1, entity_2):
        range_1 = np.arange(entity_1['start'],entity_1['end'])
        range_2 = np.arange(entity_2['start'],entity_2['end'])
        range_1_set = set(range_1)
        overlapped = range_1_set.intersection(range_2)

        # if entity_1['start'] >= entity_1['end']:
        if entity_1['start'] > entity_2['start']:
            start = entity_2['start']
        else:
            start = entity_1['start']

        if entity_1['end'] > entity_2['end']:
            end = entity_1['end']
        else:
            end = entity_2['end']

        if len(overlapped)!=0:
            return True, start, end
        else:
            return False, start, end

    def relation_overlapped(self, relation_1, relation_2, entities_1, entities_2):
        ent_1_h = entities_1[relation_1['head']]
        ent_1_t = entities_1[relation_1['tail']]
        rel_1_type = relation_1['type']

        ent_2_h = entities_2[relation_2['head']]
        ent_2_t = entities_2[relation_2['tail']]
        rel_2_type = relation_2['type']

        # check overlapped relations
        conflicing = False
        if self.entity_overlapped(ent_1_h, ent_2_h) and self.entity_overlapped(ent_1_t, ent_2_t):
            print(ent_1_h, ent_1_t)
            print(ent_2_h, ent_2_t)
            print(rel_1_type)
            print(rel_2_type)
            print("************")

            # check conflicing
            if rel_1_type!=rel_2_type:
                conflicing = True

        return conflicing
    
    def check_overlapped_relation_sample(self, sample_1, sample_2):
        relations_1 = sample_1['relations']
        relations_2 = sample_2['relations']
        entities_1 = sample_1['entities']
        entities_2 = sample_2['entities']

        conflicing = False
        for rel_1 in relations_1:
            for rel_2 in relations_2:
                conflicing = self.relation_overlapped(rel_1, rel_2, entities_1, entities_2)
                if conflicing is True:
                    break
                # break
            # break

        return conflicing
    
    def mapping_relation_with_new_entities(self, relations, mixed_entities, prefix_rel):
        mapped_relations = []
        for rel_1 in relations:
            for idx, entity in enumerate(mixed_entities):
                # if idx > 0:
                #     if self.check_duplicate_entity(entity, mixed_entities[idx-1]):
                #         value = idx - 1
                #     else:
                #         value = idx
                # else:
                #     value = idx
                value = idx
                if entity[prefix_rel]==rel_1['head']:
                    
                    # else:       
                    en_head = entity
                    idx_en_head = value
                if entity[prefix_rel]==rel_1['tail']:
                    en_tail = entity
                    idx_en_tail = value
            if rel_1['type'] not in FIX_RELATION:
                continue
            try:
                new_rel = dict(type=prefix_rel+"_"+rel_1['type'], head=idx_en_head, tail=idx_en_tail)
            except:
                print(prefix_rel)
                print("relations", relations)
                print(rel_1['head'])
                print("mixed_entities", mixed_entities)
            mapped_relations.append(new_rel)
        
        return mapped_relations
    
    def get_new_pos_entity(self, old_pos, old_entities, new_entities):
        old_entity =old_entities[old_pos]
        new_pos = 0
        for idx, entity in enumerate(new_entities):
            if entity['type']==old_entity['type'] and entity['start']==old_entity['start'] and entity['end']==old_entity['end']:
                new_pos = idx
                break

        return new_pos
    
    def match_samples(self, sample_1, sample_2, new_orig_id):
        # relabel_sample_1 = self.add_prefix_relation(sample_1, "sci")
        relabel_sample_1 = self.add_prefix_entity(sample_1, "sci")
        # relabel_sample_2 = self.add_prefix_relation(sample_2, "sem")
        relabel_sample_2 = self.add_prefix_entity(sample_2, "sem")

        entities_1 = relabel_sample_1['entities']
        entities_2 = relabel_sample_2['entities']

        entities = entities_1 + entities_2
        sorted_entities = sorted(entities, key=lambda x:x['start'])


        relations_1 = []
        for rel in relabel_sample_1['relations']:
            new_head = self.get_new_pos_entity(old_pos=rel['head'], old_entities=entities_1, new_entities=sorted_entities)
            new_tail = self.get_new_pos_entity(old_pos=rel['tail'], old_entities=entities_1, new_entities=sorted_entities)

            new_rel_1 = dict(type=rel['type'], head=new_head, tail=new_tail)
            relations_1.append(new_rel_1)


        relations_2 = []
        for rel in relabel_sample_2['relations']:
            new_head = self.get_new_pos_entity(old_pos=rel['head'], old_entities=entities_2, new_entities=sorted_entities)
            new_tail = self.get_new_pos_entity(old_pos=rel['tail'], old_entities=entities_2, new_entities=sorted_entities)

            new_rel_2 = dict(type=rel['type'], head=new_head, tail=new_tail)
            relations_2.append(new_rel_2)

        combined_relations = relations_1 + relations_2

        new_sample = dict(tokens=sample_2['tokens'], entities=sorted_entities, relations=combined_relations, orig_id=new_orig_id+"_MATCHED")

        return new_sample


    # check 2 entity
    def check_duplicate_entity(self, entity_1, entity_2):
        if entity_1['type']==entity_2['type'] and entity_1['start']==entity_2['start'] and entity_1['end']==entity_2['end']:
            return True
        else:
            return False
        
    def get_mixed_relations(self, sample_1, sample_2, mixed_entities):
        relations_1 = sample_1['relations']
        relations_2 = sample_2['relations']
        entities_1 = sample_1['entities']
        entities_2 = sample_2['entities']

        # Sci Relations
        new_relations_1 = self.mapping_relation_with_new_entities(relations=relations_1, mixed_entities=mixed_entities, prefix_rel="sci")
        # SemEval Relations
        new_relations_2 = self.mapping_relation_with_new_entities(relations=relations_2, mixed_entities=mixed_entities, prefix_rel="sem")

        # drop overlapped entity
        new_mixed_entities = [mixed_entities[0]]
        entity_pos = [[0,0]]
        count = 0
        for idx in range(1, len(mixed_entities)):
            if self.check_duplicate_entity(mixed_entities[idx], mixed_entities[idx-1]) is False:
                new_mixed_entities.append(mixed_entities[idx])
                entity_pos.append([idx, count])
            else:
                entity_pos.append([idx-1, count])
                count += 1


        # Merge
        merge_relations = new_relations_1 + new_relations_2
        merge_relations = sorted(merge_relations, key=lambda x: x['head'])

        new_relations = []
        for rel in merge_relations:
            rel_type = rel['type']
            new_head = entity_pos[rel['head']][0] - entity_pos[rel['head']][1]
            new_tail = entity_pos[rel['tail']][0] - entity_pos[rel['tail']][1]

            new_relations.append(dict(type=rel_type, head=new_head, tail=new_tail))


        return new_relations, new_mixed_entities

    
        new_relations = new_relations_1 + new_relations_2
        return new_relations, mixed_entities

    def mix_entities(self, entities_1, entities_2, entities_3):
        mixed_entities = entities_1 + entities_2 + entities_3

        return sorted(mixed_entities, key=lambda d: d['start'])

    def check_overlapped_entity_sample(self, sample_1, sample_2):
        entities_1 = sample_1['entities']
        entities_2 = sample_2['entities']
        tokens_1 = sample_1['tokens']
        tokens_2 = sample_2['tokens']

        # # print(entities_1)
        # print("entities_1", len(entities_1))
        # print("entities_2", len(entities_2))
        
        overlapped = []
        un_overlapped_SemEval = []
        un_overlapped_Sci = []
        for idx_1, entity_1 in enumerate(entities_1):
            for idx_2, entity_2 in enumerate(entities_2):
                is_entity_overlapped, start_e, end_e = self.entity_overlapped(entity_1, entity_2)
                if is_entity_overlapped:
                    # if entity_1['start']==entity_2['start'] and entity_1['end']==entity_2['end']:
                    #     print("Boundary Matcher", idx_1, tokens_1[entity_1['start']:entity_1['end']], idx_2, tokens_2[entity_2['start']:entity_2['end']])
                    # else:
                    #     print(entity_1)
                    #     print(entity_2)
                    #     print("Boundary Overlapped", idx_1, tokens_1[entity_1['start']:entity_1['end']], idx_2, tokens_2[entity_2['start']:entity_2['end']])


                    new_enity = dict(type=entity_1['type'], sci=idx_1, sem=idx_2, start=start_e, end=end_e)
                    # print(new_enity)
                    overlapped.append(new_enity)


        for i, entity_2 in enumerate(entities_2):
            if i not in [i['sem'] for i in overlapped]:
                new_enity = dict(type=entity_2['type'] + "_2", sci=None, sem=i, start=entity_2['start'], end=entity_2['end'])
                un_overlapped_SemEval.append(new_enity)

        for i, entity_1 in enumerate(entities_1):
            if i not in [i['sci'] for i in overlapped]:
                new_enity = dict(type=entity_1['type'] , sci=i, sem=None, start=entity_1['start'], end=entity_1['end'])
                un_overlapped_Sci.append(new_enity)


        # print("list of overlapped in SemEval", overlapped)
        # print("*"*20)
        # print("list of un-overlapped in SemEval", un_overlapped_SemEval)
        # print("*"*20)
        # print("list of un-overlapped in Sci", un_overlapped_Sci)

        entire_entities = self.mix_entities(overlapped, un_overlapped_SemEval, un_overlapped_Sci)

        return entire_entities
    
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
    


if __name__ == "__main__":
    semeval_path = "train/semeval-2018_train.json"
    train_scierc_path = "/Users/dongpham/Documents/NII_internship/datasets/SciERC_dataset/spert_dataset/train_data_1000_sentence.json"
    dev_scierc_path = "/Users/dongpham/Documents/NII_internship/datasets/SciERC_dataset/spert_dataset/dev_data_1000_sentence.json"
    test_scierc_path = "/Users/dongpham/Documents/NII_internship/datasets/SciERC_dataset/spert_dataset/test_data_1000_sentence.json"
    # saved_path = "overlapped_train_and_test" 
    visualized_path =  "/Users/dongpham/Documents/NII_internship/brat-1.3p1/data/examples/test_macth_sample"

    if os.path.exists(visualized_path) is False:
        os.mkdir(visualized_path)

    creator = CreateMixedDataset()
    ov_train_train_data, train_un_overlapped_sci, train_overlapped_sci_data, train_overlapped_sem_data, train_half_set, train_matched_data = creator._create_mixed_(sem_path=semeval_path, sci_path=train_scierc_path, visualized_path=visualized_path)
    ov_train_dev_data, dev_un_overlapped_sci, dev_overlapped_sci_data, dev_overlapped_sem_data, dev_half_set, dev_matched_data = creator._create_mixed_(sem_path=semeval_path, sci_path=dev_scierc_path, visualized_path=visualized_path)
    ov_train_test_data, test_un_overlapped_sci, test_overlapped_sci_data, test_overlapped_sem_data, test_half_set, test_matched_data = creator._create_mixed_(sem_path=semeval_path, sci_path=test_scierc_path, visualized_path=visualized_path)


    # ov_full_data = ov_train_train_data + ov_train_dev_data + ov_train_test_data
    # saving_data(ov_full_data, "experimental_ovp_dataset_v2/mixed_set/mixed_set.json")

    matched_set = train_matched_data + dev_matched_data + test_matched_data
    # saving_data(matched_set, "experimental_ovp_dataset_v2/matched_set/matched_set.json")

    # half_set = train_half_set + dev_half_set + test_half_set
    # saving_data(half_set, "experimental_ovp_dataset_v2/half_set/half_set.json")

    # test_sci_ovp = train_un_overlapped_sci + dev_un_overlapped_sci + test_un_overlapped_sci
    # saving_data(test_sci_ovp, "experimental_ovp_dataset_v2/sci_set/test_sci.json")

    # train_sci_ovp = train_overlapped_sci_data + dev_overlapped_sci_data + test_overlapped_sci_data
    # saving_data(train_sci_ovp, "experimental_ovp_dataset_v2/sci_set/train_sci.json")

    train_sem_ovp = train_overlapped_sem_data + dev_overlapped_sem_data + test_overlapped_sem_data
    # saving_data(train_sem_ovp, "experimental_ovp_dataset_v2/sem_set/train_sem.json")

    
    train_sem_data = creator.load_json("train/semeval-2018_train.json")
    train_ovp_sem_list = [a['orig_id'] for a in train_sem_ovp]
    the_others_train_sem_data = []
    for sample in train_sem_data:
        if sample['orig_id'] not in train_ovp_sem_list:
            the_others_train_sem_data.append(sample)

    test_sem_data = creator.load_json("test/semeval-2018_test.json")

    # test_sem_ovp = the_others_train_sem_data + test_sem_data
    # test_sem_ovp = [creator.add_prefix_entity(creator.add_prefix_relation(sample, "sem"),"sem") for sample in test_sem_ovp]
    # saving_data(test_sem_ovp, "experimental_ovp_dataset_v2/sem_set/test_sem.json")

    # print("mixed set", len(ov_full_data))
    # print("half set", len(half_set))
    # print("train sci set", len(train_sci_ovp))
    # print("test sci set", len(test_sci_ovp))
    # print("train sem set", len(train_sem_ovp))
    # print("test sem set", len(test_sem_ovp))
    # print("matched set", len(matched_set))

    # matched_set = train_matched_data + dev_matched_data + test_matched_data
    # test_sci_ovp = train_un_overlapped_sci + dev_un_overlapped_sci + test_un_overlapped_sci


    # sci_train_set = train_matched_data + train_un_overlapped_sci
    # sci_dev_set = dev_matched_data + dev_un_overlapped_sci

    # # sci_train_set = train_matched_data
    # # sci_dev_set = dev_matched_data

    # sci_train_set = sci_train_set + sci_dev_set
    # sci_test_set = test_matched_data + test_un_overlapped_sci

    # # print(train_un_overlapped_sci)
    # print("train", len(sci_train_set))
    # print("test", len(sci_test_set))

    # saving_data(sci_train_set, "experimental_ovp_dataset_v3/sci_set/train_sci.json")
    # saving_data(sci_test_set, "experimental_ovp_dataset_v3/sci_set/test_sci.json")

    # Standard SEM-EVAL 2018
    the_others_train_sem_data = [creator.add_prefix_entity(creator.add_prefix_relation(sample, "sem"),"sem") for sample in the_others_train_sem_data]
    train_sem_data_new = matched_set + the_others_train_sem_data
    test_sem_data = [creator.add_prefix_entity(creator.add_prefix_relation(sample, "sem"),"sem") for sample in test_sem_data]


    print("train_sem_ovp", len(train_sem_ovp))
    print("the_others_train_sem_data", len(the_others_train_sem_data))
    print("test_sem_data", len(test_sem_data))

    saving_data(train_sem_data_new, "standard_sem_dataset/train_sem.json")
    saving_data(test_sem_data, "standard_sem_dataset/test_sem.json")

    print_types(train_sem_data_new)
    # print_types(the_others_train_sem_data)
    print_types(test_sem_data)








