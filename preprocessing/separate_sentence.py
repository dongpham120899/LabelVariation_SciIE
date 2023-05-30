import os
import numpy as np
import json
import spacy
from spacy.tokens import Doc

nlp = spacy.load("en_core_sci_sm")
# nlp = spacy.load("en_core_web_sm")
def custom_tokenizer(text):
    text = " ".join(text.split())
    tokens = text.split(" ")
    return Doc(nlp.vocab, tokens)

# nlp.tokenizer = custom_tokenizer

from convert_json2brat import ConvertJson2Brat


class SeparateSentence():
    def __init__(self, num_sentence, padding, visualized_path) -> None:
        self.num_sents = num_sentence
        self.padding = padding
        self.visualize = ConvertJson2Brat()
        if os.path.exists(visualized_path) is False:
            os.mkdir(visualized_path)
        self.visualized_path = visualized_path
        

    def _run_separte_(self, json_path, saved_path):
        data = self.load_json(json_path)

        new_data = []
        for sample in data:
            abt = " ".join(sample['tokens']).lower()
            entities = sample['entities']
            relations = sample['relations']
            orig_id = sample['orig_id']



            # if orig_id!="D07-1073":
            #     continue
            # self.visualize.convert_each_sample(sample, self.visualized_path)
            # print(abt)

            # print(abt)

            doc = nlp(abt)
            new_sentences = []
            new_entities = []
            new_relations = []
            for sent in doc.sents:
                # sentence_text = [token.text for token in sent]
                sentence_text = sample['tokens'][sent.start:sent.end]
                # print(sent.start)
                # print(sent.end)
                # print(a)
                # print(sentence_text)
                # print("****")

                # sentence_text = sample['tokens']
                sent_entities, idx_sent_entities = self._transfer_entity(sent=sent, entities=entities)
                sent_relations = self._trasfer_relation(sent=sent, relations=relations, idx_sent_entities=idx_sent_entities)
                # print("*"*100)
                # sent_entities = list(map(dict, set(tuple(sorted(sub.items())) for sub in sent_entities)))
                # sent_entities = sorted(sent_entities, key=lambda x:x['start'])
                new_sentences.append(sentence_text)
                new_entities.append(sent_entities)
                new_relations.append(sent_relations)

                # print(sentence_text)
                
                # print(idx_sent_entities)
                # print()
                # print("entities",sent_entities)
                # print()
                # print("relations", sent_relations)
                # print()

                # print("*"*100)
            # print("*"*100)

                
            merge_data = self.merge_sentence(new_sentences, new_entities, new_relations, self.padding, orig_id)

            new_data.extend(merge_data)
            # break
        checked_data = []
        for sample in new_data:
            if self.check_correctly(sample) is True:
                checked_data.append(sample)
            else:
                print("Error")
            # break

        if saved_path:
            with open(saved_path, "w") as file:
                json.dump(checked_data, file)

        print("Number of data:", len(checked_data))

    def check_correctly(self, sample):
        tokens = sample['tokens']
        entities = sample['entities']
        # print("entities", len(entities))
        relations = sample['relations']
        orig_id = sample['orig_id']
        is_check = 0
        for idx,  entity in enumerate(entities):
            if entity['start'] > len(tokens) or entity['end'] > len(tokens):
                print("Error Entity at orig_id={} with index={}".format(orig_id, idx))
                print("tokens len", len(tokens))
                print("entities", entity)
                print("*" * 100)
                is_check+=1

        for idx, rel in enumerate(relations):
            if  rel['head']+1>len(entities) or rel['tail']+1>len(entities):
                print("Error Relation at orig_id={} with index={}".format(orig_id, idx))
                print("entities len", len(entities))
                print("relations", rel)
                print("*" * 100)
                is_check+=1

        if is_check==0:
            return True
        return False


    # def processing_text(self, text):
    #     text = text.replace("-LRB-", "(")
    #     text = text.replace("-RRB-", ")")
    #     text = text.replace("-LSB-", "[")
    #     text = text.replace("-RSB-", "]")

    #     return text
    
    def merge_sentence(self, sentence_list, entities_list, relations_list, padding=1, orig_id=None): # sliding window 
        assert padding < self.num_sents, "Padding should be smaller than num of sentences"
        # adding padding
        for idx in range(padding):
            sentence_list.insert(0, [])
            entities_list.insert(0, [])
            relations_list.insert(0, [])
            sentence_list.append([])
            entities_list.append([])
            relations_list.append([])

        if len(sentence_list) < self.num_sents:
            num_sents = len(sentence_list)
        else:
            num_sents = self.num_sents

        merge_data = []
        num_token_pre_sent = 0
        num_entities_pre = 0

        for idx in range(len(sentence_list)-(num_sents-1)):
            # print(idx)
            concat_sent = []
            concat_entity = []
            concat_relation = []
            if idx!=0:
                num_token_pre_sent += len(sentence_list[idx-1])
                num_entities_pre += len(entities_list[idx-1])
            for j in range(0, num_sents):
                convert_pos_entity = [dict(type=e['type'], start=e['start']-num_token_pre_sent, end=e['end']-num_token_pre_sent) for e in entities_list[idx+j]]
                convert_pos_rel = [dict(type=r['type'], head=r['head']-num_entities_pre, tail=r['tail']-num_entities_pre) for r in relations_list[idx+j]]
                concat_sent.extend(sentence_list[idx+j])
                concat_entity.extend(convert_pos_entity)
                concat_relation.extend(convert_pos_rel)

            # print(len(concat_sent))
            # print(concat_sent)
            # print("num_token_pre_sent", num_token_pre_sent)
            # print(concat_entity)
            # print("num_entities_pre", num_entities_pre)
            # print(concat_relation)
            # # print(len(concat_sent))
            # print("*"*100)

            # print


            new_sample = dict(tokens=concat_sent, entities=concat_entity, relations=concat_relation, orig_id="from_"+str(idx-padding)+"_to_"+str(idx-padding+num_sents)+"_"+orig_id)
            # try:
            self.visualize.convert_each_sample(new_sample, self.visualized_path)
            # except:
                # print(new_sample)
                # pass
                
            merge_data.append(new_sample)

        return merge_data

            # break




    def _transfer_entity(self, sent, entities):
        sent_entities = []
        idx_sent_entities = []
        for idx, entity in enumerate(entities):
            # print(entity)
            s_e = entity['start']
            e_e = entity['end']
            if s_e >= sent.start and e_e <= sent.end:
                sent_entities.append(entity)
                idx_sent_entities.append(idx)
            # print(s_e, e_e)

        return sent_entities, idx_sent_entities
    
    def _trasfer_relation(self, sent, relations, idx_sent_entities):
        # print(idx_sent_entities)
        sent_relations = []
        for rel in relations:
            # print(rel)
            if rel['head'] in idx_sent_entities and rel['tail'] in idx_sent_entities :
                sent_relations.append(rel)
                # print(rel)

        return sent_relations


    def load_json(self, path):
        with open(path, "r") as file:
            data = json.load(file)
        return data

if __name__ == "__main__":
    num_sentence = 1
    padding = 0
    mixed_json_path = "experimental_ovp_dataset_v2/mixed_set/mixed_set.json"
    mixed_saved_path = "experimental_ovp_dataset_v2/mixed_set/mixed_set_sentence_{}_padding_{}.json".format(num_sentence, padding)

    matched_json_path = "experimental_ovp_dataset_v2/matched_set/matched_set.json"
    matched_saved_path = "experimental_ovp_dataset_v2/matched_set/matched_set_sentence_{}_padding_{}.json".format(num_sentence, padding)

    half_json_path = "experimental_ovp_dataset_v2/half_set/half_set.json"
    half_saved_path = "experimental_ovp_dataset_v2/half_set/half_set_sentence_{}_padding_{}.json".format(num_sentence, padding)

    train_sci_json_path = "experimental_ovp_dataset_v2/sci_set/train_sci.json"
    train_sci_saved_path = "experimental_ovp_dataset_v2/sci_set/train_sci_sentence_{}_padding_{}.json".format(num_sentence, padding)

    test_sci_json_path = "experimental_ovp_dataset_v2/sci_set/test_sci.json"
    test_sci_saved_path = "experimental_ovp_dataset_v2/sci_set/test_sci_sentence_{}_padding_{}.json".format(num_sentence, padding)

    train_sem_json_path = "experimental_ovp_dataset_v2/sem_set/train_sem.json"
    train_sem_saved_path = "experimental_ovp_dataset_v2/sem_set/train_sem_sentence_{}_padding_{}.json".format(num_sentence, padding)

    test_sem_json_path = "experimental_ovp_dataset_v2/sem_set/test_sem.json"
    test_sem_saved_path = "experimental_ovp_dataset_v2/sem_set/test_sem_sentence_{}_padding_{}.json".format(num_sentence, padding)

    visualized_path =  "/Users/dongpham/Documents/NII_internship/brat-1.3p1/data/examples/check_separate_sentence"

    separator = SeparateSentence(num_sentence=1, padding=0, visualized_path=visualized_path)
    # separator._run_separte_(mixed_json_path, mixed_saved_path)

    # separator._run_separte_(matched_json_path, matched_saved_path)

    # separator._run_separte_(half_json_path, half_saved_path)

    # separator._run_separte_(train_sci_json_path, train_sci_saved_path)

    # separator._run_separte_(test_sci_json_path, test_sci_saved_path)

    # separator._run_separte_(train_sem_json_path, train_sem_saved_path)

    # separator._run_separte_(test_sem_json_path, test_sem_saved_path)


    # train_sci_path = "experimental_ovp_dataset_v3/sci_set/train_sci.json"
    # saved_train_sci_path = "experimental_ovp_dataset_v3/sci_set/train_sci_sentence_{}_padding_{}.json".format(num_sentence, padding)
    # test_sci_path = "experimental_ovp_dataset_v3/sci_set/test_sci.json"
    # saved_test_sci_path = "experimental_ovp_dataset_v3/sci_set/test_sci_sentence_{}_padding_{}.json".format(num_sentence, padding)

    # separator._run_separte_(train_sci_path, saved_train_sci_path)
    # separator._run_separte_(test_sci_path, saved_test_sci_path)

    # train_scierc_path = "/Users/dongpham/Documents/NII_internship/datasets/SciERC_dataset/spert_dataset/train_data_1000_sentence.json"
    # dev_scierc_path = "/Users/dongpham/Documents/NII_internship/datasets/SciERC_dataset/spert_dataset/dev_data_1000_sentence.json"
    # test_scierc_path = "/Users/dongpham/Documents/NII_internship/datasets/SciERC_dataset/spert_dataset/test_data_1000_sentence.json"
    # saved_test_scierc_path = "standard_sets/sci_set/test_sci_sentence_{}_padding_{}.json".format(num_sentence, padding)

    # separator._run_separte_(test_scierc_path, saved_test_scierc_path)

    train_sem_path = "standard_sem_dataset/train_sem.json"
    saved_train_sem_path = "standard_sem_dataset/train_sem_sentence_{}_padding_{}.json".format(num_sentence, padding)
    test_sem_path = "standard_sem_dataset/test_sem.json"
    saved_test_sem_path = "standard_sem_dataset/test_sem_sentence_{}_padding_{}.json".format(num_sentence, padding)

    separator._run_separte_(train_sem_path, saved_train_sem_path)
    separator._run_separte_(test_sem_path, saved_test_sem_path)
    # separator._run_separte_(test_sci_path, saved_test_sci_path)


    
