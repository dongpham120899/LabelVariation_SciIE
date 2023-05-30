import os
import os
import warnings
from typing import List, Tuple, Dict
import json
from collections import OrderedDict


import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from transformers import BertTokenizer

from spert import prediction
from spert.entities import Document, Dataset, EntityType, RelationType
from spert.input_reader import BaseInputReader
from spert.opt import jinja2

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

mapping_relation_types = {}
label_mapping = json.load(open("relation_mapping.json"), object_pairs_hook=OrderedDict)  # entity + relation types
none_relation_type = RelationType('None', 0, 'None', 'No Relation')
mapping_relation_types["None"] = none_relation_type
for i, (key, v) in enumerate(label_mapping['relations'].items()):
        relation_type = RelationType(key, i + 1, v['short'], v['verbose'], v['symmetric'])
        mapping_relation_types[key] = relation_type


# print("mapping_relation_types", mapping_relation_types)

class Evaluator:
    def __init__(self, dataset: Dataset, input_reader: BaseInputReader, text_encoder: BertTokenizer,
                 rel_filter_threshold: float, no_overlapping: bool,
                 predictions_path: str, examples_path: str, example_count: int, eval_on: str):
        self._text_encoder = text_encoder
        self._input_reader = input_reader
        self._dataset = dataset
        self._rel_filter_threshold = rel_filter_threshold
        self._no_overlapping = no_overlapping

        self._predictions_path = predictions_path
        self._examples_path = examples_path

        self._example_count = example_count

        # relations
        self._gt_relations_sci = []  # ground truth
        self._pred_relations_sci = []  # prediction
        self._gt_relations_sem = []  # ground truth
        self._pred_relations_sem = []  # prediction
        self._gt_relations_synthetic = []  # ground truth
        self._pred_relations_synthetic = []  # prediction

        # entities
        self._gt_entities_sci = []  # ground truth
        self._pred_entities_sci = []  # prediction
        self._gt_entities_sem = []  # ground truth
        self._pred_entities_sem = []  # prediction
        self._gt_entities_synthetic = []  # ground truth
        self._pred_entities_synthetic = []  # prediction

        self._pseudo_entity_type = EntityType('Entity', 1, 'Entity', 'Entity')  # for span only evaluation

        self._eval_on = eval_on
        self._convert_gt(self._dataset.documents)

    def eval_batch(self, batch_entity_clf: torch.tensor, batch_rel_clf: torch.tensor,
                   batch_rels: torch.tensor, batch_entity_sample_masks: torch.tensor, batch_entity_spans: torch.tensor, eval_on: str):
        batch_pred_entities, batch_pred_relations = prediction.convert_predictions(batch_entity_clf, batch_rel_clf,
                                                                                   batch_rels, batch_entity_sample_masks,
                                                                                   batch_entity_spans,
                                                                                   self._rel_filter_threshold,
                                                                                   self._input_reader,
                                                                                   no_overlapping=self._no_overlapping)

        if eval_on=="sci":
            self._pred_entities_sci.extend(batch_pred_entities)
            self._pred_relations_sci.extend(batch_pred_relations)
            # print("Sci pred", batch_pred_relations)
        elif eval_on=="sem":
            self._pred_entities_sem.extend(batch_pred_entities)
            self._pred_relations_sem.extend(batch_pred_relations)
            # print("Sem pred", batch_pred_relations)

        self._pred_entities_synthetic.extend(batch_pred_entities)
        self._pred_relations_synthetic.extend(batch_pred_relations)

    def eval_batch_multi_label(self, batch_entity_clf_1: torch.tensor, batch_rel_clf_1: torch.tensor,
                   batch_rels_1: torch.tensor, batch_entity_sample_masks_1: torch.tensor, batch_entity_spans_1: torch.tensor,
                   batch_entity_clf_2: torch.tensor, batch_rel_clf_2: torch.tensor,
                   batch_rels_2: torch.tensor, batch_entity_sample_masks_2: torch.tensor, batch_entity_spans_2: torch.tensor):

        batch_pred_entities_1, batch_pred_relations_1 = prediction.convert_predictions(batch_entity_clf_1, batch_rel_clf_1,
                                                                                   batch_rels_1, batch_entity_sample_masks_1,
                                                                                   batch_entity_spans_1,
                                                                                   self._rel_filter_threshold,
                                                                                   self._input_reader,
                                                                                   no_overlapping=self._no_overlapping)
        
        batch_pred_entities_2, batch_pred_relations_2 = prediction.convert_predictions(batch_entity_clf_2, batch_rel_clf_2,
                                                                                   batch_rels_2, batch_entity_sample_masks_2,
                                                                                   batch_entity_spans_2,
                                                                                   self._rel_filter_threshold,
                                                                                   self._input_reader,
                                                                                   no_overlapping=self._no_overlapping)

        self._pred_entities_sci.extend(batch_pred_entities_1)
        self._pred_relations_sci.extend(batch_pred_relations_1)

        self._pred_entities_sem.extend(batch_pred_entities_2)
        self._pred_relations_sem.extend(batch_pred_relations_2)

        syn_entities = self.mix_prediction(batch_pred_entities_1, batch_pred_entities_2, type="ent")
        syn_relations = self.mix_prediction(batch_pred_relations_1, batch_pred_relations_2, type="rel")

        # print("batch_pred_entities_1", batch_pred_entities_1)
        # print("batch_pred_relations_1", batch_pred_relations_1)
        # print("*****************************")
        # print("batch_pred_entities_2", batch_pred_entities_2)
        # print("batch_pred_relations_2", batch_pred_relations_2)
        # print("*****************************")
        # print("syn_entities", syn_entities)
        # print("syn_relations", syn_relations)
        # print("*****************************")
        # print("*****************************")
        # print("*****************************")


        self._pred_entities_synthetic.append(syn_entities)
        self._pred_relations_synthetic.append(syn_relations)

    def mix_prediction(self, batch_1, batch_2, type):
        def overlapped_entity(ent1, ent2):
            if ent1[0]==ent2[0] and ent1[1]==ent2[1] and ent1[2]==ent2[2]:
                return True
            else:
                return False
    
        def overlapped_relation(rel1, rel2):
            if overlapped_entity(rel1[0], rel2[0]) and overlapped_entity(rel1[1], rel2[1]) and rel1[2]==rel2[2]:
                return True
            else:
                return False
            
        def in_predictions(sample, predictions, type="ent"):
            if type=="ent":
                for pred in predictions:
                    if overlapped_entity(sample, pred):
                        return True
                    
                return False
            elif type=="rel":
                for pred in predictions:
                    if overlapped_relation(sample, pred):
                        return True
                return False


        predictions = []
        for interation in batch_1:
            for sample in interation:
                if in_predictions(sample, predictions, type):
                    continue
                else:
                    predictions.append(sample)

        for interation in batch_2:
            for sample in interation:
                if in_predictions(sample, predictions, type):
                    continue
                else:
                    predictions.append(sample)

        return predictions

        


        
        

    def compute_scores(self):
        print("Evaluation")

        # print("Relation Types:", self._input_reader.relation_types)

        # print("self._gt_entities_sci", self._gt_entities_sci)
        # print("self._pred_entities_sci", self._pred_entities_sci)
        # print("self._gt_entities_sem", len(self._gt_entities_sem))
        # print("self._pred_entities_sem", len(self._pred_entities_sem))

        print("")
        print("--- Entities (named entity recognition (NER)) --- SCI")
        print("An entity is considered correct if the entity type and span is predicted correctly")
        print("")
        gt, pred = self._convert_by_setting(self._gt_entities_sci, self._pred_entities_sci, include_entity_types=True)
        ner_eval = self._score(gt, pred, print_results=True)

        print("")
        print("--- Relations ---")
        print("")
        print("Without named entity classification (NEC)_ SCI")
        print("A relation is considered correct if the relation type and the spans of the two "
              "related entities are predicted correctly (entity type is not considered)")
        print("")
        gt, pred = self._convert_by_setting(self._gt_relations_sci, self._pred_relations_sci, include_entity_types=False)
        rel_eval = self._score(gt, pred, print_results=True, label_mapping=False)

        print("")
        print("--- Entities (named entity recognition (NER)) --- SEM")
        print("An entity is considered correct if the entity type and span is predicted correctly")
        print("")
        gt, pred = self._convert_by_setting(self._gt_entities_sem, self._pred_entities_sem, include_entity_types=True)
        ner_eval = self._score(gt, pred, print_results=True)

        print("")
        print("Without named entity classification (NEC) _ SEM")
        print("A relation is considered correct if the relation type and the two "
              "related entities are predicted correctly (in span and entity type)")
        print("")
        gt, pred = self._convert_by_setting(self._gt_relations_sem, self._pred_relations_sem, include_entity_types=False)
        rel_nec_eval = self._score(gt, pred, print_results=True, label_mapping=False)

        print("")
        print("--- Entities (named entity recognition (NER)) --- ALL")
        print("An entity is considered correct if the entity type and span is predicted correctly")
        print("")
        gt, pred = self._convert_by_setting(self._gt_entities_synthetic, self._pred_entities_synthetic, include_entity_types=True)
        ner_eval = self._score(gt, pred, print_results=True)

        print("")
        print("Without named entity classification (NEC) _ ALL")
        print("A relation is considered correct if the relation type and the two "
              "related entities are predicted correctly (in span and entity type)")
        print("")
        gt, pred = self._convert_by_setting(self._gt_relations_synthetic, self._pred_relations_synthetic, include_entity_types=False)
        rel_nec_eval = self._score(gt, pred, print_results=True, label_mapping=False)

        return ner_eval, rel_eval, rel_nec_eval
        # return 0, 0, 0

    def store_predictions(self):
        prediction.store_predictions(self._dataset.documents, self._pred_entities_sci,
                                     self._pred_relations_sci, self._predictions_path+"_sci"+".json")
        prediction.store_predictions(self._dataset.documents, self._pred_entities_sem,
                                     self._pred_relations_sem, self._predictions_path+"_sem"+".json")
        prediction.store_predictions(self._dataset.documents, self._pred_entities_synthetic,
                                     self._pred_relations_synthetic, self._predictions_path+"_sythetic"+".json")

    def store_examples(self):
        if jinja2 is None:
            warnings.warn("Examples cannot be stored since Jinja2 is not installed.")
            return

        entity_examples = []
        rel_examples = []
        rel_examples_nec = []

        for i, doc in enumerate(self._dataset.documents):
            # entities
            entity_example = self._convert_example(doc, self._gt_entities[i], self._pred_entities[i],
                                                   include_entity_types=True, to_html=self._entity_to_html)
            entity_examples.append(entity_example)

            # relations
            # without entity types
            rel_example = self._convert_example(doc, self._gt_relations[i], self._pred_relations[i],
                                                include_entity_types=False, to_html=self._rel_to_html)
            rel_examples.append(rel_example)

            # with entity types
            rel_example_nec = self._convert_example(doc, self._gt_relations[i], self._pred_relations[i],
                                                    include_entity_types=True, to_html=self._rel_to_html)
            rel_examples_nec.append(rel_example_nec)

        # entities
        self._store_examples(entity_examples[:self._example_count],
                             file_path=self._examples_path % 'entities',
                             template='entity_examples.html')

        self._store_examples(sorted(entity_examples[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % 'entities_sorted',
                             template='entity_examples.html')

        # relations
        # without entity types
        self._store_examples(rel_examples[:self._example_count],
                             file_path=self._examples_path % 'rel',
                             template='relation_examples.html')

        self._store_examples(sorted(rel_examples[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % 'rel_sorted',
                             template='relation_examples.html')

        # with entity types
        self._store_examples(rel_examples_nec[:self._example_count],
                             file_path=self._examples_path % 'rel_nec',
                             template='relation_examples.html')

        self._store_examples(sorted(rel_examples_nec[:self._example_count],
                                    key=lambda k: k['length']),
                             file_path=self._examples_path % 'rel_nec_sorted',
                             template='relation_examples.html')

    def _convert_gt(self, docs: List[Document]):
        if self._eval_on=="sci":
            idx_eval = 0
        elif self._eval_on=="sem":
            idx_eval = 1
        else:
            print("Don't match")

        for docs_ in docs:
            entities_ = []
            relations_ = []
            for idx, doc in enumerate(docs_):
                # doc = docs_[idx_eval]
                gt_relations = doc.relations
                gt_entities = doc.entities

                # convert ground truth relations and entities for precision/recall/f1 evaluation
                sample_gt_entities = [entity.as_tuple() for entity in gt_entities]
                sample_gt_relations = [rel.as_tuple() for rel in gt_relations]

                if self._no_overlapping:
                    sample_gt_entities, sample_gt_relations = prediction.remove_overlapping(sample_gt_entities,
                                                                                            sample_gt_relations)
                if idx==0:
                    self._gt_entities_sci.append(sample_gt_entities)
                    self._gt_relations_sci.append(sample_gt_relations)
                    # print("GT Sci", sample_gt_relations)
                elif idx==1:
                    self._gt_entities_sem.append(sample_gt_entities)
                    self._gt_relations_sem.append(sample_gt_relations)
                    # print("GT SEM", sample_gt_relations)
                entities_.append(sample_gt_entities)
                relations_.append(sample_gt_relations)
            entity_1, entity_2 = entities_
            relation_1, relation_2 = relations_

            

            syn_entity = self.mix_prediction([entity_1], [entity_2], type="ent")
            syn_relation = self.mix_prediction([relation_1], [relation_2], type="rel")

            # print("entity_1", entity_1)
            # print("entity_2", entity_2)
            # print("syn_entity", syn_entity)
            
            self._gt_entities_synthetic.append(syn_entity)
            self._gt_relations_synthetic.append(syn_relation)
            


    def _convert_by_setting(self, gt: List[List[Tuple]], pred: List[List[Tuple]],
                            include_entity_types: bool = True, include_score: bool = False):
        
        assert len(gt)==len(pred)

        # either include or remove entity types based on setting
        def convert(t):
            if not include_entity_types:
                # remove entity type and score for evaluation
                if type(t[0]) == int:  # entity
                    c = [t[0], t[1], self._pseudo_entity_type]
                else:  # relation
                    c = [(t[0][0], t[0][1], self._pseudo_entity_type),
                         (t[1][0], t[1][1], self._pseudo_entity_type), t[2]]
            else:
                c = list(t[:3])

            if include_score and len(t) > 3:
                # include prediction scores
                c.append(t[3])

            return tuple(c)

        converted_gt, converted_pred = [], []

        for sample_gt, sample_pred in zip(gt, pred):
            converted_gt.append([convert(t) for t in sample_gt])
            converted_pred.append([convert(t) for t in sample_pred])

        return converted_gt, converted_pred
    
    def mapping_relation_types(self, t, _relation_types=mapping_relation_types):
        orig_type = t.short_name
        type_split = orig_type.split("_")
        # print(type_split)
        if len(type_split) > 1:
            new_type = type_split[1]
        else:
            new_type = type_split[0]

        new_t = _relation_types[new_type]

        return new_t
        

    def _score(self, gt: List[List[Tuple]], pred: List[List[Tuple]], print_results: bool = False, label_mapping=False):
        # print(len(gt))
        # print(len(pred))
        assert len(gt) == len(pred)

        gt_flat = []
        pred_flat = []
        types = set()

        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()
            union.update(sample_gt)
            union.update(sample_pred)

            # print("sample_gt", sample_gt)
            # print("sample_pred", sample_pred)
            # print("*"*100)

            # print("union", union)

            for s in union:
                # print(s)
                if s in sample_gt:
                    t = s[2]
                    if label_mapping:
                        new_t = self.mapping_relation_types(t, mapping_relation_types)
                        gt_flat.append(new_t.index)
                        types.add(new_t)
                    else:
                        gt_flat.append(t.index)
                        types.add(t)

                else:
                    gt_flat.append(0)

                if s in sample_pred:
                    t = s[2]
                    if label_mapping:
                        # print(t.short_name)
                        new_t = self.mapping_relation_types(t, mapping_relation_types)
                        # print("new_t", new_t.short_name)
                        pred_flat.append(new_t.index)
                        types.add(new_t)
                    else:
                        pred_flat.append(t.index)
                        types.add(t)
                else:
                    pred_flat.append(0)

                # break

            # break

        # print("types", types)
        # print("prediction:")
        # print(gt_flat)
        # print(pred_flat)
        metrics = self._compute_metrics(gt_flat, pred_flat, types, print_results)

        # print("metrics", metrics)
        return metrics

    def _compute_metrics(self, gt_all, pred_all, types, print_results: bool = False):
        labels = [t.index for t in types]
        # print("labels", labels)
        # print("gt_all",gt_all)
        # print("pred_all", pred_all)
        per_type = prfs(gt_all, pred_all, labels=labels, average=None, zero_division=0)
        micro = prfs(gt_all, pred_all, labels=labels, average='micro', zero_division=0)[:-1]
        macro = prfs(gt_all, pred_all, labels=labels, average='macro', zero_division=0)[:-1]
        total_support = sum(per_type[-1])

        if print_results:
            self._print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], types)

        return [m * 100 for m in micro + macro]

    def _print_results(self, per_type: List, micro: List, macro: List, types: List):
        columns = ('type', 'precision', 'recall', 'f1-score', 'support')

        row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
        results = [row_fmt % columns, '\n']

        metrics_per_type = []
        for i, t in enumerate(types):
            metrics = []
            for j in range(len(per_type)):
                metrics.append(per_type[j][i])
            metrics_per_type.append(metrics)

        for m, t in zip(metrics_per_type, types):
            results.append(row_fmt % self._get_row(m, t.short_name))
            results.append('\n')

        results.append('\n')

        # micro
        results.append(row_fmt % self._get_row(micro, 'micro'))
        results.append('\n')

        # macro
        results.append(row_fmt % self._get_row(macro, 'macro'))

        # print(results)

        results_str = ''.join(results)
        print(results_str)

    def _get_row(self, data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i] * 100))
        row.append(data[3])
        return tuple(row)

    def _convert_example(self, doc: Document, gt: List[Tuple], pred: List[Tuple],
                         include_entity_types: bool, to_html):
        doc = doc[0]
        encoding = doc.encoding

        gt, pred = self._convert_by_setting([gt], [pred], include_entity_types=include_entity_types, include_score=True)
        gt, pred = gt[0], pred[0]

        # get micro precision/recall/f1 scores
        if gt or pred:
            pred_s = [p[:3] for p in pred]  # remove score
            precision, recall, f1 = self._score([gt], [pred_s])[:3]
        else:
            # corner case: no ground truth and no predictions
            precision, recall, f1 = [100] * 3

        scores = [p[-1] for p in pred]
        pred = [p[:-1] for p in pred]
        union = set(gt + pred)

        # true positives
        tp = []
        # false negatives
        fn = []
        # false positives
        fp = []

        for s in union:
            type_verbose = s[2].verbose_name

            if s in gt:
                if s in pred:
                    score = scores[pred.index(s)]
                    tp.append((to_html(s, encoding), type_verbose, score))
                else:
                    fn.append((to_html(s, encoding), type_verbose, -1))
            else:
                score = scores[pred.index(s)]
                fp.append((to_html(s, encoding), type_verbose, score))

        tp = sorted(tp, key=lambda p: p[-1], reverse=True)
        fp = sorted(fp, key=lambda p: p[-1], reverse=True)

        text = self._prettify(self._text_encoder.decode(encoding))
        return dict(text=text, tp=tp, fn=fn, fp=fp, precision=precision, recall=recall, f1=f1, length=len(doc.tokens))

    def _entity_to_html(self, entity: Tuple, encoding: List[int]):
        start, end = entity[:2]
        entity_type = entity[2].verbose_name

        tag_start = ' <span class="entity">'
        tag_start += '<span class="type">%s</span>' % entity_type

        ctx_before = self._text_encoder.decode(encoding[:start])
        e1 = self._text_encoder.decode(encoding[start:end])
        ctx_after = self._text_encoder.decode(encoding[end:])

        html = ctx_before + tag_start + e1 + '</span> ' + ctx_after
        html = self._prettify(html)

        return html

    def _rel_to_html(self, relation: Tuple, encoding: List[int]):
        head, tail = relation[:2]
        head_tag = ' <span class="head"><span class="type">%s</span>'
        tail_tag = ' <span class="tail"><span class="type">%s</span>'

        if head[0] < tail[0]:
            e1, e2 = head, tail
            e1_tag, e2_tag = head_tag % head[2].verbose_name, tail_tag % tail[2].verbose_name
        else:
            e1, e2 = tail, head
            e1_tag, e2_tag = tail_tag % tail[2].verbose_name, head_tag % head[2].verbose_name

        segments = [encoding[:e1[0]], encoding[e1[0]:e1[1]], encoding[e1[1]:e2[0]],
                    encoding[e2[0]:e2[1]], encoding[e2[1]:]]

        ctx_before = self._text_encoder.decode(segments[0])
        e1 = self._text_encoder.decode(segments[1])
        ctx_between = self._text_encoder.decode(segments[2])
        e2 = self._text_encoder.decode(segments[3])
        ctx_after = self._text_encoder.decode(segments[4])

        html = (ctx_before + e1_tag + e1 + '</span> '
                + ctx_between + e2_tag + e2 + '</span> ' + ctx_after)
        html = self._prettify(html)

        return html

    def _prettify(self, text: str):
        text = text.replace('_start_', '').replace('_classify_', '').replace('<unk>', '').replace('⁇', '')
        text = text.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
        return text

    def _store_examples(self, examples: List[Dict], file_path: str, template: str):
        template_path = os.path.join(SCRIPT_PATH, 'templates', template)

        # read template
        with open(os.path.join(SCRIPT_PATH, template_path)) as f:
            template = jinja2.Template(f.read())

        # write to disc
        template.stream(examples=examples).dump(file_path)
