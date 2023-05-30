import random
import numpy as np

import torch

from spert import util


def create_soft_relation_label(rel_type_index, rel_type_conflicted_index, rel_type_count, agreement):
    rel_type_index = int(rel_type_index) - 1
    soft_label = np.zeros(rel_type_count)
    for idx in range(rel_type_count):
        if idx==rel_type_index:
            if agreement==1: # high agreement
                soft_score = 0.9
            elif agreement==0: # medium agreement
                soft_score = 0.8
            elif agreement==2: # low agreement
                soft_score = 0.6
            soft_label[idx] = soft_score
        elif agreement==2 and idx==rel_type_conflicted_index:
            soft_label[idx] = 0.31
        else:
            if agreement==1: # high agreement
                the_others_score = 0.025
            elif agreement==0: # medium agreement
                the_others_score = 0.05
            elif agreement==2: # low agreement
                the_others_score = 0.03
            soft_label[idx] = the_others_score

    return list(soft_label)


def create_train_sample(doc, neg_entity_count: int, neg_rel_count: int, max_span_size: int, rel_type_count: int, overlapped_rels: list, conflicted_rels: list):
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    # print("doc", doc._doc_id)
    # print("entities", doc.entities)
    # print("relations", doc.relations)
    # print("tokens", doc.tokens)
    # print("encoding", doc.encoding)

    # positive entities
    pos_entity_spans, pos_entity_types, pos_entity_masks, pos_entity_sizes = [], [], [], []
    for e in doc.entities:
        pos_entity_spans.append(e.span)
        pos_entity_types.append(e.entity_type.index)
        pos_entity_masks.append(create_entity_mask(*e.span, context_size))
        pos_entity_sizes.append(len(e.tokens))

        # print("e.span",e.span)
        # print("e.entity_type.index",e.entity_type.index)

    # positive relations

    # collect relations between entity pairs
    entity_pair_relations = dict()
    for rel in doc.relations:
        pair = (rel.head_entity, rel.tail_entity)
        # print(pair)
        if pair not in entity_pair_relations:
            entity_pair_relations[pair] = []
        entity_pair_relations[pair].append(rel)

    # build positive relation samples
    # print("overlapped_rels", overlapped_rels)
    pos_soft_rel_types = []
    pos_rels, pos_rel_spans, pos_rel_types, pos_rel_masks = [], [], [], []
    for pair, rels in entity_pair_relations.items():
        head_entity, tail_entity = pair
        s1, s2 = head_entity.span, tail_entity.span
        pos_rels.append((pos_entity_spans.index(s1), pos_entity_spans.index(s2)))
        pos_rel_spans.append((s1, s2))

        pair_rel_types = [r.relation_type.index for r in rels]
        pair_rel_types = [int(t in pair_rel_types) for t in range(1, rel_type_count)]
        pos_rel_types.append(pair_rel_types)
        pos_rel_masks.append(create_rel_mask(s1, s2, context_size))


        soft_label_rels = []
        for rel in rels:
            if rel in overlapped_rels:
                # print("OVERLAPPED")
                rel_type_index = rel.relation_type.index
                soft_label = create_soft_relation_label(rel_type_index, None, rel_type_count-1, agreement=1)
            elif rel in conflicted_rels.keys():
                # print("CONFLICTING")
                rel_type_index = rel.relation_type.index
                rel_type_conflicted_index = conflicted_rels[rel].relation_type.index
                soft_label = create_soft_relation_label(rel_type_index, rel_type_conflicted_index, rel_type_count-1, agreement=2)
            else:
                rel_type_index = rel.relation_type.index
                soft_label = create_soft_relation_label(rel_type_index, None, rel_type_count-1, agreement=0)
            soft_label_rels.append(soft_label)
        soft_label_rel_dis = np.mean(soft_label_rels, axis=0)
        pos_soft_rel_types.append(list(soft_label_rel_dis))

        # print("(pos_entity_spans.index(s1), pos_entity_spans.index(s2))", (pos_entity_spans.index(s1), pos_entity_spans.index(s2)))
    # print("pos_rel_types", len(pos_rel_types))
    # print("pos_soft_rel_types", len(pos_soft_rel_types))

    # print("*"*100)


    # negative entities
    neg_entity_spans, neg_entity_sizes = [], []
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            if span not in pos_entity_spans:
                neg_entity_spans.append(span)
                neg_entity_sizes.append(size)

    # sample negative entities
    neg_entity_samples = random.sample(list(zip(neg_entity_spans, neg_entity_sizes)),
                                       min(len(neg_entity_spans), neg_entity_count))
    neg_entity_spans, neg_entity_sizes = zip(*neg_entity_samples) if neg_entity_samples else ([], [])

    neg_entity_masks = [create_entity_mask(*span, context_size) for span in neg_entity_spans]
    neg_entity_types = [0] * len(neg_entity_spans)

    # negative relations
    # use only strong negative relations, i.e. pairs of actual (labeled) entities that are not related
    neg_rel_spans = []

    for i1, s1 in enumerate(pos_entity_spans):
        for i2, s2 in enumerate(pos_entity_spans):
            # do not add as negative relation sample:
            # neg. relations from an entity to itself
            # entity pairs that are related according to gt
            if s1 != s2 and (s1, s2) not in pos_rel_spans:
                neg_rel_spans.append((s1, s2))

    # sample negative relations
    neg_rel_spans = random.sample(neg_rel_spans, min(len(neg_rel_spans), neg_rel_count))

    neg_rels = [(pos_entity_spans.index(s1), pos_entity_spans.index(s2)) for s1, s2 in neg_rel_spans]
    neg_rel_masks = [create_rel_mask(*spans, context_size) for spans in neg_rel_spans]
    neg_rel_types = [(0,) * (rel_type_count-1)] * len(neg_rel_spans)

    # merge
    entity_types = pos_entity_types + neg_entity_types
    entity_masks = pos_entity_masks + neg_entity_masks
    entity_sizes = pos_entity_sizes + list(neg_entity_sizes)

    rels = pos_rels + neg_rels
    rel_types = pos_rel_types + neg_rel_types
    soft_rel_types = pos_soft_rel_types + neg_rel_types
    rel_masks = pos_rel_masks + neg_rel_masks

    assert len(entity_masks) == len(entity_sizes) == len(entity_types)
    assert len(rels) == len(rel_masks) == len(rel_types) == len(soft_rel_types)

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)

    # print("encodings", encodings)
    # print("context_masks", context_masks)

    # also create samples_masks:
    # tensors to mask entity/relation samples of batch
    # since samples are stacked into batches, "padding" entities/relations possibly must be created
    # these are later masked during loss computation
    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg entities)
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    if rels:
        rels = torch.tensor(rels, dtype=torch.long)
        rel_masks = torch.stack(rel_masks)
        rel_types = torch.tensor(rel_types, dtype=torch.float32)
        soft_rel_types = torch.tensor(soft_rel_types, dtype=torch.float32)
        rel_sample_masks = torch.ones([rels.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg relations)
        rels = torch.zeros([1, 2], dtype=torch.long)
        rel_types = torch.zeros([1, rel_type_count-1], dtype=torch.float32)
        soft_rel_types = torch.zeros([1, rel_type_count-1], dtype=torch.float32)
        rel_masks = torch.zeros([1, context_size], dtype=torch.bool)
        rel_sample_masks = torch.zeros([1], dtype=torch.bool)


    # print("*"*100)

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_sizes=entity_sizes, entity_types=entity_types,
                rels=rels, rel_masks=rel_masks, rel_types=rel_types, soft_rel_types=soft_rel_types,
                entity_sample_masks=entity_sample_masks, rel_sample_masks=rel_sample_masks)


def create_eval_sample(doc, max_span_size: int):
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    # create entity candidates
    entity_spans = []
    entity_masks = []
    entity_sizes = []

    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            entity_spans.append(span)
            entity_masks.append(create_entity_mask(*span, context_size))
            entity_sizes.append(size)

    # create tensors
    # token indices
    _encoding = encodings
    encodings = torch.zeros(context_size, dtype=torch.long)
    encodings[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.zeros(context_size, dtype=torch.bool)
    context_masks[:len(_encoding)] = 1

    # entities
    if entity_masks:
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans = torch.tensor(entity_spans, dtype=torch.long)

        # tensors to mask entity samples of batch
        # since samples are stacked into batches, "padding" entities possibly must be created
        # these are later masked during evaluation
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
    else:
        # corner case handling (no entities)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_sizes=entity_sizes, entity_spans=entity_spans, entity_sample_masks=entity_sample_masks)


# def create_train_multi_task_sample(doc1, doc2, neg_entity_count: int, neg_rel_count: int, max_span_size: int, rel_type_count: int):
#     sample_1 = create_train_multi_task_sample(doc1, neg_entity_count, neg_rel_count, max_span_size, rel_type_count)
#     sample_2 = create_train_multi_task_sample(doc2, neg_entity_count, neg_rel_count, max_span_size, rel_type_count)

#     return sample_1, sample_2


def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)
    return mask


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch
