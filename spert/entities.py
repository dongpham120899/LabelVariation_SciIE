from collections import OrderedDict
from typing import List
from torch.utils.data import Dataset as TorchDataset

from spert import sampling


class RelationType:
    def __init__(self, identifier, index, short_name, verbose_name, symmetric=False):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name
        self._verbose_name = verbose_name
        self._symmetric = symmetric

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

    @property
    def symmetric(self):
        return self._symmetric

    def __int__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, RelationType):
            return self._identifier == other._identifier
        return False

    def __hash__(self):
        return hash(self._identifier)


class EntityType:
    def __init__(self, identifier, index, short_name, verbose_name):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name
        self._verbose_name = verbose_name

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

    def __int__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, EntityType):
            return self._identifier == other._identifier
        return False

    def __hash__(self):
        return hash(self._identifier)


class Token:
    def __init__(self, tid: int, index: int, span_start: int, span_end: int, phrase: str):
        self._tid = tid  # ID within the corresponding dataset
        self._index = index  # original token index in document

        self._span_start = span_start  # start of token span in document (inclusive)
        self._span_end = span_end  # end of token span in document (exclusive)
        self._phrase = phrase

    @property
    def index(self):
        return self._index

    @property
    def span_start(self):
        return self._span_start

    @property
    def span_end(self):
        return self._span_end

    @property
    def span(self):
        return self._span_start, self._span_end

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Token):
            return self._tid == other._tid
        return False

    def __hash__(self):
        return hash(self._tid)

    def __str__(self):
        return self._phrase

    def __repr__(self):
        return self._phrase


class TokenSpan:
    def __init__(self, tokens):
        self._tokens = tokens

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    def __getitem__(self, s):
        if isinstance(s, slice):
            return TokenSpan(self._tokens[s.start:s.stop:s.step])
        else:
            return self._tokens[s]

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)


class Entity:
    def __init__(self, eid: int, entity_type: EntityType, tokens: List[Token], phrase: str):
        self._eid = eid  # ID within the corresponding dataset

        self._entity_type = entity_type

        self._tokens = tokens
        self._phrase = phrase

    def as_tuple(self):
        return self.span_start, self.span_end, self._entity_type

    @property
    def entity_type(self):
        return self._entity_type

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self._eid == other._eid
        return False

    def __hash__(self):
        return hash(self._eid)

    def __str__(self):
        return self._phrase


class Relation:
    def __init__(self, rid: int, relation_type: RelationType, head_entity: Entity,
                 tail_entity: Entity, reverse: bool = False):
        self._rid = rid  # ID within the corresponding dataset
        self._relation_type = relation_type

        self._head_entity = head_entity
        self._tail_entity = tail_entity

        self._reverse = reverse

        self._first_entity = head_entity if not reverse else tail_entity
        self._second_entity = tail_entity if not reverse else head_entity

    def as_tuple(self):
        head = self._head_entity
        tail = self._tail_entity
        head_start, head_end = (head.span_start, head.span_end)
        tail_start, tail_end = (tail.span_start, tail.span_end)

        t = ((head_start, head_end, head.entity_type),
             (tail_start, tail_end, tail.entity_type), self._relation_type)
        return t

    @property
    def relation_type(self):
        return self._relation_type

    @property
    def head_entity(self):
        return self._head_entity

    @property
    def tail_entity(self):
        return self._tail_entity

    @property
    def first_entity(self):
        return self._first_entity

    @property
    def second_entity(self):
        return self._second_entity

    @property
    def reverse(self):
        return self._reverse

    def __eq__(self, other):
        if isinstance(other, Relation):
            return self._rid == other._rid
        return False

    def __hash__(self):
        return hash(self._rid)


class Document:
    def __init__(self, doc_id: int, tokens: List[Token], entities: List[Entity], relations: List[Relation],
                 encoding: List[int]):
        self._doc_id = doc_id  # ID within the corresponding dataset

        self._tokens = tokens
        self._entities = entities
        self._relations = relations

        # byte-pair document encoding including special tokens ([CLS] and [SEP])
        self._encoding = encoding

    @property
    def doc_id(self):
        return self._doc_id

    @property
    def entities(self):
        return self._entities

    @property
    def relations(self):
        return self._relations

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def encoding(self):
        return self._encoding

    @encoding.setter
    def encoding(self, value):
        self._encoding = value

    def __eq__(self, other):
        if isinstance(other, Document):
            return self._doc_id == other._doc_id
        return False

    def __hash__(self):
        return hash(self._doc_id)


class BatchIterator:
    def __init__(self, entities, batch_size, order=None, truncate=False):
        self._entities = entities
        self._batch_size = batch_size
        self._truncate = truncate
        self._length = len(self._entities)
        self._order = order

        if order is None:
            self._order = list(range(len(self._entities)))

        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._truncate and self._i + self._batch_size > self._length:
            raise StopIteration
        elif not self._truncate and self._i >= self._length:
            raise StopIteration
        else:
            entities = [self._entities[n] for n in self._order[self._i:self._i + self._batch_size]]
            self._i += self._batch_size
            return entities


class Dataset(TorchDataset):
    TRAIN_MODE = 'train'
    EVAL_MODE = 'eval'

    def __init__(self, label, rel_types, entity_types, neg_entity_count,
                 neg_rel_count, max_span_size):
        self._label = label
        self._rel_types = rel_types
        self._entity_types = entity_types
        self._neg_entity_count = neg_entity_count
        self._neg_rel_count = neg_rel_count
        self._max_span_size = max_span_size
        self._mode = Dataset.TRAIN_MODE

        self._documents = OrderedDict()
        self._entities = OrderedDict()
        self._relations = OrderedDict()

        # current ids
        self._doc_id = 0
        self._rid = 0
        self._eid = 0
        self._tid = 0

    def iterate_documents(self, batch_size, order=None, truncate=False):
        return BatchIterator(self.documents, batch_size, order=order, truncate=truncate)

    def iterate_relations(self, batch_size, order=None, truncate=False):
        return BatchIterator(self.relations, batch_size, order=order, truncate=truncate)

    def create_token(self, idx, span_start, span_end, phrase) -> Token:
        token = Token(self._tid, idx, span_start, span_end, phrase)
        self._tid += 1
        return token

    def create_document(self, tokens, entity_mentions, relations, doc_encoding) -> Document:
        document = Document(self._doc_id, tokens, entity_mentions, relations, doc_encoding)
        self._documents[self._doc_id] = document
        self._doc_id += 1

        return document

    def create_document_multi_task(self, tokens, entity_mentions_1, relations_1, entity_mentions_2 , relations_2, doc_encoding) -> Document:
        document_1 = Document(self._doc_id, tokens, entity_mentions_1, relations_1, doc_encoding)
        document_2 = Document(self._doc_id, tokens, entity_mentions_2, relations_2, doc_encoding)
        document = [document_1, document_2]
        self._documents[self._doc_id] = document
        self._doc_id += 1

        return document


    def create_entity(self, entity_type, tokens, phrase) -> Entity:
        mention = Entity(self._eid, entity_type, tokens, phrase)
        self._entities[self._eid] = mention
        self._eid += 1
        return mention

    def create_relation(self, relation_type, head_entity, tail_entity, reverse=False) -> Relation:
        relation = Relation(self._rid, relation_type, head_entity, tail_entity, reverse)
        self._relations[self._rid] = relation
        self._rid += 1
        return relation

    def check_overlapped_entities(self, ent_1, ent_2):
        if ent_1.span_start==ent_2.span_start and ent_1.span_end==ent_2.span_end:
            return True
        else:
            return False
    
    def check_overlapped_relations(self, rel_1, rel_2):
        if self.check_overlapped_entities(rel_1._head_entity, rel_2._head_entity) is True and \
            self.check_overlapped_entities(rel_1._tail_entity, rel_2._tail_entity) is True:
            if rel_1._relation_type.index==rel_2._relation_type.index:
                return 1 # overlapped
            else:
                return 2 # conflicting
        
        else:
            return 0

    def create_softlabel(self, doc1, doc2):
        overlapped_rels_1, overlapped_rels_2 = [], []
        conflicted_rels_1, conflicted_rels_2 = {}, {}
        relations_1 = doc1.relations
        relations_2 = doc2.relations
        for rel_1 in relations_1:
            for rel_2 in relations_2:
                check = self.check_overlapped_relations(rel_1, rel_2)
                if check==1:
                    overlapped_rels_1.append(rel_1)
                    overlapped_rels_2.append(rel_2)
                    break
                if check==2:
                    conflicted_rels_1[rel_1] = rel_2
                    conflicted_rels_2[rel_2] = rel_1
                    break

        return overlapped_rels_1, conflicted_rels_1, overlapped_rels_2, conflicted_rels_2
        # print("relations_1",relations_1[0])

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, index: int):
        doc_1, doc_2 = self._documents[index]
        overlapped_rels_1, conflicted_rels_1, overlapped_rels_2, conflicted_rels_2 = self.create_softlabel(doc1=doc_1, doc2=doc_2)

        if self._mode == Dataset.TRAIN_MODE:
            sample_1 = sampling.create_train_sample(doc_1, self._neg_entity_count, self._neg_rel_count,
                                                self._max_span_size, len(self._rel_types), overlapped_rels_1, conflicted_rels_1)
            sample_2 = sampling.create_train_sample(doc_2, self._neg_entity_count, self._neg_rel_count,
                                                self._max_span_size, len(self._rel_types), overlapped_rels_2, conflicted_rels_2)
            
            return dict(encodings=sample_1["encodings"], context_masks=sample_1["context_masks"], 
                        entity_masks_1=sample_1["entity_masks"], entity_sizes_1=sample_1["entity_sizes"], entity_types_1=sample_1["entity_types"], 
                        entity_masks_2=sample_2["entity_masks"], entity_sizes_2=sample_2["entity_sizes"], entity_types_2=sample_2["entity_types"], 
                        rels_1=sample_1["rels"], rel_masks_1=sample_1["rel_masks"], rel_types_1=sample_1["rel_types"], soft_rel_types_1=sample_1["soft_rel_types"],
                        rels_2=sample_2["rels"], rel_masks_2=sample_2["rel_masks"], rel_types_2=sample_2["rel_types"], soft_rel_types_2=sample_2["soft_rel_types"],
                        entity_sample_masks_1=sample_1["entity_sample_masks"], rel_sample_masks_1=sample_1["rel_sample_masks"], 
                        entity_sample_masks_2=sample_2["entity_sample_masks"], rel_sample_masks_2=sample_2["rel_sample_masks"])
        else:
            sample_1 = sampling.create_eval_sample(doc_1, self._max_span_size)
            sample_2 = sampling.create_eval_sample(doc_2, self._max_span_size)
            return dict(encodings=sample_1["encodings"], context_masks=sample_1["context_masks"], 
                        entity_masks_1=sample_1["entity_masks"], entity_sizes_1=sample_1["entity_sizes"], entity_spans_1=sample_1["entity_spans"], entity_sample_masks_1=sample_1["entity_sample_masks"],
                        entity_masks_2=sample_2["entity_masks"], entity_sizes_2=sample_2["entity_sizes"], entity_spans_2=sample_2["entity_spans"], entity_sample_masks_2=sample_2["entity_sample_masks"])

    def switch_mode(self, mode):
        self._mode = mode

    @property
    def label(self):
        return self._label

    @property
    def input_reader(self):
        return self._input_reader

    @property
    def documents(self):
        return list(self._documents.values())

    @property
    def entities(self):
        return list(self._entities.values())

    @property
    def relations(self):
        return list(self._relations.values())

    @property
    def document_count(self):
        return len(self._documents)

    @property
    def entity_count(self):
        return len(self._entities)

    @property
    def relation_count(self):
        return len(self._relations)
