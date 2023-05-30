from abc import ABC
import torch.nn.functional as F
import torch


import torch

def relation_soft_loss(rel_logits, rel_types, rel_sample_masks, _rel_criterion, output_head=None):
    # relation loss
    rel_sample_masks = rel_sample_masks.view(-1).float()
    rel_count = rel_sample_masks.sum()
    if output_head=="softmax":
        rel_logits = F.softmax(rel_logits, dim=1)
    elif output_head=="log_softmax":
        rel_logits = F.log_softmax(rel_logits, dim=1)
    elif output_head=="sigmoid":
        rel_logits = torch.sigmoid(rel_logits)


    # print("rel_logits", rel_logits.shape)
    # print("rel_types", rel_types.shape)
    # rel_logits = ((rel_logits == 0).float() * (1e-6))
    # rel_logits = torch.where(rel_logits==0.0, 1e-6, rel_logits)
    # rel_types = torch.where(rel_types==0.0, 1e-6, rel_types)
    # print("rel_logits after", rel_types[0])

    rel_logits = rel_logits.view(-1, rel_logits.shape[-1])
    rel_types = rel_types.view(-1, rel_types.shape[-1])

    # prit

    rel_loss = _rel_criterion(rel_logits, rel_types)
    rel_loss = rel_loss.sum(-1) / rel_loss.shape[-1]
    rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count

    return rel_loss

def jensen_shannon_distance_relation(x1, x2, x1_mask, x2_mask, kl_loss):
    if x1_mask.sum() > x2_mask.sum():
        mask = x1_mask
    else:
        mask = x2_mask

    x1 = x1.type(torch.cuda.FloatTensor)
    x2 = x2.type(torch.cuda.FloatTensor)

    # print("mask", mask.shape)
    # print("x1", x1.shape)
    # print("x2", x2.shape)
    _, max_pairs = mask.shape
    mask = mask.view(-1).float()
    mask_count = mask.sum()
    # print("x1____",x1[0])

    # x1 = ((x1 == 0.0).float() * (1e-30))
    # x2 = ((x2 == 0.0).float() * (1e-30))

    # print("x1",x1[0])

    x1_pad = F.pad(x1,(0,0,0,max_pairs-x1.shape[1]), "constant", 0)
    x2_pad = F.pad(x2,(0,0,0,max_pairs-x2.shape[1]), "constant", 0)
    # print("x1_pad", x1_pad.shape)
    # print("x2_pad", x2_pad.shape)
    
    x1_pad = x1_pad.view(-1, x1_pad.shape[-1])
    x2_pad = x2_pad.view(-1, x2_pad.shape[-1])

    # print("x1_pad", x1_pad.shape)
    # print("x2_pad", x2_pad.shape)
    # print("x2", torch.unique(x2_pad))
    x1_pad = F.softmax(x1_pad, dim=1)
    x1_pad = F.softmax(x1_pad, dim=1)

    v = kl_loss(x1_pad, x2_pad)
    v = v.sum(-1) / v.shape[-1]
    v = (v*mask).sum() / mask_count
    # print("*"*100)

    return v


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class SpERTLoss(Loss):
    def __init__(self, rel_criterion, entity_criterion, model, optimizer, scheduler, max_grad_norm):
        self._rel_criterion = rel_criterion
        self._entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm

    def compute(self, entity_logits, rel_logits, entity_types, rel_types, entity_sample_masks, rel_sample_masks):
        # entity loss
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()

        # print("entity_logits", entity_logits.shape)
        # print("entity_types", entity_types.shape)

        entity_loss = self._entity_criterion(entity_logits, entity_types)
        entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

        # relation loss
        rel_sample_masks = rel_sample_masks.view(-1).float()
        rel_count = rel_sample_masks.sum()

        if rel_count.item() != 0:
            rel_logits = rel_logits.view(-1, rel_logits.shape[-1])
            rel_types = rel_types.view(-1, rel_types.shape[-1])

            # prit

            rel_loss = self._rel_criterion(rel_logits, rel_types)
            rel_loss = rel_loss.sum(-1) / rel_loss.shape[-1]
            rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count

            # joint loss
            train_loss = entity_loss + rel_loss
        else:
            # corner case: no positive/negative relation samples
            train_loss = entity_loss

        # train_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        # self._optimizer.step()
        # self._scheduler.step()
        # self._model.zero_grad()
        return train_loss
    

class SpERTRelationSoftLoss(Loss):
    def __init__(self, rel_criterion, entity_criterion, model, optimizer, scheduler, max_grad_norm, output_head, inverse=False):
        self._rel_criterion = rel_criterion
        self._entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm
        self.output_head = output_head
        self.inverse = inverse

    def compute(self, entity_logits, rel_logits, entity_types, rel_types, entity_sample_masks, rel_sample_masks):
        # entity loss
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()

        entity_loss = self._entity_criterion(entity_logits, entity_types)
        entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

        if self.output_head=="softmax":
            rel_logits = F.softmax(rel_logits, dim=1)
        elif self.output_head=="log_softmax":
            rel_logits = F.log_softmax(rel_logits, dim=1)
        elif self.output_head=="sigmoid":
            rel_logits = torch.sigmoid(rel_logits)

        # relation loss
        rel_sample_masks = rel_sample_masks.view(-1).float()
        rel_count = rel_sample_masks.sum()

        if rel_count.item() != 0:
            rel_logits = rel_logits.view(-1, rel_logits.shape[-1])
            rel_types = rel_types.view(-1, rel_types.shape[-1])

            # prit

            # rel_loss = self._rel_criterion(rel_logits, rel_types)
            if self.inverse:
                rel_loss = self._rel_criterion(rel_logits, rel_types)
            else:   
                rel_loss = self._rel_criterion(rel_types, rel_logits)
            rel_loss = rel_loss.sum(-1) / rel_loss.shape[-1]
            rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count

        # joint loss
            train_loss = entity_loss + rel_loss
        else:
            # corner case: no positive/negative relation samples
            train_loss = entity_loss

        return train_loss