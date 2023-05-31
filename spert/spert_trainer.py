import argparse
import math
import os
from typing import Type

import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
import transformers
from torch.utils.data import DataLoader
from transformers import AdamW, BertConfig
from transformers import BertTokenizer, AutoTokenizer
import torch.nn.functional as F

from spert import models, prediction
from spert import sampling
from spert import util
from spert.entities import Dataset
from spert.evaluator import Evaluator
from spert.input_reader import JsonInputReader, BaseInputReader
from spert.loss import SpERTLoss, Loss, SpERTRelationSoftLoss
from tqdm import tqdm
from spert.trainer import BaseTrainer

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class SpERTTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # byte-pair encoding
        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                        do_lower_case=args.lowercase,
                                                        cache_dir=args.cache_path)

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args
        train_label, valid_label = 'train', 'valid'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("Model type: %s" % args.model_type)

        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer, args.neg_entity_count,
                                        args.neg_relation_count, args.max_span_size, self._logger)
        train_dataset = input_reader.read(train_path, train_label)
        validation_dataset = input_reader.read(valid_path, valid_label)
        self._log_datasets(input_reader)

        # print(list(train_dataset))

        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // args.train_batch_size
        updates_total = updates_epoch * args.epochs

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        # load model
        model = self._load_model(input_reader)

        # SpERT is currently optimized on a single GPU and not thoroughly tested in a multi GPU setup
        # If you still want to train SpERT on multiple GPUs, uncomment the following lines
        # # parallelize model
        # if self._device.type != 'cpu':
        #     model = torch.nn.DataParallel(model)

        model.to(self._device)

        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        # create scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total)
        # create loss function
        rel_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')

        # rel_soft_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        rel_soft_criterion = torch.nn.KLDivLoss(reduction="none", log_target=False) # If you compute KL standard, let set log_target True and output_head="log_softmax"
                                                                                    # KL inverse don't need log normalization
        # rel_soft_criterion = torch.nn.BCELoss(reduction='none')
        compute_loss = SpERTLoss(rel_criterion, entity_criterion, model, optimizer, scheduler, args.max_grad_norm)
        compute_soft_loss = SpERTRelationSoftLoss(rel_soft_criterion, entity_criterion, model, optimizer, scheduler, args.max_grad_norm, output_head="softmax", inverse=args.inverse_soft_loss)
        # compute_soft_loss = SpERTRelationSoftLoss(rel_soft_criterion, model, optimizer, scheduler, args.max_grad_norm)

        # eval validation set
        if args.init_eval:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch, eval_on=args.eval_on)

        # train
        for epoch in range(args.epochs):
            # train epoch
            self._train_epoch(model, compute_loss, compute_soft_loss, optimizer, train_dataset, updates_epoch, epoch, alpha=args.alpha, soft_label=args.use_soft)

            # eval validation sets
            if not args.final_eval or (epoch == args.epochs - 1):
                self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch, eval_on=args.eval_on)

        # save final model
        extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
        global_iteration = args.epochs * updates_epoch
        self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                         optimizer=optimizer if self._args.save_optimizer else None, extra=extra,
                         include_iteration=False, name='final_model')

        self._logger.info("Logged in: %s" % self._log_path)
        self._logger.info("Saved in: %s" % self._save_path)
        self._close_summary_writer()

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args
        print("EVAL_ON args", args.eval_on)

        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer,
                                        max_span_size=args.max_span_size, logger=self._logger)
        test_dataset = input_reader.read(dataset_path, dataset_label)
        self._log_datasets(input_reader)

        # load model
        model = self._load_model(input_reader)
        model.to(self._device)

        # evaluate
        self._eval(model, test_dataset, input_reader, eval_on=args.eval_on)

        self._logger.info("Logged in: %s" % self._log_path)
        self._close_summary_writer()

    def predict(self, dataset_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer,
                                        max_span_size=args.max_span_size,
                                        spacy_model=args.spacy_model)
        dataset = input_reader.read(dataset_path, 'dataset')

        model = self._load_model(input_reader)
        model.to(self._device)

        self._predict(model, dataset, input_reader)

    def _load_model(self, input_reader):
        model_class = models.get_model(self._args.model_type)

        config = BertConfig.from_pretrained(self._args.model_path, cache_dir=self._args.cache_path)
        util.check_version(config, model_class, self._args.model_path)

        config.spert_version = model_class.VERSION
        model = model_class.from_pretrained(self._args.model_path,
                                            config=config,
                                            # SpERT model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            relation_types=input_reader.relation_type_count - 1,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self._args.max_pairs,
                                            prop_drop=self._args.prop_drop,
                                            size_embedding=self._args.size_embedding,
                                            freeze_transformer=self._args.freeze_transformer,
                                            cache_dir=self._args.cache_path)

        return model

    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, compute_soft_loss: Loss, optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int, alpha: float, soft_label: bool):
        self._logger.info("Train epoch: %s" % epoch)

        # create data loader
        dataset.switch_mode(Dataset.TRAIN_MODE)
        # print(list(dataset)[0])
        data_loader = DataLoader(dataset, batch_size=self._args.train_batch_size, shuffle=True, drop_last=True,
                                 num_workers=self._args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        # print(list(data_loader)[0])
        model.zero_grad()

        iteration = 0
        total = dataset.document_count // self._args.train_batch_size
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            model.train()
            batch = util.to_device(batch, self._device)
            torch.autograd.set_detect_anomaly( True )
            # compute_loss._optimizer.zero_grad()


            # print("keys", batch.keys())

            # print("batch['encodings']", batch['encodings'].shape)
            # print("batch['rel_types']", batch['rel_types'].shape)

            # forward step
            entity_logits_1, rel_logits_1, entity_logits_2, rel_logits_2 = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                                                                                entity_masks_1=batch['entity_masks_1'], entity_sizes_1=batch['entity_sizes_1'],
                                                                                relations_1=batch['rels_1'], rel_masks_1=batch['rel_masks_1'],
                                                                                entity_masks_2=batch['entity_masks_2'], entity_sizes_2=batch['entity_sizes_2'],
                                                                                relations_2=batch['rels_2'], rel_masks_2=batch['rel_masks_2'])

            # print("entity_logits_1", entity_logits_1.shape)
            # print("rel_logits_1", rel_logits_1.shape)
            # print("entity_logits_2", entity_logits_2.shape)
            # print("rel_logits_2", rel_logits_2.shape)

            # compute loss and optimize parameters
            batch_loss_1 = compute_loss.compute(entity_logits=entity_logits_1, rel_logits=rel_logits_1,
                                              rel_types=batch['rel_types_1'], entity_types=batch['entity_types_1'],
                                              entity_sample_masks=batch['entity_sample_masks_1'],
                                              rel_sample_masks=batch['rel_sample_masks_1'])
            
            batch_loss_2 = compute_loss.compute(entity_logits=entity_logits_2, rel_logits=rel_logits_2,
                                              rel_types=batch['rel_types_2'], entity_types=batch['entity_types_2'],
                                              entity_sample_masks=batch['entity_sample_masks_2'],
                                              rel_sample_masks=batch['rel_sample_masks_2'])
        
            if soft_label:
                rel_soft_loss_1 = compute_soft_loss.compute(entity_logits=entity_logits_1, rel_logits=rel_logits_1,
                                              rel_types=batch['soft_rel_types_1'], entity_types=batch['entity_types_1'],
                                              entity_sample_masks=batch['entity_sample_masks_1'],
                                              rel_sample_masks=batch['rel_sample_masks_1'])
            
                rel_soft_loss_2 = compute_soft_loss.compute(entity_logits=entity_logits_2, rel_logits=rel_logits_2,
                                              rel_types=batch['soft_rel_types_2'], entity_types=batch['entity_types_2'],
                                              entity_sample_masks=batch['entity_sample_masks_2'],
                                              rel_sample_masks=batch['rel_sample_masks_2'])
                
                batch_loss = (alpha)*batch_loss_1 + (1-alpha)*batch_loss_2 + 0.1*rel_soft_loss_1 + 0.1*rel_soft_loss_2
                # batch_loss = (alpha)*batch_loss_1 + (1-alpha)*batch_loss_2 + (alpha)*rel_soft_loss_1 + (1-alpha)*rel_soft_loss_2

            else:
                batch_loss = (alpha)*batch_loss_1 + (1-alpha)*batch_loss_2  

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), compute_loss._max_grad_norm)
            compute_loss._optimizer.step()
            compute_loss._scheduler.step()
            compute_loss._model.zero_grad()

            # logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self._args.train_log_iter == 0:
                self._log_train(optimizer, batch_loss.item(), epoch, iteration, global_iteration, dataset.label)

            # break

        return iteration

    def _eval(self, model: torch.nn.Module, dataset: Dataset, input_reader: BaseInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0, eval_on: str = "sci"):
        self._logger.info("Evaluate: %s" % dataset.label)

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module

        # create evaluator
        predictions_path = os.path.join(self._log_path, f'predictions_{dataset.label}_epoch_{epoch}')
        examples_path = os.path.join(self._log_path, f'examples_%s_{dataset.label}_epoch_{epoch}.html')
        evaluator = Evaluator(dataset, input_reader, self._tokenizer,
                              self._args.rel_filter_threshold, self._args.no_overlapping, predictions_path,
                              examples_path, self._args.example_count, eval_on=eval_on)

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self._args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self._args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self._args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                               entity_masks_1=batch['entity_masks_1'], entity_sizes_1=batch['entity_sizes_1'],
                               entity_spans_1=batch['entity_spans_1'], entity_sample_masks_1=batch['entity_sample_masks_1'],
                               entity_masks_2=batch['entity_masks_2'], entity_sizes_2=batch['entity_sizes_2'],
                               entity_spans_2=batch['entity_spans_2'], entity_sample_masks_2=batch['entity_sample_masks_2'],
                               inference=True)
                entity_clf_1, rel_clf_1, rels_1, entity_clf_2, rel_clf_2, rels_2 = result

                # print("batch['entity_masks_1']", batch['entity_masks_1'][0])
                # # print("batch['entity_masks_1']", batch['entity_spans_1'][0])
                # print("batch['entity_masks_2']", batch['entity_masks_2'][0])
                # # print("batch['entity_masks_2']", batch['entity_spans_2'][0])
                # print("*"*100)

                # print("entity_clf_1", entity_clf_1)
                # print("entity_clf_1", entity_clf_1.shape)
                # print("entity_clf_2", entity_clf_2)


                # evaluate batch
                # if eval_on=="sci":
                # evaluator.eval_batch(entity_clf_1, rel_clf_1, rels_1, batch['entity_sample_masks_1'], batch['entity_spans_1'], eval_on="sci")
                # elif eval_on=="sem":
                # evaluator.eval_batch(entity_clf_2, rel_clf_2, rels_2, batch['entity_sample_masks_2'], batch['entity_spans_2'], eval_on="sem")

                evaluator.eval_batch_multi_label(entity_clf_1, rel_clf_1, rels_1, batch['entity_sample_masks_1'], batch['entity_spans_1'],
                                                 entity_clf_2, rel_clf_2, rels_2, batch['entity_sample_masks_2'], batch['entity_spans_2'])

                # break

        global_iteration = epoch * updates_epoch + iteration
        ner_eval, rel_eval, rel_nec_eval = evaluator.compute_scores()
        self._log_eval(*ner_eval, *rel_eval, *rel_nec_eval,
                       epoch, iteration, global_iteration, dataset.label)

        if self._args.store_predictions and not self._args.no_overlapping:
            evaluator.store_predictions()

        if self._args.store_examples:
            evaluator.store_examples()

    def _predict(self, model: torch.nn.Module, dataset: Dataset, input_reader: BaseInputReader):
        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self._args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self._args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        pred_entities = []
        pred_relations = []

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self._args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Predict'):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                               entity_masks_1=batch['entity_masks_1'], entity_sizes_1=batch['entity_sizes_1'],
                               entity_spans_1=batch['entity_spans_1'], entity_sample_masks_1=batch['entity_sample_masks_1'],
                               entity_masks_2=batch['entity_masks_2'], entity_sizes_2=batch['entity_sizes_2'],
                               entity_spans_2=batch['entity_spans_2'], entity_sample_masks_2=batch['entity_sample_masks_2'],
                               inference=True)
                entity_clf_1, rel_clf_1, rels_1, entity_clf_2, rel_clf_2, rels_2 = result

                # convert predictions
                predictions_1 = prediction.convert_predictions(entity_clf_1, rel_clf_1, rels_1,
                                                             batch, self._args.rel_filter_threshold,
                                                             input_reader)

                batch_pred_entities_1, batch_pred_relations_1 = predictions_1
                pred_entities.extend(batch_pred_entities_1)
                pred_relations.extend(batch_pred_relations_1)

                predictions_2 = prediction.convert_predictions(entity_clf_2, rel_clf_2, rels_2,
                                                             batch, self._args.rel_filter_threshold,
                                                             input_reader)

                batch_pred_entities_2, batch_pred_relations_2 = predictions_2
                pred_entities.extend(batch_pred_entities_2)
                pred_relations.extend(batch_pred_relations_2)

        prediction.store_predictions(dataset.documents, pred_entities, pred_relations, self._args.predictions_path)

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self._args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self._args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,

                  rel_prec_micro: float, rel_rec_micro: float, rel_f1_micro: float,
                  rel_prec_macro: float, rel_rec_macro: float, rel_f1_macro: float,

                  rel_nec_prec_micro: float, rel_nec_rec_micro: float, rel_nec_f1_micro: float,
                  rel_nec_prec_macro: float, rel_nec_rec_macro: float, rel_nec_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_prec_micro', rel_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_micro', rel_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_micro', rel_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_prec_macro', rel_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_macro', rel_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_macro', rel_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_nec_prec_micro', rel_nec_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_micro', rel_nec_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_micro', rel_nec_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_prec_macro', rel_nec_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_macro', rel_nec_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_macro', rel_nec_f1_macro, global_iteration)

        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,

                      rel_prec_micro, rel_rec_micro, rel_f1_micro,
                      rel_prec_macro, rel_rec_macro, rel_f1_macro,

                      rel_nec_prec_micro, rel_nec_rec_micro, rel_nec_f1_micro,
                      rel_nec_prec_macro, rel_nec_rec_macro, rel_nec_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        self._logger.info("Relations:")
        for r in input_reader.relation_types.values():
            self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro',
                                                 'rel_prec_macro', 'rel_rec_macro', 'rel_f1_macro',
                                                 'rel_nec_prec_micro', 'rel_nec_rec_micro', 'rel_nec_f1_micro',
                                                 'rel_nec_prec_macro', 'rel_nec_rec_macro', 'rel_nec_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})
