# LabelVariation_SciIE

This repository contains the implementation of the following work: 
"Solving Label Variation in Scientific Information Extraction Via Multi-Task Learning".

We based the code on [SpERT](https://github.com/lavis-nlp/spert) network.

## Setup
### Requirements
- Required
  - Python 3.5+
  - PyTorch (tested with version 1.4.0)
  - transformers (+sentencepiece, e.g. with 'pip install transformers[sentencepiece]', tested with version 4.1.1)
  - scikit-learn (tested with version 0.24.0)
  - tqdm (tested with version 4.55.1)
  - numpy (tested with version 1.17.4)
- Optional
  - jinja2 (tested with version 2.10.3) - if installed, used to export relation extraction examples
  - tensorboardX (tested with version 1.6) - if installed, used to save training process to tensorboard
  - spacy (tested with version 3.0.1) - if installed, used to tokenize sentences for prediction

### Download and preprocess the datasets
We used three datasets in our experiments including:
- [SciERC](http://nlp.cs.washington.edu/sciIE/)
- [SemEval-2018 Task 7](https://lipn.univ-paris13.fr/~gabor/semeval2018task7/)
- [SciREX](https://github.com/allenai/SciREX)

In addition, we prepared the processed datasets for running the experiments as in our document.
- For the overlap and the non-overlap ([experiment_v2](https://github.com/dongpham120899/LabelVariation_SciIE/tree/main/experimental_ovp_dataset_v2))
- For standard SciERC splitting and cross-dataset on SciREX ([experiment_v4](https://github.com/dongpham120899/LabelVariation_SciIE/tree/main/experimental_ovp_dataset_v4))

A preprocessing code is provided in file [preprocessing](https://github.com/dongpham120899/LabelVariation_SciIE/tree/main/preprocessing)

## How to run
(1) Training on the overlap and testing on the non-overlap.
```
python ./spert.py train --config configs/train_fix_label/train_matched_set.conf
```

(2) Training and Testing on standard splitting set in SciERC
```
python ./spert.py train --config configs/train_standard_sci/train_0.conf 
```

## References
```
[1]: Eberts, Markus, and Adrian Ulges. "Span-based joint entity and relation extraction with transformer pre-training." arXiv preprint arXiv:1909.07755 (2019).
[2]: Luan, Yi, et al. "Multi-task identification of entities, relations, and coreference for scientific knowledge graph construction." arXiv preprint arXiv:1808.09602 (2018).
[3]: Buscaldi, Davide, et al. "Semeval-2018 task 7: Semantic relation extraction and classification in scientific papers." International Workshop on Semantic Evaluation (SemEval-2018). 2017.
[4]: Jain, Sarthak, et al. "Scirex: A challenge dataset for document-level information extraction." arXiv preprint arXiv:2005.00512 (2020).
```