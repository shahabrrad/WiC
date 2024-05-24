# CS 577 Homework 2 Submission

## Overview
This repository contains the homework submission for CS 577. The assignment focuses on exploring advanced topics in natural language processing, including LSTM and Word2Vec embeddings, textual entailment, and the analysis of a paper introducing a new benchmark dataset, WiC.

The main goal of this assignment is to make RNN, LSTM, and DAN ([Deep Averaging Network](https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf)) classifiers for the [WiC: The Word-in-Context Dataset](https://pilehvar.github.io/wic/). 

## Files
- `Homework_2_Report.pdf` - Detailed report of all answers and experiments.
- `neural_archs.py` - The architecture for the neural network models are implemented in this file. Includes DAN, RNN, and LSTM architectures.
- `utils.py` - The utilitary function and classes for the datasets, vocabulary, embedding, ploting, etc. are in this file.
- `train.py` - The training, validation, testing and fine-tuning process are implemented in this file.

## Experiments
### Experiment Setup
Two layers of  LSTM/RNN learns on each one of phrases that are then connected through a fully-connected layer. Dropout is utilized to avoid overfitting.

### Key Findings
- Impact of bidirectionality on model performance.
- Effectiveness of different types of embeddings like GloVe.
- Role of additional features like WordNet lemmas in performance.

## Paper Analysis
Review and critique of the WiC paper, highlighting its contributions to context-sensitive word embeddings and discussing the strengths and weaknesses of the proposed dataset.

## Conclusion
Summarizes the learnings and outcomes from the written answers, experiments, and paper analysis.

## How to Run
run the `train.py` file to run the best performing model. To run specific architectures you can use `--neural_arch` flag with `dan`, `rnn`, `lstm` as arguments. To control bidirectionality you can set the `--rnn_bidirect` flag with `True` and `False`. You can set the use of word embeddings with `--init_word_embs` with either `scratch` to train a word embedding from scratch or `glove` to use a glove embedding.
