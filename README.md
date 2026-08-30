# BrainSSL FOMO 26 Challenge

## Introduction
This repository contains the code developed by the BrainSSL team for its participation in the [FOMO26 challenge](https://fomo26.github.io/).

## Scope of the challenge
The mainaim of the FOMO26 challenge is the development of a foundation model of the brain trained on neuroimaging data.
The foundation modelling approach, first introduced in natural language processing and computer vision, has gained a lot of attention in the neuroimaging community thanks to its potential for both in-domain and out-of-domain performance.
Despite the recent progresses, much still has to be done and this challenge represents the ideal evironment to bring forward and test new ideas in the field.

## The BrainSSL approach
We set out to test two related hypotheses: I) that a Vision Transformer pretrained on a medium-sized neuroimaging dataset can transfer competitively to small clinical cohorts, and reach the performance level of traditional CNN-based approaches, and II) that treating the different MRI sequences acquired in a single session jointly rather than one at a time yields a richer and more robust latent representation. To test them, we combine two ingredients. The first is [NeuroJEPA](https://arxiv.org/abs/2606.14957), a joint-embedding predictive architecture specifically adapted to neuroimaging data, that learns by masking part of a volume and predicting the missing content in latent space rather than in voxel space, which makes the model capable of ignoring the noise tipical of the latter and focus only on structure that actually matters. The second is [CoMM](https://arxiv.org/abs/2409.07402), a contrastive fusion module that sits on top of the frozen encoder and learns to combine the per-sequence embeddings of a session into a single multimodal one. We chose CoMM over standard contrastive fusion because it is designed to capture all the ways modalities relate, what they share, what each one uniquely contributes, and what only emerges from their combination, instead of just the redundant overlap.

## Results

## Repository organization
A preliminary step needed to run any code in this repository is to set up a .env file as explained in the dedicated section.
The pretraining folders contains the code and the configuration files needed to run pre-training experiments.
The content of the configurations files is set to match what was used to pre-traing our final submission to the FOMO26 test stage.
Two pre-training experiments are needed to reproduce our final submission: one for the NeuroJEPA encoder, and the other for the CoMM fusion module.
A pre-training of the NeuroJEPA encoder be launched by executing the following steps:
- run pretraining/script/split_pretraining_data.py
- run pretraining/script/prepare_pretraining_data.py on the output files produced by step 2
- run pretraining/src/pretraining_driver.py
The CoMM fusion module can be pre-trained by:
- setting PRETRAINING_CONF=pretrain_neurojepa in pretraining_multimodal/src/pretraining_driver.py 
- setting model.enc_ckpt in pretraining_multimodal/configs/model to the path of the NeuroJEPA encoder's checkpoint
- running pretraining_multimodal/src/pretraining_driver.py
- setting PRETRAINING_CONF=pretrain_neurojepa_high_res in pretraining_multimodal/src/pretraining_driver.py 
- setting warm_start_ckpt in pretraining_multimodal/configs/pretrain_neurojepa_high_res to the path of the checkpoint resulting from step 3

Similarly, the fine_tuning folder contains code and configs file needed to run a fine-tuning experiment, with default values set to replicate our final submission.
A fine-tuning + evaluation experiment can be launched as follows:
- run script/prepare_finetuning_data.py
- run script/evaluate.py
- run script/gather_results.py

The fine-tuning of models submitted to the challenge can be reproduced as follows:
- run script/prepare_finetuning_data.py
- run script/final_finetuning.py

Fine-tuning experiment require a pre-trained checkpoint, whose location can be specified by setting experiment.warm_start_ckpt in fine_tuning/configs/eval

## Setting up the .env file
A .env file with the following content must be set up in both the pretraining and fine-tuning folder
with the following content:
PROJECT_ROOT=[path to project folder]
FOMO_PRETRAINING_DATA_ROOT=[path to pretraining data]
FOMO_FINETUNING_DATA_ROOT=[path to finetuning data]
PRETRAINED_CHECKPOINTS_ROOT=[path to directory where checkpoints are stored]
WANDB_PROJECT=[name of the wandb project where logs will be stored]
LOGS_PATH=[path to folder where log files will be stored]

# Dependencies
Dependencies are listed in the requirements.txt file of each experiment's folder.
The execution of this code requires a special version of the [NIDL library](https://neurospin-deepinsight.github.io/nidl/index.html)

## Team Members
- Benoit Dufumier, Neurospin CEA
- Carlo Alberto Barbano, INRIA Saclay
- Pauline Amrouche, Neurospin CEA
- Akshita Kumar, Telecom Paris & Neurospin CEA
- Michele Cannito, Università degli Studi di Torino
- Santiago Cifuentes Almanza, Université Lyon 1 Claude Bernard
- Federico Giacardi, Università degli Studi di Torino