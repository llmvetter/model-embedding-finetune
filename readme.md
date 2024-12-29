# E-commerce Sentence Embedding Model

## Project Overview

This repository contains code to train a sentence embedding model using modal as infrastructure. The model aims to generate high-quality embeddings for product classification.

## Key Features

Hyperparameter optimization via grid search
Custom loss function adaptation for multiple positive anchors
Evaluation using information retrieval metrics

## Dataset Generation

The dataset is synthetically generated using an inverse approach:
A set of e-commerce categories is provided as input.
A teacher model (e.g., GPT-3 or a fine-tuned BERT model) generates 10 example items for each category.
This synthetic data simulates real-world e-commerce product descriptions and categories.

## Model Training

The sentence embedding model is trained using the following approach:
Loss Function: MultipleNegativesRankingLoss
Potential adaptation to allow for multiple positive anchors in batch negative sampling
Hyperparameter Optimization: Grid search to find the optimal combination of hyperparameters

## Evaluation

The model's performance is evaluated using:
Evaluator: InformationRetrievalEvaluator
Metrics: Retrieval @N (e.g., @1, @5, @10)
