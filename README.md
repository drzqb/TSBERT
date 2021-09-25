# TSBERT：Text Similarity using BERT 基于BERT的文本相似度模型

## 无监督学习：向量白化、对比学习
    
    bertwhitening：bert输出向量白化
    论文：Whitening Sentence Representations for Better Semantics and Faster Retrieval
    训练数据：lcqmc随机选取10000语句，抛弃标签。
    
    SimCSE_unsupervised：采用与论文相同的损失函数
    论文：SimCSE: Simple Contrastive Learning of Sentence Embeddings
    训练数据：lcqmc随机选取10000语句，抛弃标签。
    
    SimCSE_unsupervised_sp：采用与苏剑林相同的损失函数
    训练数据：同上
    
    SimCSE_unsupervised_sp_simplified：采用与苏剑林相同的损失函数，从transformers加载bert
    训练数据：同上

    SimCSE_unsupervised_simplified：采用与论文相同的损失函数，从transformers加载bert
    训练数据：同上
    
    ConSERT_unsupervised_shuffle：对posids进行shuffle
    论文：ConSERT: A Contrastive Framework for Self-Supervised Sentence Representation Transfer
    训练数据：同上
    
    ESimCSE_unsupervised_endpoints: 采用与论文相同的损失函数
    论文：ESimCSE: Enhanced Sample Building Method for Contrastive Learning of Unsupervised Sentence Embedding
    训练数据：同上
    
## 监督学习：双塔模型、对比学习
    
    SBERT：SentenceBERT
    论文：Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
    训练数据：lcqmc训练集
    
    SBERT：SentenceBERT_simplified, 从transformers加载bert
    论文：同上
    训练数据：同上
    
    SimCSE_supervised：采用与论文相同的损失函数
    训练数据：snli随机选取10000条数据，数据格式[sentence,sentence_entailment,sentence_contradiction]