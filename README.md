## Overview
The Markov Random Field (MRF), although being widely used for contact prediction, suffers from the following dilemma: the actual likelihood function of MRF is accurate but time-consuming to calculate, in contrast, approximations to the actual likelihood, say pseudo-likelihood, are efficient to calculate but inaccurate.

MRF-suite is a tool to estimate the parameters of MRF efficiently. Three algorhims are included in MRF-suite-V0.1

- clmMRF. Composite likelihood maximization is used to estimate the parameters of MRFs.
- glassoMRF. Group lasso regularization is applied.
- fmMRF. Factorization machine (FM) is a very popular model in recommendation system, advaertising and other machine learning fileds. Here  we integrate FM into MRF to reduce the number of parameters. fmMRF works wery well when only limited samples are available.

## Citation
Haicang Zhang, Qi Zhang, Fusong Ju, Jianwei Zhu, Shiwei Sun, Yujuan Gao,ZiweiXie,MinghuaDeng,WeiMouZheng,andDongboBu. Predictingproteininter-residuecontacts using composite likelihood maximization and deep learning.arXiv preprintarXiv:1809.00083, 2018. https://arxiv.org/abs/1809.00083

## Build
1. liblbfgs is required. Please refer to http://www.chokkan.org/software/liblbfgs/ for details.
2. cd MRF-suite/src; make

## Usage
Usage: clm_singlet [options] aln-file out-prefix dis-file
