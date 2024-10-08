# Multi-scale Feature Contrastive Loss
This GitHub repository contains code and resources for the paper titled **"Improving Speaker Representations Using Contrastive Losses on Multi-scale Features"**. In this project, we introduced the Multi-scale Feature Contrastive (MFCon) loss, a novel approach that significantly enhances speaker verification systems by leveraging contrastive learning on multi-scale features. This repository is based on [MFA Conformer](https://github.com/zyzisyz/mfa_conformer)
<p align="center"><img src="docs/idea.png" width="500"/></p>

## Abstract
Speaker verification systems have seen significant advancements with the introduction of Multi-scale Feature Aggregation (MFA) architectures, such as MFA-Conformer and ECAPA-TDNN. These models leverage information from various network depths by concatenating intermediate feature maps before the pooling and projection layers, demonstrating that even shallower feature maps encode valuable speaker-specific information. Building upon this foundation, we propose a Multi-scale Feature Contrastive (MFCon) loss that directly enhances the quality of these intermediate representations.
Our MFCon loss applies contrastive learning to all feature maps within the network, encouraging the model to learn more discriminative representations at the intermediate stage itself. By enforcing better feature map learning, we show that the resulting speaker embeddings exhibit increased discriminative power. Our method achieves a 9.05\% improvement in equal error rate (EER) compared to the standard MFA-Conformer on the VoxCeleb-1o test set.
<p align="center"><img src="docs/MFCon.png" /></p>

## Getting Started
To get started with the code and replicate our experiments, follow these steps:
```
pip install -r req.txt 
./data.sh # change path
./data_2.sh # change path

sbatch -p GPU-shared -N 1 --gpus=v100-32:1 --cpus-per-gpu 5 -t 48:00:00 start.sh
```

## Citation
```
@misc{dixit2024improvingspeakerrepresentationsusing,
      title={Improving Speaker Representations Using Contrastive Losses on Multi-scale Features}, 
      author={Satvik Dixit and Massa Baali and Rita Singh and Bhiksha Raj},
      year={2024},
      eprint={2410.05037},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2410.05037}, 
}
```


