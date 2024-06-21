# An audio-visual dataset in the wild for cross-modal retrieval

##Deep triplet neural networks with cluster-cca for audio-visual cross-modal retrieval (TNN-C-CCA)

This paper was published in ACM TOMM 2020, research article: https://dl.acm.org/doi/abs/10.1145/3387164

Arxiv link: https://arxiv.org/pdf/1908.03737.pdf

## Cite this paper

Donghuo Zeng, Yi Yu, Keizo Oyama. Deep triplet neural networks with cluster-cca for audio-visual cross-modal retrieval[J]. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 2020, 16(3), pp:1-23.

```bash
@article{zeng2020deep,
  title={Deep triplet neural networks with cluster-cca for audio-visual cross-modal retrieval},
  author={Zeng, Donghuo and Yu, Yi and Oyama, Keizo},
  journal={ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)},
  volume={16},
  number={3},
  pages={1--23},
  year={2020},
  publisher={ACM New York, NY, USA}
}
```

# Introduction

This paper is to address cross-modal retrieval between audio and visual (silence) with audio-visual one-to-one paired datasets. 
The main work focuses on learning joint embeddings from a shared subspace for computing the similarity across different modalities, 
where generating new representations is to maximize the correlation between audio and visual modalities space. There are three steps
to implement our model:

**#Feature Extraction**: extract features from raw data to reduce the dimensionality of inpu data by removing the redundant data, so that the correlation between
audio and visual modality can be learn in a common space more effective and accurate. We used vggish pre-trained model to extract audio features, while used Inception
V3 pre-trained model to extract visual features, more detail seen in the following Section Pre-trained Models and the original paper.

**#TNN-C-CCA Model**: this model contain two parts, cluter-cca and triplet neural network. 
  
    Cluster-CCA [1] is a supervised Conanical Correlation Anlysis (CCA), Unlike the standard pairwise correspondence 
    between the data points as CCA, in Cluster-CCA learning process, each set is partitioned into multiple clusters or 
    classes, where the class labels define correspondences between the sets. The reason why we use this model is that 
    Cluster-CCA is able to learn discriminant low dimensional representations that maximize the correlation between the 
    audio and visual sets while segregating the different classes on the common space. 
    
    Triplet Neural Network [2] uses triplet loss as function loss to learn a triplet networks for three branches. The 
    Triplet Loss minimizes the distance between an anchor and a positive, both of which belong to the same class, and 
    maximizes the distance between the anchor and a negative of a different class.

**#Evelation**:
    we use MAP and PRC as metrics to evaluate our architecture, when the system generates a ranked list in one modality 
    for a query in another modality. Those documents in the ranked list with the same class are regarded as relevant or correct.


# Requirements
    
    conda 4.8.5: https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html
    python 3.5
    keras 2.0.5
    tensorflow 1.4.1 (pip install tensorflow==1.4.1 --ignore-installed)
    theano 1.0.5
    scipy 1.4.1
    numpy 1.18.5
    h5py 2.10.0
    pip 20.3.4
    sklearn
    
# Usage
## Dataset Download:https://drive.google.com/drive/u/1/folders/1p7QyvYSaJ-0vJla7N9UQY1XS3lTSzgdo

**MV-10K dataset**: 
    https://drive.google.com/drive/u/1/folders/1-N1uQDkwvWEBJmzRexHUA0QrRFDuvKBu 
    
&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;5-folds:
    https://drive.google.com/drive/u/1/folders/1-N1uQDkwvWEBJmzRexHUA0QrRFDuvKBu
      
      MV-10K dataset includes pre-trained model extracted audio-visual features. 
      More detail seen in the link:http://research.google.com/youtube8m/download.html
      
**VEGAS dataset**:

The Raw dataset from: https://arxiv.org/abs/1712.01393, now is not available, I didn't copy it out from my past desktop. 
But you can download from my google drive named vegas.zip: https://drive.google.com/drive/folders/1-N1uQDkwvWEBJmzRexHUA0QrRFDuvKBu?usp=sharing
       
       1. vgg file contains audio features are extracted by pre-trained vggish model.
       2. inception file is visual features are extracted by pre-trained inception V3 model.
       3. folds file is used for 5-folds cross-validation, and we keep the class balance in each fold.

Or you can use my extracted feature named [vegas_feature.h5] and please kindly cite my paper, thank you!

**AVE dataset**:

Original Dataset homepage:https://sites.google.com/view/audiovisualresearch

## Feature Extraction
The pre-trained model is in the google drive the same as above, named pretrain.zip

**ffmpeg**:

It is a tool to edit the video or audio, more detail seen: http://ffmpeg.org/. Here, I use the tool to extract audio track from video.


# Contact
If you have any questions, please email donghuozeng@gmail.com (I am Zen).

## References：

<a id="1">[1]</a>Rasiwasia, Nikhil, et al. "Cluster canonical correlation analysis." Artificial intelligence and statistics. PMLR, 2014.
<a id="2">[2]</a>Hermans, Alexander, Lucas Beyer, and Bastian Leibe. "In defense of the triplet loss for person re-identification." arXiv preprint arXiv:1703.07737 (2017).
