## It’s Just a Matter of Time: Detecting Depression with Time-Enriched Multimodal Transformers
### Ana-Maria Bucur, Adrian Cosma, Paolo Rosso and Liviu P. Dinu

This repository contains the official source code for the paper **"It’s Just a Matter of Time: Detecting Depression with Time-Enriched Multimodal Transformers"***, accepted at the 2023 edition of European Conference on Information Retrieval (ECIR).

#### Abstract
*Depression detection from user-generated content on the internet has
been a long-lasting topic of interest in the research community. Current
methods for depression detection from social media are mainly focused
on text, and only a few also utilize images posted by users. In this work,
we propose a flexible multimodal transformer architecture for classifying
depression from social media posts containing text and images. We show
that our model, using EmoBERTa and CLIP embeddings, surpasses other
methods on two multimodal datasets, obtaining state-of-the-art results
of 0.931 F1 score on a popular multimodal Twitter dataset, surpassing
previous methods by 2.3%, and 0.900 F1 score on the only multimodal
dataset with Reddit data. Our model is flexible and can easily incorpo-
rate temporal information by manipulating the transformer’s positional
encodings. Consequently, our model can operate both on randomly sam-
pled and unordered sets of posts to be more robust to dataset noise
and on ordered collections of posts, accommodating the relative posting
intervals without any major architectural modifications.*

#### Data




#### Running experiments

Our model definition can be found in the `models/` folder. The multimodal transformer model is based on [LXMERT](https://github.com/airsplay/lxmert) for the cross-encoder definition. The T-LSTM model is adapted from this [repo](https://github.com/duskybomb/tlstm). Time2Vec positional embeddings are adapted from this [repo](https://github.com/ojus1/Time2Vec-PyTorch).

Experiments used to produce the results from the paper are defined in the bash script in `experiments/run_experiments.sh`.


This repo is based on [acumen-template](https://github.com/cosmaadrian/acumen-template) to organise the project, and uses [wandb.ai](https://wandb.ai/) for experiment tracking.


#### Citation
If you find our work useful, please cite us:

```
@article{matter-of-time-2023,
  author  = "Bucur, Ana-Maria and Cosma, Adrian and Rosso, Paolo and P. Dinu, Liviu",
  title   = "It’s Just a Matter of Time: Detecting Depression with Time-Enriched Multimodal Transformers",
  journal = "European Conference on Information Retrieval (ECIR)",
  year    = 2023,
}
```
