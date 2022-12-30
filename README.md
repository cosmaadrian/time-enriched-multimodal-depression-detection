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
dataset with Reddit data. Our model is flexible and can easily incorporate temporal information by manipulating the transformer’s positional
encodings. Consequently, our model can operate both on randomly sampled and unordered sets of posts to be more robust to dataset noise
and on ordered collections of posts, accommodating the relative posting
intervals without any major architectural modifications.*

#### Data

The Reddit and Twitter multimodal data used in our experiments are from the work of:

Uban, Ana-Sabina, Berta Chulvi, and Paolo Rosso. [Explainability of Depression Detection on Social Media: From Deep Learning Models to Psychological Interpretations and Multimodality](https://link.springer.com/chapter/10.1007/978-3-031-04431-1_13). In Early Detection of Mental Health Disorders by Social Media Monitoring, pp. 289-320. Springer, Cham, 2022.

Gui, Tao, Liang Zhu, Qi Zhang, Minlong Peng, Xu Zhou, Keyu Ding, and Zhigang Chen. [Cooperative Multimodal Approach to Depression Detection in Twitter](https://ojs.aaai.org/index.php/AAAI/article/view/3775). In Proceedings of the AAAI conference on Artificial Intelligence, vol. 33, no. 01, pp. 110-117. 2019.

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
