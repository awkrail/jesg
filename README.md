Recipe generation from unsegmented cooking videos
=====
PyTorch code for our arXiv 2022 paper ["Recipe generation from unsegmented cooking videos"](https://arxiv.org/abs/2209.10134)
by [Taichi Nishimura](https://misogil0116.github.io/nishimura/), Atsushi Hashimoto, Yoshitaka Ushiku, Hirotaka Kameko, and Shinsuke Mori.

This paper tackles recipe generation from unsegmented cooking videos, a task that requires agents to (1) extract key events in completing the dish and (2) generate sentences for the extracted events. Our task is similar to dense video captioning (DVC), which aims at detecting events thoroughly and generating sentences for them. However, unlike DVC, in recipe generation, recipe story awareness is crucial, and a model should output an appropriate number of key events in the correct order. We analyze the output of the DVC model and observe that although (1) several events are adoptable as a recipe story, (2) the generated sentences for such events are not grounded in the visual content.
Based on this, we hypothesize that we can obtain correct recipes by selecting oracle events from the output events of the DVC model and re-generating sentences for them. To achieve this, we propose a novel transformer-based joint approach of training event selector and sentence generator for selecting oracle events from the outputs of the DVC model and generating grounded sentences for the events, respectively. In addition, we extend the model by including ingredients to generate more accurate recipes. The experimental results show that the proposed method outperforms state-of-the-art DVC models. We also confirm that, by modeling the recipe in a story-aware manner, the proposed model output the appropriate number of events in the correct order.

## Getting started
### Features
#### Word embedding
Pre-trained GloVe word embedding is necessary for training our model.
Download from [here](https://nlp.stanford.edu/data/glove.6B.zip) and unzip it.
In our experiments, we use `glove.6B.300d.txt` so save it to `/path/to/glove.6B.300d.txt`.

## Event features encoded by the ResNet and MIL-NCE
#### ResNet
Download [features.tar.gz](https://drive.google.com/file/d/1T5COAiqhIgqKvHzzsY2bw29fSuX68E39/view?usp=sharing) from Google drive.
The features/ directory stores ResNet + BN-Inception features for each video.
```
features
├── testing
├── training
├── validation
└── yc2
```

#### MIL-NCE
Download []() from Google drive.
```
[TBD]
```

### Training
0. Build Vocabulary
Pre-compute the vocabulary embedding via GloVe. Run the following command with the saved GloVe file like:
```
bash scripts/build_vocab.sh /path/to/glove.6B.300d.txt
```


1. Training

The general training command is:
```
bash scripts/train.sh FEATURE IS_JOINT QUERY_NUM TAU MODALITY MODEL_PATH FEATURE_PATH
```
`FEATURE` is related to the types of the event encoder and has two options: resnet and MIL-NCE.
MIL-NCE achieves the better performance than the ResNet features.
| FEATURE            | Description                            |
|--------------------|----------------------------------------|
| resnet             | Use resnet features as the inputs      |
| mil                | Use the MIL-NCE features as the inputs |

`IS_JOINT` decides whether you train the model with jointly fusing memories or sepearately learning them.
Jointly learning them achieves the higher.
| FEATURE            | Description                            |
|--------------------|----------------------------------------|
| joint              | Joint learning of memories             |
| seperate           | Seperately learning them               |

`QUERY_NUM` and `TAU` can be selectable from `[25, 50, 100, 200]` and `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`, respectively.
In our experiments, `QUERY_NUM=25` and `TAU=0.5` achieves the best performance.

For `MODEL_PATH`, please specify the directory you want to save the model's parameters. 
For `FEATURE_PATH`, please specify the directory, whcih contains features you saved (MIL-NCE or ResNet).

### How to reproduce the experiments?
If you want to acheive a comparable result to Table 2, run
```
bash scripts/train.sh mil joint 100 0.5 /path/to/model/ /path/to/features/
```
If you want to reproduce Table 7, change the `QUERY_NUM` from `[25, 50, 100, 200]`.

### Pre-trained model
TBD

### Misc
The code of BIVT is under construction due to our legacy reason.

## Citations
If you find this code useful for your research, please cite our paper:
```
@inproceedings{taichi2022arxiv,
  title={Recipe Generation from Unsegmented Cooking Videos},
  author={Taichi Nishimura and Atsushi Hashimoto and Yoshitaka Ushiku and Hirotaka Kameko and Shinsuke Mori},
  booktitle={arXiv},
  year={2022}
}
```

## Reference
This code used resources from the following projects: 
[MART](https://arxiv.org/abs/2005.05402),
[svpc](https://github.com/misogil0116/svpc).

## Media (Japanese)
[料理動画からAIでレシピ生成　オムロンと京大の新技術](https://www.nikkei.com/article/DGXZQOUC297B00Z20C22A9000000/)

## Contact
taichitary@gmail.com
