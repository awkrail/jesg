Recipe generation from unsegmented cooking videos
=====
PyTorch code for our arXiv 2022 paper ["Recipe generation from unsegmented cooking videos"](https://arxiv.org/abs/2209.10134)
by [Taichi Nishimura](https://misogil0116.github.io/nishimura/), Atsushi Hashimoto, Yoshitaka Ushiku, Hirotaka Kameko, and Shinsuke Mori.

This paper tackles recipe generation from unsegmented cooking videos, a task that requires agents to (1) extract key events in completing the dish and (2) generate sentences for the extracted events. Our task is similar to dense video captioning (DVC), which aims at detecting events thoroughly and generating sentences for them. However, unlike DVC, in recipe generation, recipe story awareness is crucial, and a model should output an appropriate number of key events in the correct order. We analyze the output of the DVC model and observe that although (1) several events are adoptable as a recipe story, (2) the generated sentences for such events are not grounded in the visual content.
Based on this, we hypothesize that we can obtain correct recipes by selecting oracle events from the output events of the DVC model and re-generating sentences for them. To achieve this, we propose a novel transformer-based joint approach of training event selector and sentence generator for selecting oracle events from the outputs of the DVC model and generating grounded sentences for the events, respectively. In addition, we extend the model by including ingredients to generate more accurate recipes. The experimental results show that the proposed method outperforms state-of-the-art DVC models. We also confirm that, by modeling the recipe in a story-aware manner, the proposed model output the appropriate number of events in the correct order.

## Getting started
### Features
```
[TBD]
```

### Training and Inference
We give examples on how to perform training and inference with MART.

0. Build Vocabulary
```
bash scripts/build_vocab.sh DATASET_NAME
```
`DATASET_NAME` can be `anet` for ActivityNet Captions or `yc2` for YouCookII.


1. MART training

The general training command is:
```
bash scripts/train.sh DATASET_NAME MODEL_TYPE
```
| MODEL_TYPE         | Description                            |
|--------------------|----------------------------------------|
| mart               | Memory Augmented Recurrent Transformer |
| xl                 | Transformer-XL                         |
| xlrg               | Transformer-XL with recurrent gradient |
| mtrans             | Vanilla Transformer                    |
| mart_no_recurrence | mart with recurrence disabled          |


To train our MART model on ActivityNet Captions:
```
bash scripts/train.sh anet mart
```
Training log and model will be saved at `results/anet_re_*`.  
Once you have a trained model, you can follow the instructions below to generate captions. 

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

## Others
This code used resources from the following projects: 
[MART](https://arxiv.org/abs/2005.05402),
[svpc](https://github.com/misogil0116/svpc).

## Contact
taichitary@gmail.com
