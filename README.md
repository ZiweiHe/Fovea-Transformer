# Fovea-Transformer
This is the official Pytorch implementation of paper [Fovea Transformer: Efficient Long-Context Modeling with Structured Fine-to-Coarse Attention](https://arxiv.org/html/2311.07102v2)


## Install
For all the dependencies.
```bash
transformers==4.27.3
deepspeed==0.9.4
datasets==2.19.1
evaluate==0.4.0
accelerate==0.19.0
jax==0.4.10
jaxlib==0.4.10
flax==0.6.10
rouge-score==0.1.2
```

To run summarization tasks including Multi-News, Pubmed and WCEP-10.
```bash
sh scripts/submit.sh
```

If you find the code and paper helpful, pelase kindly cite our work.
```bash
@inproceedings{he2024fovea,
  title={Fovea Transformer: Efficient Long-Context Modeling with Structured Fine-To-Coarse Attention},
  author={He, Ziwei and Yuan, Jian and Zhou, Le and Leng, Jingwen and Jiang, Bo},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={12261--12265},
  year={2024},
  organization={IEEE}
}
```
