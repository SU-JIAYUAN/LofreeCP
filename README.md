# LofreeCP
LofreeCP aims to address the pervasive challenge of quantifying uncertainty in large language models (LLMs) without logit-access by formulating nonconformity measures using both coarse-grained (i.e., sample frequency) and fine-grained uncertainty notions (e.g., normalized entropy & semantic similarity). 

## **📄** Paper
For a detailed explanation of LofreeCP, please refer to the paper:  
[API Is Enough: Conformal Prediction for Large Language Models Without Logit-Access](https://arxiv.org/abs/2403.01216v2)

## **🛠️** About LofreeCP

1. **First Step: Sampling Responses**
    - Run `run_generation.py` to generate and sample responses. I suggest you use transformers Version: 4.32.0.
  
2. **Second Step: Conformal Prediction**
    - After collecting all the responses, run `run_cp.py` to apply our conformal prediction method.

## Citation<a name="cita"></a>
```latex
@inproceedings{su-etal-2024-api,
  title={API Is Enough: Conformal Prediction for Large Language Models Without Logit-Access},
  author={Su, Jiayuan and Luo, Jing and Wang, Hongwei and Cheng, Lu},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2024},
  pages={979--995},
  year={2024}
}
```
