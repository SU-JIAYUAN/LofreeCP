# LofreeCP
LofreeCP addresses the critical challenge of quantifying uncertainty in large language models (LLMs) without logit access. By leveraging ​nonconformity measures​ that combine coarse-grained (e.g., sample frequency) and fine-grained uncertainty signals (e.g., normalized entropy, semantic similarity), our method provides statistically rigorous uncertainty estimates for black-box LLMs.


## **📄** Paper
For a detailed explanation of LofreeCP, please refer to the paper:  
[API Is Enough: Conformal Prediction for Large Language Models Without Logit-Access](https://arxiv.org/abs/2403.01216v2)

## **🛠️** About LofreeCP

![image](https://github.com/user-attachments/assets/298f5751-435b-432f-bd06-c6673d5d46ac)

## Why LofreeCP?

## Why LofreeCP?

<div align="center">

| Feature               | Traditional CP         | LofreeCP               |
|-----------------------|------------------------|------------------------|
| ​**Logit Access**​      | ❌ Required           | ✅ Not needed          |
| ​**API-Only LLMs**​     | ❌ Incompatible       | ✅ Supported          |
| ​**Prediction Set Efficiency**​ | ⚠️ Suboptimal       | 🎯 Optimized          |


</div>

## Run
**First Step: Sampling Responses**
   
Run `run_generation.py` to generate and sample responses. I suggest you use transformers Version: 4.32.0.
```shell
python run_generation.py
```
  
**Second Step: Conformal Prediction**
   
After collecting all the responses, run `run_cp.py` to apply our conformal prediction method.
```shell
python run_cp.py
```
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
