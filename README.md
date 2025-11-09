# Signs of Struggle: Spotting Distorted Thoughts in Social Media Text

This repository contains the code accompanying the paper "Signs of Struggle: Spotting Distorted Thoughts in Social Media Text", accepted at IJCNLP-AACL 2025 Findings. The paper is currently available as a pre-print on [arXiv](https://arxiv.org/abs/2508.20771).

## Setup
To set up the environment, install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Each script in this repository runs end-to-end and corresponds to specific experimental rows reported in the paper.

Refer to the script comments for details on which section or experiment each file implements.

> [!IMPORTANT]
> Running the LLaMA models requires a valid HuggingFace token with permission to the Meta LLaMA [repositories](https://huggingface.co/meta-llama/Llama-3.1-8B).

## Data Access
Due to the sensitive nature of the data used in this paper, it cannot be published online. Researchers interested in obtaining access for replication or further study can contact the authors at e.liscio@tudelft.nl

## Citation
If you find this work useful, please cite 
```
@misc{kuber2025signsstrugglespottingcognitive,
      title={Signs of Struggle: Spotting Cognitive Distortions across Language and Register}, 
      author={Abhishek Kuber and Enrico Liscio and Ruixuan Zhang and Caroline Figueroa and Pradeep K. Murukannaiah},
      year={2025},
      eprint={2508.20771},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.20771}, 
}
```