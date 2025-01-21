# DeepTGIN
 DeepTGIN: a novel hybrid multimodal approach using Transformers and graph isomorphism networks for protein-ligand binding affinity prediction
```
conda create -n DeepTGIN poython=3.8
```
 
## Data
We use PDBBind2020 as our dataset，http://pdbbind.org.cn/
The code for data preprocessing is [create_data.py](create_data.py)
## Train
If you want to train the model, run python [main.py](main.py)
## Cite
If our work is helpful to you, we encourage you to cite our paper:
@article{wang2024deeptgin,
  title={DeepTGIN: a novel hybrid multimodal approach using transformers and graph isomorphism networks for protein-ligand binding affinity prediction},
  author={Wang, Guishen and Zhang, Hangchen and Shao, Mengting and Feng, Yuncong and Cao, Chen and Hu, Xiaowen},
  journal={Journal of Cheminformatics},
  volume={16},
  number={1},
  pages={147},
  year={2024},
  publisher={Springer}
}
