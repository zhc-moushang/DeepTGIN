# DeepTGIN
We propose a novel protein-ligand binding affinity prediction model (DeepTGIN), to extract sequential features and graph features efficiently. Our DeepTGIN model consists of three layers. The three layers are the input layer, the feature representation layer, and the output layer. Our DeepTGIN model first employs the transformer encoder to extract sequential features from protein and pocket separately. Meanwhile, our DeepTGIN model introduces the graph isomorphism network to extract graph features from the ligand.
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
