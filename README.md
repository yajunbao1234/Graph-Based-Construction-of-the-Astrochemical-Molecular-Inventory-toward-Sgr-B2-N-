# Graph-Based-Construction-of-the-Astrochemical-Molecular-Inventory-toward-Sgr-B2-N

This repository provides the core code for the Graph-Based-Construction-of-the-Astrochemical-Molecular-Inventory-toward-Sgr-B2-N.
The workflow includes data splitting, heterogeneous graph construction, metapath-guided random walks, metapath2vec embedding, column-density prediction, ablation analysis, and several interpretation scripts.

## 1. Data preparation

Put the regional data files in:

```text
data/test1/areas_split/
```

For example:

```text
data/test1/areas_split/A01.txt
data/test1/areas_split/A02.txt
...
data/test1/areas_split/A20.txt
```

Each `Axx.txt` file should be a tab-separated text file. The current code assumes the following columns:

```text
0  sample ID
1  source size
2  rotational temperature
3  line width
4  velocity offset
5  vibrational state
6  auxiliary column
7  SELFIES
8  column density label
9  molecule name
10 SMILES
```

The main model uses columns `1, 2, 3, 4, 5, 7` as input features and column `8` as the prediction label.

For the interpretation scripts, also put the molecule lookup table here:

```text
data/test1/Core_Components_reordered.txt
```

## 2. Environment

Install the required packages:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm torch dgl networkx
```

## 3. Select the region

The example code uses `A01` by default.
To run another region, modify the region name in `01_split_data.py`, for example:

```python
A_name = "A01"
```

Change it to:

```python
A_name = "A02"
```

or any other region from `A01` to `A20`.

## 4. Run the main pipeline

Run the scripts in order:

```bash
python 01_split_data.py
python 02_combine_train_test.py
python 03_encode_nodes.py
python 04_build_edges.py
python 05_generate_metapaths.py
python 06_train_metapath2vec.py
python 07_build_sample_embeddings.py
python 08_predict_and_ablation.py 1 12345 A01
```


## 5. Optional analysis scripts

After the main pipeline is finished, the following scripts can be run if needed:

```bash
python 09_attribution.py
python 10_prototype_retrieval.py
python 11_contrastive_retrieval.py
```

Before running `12_prioritization.py`, copy the Random Forest prediction result to the expected path:

```bash
cp data/test8/A01/iter_001_seed_12345/prediction/testing_label_pred_rf.txt data/test7/testing_label_pred.txt
```

Then run:

```bash
python 12_prioritization.py
```

## 6. Main outputs

Important output files include:

```text
metrics_summary.csv
prediction_detail.csv
ablation_summary.csv
fig_variable_importance.png
tsne_clusters.png
contrastive_pairs.csv
priority_ranking.csv
```

## 7. Notes

* The code is written for one region at a time.
* To process all regions from `A01` to `A20`, repeat the workflow after changing the region name.
* The random split in `01_split_data.py` is generated automatically, so results may vary between runs.
* The input data and generated result tables may contain molecule names, SELFIES, SMILES, physical parameters, true column densities, and predicted column densities. Please check the data license before redistribution.
 
