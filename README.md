# SoftMax-Cosine distance Rescaling method
**Abstract:**

* Efficiently rescaling a large dataset by filtering samples based on softmax-cosine scores computed from cosine distance to class centroids in feature space, using a lightweight backbone network (MobileNetV3).
* A unified collection of filtered images forming a compact yet challenging subset that preserves the diversity, difficulty, and representativeness of the full dataset.
* An application to rescaling two large datasets — ImageNet and Places365 — to obtain rescaled subsets at multiple preservation rates $r$.
* Experimental results for image classification validate the criteria of a good rescaled subsets across multiple CNN backbones.
* Strong performance on a rescaled subset is indicative of strong performance on the full dataset, allowing researchers to save time and computational cost during early network development.

**Note**: 
* Each rescaled dataset is stored as a text file of extracted relative paths with respect to the directory of the original dataset, and not the images themselves.

---

## Rescaling Explanations

* **Rescaled subsets of ImageNet** — subsets $IN^{r}$ at preservation rates $r \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$ stored under `Rescaled_ImageNet/`
* **Rescaled subsets of Places365** — subsets $PL^{r}$ at preservation rates $r \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$ stored under `Rescaled_Places365/`

**Phase 1 — Extract validation outcomes and prune big datasets:**

Validation outcomes for each image include: softmax scores, extracted features, and ground truths, along with class centroids. The model prediction can be inferred from the softmax scores, i.e., via argmax. The extraction are then passed into FeaturePruner, with specified rates and number of quantized bins $q$, to filter images. The main algorithm is run for ImageNet as: 

```
$ python pruning_softmax.py -d path/to/ImageNet -r1 0.1 -r2 0.1 -q 10 -m "ImageNet"
```

The results are selected filepaths of images, stored in text file in `reduce_result` folder, by default.

Modify the script accordingly for Places365 or any other datasets.


**Phase 2 — Train classifiers on rescaled subsets of ImageNet and Places365:**

* For training CNN-backbones on rescaled sub-datasets of ImageNet

```
$ python classifiers_imgnet.py -d path/to/ImageNet --rates 0.1 -q 10 --prune_out reduce_result 
```

* For training CNN-backbones on rescaled sub-datasets of Places365

```
$ python classifiers_places365.py --data path/to/Rescaled_Places365 
```

The argument -pm is for conducting ablation studies, where one wants to apply stratified sampling using cosine distances on both $V_{correct}$ and $V_{incorrect}$ (`-pm cosine_all`).
---

## Experimental Results — Top-1 Accuracy (%) on Rescaled Subsets

Top-1 accuracy (%) across backbones on the $r=0.1$ rescaled subsets vs. full datasets:

| Network | IN$^{r=0.1}$ | ImageNet | PL$^{r=0.1}$ | Places365 |
|:---|---:|---:|---:|---:|
| GoogLeNet     | 23.74 | 68.30 | 43.43 | 53.63 |
| ShuffleNetV1  | 29.12 | 67.80 | 50.56 | 51.36 |
| ShuffleNetV2  | 30.21 | 69.36 | 50.99 | 50.80 |
| MobileNetV2   | 32.01 | 72.00 | 51.19 | 52.19 |
| MobileNetV1   | 32.35 | 70.60 | 52.81 | 53.50 |
| MobileNetV3   | 32.61 | 71.50 | 50.20 | 53.53 |


MobileNetV1 on rescaled subsets $\overline{\mathcal{D}^r}$ of ImageNet and Places365:

| Rescaled subset | $r$ | $\overline{\mathcal{D}^r_{train}}$ images | $\overline{\mathcal{D}^r_{valid}}$ images | MobileNetV1 |
|:---|---:|---:|---:|---:|
| IN$^{r=0.1}$ | 0.1 | 121,496 | 17,819 | 32.35 |
| IN$^{r=0.2}$ | 0.2 | 248,914 | 17,819 | 41.05 |
| IN$^{r=0.3}$ | 0.3 | 375,516 | 17,819 | 45.02 |
| IN$^{r=0.4}$ | 0.4 | 504,569 | 20,000 | 49.94 |
| IN$^{r=0.5}$ | 0.5 | 635,598 | 24,585 | 55.11 |
| PL$^{r=0.1}$ | 0.1 | 177,104 | 7,101 | 52.81 |
| PL$^{r=0.2}$ | 0.2 | 357,713 | 7,300 | 55.48 |
| PL$^{r=0.3}$ | 0.3 | 537,751 | 9,502 | 51.41 |
| PL$^{r=0.4}$ | 0.4 | 718,377 | 12,827 | 51.85 |
| PL$^{r=0.5}$ | 0.5 | 899,952 | 17,039 | 51.82 |


---

## Related Citations

If you use any materials from this repository, please cite the following relevant works.

```bibtex
@unpublished{...Nguyen26,
  author = {Nguyen, Trung Duc and Nguyen, Thanh Tuan and Borgi, Mohamed Anouar and Nguyen, Thanh Phuong},
  title  = {Rescaling Huge Datasets for Rapid Evaluation of Deep Models},
  note   = {Manuscript submitted for publication to ...},
  year   = {2026},
}
```

```bibtex
@inproceedings{cvprDengDSLL009,
  author       = {Jia Deng and Wei Dong and Richard Socher and Li{-}Jia Li and Kai Li and Li Fei{-}Fei},
  title        = {ImageNet: {A} large-scale hierarchical image database},
  booktitle    = {CVPR},
  pages        = {248--255},  
  year         = {2009}
}
```

```bibtex
@article{pamiZhouLKO018,
  author    = {Bolei Zhou and {\`{A}}gata Lapedriza and Aditya Khosla and Aude Oliva and Antonio Torralba},
  title     = {Places: {A} 10 Million Image Database for Scene Recognition},
  journal   = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
  volume    = {40},
  number    = {6},
  pages     = {1452--1464},
  year      = {2018}
}
```

```bibtex
@article{prlNguyen23,
  author  = {Thanh Tuan Nguyen and Thanh Phuong Nguyen},
  title   = {Rescaling Large Datasets Based on Validation Outcomes of a Pre-trained Network},
  journal = {Pattern Recognition Letters},
  volume  = {185},
  pages   = {73--80},
  year    = {2024},
  url     = {https://doi.org/10.1016/j.patrec.2024.07.001},
}
```

