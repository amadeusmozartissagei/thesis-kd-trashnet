# Dataset Information

## Dataset
Penelitian ini menggunakan **TrashNet dataset** dari Kaggle sebagai dataset utama untuk tugas klasifikasi citra sampah.

Source:
https://www.kaggle.com/datasets/feyzazkefe/trashnet

Kaggle Dataset Path:
`/kaggle/input/datasets/feyzazkefe/trashnet/dataset-resized`

---

## Dataset Description
Dataset terdiri dari citra RGB berbagai jenis sampah rumah tangga yang dikategorikan berdasarkan material.

Total:
```text
2527 images
```

Total classes:
```text
6
```

---

## Class Distribution

| Class | Number of Images |
|------|------------------|
| Cardboard | 403 |
| Glass | 501 |
| Metal | 410 |
| Paper | 594 |
| Plastic | 482 |
| Trash | 137 |

Total:
```text
2527 images
```

---

## Class Description

### Cardboard
Berisi sampah berbahan karton/kardus.

Contoh:
- cardboard box
- packaging cardboard
- folded carton

---

### Glass
Berisi objek berbahan kaca.

Contoh:
- glass bottles
- glass containers
- broken glass objects

---

### Metal
Berisi sampah berbahan logam.

Contoh:
- aluminium cans
- metal containers
- beverage cans

---

### Paper
Berisi objek berbahan kertas.

Contoh:
- newspaper
- office paper
- magazines
- paper packaging

---

### Plastic
Berisi objek berbahan plastik.

Contoh:
- plastic bottles
- plastic packaging
- plastic containers

---

### Trash
Kategori miscellaneous / non-recyclable waste.

Contoh:
- dirty tissue
- mixed waste
- ambiguous waste objects

Catatan:
Class ini merupakan class dengan jumlah sampel paling sedikit dan memiliki ambiguity tinggi, sehingga menjadi class yang paling challenging untuk diklasifikasikan.

---

## Dataset Characteristics
Karakteristik dataset:

- relatively small-scale dataset
- imbalanced class distribution
- high inter-class similarity
- background clutter
- varying object scales
- real-world image variation

---

## Dataset Split
Menggunakan split:

```python
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.3
SEED = 42
```

Approximate distribution:

| Class | Train (70%) | Validation (30%) |
|------|-------------|------------------|
| Paper | 415 | 179 |
| Glass | 350 | 151 |
| Plastic | 337 | 145 |
| Metal | 287 | 123 |
| Cardboard | 282 | 121 |
| Trash | 96 | 41 |

Approximate total:
```text
Train: 1768
Validation: 759
```

Catatan:
Exact distribution mengikuti hasil seeded split (`SEED = 42`) pada implementasi eksperimen.

---

## Experimental Fairness Rule
Untuk menjaga fairness:

- dataset split dibuat satu kali
- split dikunci menggunakan fixed random seed
- seluruh eksperimen menggunakan exact same train/validation split
- tidak dilakukan reshuffling antar eksperimen