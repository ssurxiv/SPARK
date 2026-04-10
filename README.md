
⭐ Official implementation of the paper:

**Brain Topology-Driven Graph Structure Learning for Functional Brain Network-Based Major Depressive Disorder Diagnosis**  
Suyeon Kwak, Ji-Hye Oh, Chang-Hoon Ji, Yu-Kyum Kang, Jaeyong Chang, Tae-Eui Kam  
📍 ISBI 2026

---

## 🧠 Overview

Resting-state fMRI is widely used to model the brain as a graph for disorder diagnosis.  
However, most existing methods rely on predefined or correlation-based graph structures, which often fail to reflect the underlying organization of the human brain.

In this work, we propose **SPARK (Small-world Pattern Aware Reconstructed networK)**,  
a graph structure learning framework that incorporates the **small-world property** into the learning process.

SPARK is built on three key ideas:

- **BRW-enhanced node representation**  
  capturing multi-scale and long-range dependencies between brain regions  

- **Dual-geometry structure learning**  
  modeling brain connectivity in both Euclidean and hyperbolic spaces  

- **Task-oriented graph construction**  
  learning subject-specific structures for diagnosis  

This framework explicitly models the balance between local specialization and global integration,  
a defining characteristic of functional brain networks.

---

## 📁 Repository Structure

```text
SPARK/
├── configs/
│   └── spark.json
├── networks/
│   ├── gcn_utils.py
│   ├── gnns.py
│   ├── model.py
│   └── utils.py
├── utils/
│   ├── __init__.py
│   ├── device.py
│   ├── metrics.py
│   ├── results.py
│   └── seed.py
├── doc/
│   └── paper.pdf
└── README.md
```

---

## ☀️ Citation

If you find this work useful, please cite:
```bibtex
@inproceedings{kwak2026spark,
  title={Brain Topology-Driven Graph Structure Learning for Functional Brain Network-Based Major Depressive Disorder Diagnosis},
  author={Kwak, Suyeon and Oh, Ji-Hye and Ji, Chang-Hoon and Kang, Yu-Kyum and Chang, Jaeyong and Kam, Tae-Eui},
  booktitle={IEEE International Symposium on Biomedical Imaging (ISBI)},
  year={2026},
  address={London, United Kingdom}
}
```
