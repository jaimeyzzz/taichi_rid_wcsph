# Taichi RID SPH Fluid Simulation

A GPU-accelerated fluid simulation using the Reliable Iterative Dynamics (RID) variant of Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH), implemented with Taichi.

## Requirements

```bash
pip install taichi matplotlib numpy
```

## Usage
```bash
python taichi_rid_wcsph.py
```

## Reference
This implementation is based on the paper:
Reliable Iterative Dynamics: A Versatile Method for Fast and Robust Simulation (https://dl.acm.org/doi/10.1145/3734518):

```bibtex
@article{10.1145/3734518,
    author = {Lu, Jia-Ming and Hu, Shi-Min},
    title = {Reliable Iterative Dynamics: A Versatile Method for Fast and Robust Simulation},
    year = {2025},
    issue_date = {June 2025},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    volume = {44},
    number = {3},
    issn = {0730-0301},
    url = {https://doi.org/10.1145/3734518},
    doi = {10.1145/3734518},
    journal = {ACM Trans. Graph.},
    month = may,
    articleno = {29},
    numpages = {18},
    keywords = {Physics-based animation}
}
```