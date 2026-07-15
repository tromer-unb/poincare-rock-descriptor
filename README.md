# Poincaré Rock Descriptor

Python implementation of the multilevel hyperbolic descriptor introduced in:

**Mapping Rock Pore Space into the Poincaré Disk for Multiscale
Microstructure Characterization**

The method maps the pore phase of binary two-dimensional rock images into
the Poincaré disk and combines three levels of structural information:

1. Global pore occupation in hyperbolic space;
2. Connected-component organization;
3. Skeleton-based pore-network geometry.

## Repository structure

```text
.
├── descriptor.py
├── structures/
│   ├── patch_y3800_x3800_c0_mask.png
│   ├── patch_y7600_x19000_c0_mask.png
│   └── patch_y7600_x53200_c0_mask.png
├── requirements.txt
└── README.md
