# OIC-264

The goal of this project is to quantify zebrafish macrophage biology following phagocytosis of an RFP protein cargo in the caudal hematopoietic tissue. In the dataset for this project, macrophages are labeled in green and the toxic protein is tagged with an RFP to study its uptake and degradation. Currently, the dataset is of single images captured at the same time post exposure.

The goal is to establish a robust and reproducible pipeline capable of extracting the following quantitative parameters:

1. Macrophage morphology / activation profile - Measurements of cell shape descriptors (area, circularity, elongation, branching, etc.) to assess activation state.
2. Intracellular protein load - Quantification of red fluorescent signal contained within macrophages, including vesicle number, size, and intensity per cell.
3. Extracellular protein load - Quantification of red fluorescent signal outside macrophages within the CHT region to estimate non-phagocytosed material.

This project is in active development and things might change rapidly.

## Getting started

### Prerequisites

- [Python](https://www.python.org/downloads/) version 3.14.0

### Installation

1. Download or clone the GitHub repository
   ```bash
   git clone git@github.com:vaioic/OIC-264.git
   cd OIC-264
   ```

2. Create a python virtual environment
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment
   ```bash
   .\venv\Scripts\activate
   ```

4. Install the dependencies using Pip
   ```bash
   python -m pip install -r .\requirements.txt
   ```

### Running the code

1. Start the virtual environment if not already loaded
   ```bash
   .\venv\Scripts\activate
   ```

... Coming soon


## Issues

If you encounter any issues with running the code or have any questions, please create an [Issue](https://github.com/vaioic/OIC-264/issues) or send an email to opticalimaging@vai.org. If you are reporting a programmatic bug, please include any error messages to aid with troubleshooting.

## Acknowledgements

### Dependencies

This project relies on the following packages:

TBD

**Note:** For full dependency list, see [requirements.txt](requirements.txt).