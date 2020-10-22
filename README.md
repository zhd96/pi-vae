## Poisson Identifiable VAE (pi-VAE)

This code implements pi-VAE, VAE and all the examples in paper: Learning identifiable and interpretable latent models of high-dimensional neural activity using pi-VAE.

To check source code for pi-VAE and VAE, see folder ./code/

To check examples in paper, see folder ./examples/

We are working on adding more comments to the code.

## Installation

The code is written in Python 3.6. In addition to standard scientific Python libraries (numpy, scipy, matplotlib), the code expects the tensorflow (1.13.1) and keras (2.3.1) packages.

To download this code, run `git clone https://github.com/zhd96/pi-vae.git`

## Data
You can find ways to simulate data in ./examples/simulate_data.ipynb. You can find the rat hippocampus data in https://portal.nersc.gov/project/crcns/download/hc-11/data/. We use session Achilles_10252013.

## Reference

If you use this code please cite the paper:

Zhou, D., Wei, X. Learning identifiable and interpretable latent models of high-dimensional neural activity using pi-VAE. NeurIPS 2020.
