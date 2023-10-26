<h1 align="center">
    <img src="jaxqualin_logo.png" alt="jaxqualin" width="500">
</h1>

<h4 align="center"> A python package for extracting quasinormal modes from black-hole ringdown simulations.</h4>

<p align="center">
    <a href = "https://arxiv.org/abs/2310.04489"><img src="https://img.shields.io/badge/arXiv-2310.04489-b31b1b.svg"></a>
    <a href="https://badge.fury.io/py/jaxqualin"><img src="https://badge.fury.io/py/jaxqualin.svg"></a>
    <a href="https://github.com/mhycheung/jaxqualin/actions/workflows/pytest.yml"><img src="https://github.com/mhycheung/jaxqualin/actions/workflows/pytest.yml/badge.svg"></a>
    <a href="https://github.com/mhycheung/jaxqualin/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
    <a href="https://pypi.org/project/jaxqualin/"><img src="https://img.shields.io/pypi/pyversions/jaxqualin"></a>
</p>

## Key Features

* Fit a ringdown waveform with quasinormal modes (QNMs) of fixed or free frequencies
* Nonlinear least-squares fitting with automatic differentiation via <a href="https://github.com/Dipolar-Quantum-Gases/jaxfit">JaxFit</a>
* Agnostic identification of QNMs within the waveform
* Saving and reusing results with `pickle`
* Easy visualization of results
* Call hyperfit polynomials to approximate QNM amplitudes in the ringdown of binary black hole (BBH) mergers

## Installation

```shell
pip install jaxqualin
```

## Usage

Basic usage examples can be found under the Examples tab on the left.

> **Note**
> We did not extensively test and do not recommend running `jaxqualin` on a GPU

## Paper Results

Interactive plots of the methods paper results can be found under the Results tab on the left.

## Coming Soon

* Full API
* Support for real (Schwarzshild) ringdown waveforms
* Fitting for the mass and spin of the remnant

## How to Cite
Please cite the methods paper if you used our package to produce results in your publication.
Here is the BibTeX entry:
```
@misc{cheung2023extracting,
      title={Extracting linear and nonlinear quasinormal
      modes from black hole merger simulations}, 
      author={Mark Ho-Yeuk Cheung and Emanuele Berti and 
      Vishal Baibhav and Roberto Cotesta},
      year={2023},
      eprint={2310.04489},
      archivePrefix={arXiv},
      primaryClass={gr-qc}
}
```

## License

MIT

---

> GitHub [@mhycheung](https://github.com/mhycheung)