<h1 align="center">
    <img src="jaxqualin_logo.png" alt="jaxqualin" width="500">
</h1>

<h4 align="center"> A python package for extracting quasinormal modes from black-hole ringdown simulations.</h4>

<p align="center">
    <a href = ""><img src="https://img.shields.io/badge/arXiv-0000.00000-b31b1b.svg"></a>
    <a href="https://badge.fury.io/py/jaxqualin"><img src="https://badge.fury.io/py/jaxqualin.svg"></a>
    <a href="https://github.com/mhycheung/jaxqualin/actions/workflows/pytest.yml"><img src="https://github.com/mhycheung/jaxqualin/actions/workflows/pytest.yml/badge.svg"></a>
    <a href="https://github.com/mhycheung/jaxqualin/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

## Key Features

* Fit a ringdown waveform with quasinormal modes (QNMs) of fixed or free frequencies
* Nonlinear least-squares fitting with automatic differentiation via <a href="https://github.com/Dipolar-Quantum-Gases/jaxfit">JaxFit</a>
* Agnostic identification of QNMs within the waveform
* Saving and reusing results with pickle
* Easy visualization of results
* Call hyperfit models of QNM amplitudes in the ringdown of binary black hole (BBH) mergers

## Installation

```shell
pip install jaxqualin
```

## Usage

Basic usage of the package are showcased under the Examples tab on the left.

> **Note**
> We did not extensively test and do not recommend running `jaxqualin` on a GPU

## Paper Results

Interactive plots of the methods paper can be found under the Results tab on the left.

## Coming Soon

* Full API
* Support for real (Schwarzshild) ringdown waveforms
* Fitting for the mass and spin of the remnant
* Fitting the (noiseless) detector response 

## License

MIT

---

> GitHub [@mhycheung](https://github.com/mhycheung)