# LIGO Homework 3 – MyST Site

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ucb-stat-159-f25/hw3-beccaashley/HEAD)

This project reproduces parts of the LIGO Gravitational Wave Event tutorial, including:
- Implementation of `whiten`, `write_wavfile`, `reqshift`, and `plot_spectrogram` utilities.
- Modularization of code into `ligotools/utils.py`.
- Testing of these functions with Pytest.
- A full MyST website built from Jupyter Notebooks.

---

### Repository Contents

- `LOSC_Event_tutorial.ipynb`: main notebook demonstrating the workflow.
- `ligotools/`: package containing reusable code.
- `tests/`: unit tests for utilities.
- `myst.yml`: configuration file for the MyST site.
- `_build/`: auto-generated HTML site output.

---

### View the MyST Site

Once deployed via GitHub Pages, your site will be available at:
 **https://beccaashley.github.io/hw3-beccaashley/**

---

### Run Locally

To build locally:
```bash
myst build
