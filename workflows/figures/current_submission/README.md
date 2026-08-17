# Manuscript figure workflow

This directory contains the Python/Matplotlib workflow for the five main
figures and seven Extended Data figures. It reads only released source-data
tables and released vector-panel sources; it does not read author workstation
paths.

Run from the code-repository root:

```bash
python workflows/figures/current_submission/make_submission_figures.py \
  --data-package /path/to/released_data_package \
  --output-dir outputs/manuscript_figures
```

The data package must contain `source_data/` and `figure_panel_sources/`.
The workflow writes exactly 12 figures in SVG, PDF, PNG and RGB/LZW TIFF and
then checks both the figure set and the source-data manifest.
