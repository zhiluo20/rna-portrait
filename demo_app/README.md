# RNA Portrait — interactive demo

A tiny local web app for demonstrating the pretrained RNA Portrait model. Two modes:

- **Single profile** — paste/upload one two-column profile and see the
  natural-language portrait text, **gauge dials** for tumor probability and the
  age-like readout, summary cards, and a bar chart of the top prototype weights.
  Export the prototype components as CSV.
- **Batch compare** — upload several profiles (one sample each) and/or paste an
  expression matrix (gene column + one column per sample). Get a side-by-side
  comparison table (disease family, organ, tumor probability, age-like, top
  weight, gene coverage), a **disease-family weight heatmap** (samples × disease
  families), and a **sample-clustering dendrogram** (average-linkage hierarchical
  clustering on the per-sample disease-family weight vectors).

All charts export: the single-profile chart and the batch heatmap/dendrogram
download as **PNG**, and both the prototype table and comparison table download
as **CSV**. Export is done client-side via an offline SVG→canvas pipeline — no
network or extra libraries.

## Run

From the repository root:

```bash
python demo_app/server.py
```

Then open <http://127.0.0.1:8000> in a browser.

- **Load example** fills the box with `examples/example_gene_expression.tsv`.
- **Upload file** accepts `.tsv` / `.csv` / `.txt` (gene symbol in column 1, value in column 2).
- **Top prototypes (K)** controls how many prototype components are shown.

Press `Ctrl+C` in the terminal to stop the server.

## How it works

- `server.py` — standard-library `http.server`. Loads the model once at startup
  via `rna_portrait.load_model("models")` and exposes:
  - `POST /api/describe` — one two-column profile.
  - `POST /api/describe_batch` — `{"samples":[{name,text}], "matrix": "...", "top_k": N}`;
    each sample text is one profile, and `matrix` is split into one sample per column.
  Pasted two-column text is written to a temp file and parsed by the package's own
  `describe_file`, so results match the CLI exactly; matrix columns are fed straight
  to `MODEL.describe`.
- `index.html` — self-contained front-end (no CDN / network needed). Gauges, bar
  charts, comparison table and CSV export are all plain HTML/CSS/SVG, so it works
  fully offline for live demos.

No extra dependencies beyond what `rna_portrait` already requires
(torch / numpy / pandas).

## Note on gene coverage

The bundled example file has only 20 genes, so the model matches <1% of its
vocabulary and the prediction is not biologically meaningful — the UI shows a
coverage warning when matched genes are very low. Use a full expression profile
for a real prediction.
