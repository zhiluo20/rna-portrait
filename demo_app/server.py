"""Local demo web app for the RNA Portrait model.

Run:
    python demo_app/server.py
Then open http://127.0.0.1:8000 in a browser.

Paste or upload a two-column gene-expression profile (gene symbol, value)
and the page shows the natural-language portrait text plus a bar chart of
the top prototype weights.

Zero extra dependencies: uses only the Python standard library plus the
already-installed rna_portrait package (torch / numpy / pandas).
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rna_portrait as rp  # noqa: E402

HOST = "127.0.0.1"
PORT = 8000

print("Loading RNA Portrait model from", REPO_ROOT / "models", "...")
MODEL = rp.load_model(str(REPO_ROOT / "models"))
print("Model loaded. Selected genes:", len(MODEL.selected_genes))


def describe_text(raw_text: str, top_k: int) -> dict:
    """Parse a pasted/uploaded two-column profile and run the model.

    Reuses the package's own file reader so parsing matches the CLI exactly.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8") as handle:
        handle.write(raw_text)
        temp_path = handle.name
    try:
        return MODEL.describe_file(temp_path, top_k=top_k)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def parse_matrix(raw_text: str) -> dict[str, dict[str, float]]:
    """Parse an expression matrix into {sample_name: {gene: value}}.

    First column is the gene symbol; every remaining column is one sample.
    Accepts comma-, tab- or whitespace-delimited text with a header row.
    """
    frame = pd.read_csv(StringIO(raw_text), sep=None, engine="python", header=0)
    if frame.shape[1] < 2:
        raise ValueError("Matrix needs a gene column plus at least one sample column.")
    gene_col = frame.columns[0]
    samples: dict[str, dict[str, float]] = {}
    for col in frame.columns[1:]:
        values: dict[str, float] = {}
        for _, row in frame.iterrows():
            gene = str(row[gene_col]).strip().upper()
            if not gene or gene in {"GENE", "GENES", "SYMBOL"}:
                continue
            try:
                values[gene] = float(row[col])
            except (TypeError, ValueError):
                continue
        samples[str(col)] = values
    return samples


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path in ("/", "/index.html"):
            html = (APP_DIR / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/example":
            example = (REPO_ROOT / "examples" / "example_gene_expression.tsv").read_text(encoding="utf-8")
            self._send(200, example.encode("utf-8"), "text/plain; charset=utf-8")
        else:
            self._send(404, b"Not found", "text/plain; charset=utf-8")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def _error(self, message: str) -> None:
        self._send(400, json.dumps({"error": message}).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        try:
            if self.path == "/api/describe":
                self._handle_single()
            elif self.path == "/api/describe_batch":
                self._handle_batch()
            else:
                self._send(404, b"Not found", "text/plain; charset=utf-8")
        except Exception as exc:  # surface parse/model errors to the UI
            traceback.print_exc()
            self._error(f"{type(exc).__name__}: {exc}")

    def _handle_single(self) -> None:
        payload = self._read_json()
        raw_text = str(payload.get("text", "")).strip()
        top_k = max(1, min(int(payload.get("top_k", 5)), 15))
        if not raw_text:
            self._error("Empty profile.")
            return
        result = describe_text(raw_text, top_k)
        self._send(200, json.dumps(result).encode("utf-8"), "application/json; charset=utf-8")

    def _handle_batch(self) -> None:
        payload = self._read_json()
        top_k = max(1, min(int(payload.get("top_k", 5)), 15))
        samples = payload.get("samples") or []  # [{name, text}] — one two-column profile each
        matrix_text = str(payload.get("matrix", "")).strip()

        results = []
        errors = []
        for entry in samples:
            name = str(entry.get("name", "sample")).strip() or "sample"
            text = str(entry.get("text", "")).strip()
            if not text:
                errors.append({"name": name, "error": "empty"})
                continue
            try:
                results.append({"name": name, "result": describe_text(text, top_k)})
            except Exception as exc:
                errors.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})

        if matrix_text:
            for name, values in parse_matrix(matrix_text).items():
                try:
                    results.append({"name": name, "result": MODEL.describe(values, top_k=top_k)})
                except Exception as exc:
                    errors.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})

        if not results and not errors:
            self._error("No samples provided.")
            return
        body = json.dumps({"results": results, "errors": errors}).encode("utf-8")
        self._send(200, body, "application/json; charset=utf-8")

    def log_message(self, *args) -> None:  # quieter console
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"\nRNA Portrait demo running at {url}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
