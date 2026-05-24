"""Dataset acquisition and manifest construction.

`sources` declares where each corpus comes from; `download` fetches and extracts
the archive-based ones; `manifest` turns an extracted corpus into a normalized
JSON-lines manifest that the evaluation harness consumes.
"""
