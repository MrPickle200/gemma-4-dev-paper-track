"""Harness for the Micro-Model Native Harness experiment (IDEA.md).

Modules are added incrementally by ablation variant:
  B0  bash-style text tools + raw search/replace edit          (Bước 2, this commit)
  M1  + constrained JSON edit schema + in-memory syntax gate    (Bước 3)
  M2  + tree-sitter navigation tools (repo map, read symbol)     (Bước 3)
  M3  + episodic context compaction between phases               (Bước 3)
"""
