"""Keypoint skeleton definition (nodes + edges + L/R symmetries).

The skeleton is a small graph, so it is kept out of ``config.toml`` (which is
scalar-only) and lives either as the built-in default or an external JSON file
that users can copy and edit. Convertible to a ``sleap_io`` Skeleton at training
time (Milestone B3).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class Skeleton:
    """An ordered keypoint schema.

    ``nodes`` is the canonical keypoint order (defines the K axis of every pose
    array). ``edges`` connect nodes for drawing/structure; ``symmetries`` are
    left/right pairs used for augmentation and swap detection.
    """
    name: str
    nodes: List[str]
    edges: List[Tuple[str, str]] = field(default_factory=list)
    symmetries: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def index(self, node: str) -> int:
        return self.nodes.index(node)

    def edge_indices(self) -> List[Tuple[int, int]]:
        return [(self.index(a), self.index(b)) for a, b in self.edges]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "nodes": list(self.nodes),
            "edges": [list(e) for e in self.edges],
            "symmetries": [list(s) for s in self.symmetries],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Skeleton":
        nodes = list(d["nodes"])
        edges = [tuple(e) for e in d.get("edges", [])]
        syms = [tuple(s) for s in d.get("symmetries", [])]
        known = set(nodes)
        for a, b in edges + syms:
            if a not in known or b not in known:
                raise ValueError(f"skeleton {d.get('name')!r}: edge/symmetry references unknown node ({a},{b})")
        return cls(name=str(d.get("name", "skeleton")), nodes=nodes, edges=edges, symmetries=syms)

    def to_json_file(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


# Built-in default: 5-node fly skeleton (head → thorax → abdomen tip, plus wings).
DEFAULT_SKELETON = Skeleton(
    name="fly5",
    nodes=["head", "thorax", "abdomen_tip", "wingL", "wingR"],
    edges=[("head", "thorax"), ("thorax", "abdomen_tip"),
           ("thorax", "wingL"), ("thorax", "wingR")],
    symmetries=[("wingL", "wingR")],
)


def load_skeleton(path) -> Skeleton:
    """Load a skeleton from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Skeleton.from_dict(data)


def get_skeleton(cfg) -> Skeleton:
    """Resolve the skeleton for a run: ``cfg.pose_skeleton`` JSON if set/exists,
    else the built-in :data:`DEFAULT_SKELETON`."""
    p = getattr(cfg, "pose_skeleton", "") or ""
    if p and Path(p).exists():
        return load_skeleton(p)
    return DEFAULT_SKELETON
