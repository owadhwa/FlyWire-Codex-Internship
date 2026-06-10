# FlyWire Codex Internship

# Conserved Directed Microcircuit Search Across Connectomes

## Overview

This project identifies candidate conserved directed microcircuits across multiple large-scale Drosophila connectomic datasets by searching for recurrent induced subgraph structures shared between independently reconstructed connectomes.

The pipeline combines:

* graph-theoretic preprocessing,
* stochastic local subgraph growth,
* canonical structural signatures,
* cross-dataset motif matching,
* and exact directed graph isomorphism verification.

The analysis was performed across three FlyWire Codex datasets:

* MCNS
* BANC
* FAFB

Each dataset is represented as a directed graph in which:

* nodes correspond to neurons,
* directed edges correspond to synaptic connectivity.

Edge weights (synapse counts) were ignored for this analysis.

---

# 1. Graph Construction

Each connectome edge list is loaded into a directed graph (`networkx.DiGraph`) using neuron identifiers as graph nodes and synaptic connections as directed edges.

Self-loops are removed to avoid trivial recurrence motifs.

```python
G = nx.DiGraph()

G.add_edges_from(
    zip(source_ids, target_ids)
)

G.remove_edges_from(
    nx.selfloop_edges(G)
)
```

The resulting graph preserves directed connectivity structure necessary for identifying recurrent local circuit motifs.

---

# 2. Restricting the Search Space

The full connectomes contain hundreds of thousands of neurons and millions of edges, making exhaustive induced-subgraph search computationally intractable.

To focus the search on biologically relevant local circuitry, the analysis was restricted to neighborhoods surrounding manually identified corresponding neurons across datasets.

For each dataset:

1. seed neurons were selected,
2. local neighborhoods were expanded by traversing incoming and outgoing connections,
3. induced local subgraphs were constructed.

```python
new_nodes.update(G.predecessors(n))
new_nodes.update(G.successors(n))
```

This procedure enriches the search space for recurrent local circuitry while substantially reducing computational complexity.

The resulting search graphs preserve directed connectivity structure within local neighborhoods surrounding putatively homologous neurons.

---

# 3. Candidate Subgraph Generation

The pipeline searches for conserved induced subgraphs of size `N` using a stochastic graph-growth procedure.

## 3.1 Seed Selection

Candidate generation begins from high-degree nodes under the assumption that densely connected neurons are more likely to participate in recurrent motifs.

```python
high_degree_nodes = sorted(
    G.nodes(),
    key=lambda n: G.in_degree(n) + G.out_degree(n),
    reverse=True
)
```

Additional random seeds are sampled to improve exploration of graph space.

---

## 3.2 Stochastic Neighborhood Expansion

Subgraphs are grown iteratively from a seed node by adding neighboring nodes connected through incoming or outgoing edges.

At each step:

* neighboring nodes are collected,
* previously selected nodes are excluded,
* the next node is chosen either:

  * greedily (high-degree preference),
  * or randomly.

```python
neighbors.update(G.predecessors(n))
neighbors.update(G.successors(n))
```

This biases the search toward densely interconnected local circuits while still allowing stochastic exploration of graph topology.

The resulting subgraph is treated as a directed induced subgraph.

---

# 4. Canonical Structural Signatures

Each candidate induced subgraph is converted into a canonical-like structural signature that approximates graph isomorphism while remaining computationally efficient.

The signature incorporates:

* number of nodes,
* number of directed edges,
* node in/out degrees,
* predecessor degree distributions,
* successor degree distributions,
* relabeled edge structure.

```python
node_features[n] = (
    S.in_degree(n),
    S.out_degree(n),
    pred_degrees,
    succ_degrees
)
```

Nodes are ordered according to these local structural features, and edges are relabeled into canonical index space.

The resulting tuple acts as a hashable structural fingerprint.

This allows rapid grouping of candidate motifs without performing expensive exact graph-isomorphism comparisons across all candidate subgraphs.

---

# 5. Cross-Dataset Motif Matching

Candidate signatures from all datasets are pooled into shared buckets.

```python
buckets[sig].append({...})
```

A motif is considered conserved if the same signature appears across at least three datasets.

```python
if len(datasets_present) >= min_datasets
```

The search proceeds from larger motif sizes to smaller motif sizes:

```python
for size in range(max_size, min_size - 1, -1)
```

Thus, the first successful match corresponds to the largest conserved candidate motif identified under the search constraints.

---

# 6. Exact Directed Isomorphism Verification

Because the canonical signature is only an approximation of graph isomorphism, all candidate motifs are verified using exact directed graph isomorphism.

Verification is performed using the VF2 algorithm implemented in NetworkX:

```python
GM = iso.DiGraphMatcher(S1, S2)
GM.is_isomorphic()
```

This ensures that:

* edge directionality is preserved,
* induced connectivity structure is identical,
* node labels themselves are ignored.

Thus, motifs are matched based purely on directed graph topology.

---

# 7. Output

For the best conserved motif:

* neuron identities are exported,
* dataset membership is recorded,
* matched node correspondences are saved as CSV.

```python
solution.to_csv(out, index=False)
```

The output enables downstream:

* manual biological inspection,
* visualization,
* neurotransmitter annotation,
* cell-type analysis,
* and connectomic interpretation.

---

# Limitations

Several limitations should be noted:

1. The canonical structural signature is an approximation and does not guarantee graph isomorphism.
2. Stochastic candidate growth may miss rare motifs.
3. Degree-biased exploration favors dense recurrent circuits over sparse motifs.
4. Local-neighborhood restriction biases the search toward motifs surrounding manually selected seed neurons.
5. Connectivity alone does not capture neurotransmitter identity, synaptic strength, physiology, or morphology.

Thus, all candidate motifs should be interpreted as putative conserved microcircuits requiring additional biological validation.
