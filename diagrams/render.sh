#!/usr/bin/env bash
# Renders the conceptual diagrams (Figs. 1-3) with Graphviz (tested with 2.43).
set -e; cd "$(dirname "$0")"; mkdir -p ../figures
dot   -Tpng -Gdpi=600 fig1_framework.dot  -o ../figures/fig1_framework.png
neato -Tpng -Gdpi=600 fig2_causal.dot     -o ../figures/fig2_causal.png   # pinned node positions
dot   -Tpng -Gdpi=600 fig3_event_flow.dot -o ../figures/fig3_event_flow.png
