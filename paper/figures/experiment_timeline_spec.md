# Figure Spec: Experiment Timeline

**Figure type:** Horizontal timeline / bar chart
**Caption:** AetherForge experimental timeline (v2.5–v2.11). Bars show benchmark scores; dashed line marks the clean champion (23/28 = 82.1%). Grey bars are negative results; the green bar is the champion; the orange bar is the diagnostic repair result.

## Data points (left to right)

| Label | Score | Benchmark | Type | Colour |
|---|---|---|---|---|
| v2.5 data-mixture | 53.6% | 28-task | Negative | Grey |
| v2.6 traces=0% | 57.1% | 28-task | Negative | Grey |
| v2.7 adapter-only | 64.3% | 28-task | Negative | Grey |
| **v2.7 champion (+memory)** | **82.1%** | **28-task** | **Clean champion** | **Green** |
| v2.8 k=1 | 71.4% | 28-task | Negative | Grey |
| v2.8 k=5 | 78.6% | 28-task | Negative | Grey |
| v2.9 repair diagnostic | 96.4% | 28-task | Diagnostic | Orange |
| v2.9 clean transfer | 80.0% | 5-task | Positive signal | Light green |
| v2.10 champion (32-task) | 62.5% | 32-task | Clean | Green |
| v2.10 repair (32-task) | 56.2% | 32-task | Negative | Grey |
| v2.11 confidence router | 62.5% | 32-task | Neutral | Grey |
| v2.11 oracle ceiling | 71.9% | 32-task | Diagnostic | Orange |

## Annotations

- Dashed horizontal line at 82.1% labelled "Clean champion (23/28)"
- Label on v2.9 bar: "Diagnostic only — benchmark not independent"
- Label on v2.11 oracle bar: "Theoretical ceiling — not deployable"
- X-axis: experiment version
- Y-axis: score (%)
- Two sub-groups: "28-task frozen benchmark" and "32-task clean benchmark"

## Key visual message

The champion (green bar, v2.7) is the highest clean result. All retraining bars (grey, v2.5/v2.6) fall below. The repair diagnostic bar (orange, v2.9) is taller but visually distinct as non-clean. The 32-task clean results (right group) show champion and repair converging, with oracle as the ceiling.
