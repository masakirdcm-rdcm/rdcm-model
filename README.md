# RDCM Model

A minimal computational model where **persistent error does not vanish but forms structure**.

## Overview
RDCM (Recursive Differentiation–Convergence Model) describes a non-standard convergence process:

- Error is not minimized
- Variability is preserved
- Stable structure emerges from accumulated differences

This repository provides minimal simulations demonstrating how dynamic systems can converge to structured patterns without eliminating fluctuations.

## Demo (core idea)
Depending on parameter settings:

- Low variability → trivial uniform convergence  
- Moderate variability → structured pattern formation (RDCM regime)  
- High variability → collapse / chaos  

## Files
- `fig1a.py` : conceptual structure (Figure 1A)
- `fig1b.py` : feedback structure (Figure 1B)
- `timeseries.py` : temporal evolution
- `phase.py` : phase-space behavior

## Usage
```bash
python timeseries.py
python phase.py
```

## Notes
- Stochastic process (fixed seed for reproducibility)
- Phase diagram is averaged over multiple runs

## Associated Paper
arXiv preprint (coming soon)

## Author
T. Masaki
