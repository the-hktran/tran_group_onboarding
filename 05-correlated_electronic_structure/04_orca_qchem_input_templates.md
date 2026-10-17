# Module 05 Example 04: Standalone Input Deck Templates for ORCA and Q-Chem

Alongside Python-based workflows in [PySCF](https://www.google.com/search?q=https%3A%2F%2Fpyscf.org%2Fuser%2Findex.html) and [QuEmb](https://github.com/troyvvgroup/quemb), our group frequently runs DFT geometry optimizations, broken-symmetry transition-metal calculations, and coupled-cluster benchmarks using [ORCA](https://www.faccts.de/orca/) and [Q-Chem](https://manual.q-chem.com/latest/qchem_manual.pdf).

## 1. [ORCA](https://www.faccts.de/orca/) Input Templates

### 1A. Range-Separated Hybrid DFT Geometry Optimization + Harmonic Frequencies (`h2o_wb97xd3_opt.inp`)
```text
! wB97X-D3 def2-TZVPP def2/J RIJCOSX TightSCF TightOpt Freq
%pal nprocs 8 end
%maxcore 3500

* xyz 0 1
O    0.00000000    0.00000000    0.00000000
H    0.00000000   -0.75716000    0.58626000
H    0.00000000    0.75716000    0.58626000
*
```

### 1B. Domain-Based Local Pair Natural Orbital Coupled Cluster (`DLPNO-CCSD(T)`) Single-Point (`h2o_dlpno_ccsdt.inp`)
```text
! DLPNO-CCSD(T) cc-pVTZ cc-pVTZ/C TightPNO TightSCF
%pal nprocs 8 end
%maxcore 4000

* xyz 0 1
O    0.00000000    0.00000000    0.00000000
H    0.00000000   -0.75716000    0.58626000
H    0.00000000    0.75716000    0.58626000
*
```

## 2. [Q-Chem](https://manual.q-chem.com/latest/qchem_manual.pdf) Input Templates

### 2A. Canonical RHF / CCSD(T) Benchmark (`h2o_qchem_ccsdt.in`)
```text
$molecule
0 1
O    0.00000000    0.00000000    0.00000000
H    0.00000000   -0.75716000    0.58626000
H    0.00000000    0.75716000    0.58626000
$end

$rem
   METHOD          CCSD(T)
   BASIS           cc-pVTZ
   SCF_CONVERGENCE 10
   CC_CONVERGENCE  9
   THRESH          14
   N_FROZEN_CORE   FC
$end
```
