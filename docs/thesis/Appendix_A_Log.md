# Appendix A: SpECTRE Simulation Run Log

**SpECTRE Baseline Tag:** `v2024.09.29`
**Cluster:** _______________
**MPI/Compiler Modules:** _______________

| Run ID | Execution Date | Git Commit Hash | Slurm Job ID | Nodes/Cores | $s_\gamma, s_\mu$ config | Output HDF5 Path | Notes |
|--------|----------------|-----------------|--------------|-------------|--------------------------|------------------|-------|
| Tier0_1|                |                 |              | 10 / 560    | Baseline                 |                  | Single BH Verification |
| Tier1_B|                |                 |              | 10 / 560    | Baseline                 |                  | Inspiral to 2400M |
| Tier1_S1|               |                 |              | 10 / 560    | Modified Damping (-/+)   |                  | Sensitivity Test 1 |
| Tier1_L|                |                 |              | 10 / 560    | Feedback Enabled         |                  | Dynamic Scale Factors |
| Tier2_B|                |                 |              | 10 / 560    | Baseline                 |                  | Full Merger to Ringdown |
