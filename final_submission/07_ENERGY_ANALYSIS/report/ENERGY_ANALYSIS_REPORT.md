# Requirement 7: Energy Analysis (Native Model & Calibrated Transport)

## 1. Native Model Inventory
The simulator natively models on-chip HBM access and MAC computation:
- **DRAM Activation Energy:** $0.452\text{ J}$ per 4 decoding tokens.
- **DRAM Read Energy:** $3.815\text{ J}$ per 4 decoding tokens.
- **DRAM Write Energy:** $1.280\text{ J}$ per 4 decoding tokens.
- **MAC Compute Energy:** $3.521\text{ J}$ per 4 decoding tokens.
- **Total Native Energy:** $9.068\text{ J}$.

## 2. Calibrated Off-Chip DDR5 Transport Energy
Speculative transfers traversing the off-chip DDR5 bus are calibrated at standard DDR5 energy consumption:
$$E_{\text{expert}} = 88,080,384\text{ bytes} \times 28\text{ pJ/byte} = 2.466\text{ mJ/expert}$$
For ST-MoE (395 speculative transfers), estimated off-chip DDR5 transport energy is $0.974\text{ J}$ ($10.7\%$ total system energy increment).
