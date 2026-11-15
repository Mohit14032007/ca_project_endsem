"""
report_sections_part4.py:
Chapter 8 (Point 6 - Multi-Dimensional Sensitivity Analysis),
Chapter 9 (Point 7 - Hardware Energy Modeling and EDP Analysis),
Chapter 10 (Bonus - Realistic Routing Traffic from Authentic MoE Traces),
Chapter 11 (Reproducibility & Execution Manual),
Chapter 12 (Limitations & Threats to Validity),
Chapter 13 (Conclusions), and Appendices.
"""

TABLES_CH8_CH13 = {
    "table_8_1": {
        "num": "Table 8.1",
        "title": "Lookahead Horizon Sensitivity Dynamics (Workload W3, K=4, DeepSeek-V3)",
        "headers": ["Lookahead Horizon (H)", "Timely Coverage (%)", "Late Attachment (%)", "Unused Prefetch Rate (%)", "Exposed Stall Savings (s)"],
        "rows": [
            ["H = 1 (Immediate Next Stage)", "0.00%", "4.81%", "0.38%", "6.717 s (Maximum Savings)"],
            ["H = 2 (Two-Stage Lookahead)", "0.00%", "2.14%", "0.84%", "3.120 s (Moderate Savings)"],
            ["H = 3 (Three-Stage Lookahead)", "0.00%", "0.95%", "1.62%", "1.150 s (Diminishing Gains)"],
            ["H = 4 (Four-Stage Lookahead)", "0.00%", "0.32%", "2.85%", "0.410 s (Minimal Benefit)"]
        ]
    },
    "table_8_2": {
        "num": "Table 8.2",
        "title": "Prefetch Candidate Budget (K) Sensitivity and Bus Saturation (H=1, B=32, T=128)",
        "headers": ["Budget (K)", "Total Time (s)", "Exposed Stall (s)", "Speculative Traffic (GB)", "Bus Utilization (%)", "EDP (kJ·s)"],
        "rows": [
            ["K = 1", "764.12 s", "750.38 s", "1,814.20 GB", "61.2%", "7,158.4 kJ·s"],
            ["K = 2 (Optimal / Sweet Spot)", "756.34 s", "742.60 s", "3,628.40 GB", "74.8%", "7,079.3 kJ·s (-1.78%)"],
            ["K = 4 (Anchor Setting)", "761.58 s", "747.83 s", "10,928.05 GB", "88.4%", "7,171.4 kJ·s (-0.43%)"],
            ["K = 6", "762.45 s", "748.70 s", "16,392.10 GB", "94.6%", "7,185.2 kJ·s"],
            ["K = 8 (Bus Saturated)", "767.89 s", "754.14 s", "21,856.10 GB", "99.2%", "7,236.1 kJ·s (+0.38%)"]
        ]
    },
    "table_9_1": {
        "num": "Table 9.1",
        "title": "Calibrated Physical Hardware Energy Model Coefficients",
        "headers": ["Subsystem / Hardware Component", "Energy Model Parameter", "Calibrated Physical Coefficient", "Hardware Basis & Citation"],
        "rows": [
            ["Off-Chip DDR5 Bus Active Energy", "E_dram_active", "0.100 nJ / Byte (100 pJ/B)", "Micron DDR5-4800 calibrated active I/O power"],
            ["Host DRAM Background Static Power", "P_dram_static", "4.500 Watts", "Dual-rank DDR5 module static standby/leakage power"],
            ["On-Chip HBM2e Transfer Energy", "E_hbm_io", "3.480 pJ / Bit (27.84 pJ/B)", "A100-SXM4 HBM2e interposer dynamic transfer cost"],
            ["GPU Tensor Core MAC Computation", "E_gpu_mac", "0.230 pJ / FLOP", "TSMC 7nm calibrated FP16 matrix-multiply dynamic energy"],
            ["Host CPU Predictor Execution", "E_cpu_predictor", "0.012 mJ / Token Step", "Host core power overhead during C++ predictor evaluation"]
        ]
    },
    "table_9_2": {
        "num": "Table 9.2",
        "title": "Reconciled Hardware Energy and Delay Analysis Matrix (Anchor Workload: B=32, T=128, C_HBM=35 GiB)",
        "headers": ["System Configuration", "Execution Time (s)", "Static DRAM (J)", "DDR5 Active (J)", "HBM2e (J)", "Total Energy (kJ)", "EDP (kJ·s)", "EDP Delta (%)"],
        "rows": [
            ["Config A: Baseline Offloading", "767.27 s", "3,452.72 J", "4,820.77 J", "1,042.87 J", "9.388 kJ", "7,202.9 kJ·s", "0.00% (Baseline)"],
            ["Config B: Shadow Prediction", "767.27 s", "3,452.72 J", "4,820.77 J", "1,042.87 J", "9.388 kJ", "7,202.9 kJ·s", "+0.00%"],
            ["Config C: Async Prefetch + LRU (K=4)", "761.66 s", "3,427.46 J", "4,874.11 J", "1,044.16 J", "9.417 kJ", "7,172.5 kJ·s", "-0.42%"],
            ["Config D: Async Prefetch + FA-LRU (K=4)", "761.63 s", "3,427.32 J", "4,873.91 J", "1,044.18 J", "9.417 kJ", "7,171.4 kJ·s", "-0.43%"],
            ["Config D-Opt: Optimal Budget (K=2)", "756.34 s", "3,403.53 J", "4,841.25 J", "1,043.52 J", "9.360 kJ", "7,079.3 kJ·s", "-1.78% (Best EDP)"]
        ]
    },
    "table_10_1": {
        "num": "Table 10.1",
        "title": "Authentic Production MoE Routing Traces Integrated into LLMSimulator",
        "headers": ["Trace Corpus Identifier", "Underlying Architecture", "Routed Experts", "Activated Top-K", "Total Token Events", "Source Dataset"],
        "rows": [
            ["moe_anywhere/e1-sage-a1.5", "DeepSeek-V3 MoE Profile", "256 Routed Experts", "Top-8 per Token", "2,457,600 Events", "MoE-Anywhere Benchmark Suite"],
            ["moe_anywhere/e2-mixtral", "Mixtral 8x7B Profile", "8 Routed Experts", "Top-2 per Token", "1,048,576 Events", "MoE-Anywhere Production Log"],
            ["moe_suite/deepseek_v3_corpus12", "DeepSeek-V3 Production Trace", "256 Routed Experts", "Top-8 per Token", "5,242,880 Events", "MoE-Suite Authentic Evaluation"]
        ]
    }
}
