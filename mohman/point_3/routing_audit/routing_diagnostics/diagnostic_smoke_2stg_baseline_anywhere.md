# Routing and Prefetch Pipeline Forensic Diagnostic

**Run Directory:** `/home/mohman/ca_project/mohman/point_3/routing_audit/results/smoke_2stg_baseline_anywhere`  
**Ground Truth Trace:** `/home/mohman/ca_project/mohman/point_3/routing_audit/traces/moe_anywhere_2stg_t4.csv`  

| Seq | RelTok | SimTok | MoE Stage | Routing Source | Ground Truth Top-K | update_expert() Routing | GEMM Demanded Experts | Match? | Predicted Targets | Admitted | Dispatched |
|---|---|---|---|---|---|---|---|---|---|---|---||
| 0 | 0 | 128 | 0 | REAL_TRACE | 140;81;206;95;67;200;30;115 | 140;81;206;95;67;200;30;115 | 30;67;81;95;115;140;200;206 | EXACT_MATCH |  |  |  |
| 0 | 0 | 128 | 1 | REAL_TRACE | 233;94;11;112;250;224;172;128 | 233;94;11;112;250;224;172;128 | 11;94;112;128;172;224;233;250 | EXACT_MATCH |  |  |  |
| 0 | 1 | 129 | 0 | REAL_TRACE | 77;184;37;138;75;173;26;232 | 77;184;37;138;75;173;26;232 | 26;37;75;77;138;173;184;232 | EXACT_MATCH |  |  |  |
| 0 | 1 | 129 | 1 | REAL_TRACE | 214;112;172;147;129;224;250;24 | 214;112;172;147;129;224;250;24 | 24;112;129;147;172;214;224;250 | EXACT_MATCH |  |  |  |
| 0 | 2 | 130 | 0 | REAL_TRACE | 140;81;115;95;206;86;235;177 | 140;81;115;95;206;86;235;177 | 81;86;95;115;140;177;206;235 | EXACT_MATCH |  |  |  |
| 0 | 2 | 130 | 1 | REAL_TRACE | 112;94;172;224;147;250;129;19 | 112;94;172;224;147;250;129;19 | 19;94;112;129;147;172;224;250 | EXACT_MATCH |  |  |  |
| 0 | 3 | 131 | 0 | REAL_TRACE | 43;236;30;238;147;73;124;191 | 43;236;30;238;147;73;124;191 | 30;43;73;124;147;191;236;238 | EXACT_MATCH |  |  |  |
| 0 | 3 | 131 | 1 | REAL_TRACE | 112;19;33;86;147;224;129;167 | 112;19;33;86;147;224;129;167 | 19;33;86;112;129;147;167;224 | EXACT_MATCH |  |  |  |
