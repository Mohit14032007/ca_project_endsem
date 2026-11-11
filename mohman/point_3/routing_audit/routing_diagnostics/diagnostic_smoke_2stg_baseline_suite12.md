# Routing and Prefetch Pipeline Forensic Diagnostic

**Run Directory:** `/home/mohman/ca_project/mohman/point_3/routing_audit/results/smoke_2stg_baseline_suite12`  
**Ground Truth Trace:** `/home/mohman/ca_project/mohman/point_3/routing_audit/traces/moe_suite12_2stg_t4.csv`  

| Seq | RelTok | SimTok | MoE Stage | Routing Source | Ground Truth Top-K | update_expert() Routing | GEMM Demanded Experts | Match? | Predicted Targets | Admitted | Dispatched |
|---|---|---|---|---|---|---|---|---|---|---|---||
| 0 | 0 | 128 | 0 | REAL_TRACE | 142;71;68;208;77;240;209;94 | 142;71;68;208;77;240;209;94 | 68;71;77;94;142;208;209;240 | EXACT_MATCH |  |  |  |
| 0 | 0 | 128 | 1 | REAL_TRACE | 230;234;141;70;115;220;225;37 | 230;234;141;70;115;220;225;37 | 37;70;115;141;220;225;230;234 | EXACT_MATCH |  |  |  |
| 0 | 1 | 129 | 0 | REAL_TRACE | 71;150;28;19;191;167;30;248 | 71;150;28;19;191;167;30;248 | 19;28;30;71;150;167;191;248 | EXACT_MATCH |  |  |  |
| 0 | 1 | 129 | 1 | REAL_TRACE | 209;101;169;63;235;87;36;41 | 209;101;169;63;235;87;36;41 | 36;41;63;87;101;169;209;235 | EXACT_MATCH |  |  |  |
| 0 | 2 | 130 | 0 | REAL_TRACE | 110;34;83;14;128;109;88;185 | 110;34;83;14;128;109;88;185 | 14;34;83;88;109;110;128;185 | EXACT_MATCH |  |  |  |
| 0 | 2 | 130 | 1 | REAL_TRACE | 219;140;70;102;169;108;11;27 | 219;140;70;102;169;108;11;27 | 11;27;70;102;108;140;169;219 | EXACT_MATCH |  |  |  |
| 0 | 3 | 131 | 0 | REAL_TRACE | 230;99;132;252;155;197;75;150 | 230;99;132;252;155;197;75;150 | 75;99;132;150;155;197;230;252 | EXACT_MATCH |  |  |  |
| 0 | 3 | 131 | 1 | REAL_TRACE | 252;25;80;218;100;157;155;156 | 252;25;80;218;100;157;155;156 | 25;80;100;155;156;157;218;252 | EXACT_MATCH |  |  |  |
