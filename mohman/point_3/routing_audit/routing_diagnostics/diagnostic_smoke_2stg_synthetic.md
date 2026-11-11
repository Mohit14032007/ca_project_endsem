# Routing and Prefetch Pipeline Forensic Diagnostic

**Run Directory:** `/home/mohman/ca_project/mohman/point_3/routing_audit/results/smoke_2stg_synthetic`  
**Ground Truth Trace:** ``  

| Seq | RelTok | SimTok | MoE Stage | Routing Source | Ground Truth Top-K | update_expert() Routing | GEMM Demanded Experts | Match? | Predicted Targets | Admitted | Dispatched |
|---|---|---|---|---|---|---|---|---|---|---|---||
| 0 | 0 | 128 | 0 | SYNTHETIC | N/A (SYNTHETIC) | 15;39;51;63;77;117;162;165 | 15;39;51;63;77;117;162;165 | EXACT_MATCH |  |  |  |
| 0 | 0 | 128 | 1 | SYNTHETIC | N/A (SYNTHETIC) | 144;156;186;193;196;213;225;237 | 144;156;186;193;196;213;225;237 | EXACT_MATCH |  |  |  |
| 0 | 1 | 129 | 0 | SYNTHETIC | N/A (SYNTHETIC) | 20;23;68;88;99;133;150;164 | 20;23;68;88;99;133;150;164 | EXACT_MATCH |  |  |  |
| 0 | 1 | 129 | 1 | SYNTHETIC | N/A (SYNTHETIC) | 87;98;100;160;174;239;248;253 | 87;98;100;160;174;239;248;253 | EXACT_MATCH |  |  |  |
| 0 | 2 | 130 | 0 | SYNTHETIC | N/A (SYNTHETIC) | 30;57;68;95;141;205;207;237 | 30;57;68;95;141;205;207;237 | EXACT_MATCH |  |  |  |
| 0 | 2 | 130 | 1 | SYNTHETIC | N/A (SYNTHETIC) | 47;49;55;56;100;141;156;244 | 47;49;55;56;100;141;156;244 | EXACT_MATCH |  |  |  |
| 0 | 3 | 131 | 0 | SYNTHETIC | N/A (SYNTHETIC) | 11;13;46;64;154;159;225;247 | 11;13;46;64;154;159;225;247 | EXACT_MATCH |  |  |  |
| 0 | 3 | 131 | 1 | SYNTHETIC | N/A (SYNTHETIC) | 8;10;57;132;153;208;209;226 | 8;10;57;132;153;208;209;226 | EXACT_MATCH |  |  |  |
