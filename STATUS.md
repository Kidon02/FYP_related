# Project status (updated 2026-11-07, after Part A)

## State
- Panel: data/processed/panel_full_2026-11-07.csv (6288 days x 19 columns) = the 2026-10-27 panel plus
  BAA10Y. Its 18 old columns are exactly identical to the 2026-10-27 panel.
- Macro features are stationary (configs/features.yaml `macro.transform: stationary`): DGS10 and BAA10Y
  daily change, USDJPY log change, EPU US/JP log of the previous-month value standardised over a trailing
  252-day window. The level definitions are kept as `transform: levels`.
- Feature rows: GSPC 6016, FTSE 6016 (20 and 19 fewer, from the EPU standardisation warm-up), N225 5427.
  The no-lookahead test passes for all three indices.
- Full rerun of code/train.py and code/test.py finished on 2026-11-07 (18 minutes, CPU, exit 0;
  logs/partA_rerun_2026-11-07.log). All results and tables in results/ are from this run.
- New tables: results/tables/table_macro_range.csv, table_event_matrix.csv, table_matched_control.csv,
  each with a settings file in thesis_tables/ (code/macro_range.py, code/event_tables.py, run by test.py).

## Findings
- Out-of-range: share of 2019-2024 test days outside the 2000-2015 training range, levels -> stationary:
  DGS10 20.9% -> 0.1%, EPU_US 12.3% -> 1.3%, BAA10Y 4.8% -> 0.07%, EPU_JP 0% -> 0%.
- Feature sets vs Domestic-Only (Wilcoxon signed-rank over 9 blocks, Holm), new run:
  - XGBoost: Domestic + macro worse in 9 of 9 blocks (median -0.039, p_holm 0.020); Cross-Market (lag 1)
    worse in 8 of 9 (p_holm 0.082, was 0.035).
  - Logistic regression: Domestic + macro and Cross-Market (lag 1) now worse in 9 of 9 (p_holm 0.020 each).
    With the level features they were better in 7 and 6 of 9 (p_holm 0.098 and 0.719). The earlier LR
    gain from macro inputs does not survive the stationary transform.
  - Isolation Forest: Cross-Market better than Domestic-Only in 9 of 9 for lag 1 and lag 0+1
    (p_holm 0.008 each; lag 1 was 7 of 9, 0.074).
- Causal vs full-sample HMM: S&P 500 XGBoost Domestic-Only PR-AUC 0.157 (causal) vs 0.268 (full-sample)
  in this run; 0.170 vs 0.268 in the 3 Nov run.
- Event matrix: every scheme detects Lehman, European debt crisis and COVID on all three indices. Misses:
  N225 Dot-com (all schemes); N225 and FTSE 2022 (baseline and regime-dependent threshold, not
  regime-standardised); Flash Crash on N225 and FTSE (regime-standardised only).
- Matched single-threshold control: a single |z| threshold (2.49-2.71) that flags as many days as the
  scheme puts 35-51% of its flags in the high-volatility regime, against 24-33% for the hmm and
  regime_std schemes (the regime covers 17-23% of days). The regime schemes move flags out of the
  high-volatility regime rather than just changing how many days are flagged.

## Decisions
- BAA10Y replaces TED as the credit spread (supervisor approved, Nov 2026). HY OAS rejected (FRED serves
  only a rolling 3-year window).
- Macro features made stationary (supervisor request).
- Reference [16] confirmed: Mostafavi and Hooman (2025).
- Same-day foreign features are nowcasting inputs, not a next-day early warning.
- Attachments to the supervisor always get a distinct dated or versioned file name.

## Next steps (Part B)
1. Decision Tree model.
2. Hyperparameter tuning.
3. Expanding-window evaluation.
4. RF-LSTM benchmark (Pandey & Sharma), in the consolidated table.
5. SHAP and LIME explanations.
6. Rolling 60-day SHAP share of foreign features against VIX (2019-2024).
- Still open: confirm the six crisis date windows; interim report due 10 Nov.
