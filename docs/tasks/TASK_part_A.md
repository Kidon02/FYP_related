# TASK: Part A of the supervisor's reply (due Sat 7 Nov 2026)

How to use: save this file as `docs/tasks/TASK_part_A.md` in the repo. In Claude Code, say:
"Do the task in docs/tasks/TASK_part_A.md." If a session ends early, start a new one and say:
"Continue docs/tasks/TASK_part_A.md from the first unticked step." The progress log at the
bottom is the resume point.

## Context
The supervisor approved BAA10Y (Moody's Baa yield minus the 10-year Treasury, FRED) as the
replacement for the discontinued TED spread, and asked for stationary macro features, because the
10-year yield and US EPU leave their training ranges in the 2019-2024 test period (yield test
minimum 0.52 vs training minimum 1.43; EPU_US test maximum 504 vs 284). Reference [16] is
Mostafavi and Hooman (2025). Part B (Decision Tree, tuning, expanding window, RF-LSTM, SHAP/LIME,
rolling SHAP) comes later and is NOT part of this task.

## Hard rules (never break, even to make a check pass)
1. Download nothing except the single FRED series BAA10Y. Never run data_pipeline.py --refresh:
   it would re-download Yahoo data, whose history can change. Never modify existing files in data/raw/.
2. Keep the date range 2000-01-01 to 2024-12-31.
3. Do not overwrite data/processed/panel_full_2026-10-27.csv. Write a new dated panel.
4. No future information in any feature, label, threshold or standardisation statistic.
5. Never edit result files by hand. Report results as they come out, including negative ones.
6. No empty or backdated commits. Commit after every step with a descriptive message. Do not push to
   any branch other than main from this laptop session, and never merge claude/* branches.
7. CPU only. Never commit secrets.

## Steps (tick each in the progress log when its check passes, then commit)

A1. Fetch BAA10Y only.
- Add a targeted option to code/data_pipeline.py, e.g. `--fetch-series BAA10Y`, that downloads only that
  FRED series to data/raw/fred_BAA10Y.csv (needs FRED_API_KEY) and leaves every other raw file untouched.
- Check: the series covers 2000-01-03 to 2024-12-31 with no gap longer than 5 trading days; report its
  first/last date and missing share. Record the substitution (why TED was dropped, why not HY OAS) in
  data_readme.txt.

A2. Build a new panel and prove nothing else changed.
- Rebuild from cache into data/processed/panel_full_2026-11-<DD>.csv with BAA10Y added. Point configs
  and run_all.sh at the new file.
- Check: every column that exists in panel_full_2026-10-27.csv is identical in the new panel (same rows,
  same values). If not, stop and report the root cause.

A3. Stationary macro features (configs/features.yaml, config flags, old definitions kept as an option).
- 10-year yield and BAA10Y: daily change. USD/JPY: log change (already). EPU US and Japan: log of the
  previous-month value, then standardised with a trailing 252-day mean and standard deviation.
- Every trailing statistic uses data up to t-1 only. The existing no-lookahead test must cover the new
  features and must pass for all three indices.
- Report how many extra warm-up rows the new features drop. If the row set changes, all feature sets
  (including Domestic-Only) must be rerun in A5 so every comparison uses the same rows.

A4. Out-of-range table (new file results/tables/table_macro_range.csv plus a settings file).
- For each macro feature, before and after the transformation: training (2000-2015) minimum and maximum,
  test (2019-2024) minimum and maximum, and the share of test days outside the training range.

A5. Rerun and retest.
- Run `python code/train.py` then `python code/test.py` (full rerun, all three indices). Plug in the
  laptop and stop it from sleeping; estimate the runtime first.
- Report Domestic + macro and Cross-Market against Domestic-Only again (Wilcoxon, Holm), for XGBoost,
  LR and Isolation Forest, next to the previous values. Label the DM and significance tables
  "provisional" if any step above did not finish.

A6. Event-by-scheme matrix (results/tables/table_event_matrix.csv plus a settings file).
- For each index, labelling scheme and crisis window in configs/labelling.yaml: number of flagged days
  inside the window, detected or missed, and the distance in trading days from the window to the
  nearest flagged day.

A7. Matched single-threshold control (results/tables/table_matched_control.csv plus a settings file).
- For each index and for the hmm and regime_std schemes: on rows where the 252-day rolling z-score and
  regime_filtered are both defined, choose the single |z| threshold that gives the same number of flags
  as the scheme; report that threshold, the share of its flags in the high-volatility regime, the
  scheme's share, and the regime's share of days. (Expected roughly 37-51% vs 24-33%; if very different,
  explain why before continuing.)

A8. Records.
- CLAUDE.md: same-day foreign features are nowcasting inputs (available a few hours before the target
  close), not a next-day early warning; S&P 500 XGBoost Domestic-Only PR-AUC is 0.170 with the causal
  HMM vs 0.268 with the full-sample HMM; reference [16] is Mostafavi and Hooman (2025); attachments sent
  to the supervisor always get a distinct dated or versioned file name.
- STATUS.md: update the state, findings, decisions (BAA10Y approved, [16] confirmed) and next steps
  (Part B list).

## Failure rule (root-cause analysis)
For any failed check or error: state what failed, why, and the evidence (file, line, values). Apply the
smallest fix inside these rules, rerun the check, and retry up to 3 times. If it still fails, or the fix
needs something outside the rules (a decision about method, a new download, admin rights), stop, record
it in the progress log, and ask.

## Report back (paste into the technical chat when done)
1. The progress log below.
2. table_macro_range.csv.
3. The new significance table and the Domestic + macro / Cross-Market comparison, old vs new.
4. table_event_matrix.csv and table_matched_control.csv.
5. Anything that failed, with its root cause.

## Progress log (update after every step; this is the resume point)
- [x] A1 BAA10Y fetched and documented (2000-01-03 to 2024-12-31; 0.78% of panel days missing, longest gap 1 day)
- [ ] A2 new panel built; old columns identical
- [ ] A3 stationary macro features; no-lookahead test passes (GSPC, N225, FTSE)
- [ ] A4 out-of-range table
- [ ] A5 full rerun of train.py and test.py; comparison reported
- [ ] A6 event-by-scheme matrix
- [ ] A7 matched control table
- [ ] A8 CLAUDE.md and STATUS.md updated
Notes:
