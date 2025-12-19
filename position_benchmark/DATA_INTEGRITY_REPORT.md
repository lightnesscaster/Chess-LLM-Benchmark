# Position Benchmark Data Integrity Report
Generated: 2025-12-19

## Summary

### Overall Status: ⚠️ ISSUES FOUND

The position benchmark results have **1 critical data integrity issue** affecting 20 out of 24 models.

---

## Files Analyzed

1. **equal_results.json** - Equal position test results (100 positions per model)
2. **results.json** - Blunder position test results (100 positions per model)
3. **results_combined.json** - Summary statistics aggregation

---

## Verification Checks Performed

### ✅ 1. Model Count Check
- **equal_results.json**: 24 models ✓
- **results.json**: 24 models ✓
- **results_combined.json**: 24 models ✓

All three files have the expected 24 models.

---

### ✅ 2. Model Name Consistency
All 24 models are present in all three files with matching names:

1. eubos
2. survival-bot
3. maia-1900
4. maia-1100
5. gemini-2.5-flash (no thinking)
6. random-bot
7. deepseek-chat-v3-0324
8. kimi-k2-0905
9. gemini-2.0-flash-001
10. llama-4-maverick
11. gpt-3.5-turbo-0613
12. kimi-k2
13. deepseek-chat-v3.1 (no thinking)
14. deepseek-v3.1-terminus (no thinking)
15. qwen3-235b-a22b-2507
16. deepseek-v3.2 (no thinking)
17. deepseek-r1-distill-qwen-32b
18. glm-4.6 (no thinking)
19. mistral-medium-3
20. gpt-3.5-turbo
21. llama-3.3-70b-instruct
22. gpt-5-chat
23. grok-3-mini
24. glm-4.6 (thinking)

---

### ✅ 3. Position Count Check

**equal_results.json (Equal Positions)**
- All 24 models have exactly 100 positions ✓

**results.json (Blunder Positions)**
- All 24 models have exactly 100 positions ✓

---

### ✅ 4. Required Fields Check

All per-position data includes the required fields:
- `position_idx` ✓
- `fen` ✓
- `model_move` ✓
- `best_move` ✓
- `cpl` ✓
- `is_legal` ✓
- `is_best` ✓

---

### ✅ 5. Summary Statistics Check

All models in `results_combined.json` have the required summary fields:
- `legal_pct` ✓
- `best_pct` ✓
- `avg_cpl` ✓
- `avoided_pct` ✓
- `median_cpl` ✓
- `type` ✓

---

### ❌ 6. CRITICAL ISSUE: Duplicate Position Indices in results.json

**Problem**: 20 out of 24 models have duplicate `position_idx` values in results.json

**Pattern**: For affected models, positions 93-99 (the last 7 positions) all have `position_idx: 0` instead of their correct indices (93, 94, 95, 96, 97, 98, 99).

**Affected Models** (20):
1. random-bot
2. maia-1100
3. maia-1900
4. gemini-2.5-flash (no thinking)
5. gemini-2.0-flash-001
6. glm-4.6 (no thinking)
7. deepseek-v3.2 (no thinking)
8. kimi-k2-0905
9. deepseek-v3.1-terminus (no thinking)
10. kimi-k2
11. qwen3-235b-a22b-2507
12. deepseek-chat-v3-0324
13. mistral-medium-3
14. gpt-3.5-turbo-0613
15. deepseek-chat-v3.1 (no thinking)
16. llama-3.3-70b-instruct
17. gpt-3.5-turbo
18. llama-4-maverick
19. deepseek-r1-distill-qwen-32b
20. survival-bot

**Unaffected Models** (4):
1. eubos ✓
2. gpt-5-chat ✓
3. grok-3-mini ✓
4. glm-4.6 (thinking) ✓

**Impact**:
- Each affected model has 8 positions with `position_idx: 0` (the original position 0 + positions 93-99)
- This means 7 unique position indices are missing from each affected model (82, 94, 95, 96, 97, 98, 99)
- The actual position data (FEN, moves, etc.) appears to be present and correct
- Only the metadata field `position_idx` is incorrect

**Example from random-bot**:
```
Position 0:   position_idx=0 ✓ (correct)
Position 93:  position_idx=0 ✗ (should be 93)
Position 94:  position_idx=0 ✗ (should be 94)
Position 95:  position_idx=0 ✗ (should be 95)
Position 96:  position_idx=0 ✗ (should be 96)
Position 97:  position_idx=0 ✗ (should be 97)
Position 98:  position_idx=0 ✗ (should be 98)
Position 99:  position_idx=0 ✗ (should be 99)
```

---

### ✅ 7. equal_results.json - No Duplicate Indices

All 24 models in `equal_results.json` have unique position indices with no duplicates.

---

### ✅ 8. Summary Statistics Accuracy

Verified that summary statistics in both `results.json` and `results_combined.json` accurately match the calculated values from per-position data. All percentages and aggregations are correct.

---

## Root Cause Analysis

The duplicate index issue suggests a bug in the code that generates `results.json`, specifically affecting positions 93-99. The fact that exactly 4 models are unaffected (and they appear to be tested later based on their position in the results) suggests:

1. The bug may have been introduced partway through the benchmark run
2. OR these 4 models were tested in a different batch/run where the bug was fixed
3. The bug appears to set `position_idx` to 0 as a default value when something goes wrong

The fact that the actual position data (FEN, moves, CPL) is correct but only the `position_idx` field is wrong suggests this is a metadata assignment issue rather than a data collection issue.

---

## Recommendations

### Immediate Actions Required

1. **Fix the duplicate position_idx values** in results.json for the 20 affected models
   - Positions 93-99 should have `position_idx` values of 93, 94, 95, 96, 97, 98, 99 respectively

2. **Investigate the code** that generates results.json to find the bug causing this issue
   - Look for default value assignments or initialization issues around position index tracking
   - Check if there's special handling for the last 7 positions

3. **Re-run verification** after fixing the data to ensure all checks pass

### Data Quality Impact

Despite this issue:
- ✓ All 100 positions are actually tested for each model
- ✓ The actual chess data (FEN, moves, evaluations) is correct
- ✓ Summary statistics are calculated correctly
- ⚠️ The position_idx field is unreliable for 20 models in results.json
- ⚠️ Any code that relies on position_idx being unique will fail for affected models

---

## Conclusion

The position benchmark data is **mostly sound** but has a **critical metadata issue** that needs to be corrected. The issue is isolated to the `position_idx` field in `results.json` and does not affect the actual chess evaluation data or summary statistics. However, this should be fixed to maintain data integrity and prevent potential issues with any analysis code that assumes unique position indices.
