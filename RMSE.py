# One-shot script to compute Table 3 (correct values) + supplemental by Approach x Prompt_Type

import pandas as pd
import numpy as np

# 1) Load data
df = pd.read_excel('Data.xlsx')

# 2) Clean Prompt Type like in your analysis
prompt_type_mapping = {
    "1. Zero-Shot (ZS) (step-by-step instructions)": "Zero-Shot",
    "2. Few-shot (example of expected behaviors)": "Few-Shot",
    "3. Chain-of-thoughts (sequence of intermediary reasoning": "CoT",
    "4.Chain of thought including Problem context and rubric (CR)": "CoT-CR"
}
df['Prompt Type'] = df['Prompt Type'].map(lambda x: prompt_type_mapping.get(x, x))
df['Prompt Type'] = df['Prompt Type'].astype(str)
df['Prompt Type'] = df['Prompt Type'].str.replace(
    "3. Chain-of-thoughts (sequence of intermediary reasoning", "CoT", regex=False
)
df['Prompt Type'] = df['Prompt Type'].str.replace(
    "3. Chain-of-thoughts (sequence of intermediary reasoning steps)", "CoT", regex=False
)

# Rename for statsmodels-style (and consistency with your code)
df = df.rename(columns={'Prompt Type': 'Prompt_Type'})

# 3) Ensure numeric columns
df['Human Rating'] = pd.to_numeric(df.get('Human Rating'), errors='coerce')
df['Extracted Rating'] = pd.to_numeric(df.get('Extracted Rating'), errors='coerce')

# 4) Keep only complete cases for performance metrics
perf_df = df.dropna(subset=['Human Rating', 'Extracted Rating', 'Approach']).copy()

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

# 5) Table 3: Performance by Approach
rows = []
for approach, g in perf_df.groupby('Approach'):
    y_true = g['Human Rating'].astype(float).values
    y_pred = g['Extracted Rating'].astype(float).values
    r = np.corrcoef(y_true, y_pred)[0, 1] if len(g) > 1 else np.nan
    rows.append({
        'Approach': int(approach),
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'Correlation (r)': float(r) if np.isfinite(r) else np.nan,
        'N': int(len(g))
    })

table3_df = pd.DataFrame(rows).sort_values('Approach')

# Optional interpretation column (static text for your paper)
interpretation_map = {
    1: "Generic Summary — Overestimates quality, low alignment",
    2: "Structured Summary — Slight improvement via structured prompts",
    3: "Aspect-Focused Summary — Closest to human ratings; interpretable variance"
}
table3_df['Interpretation'] = table3_df['Approach'].map(interpretation_map)

# Pretty print
pd.options.display.float_format = '{:,.3f}'.format
print("\nTable 3: Performance Comparison of Summarization Approaches (computed from data)")
print(table3_df[['Approach', 'MAE', 'RMSE', 'Correlation (r)', 'N', 'Interpretation']])

# Save
table3_df.to_csv('table3_performance_by_approach.csv', index=False)

# 6) Supplemental: Performance by Approach x Prompt_Type (optional for appendix)
perf_axp = []
group_cols = ['Approach', 'Prompt_Type']
missing_cols = [c for c in group_cols if c not in perf_df.columns]
if not missing_cols:
    for (approach, ptype), g in perf_df.groupby(group_cols):
        y_true = g['Human Rating'].astype(float).values
        y_pred = g['Extracted Rating'].astype(float).values
        if len(g) >= 2:
            r = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            r = np.nan
        perf_axp.append({
            'Approach': int(approach),
            'Prompt_Type': ptype,
            'MAE': mae(y_true, y_pred),
            'RMSE': rmse(y_true, y_pred),
            'Correlation (r)': float(r) if np.isfinite(r) else np.nan,
            'N': int(len(g))
        })

    table3_axp = pd.DataFrame(perf_axp).sort_values(['Approach', 'Prompt_Type'])
    print("\nSupplement: Performance by Approach x Prompt_Type")
    print(table3_axp)
    table3_axp.to_csv('table3_performance_by_approach_prompt.csv', index=False)
else:
    print(f"\nSkipping Approach x Prompt_Type supplement because missing columns: {missing_cols}")