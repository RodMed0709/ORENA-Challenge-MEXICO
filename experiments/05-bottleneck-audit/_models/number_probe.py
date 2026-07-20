import re
import numpy as np
import pandas as pd
from typing import Union

# `parse_number` moved to `frame.parsing` when a third rung (10-self-consistency)
# came to depend on it — `_models/` is folder-private by repo rule. Re-exported
# here VERBATIM so 05b and 05c import exactly what they always did.
from frame.parsing import WORD_TO_NUM, parse_number  # noqa: F401

def probe_frame(responses: list, references: list, results_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Builds the 8-column contract DataFrame:
    qID, distribution, pred_text, pred_value, true_value, parsed, is_mode, correctness
    """
    rows = []
    
    ref_dict = {ref.qID: ref for ref in references}
    
    correctness_dict = {}
    if results_df is not None and not results_df.empty:
        if "qID" not in results_df.columns or "correctness" not in results_df.columns:
            raise ValueError("results_df must contain the 'qID' and 'correctness' columns.")
            
        for _, row in results_df.iterrows():
            correctness_dict[row["qID"]] = bool(row["correctness"])
                
    for resp in responses:
        qID = resp.qID
        ref = ref_dict[qID]
        
        # OOD if qID starts with heico
        distribution = "OOD" if qID.startswith("heico") else "ID"
        
        pred_text = resp.content
        pred_value = parse_number(pred_text)
        
        try:
            true_value = float(ref.answer)
        except (ValueError, TypeError):
            true_value = parse_number(str(ref.answer))
            
        parsed = not np.isnan(pred_value)
        
        # C9: Detect ambiguous negation
        text_lower = str(pred_text).lower()
        has_negation = bool(re.search(r'\b(no|none)\b', text_lower))
        has_other_num = bool(re.search(r'\d+|\b(zero|one|two|three|four|five|six|seven|eight|nine|ten)\b', text_lower))
        ambiguous_negation = has_negation and has_other_num
        
        correctness = correctness_dict.get(qID, np.nan)
        
        rows.append({
            "qID": qID,
            "distribution": distribution,
            "pred_text": pred_text,
            "pred_value": pred_value,
            "true_value": true_value,
            "parsed": parsed,
            "is_mode": False,  # Will update after gathering all rows
            "correctness": correctness,
            "ambiguous_negation": ambiguous_negation
        })
        
    df = pd.DataFrame(rows)
    if not df.empty:
        mode_val = df["true_value"].mode().iloc[0]
        df["is_mode"] = df["true_value"] == mode_val
        
    return df

def probe_metrics(df: pd.DataFrame, source: str) -> dict:
    """
    Computes the metrics minus verdict:
    source, n, n_no_parseable, n_ambiguous_negation, mode_value, mode_rate, acc_overall, 
    acc_mode, acc_non_mode, spearman_r, pred_mode_share, pred_entropy
    """
    n = len(df)
    n_no_parseable = (~df["parsed"]).sum()
    n_ambiguous_negation = df["ambiguous_negation"].sum() if "ambiguous_negation" in df.columns else 0
    
    if n > 0:
        mode_value = df["true_value"].mode().iloc[0]
        mode_rate = df["is_mode"].mean()
        acc_overall = df["correctness"].mean()
        acc_mode = df[df["is_mode"]]["correctness"].mean() if df["is_mode"].any() else np.nan
        acc_non_mode = df[~df["is_mode"]]["correctness"].mean() if (~df["is_mode"]).any() else np.nan
        
        parsed_df = df[df["parsed"]]
        if len(parsed_df) > 1:
            spearman_r = parsed_df["pred_value"].corr(parsed_df["true_value"], method="spearman")
        else:
            spearman_r = np.nan
            
        pred_mode_share = (parsed_df["pred_value"] == mode_value).mean() if len(parsed_df) > 0 else np.nan
        
        if len(parsed_df) > 0:
            value_counts = parsed_df["pred_value"].value_counts(normalize=True)
            pred_entropy = -(value_counts * np.log2(value_counts)).sum()
        else:
            pred_entropy = np.nan
            
    else:
        mode_value = np.nan
        mode_rate = np.nan
        acc_overall = np.nan
        acc_mode = np.nan
        acc_non_mode = np.nan
        spearman_r = np.nan
        pred_mode_share = np.nan
        pred_entropy = np.nan

    return {
        "source": source,
        "n": n,
        "n_no_parseable": n_no_parseable,
        "n_ambiguous_negation": n_ambiguous_negation,
        "mode_value": mode_value,
        "mode_rate": mode_rate,
        "acc_overall": acc_overall,
        "acc_mode": acc_mode,
        "acc_non_mode": acc_non_mode,
        "spearman_r": spearman_r,
        "pred_mode_share": pred_mode_share,
        "pred_entropy": pred_entropy
    }
