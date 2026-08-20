import json
import sys

try:
    with open('results/5fold_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print("=== Zero-Shot (M0) Test Set CPC ===")
    for c in data.get('city_level_results', []):
        fold = c.get('fold', 'N/A')
        city = c.get('city', 'Unknown')
        m0_cpc = c.get('M0', {}).get('cpc_inter', 0.0)
        print(f"Fold {fold} | City: {city:15} | M0 CPC: {m0_cpc:.4f}")
except Exception as e:
    print(f"Error: {e}")
