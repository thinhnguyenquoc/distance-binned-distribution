import json
import sys

try:
    with open('results/5fold_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    rq1 = data.get('rq1_delta_r', {})
    
    sys.stdout.buffer.write(b"=== SUMMARY M0 & M1 CPC (200 Epochs) ===\n")
    
    for geo in ['city', 'county', 'subzone']:
        if geo in rq1:
            sys.stdout.buffer.write(f"\n[{geo.upper()} LEVEL]\n".encode('utf-8'))
            geo_data = rq1[geo]
            m0_mean = geo_data.get('m0_cpc_inter', {}).get('mean', 0)
            m1_mean = geo_data.get('m1_cpc_inter', {}).get('mean', 0)
            diff_mean = geo_data.get('delta_cpc_inter', {}).get('mean', 0)
            p_improved = geo_data.get('p_improved', 0)
            
            sys.stdout.buffer.write(f"  M0 Baseline CPC : {m0_mean:.4f}\n".encode('utf-8'))
            sys.stdout.buffer.write(f"  M1 Oracle CPC   : {m1_mean:.4f}\n".encode('utf-8'))
            sys.stdout.buffer.write(f"  Delta (M1 - M0) : +{diff_mean:.4f}  ({(diff_mean/m0_mean*100) if m0_mean else 0:.2f}%)\n".encode('utf-8'))
            sys.stdout.buffer.write(f"  P(Delta > 0)    : {p_improved*100:.1f}%\n".encode('utf-8'))
            
except Exception as e:
    sys.stdout.buffer.write(f"Error reading file: {e}\n".encode('utf-8'))
