"""Check manuscript source data and regenerate its statistical audit, without inference."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/interzonal_only'
ART = BASE / 'artifacts'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    sources = {}
    def read(name):
        path = ART / name
        sources[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text())
    protocol = json.loads((BASE/'interzonal_protocol.json').read_text())
    actual = {str(p.relative_to(BASE/'data')): sha(p) for p in sorted((BASE/'data').rglob('*')) if p.is_file()}
    assert actual == protocol['filtered_files'], 'Data inventory changed'
    counts = {}
    noninteger = 0
    for p in sorted((BASE/'data').glob('*/pairs/od.csv')):
        total = 0
        for frame in pd.read_csv(p, chunksize=200000):
            assert (frame.o_idx != frame.d_idx).all(), p
            assert np.isfinite(frame.trip_count).all() and (frame.trip_count > 0).all(), p
            noninteger += int((frame.trip_count != np.floor(frame.trip_count)).sum())
            total += len(frame)
        counts[p.parents[1].name] = total
    assert len(counts) == 50 and sum(counts.values()) == 6065339
    g = read('5fold_results.json')
    m = read('mlp_backbone_results.json')
    grav = read('gravity_backbone_results.json')
    read('backbone_comparison.json')
    for data in [g,m]:
        rows=data['city_level_results']
        assert len(rows)==len({r['city'] for r in rows})==50
        assert {r['city']: r['n_inter_pairs'] for r in rows}==counts
        assert all(r['n_pairs']==r['n_inter_pairs'] for r in rows)
    grav_rows = grav['city_level_results']
    assert len(grav_rows) == len({r['city'] for r in grav_rows}) == 50
    assert {r['city']: r['n_pairs'] for r in grav_rows} == counts
    assert grav['training_support'] == 'positive_interzonal' and grav['K'] == 8
    assert grav['folds'] == [1, 2, 3, 4, 5]
    assert {r['city']: r['fold'] for r in grav_rows} == {r['city']: r['fold'] for r in g['city_level_results']}
    comparison = json.loads((ART/'backbone_comparison.json').read_text())['summaries']['gravity']
    gravity_delta = np.array([r['M1_city_oracle_obs']['cpc_inter'] - r['M0']['cpc_inter'] for r in grav_rows])
    np.testing.assert_allclose(gravity_delta, [r['delta_city'] for r in grav_rows], atol=1e-15)
    assert abs(gravity_delta.mean() - comparison['mean_delta_cpc']) < 1e-15
    assert int((gravity_delta > 0).sum()) == comparison['cities_improved']
    assert abs(wilcoxon(gravity_delta, alternative='two-sided').pvalue - comparison['wilcoxon_two_sided_p']) < 1e-15

    target={r['city']:r['delta_city'] for r in g['city_level_results']}
    intra=read('intra_bin_mechanism_diagnostic.json')
    assert len(intra['per_city'])==50 and len(intra['per_seed'])==150
    for r in intra['per_city']:
        assert abs(r['delta_cpc']-target[r['city']])<1e-12
    bins={r['city']:sum(x['n_pairs']>0 for x in r['bins']) for r in intra['per_seed']}
    placebo=read('unified_placebo/bootstrap_fold_stratified/unified_placebo_summary.json')
    read('unified_placebo/bootstrap_fold_stratified/bootstrap_method.json')
    path=ART/'unified_placebo/unified_placebo_per_city.csv'
    sources[str(path.relative_to(ROOT))]=sha(path)
    pc=pd.read_csv(path)
    assert len(pc)==pc.city.nunique()==50
    assert max(abs(r.d_cpc_target-target[r.city]) for r in pc.itertuples())<1e-12
    kr=read('k_sensitivity/k_sensitivity_raw.json')
    k=read('k_sensitivity/k_sensitivity_summary.json')
    kd=pd.DataFrame(kr)
    for kval, group in kd.groupby('K'):
        assert len(group)==150 and group.city.nunique()==50
    for city, delta in kd[kd.K==8].groupby('city').delta_cpc.mean().items():
        assert abs(delta-target[city])<1e-12
    read('spatial_resolution/spatial_resolution_summary.json')
    spatial=read('spatial_resolution/spatial_resolution_per_city.json')
    spatial_discrepancy=max(abs(r['delta_cpc_city']-target[r['city']]) for r in spatial)
    assert spatial_discrepancy<1e-5
    assert all(r['delta_cpc_resolution']==0 for r in spatial if not r['is_multi_county'])
    read('audit/dpre_mechanism_summary.json')
    read('noise_robustness/noise_summary.json')
    path=ART/'noise_robustness/noise_per_city.csv'
    sources[str(path.relative_to(ROOT))]=sha(path)
    noise=pd.read_csv(path)
    noise_discrepancy=max(abs(r.delta_cpc_mean-target[r.target_city]) for r in noise[noise.epsilon==0].itertuples())
    assert noise_discrepancy<1e-5
    eps=[];ps=[]
    for e,group in noise.groupby('epsilon'):
        assert len(group)==group.target_city.nunique()==50
        if e>0:
            eps.append(float(e));ps.append(float(wilcoxon(group.delta_cpc_mean,alternative='greater').pvalue))
    ps=np.array(ps);order=np.argsort(ps);adj=np.empty(len(ps));adj[order]=np.minimum(1,np.maximum.accumulate(ps[order]*np.arange(len(ps),0,-1)))
    report={'data_inventory_verified':True,'cities':50,'gravity_support_and_statistics_verified':True,'interzonal_od_pairs':sum(counts.values()),
            'intrazonal_od_pairs':0,'nonpositive_od_flows':0,'noninteger_positive_flows':noninteger,
            'active_bins_city_counts':{str(k):list(bins.values()).count(k) for k in sorted(set(bins.values()))},
            'main_k8_placebo_mechanism_agreement_tolerance':1e-12,
            'max_abs_noise_zero_vs_main':noise_discrepancy,
            'max_abs_spatial_city_vs_main':spatial_discrepancy,
            'noise_paper_test':{'alternative':'greater','adjustment':'Holm across five positive epsilon levels',
                                'values':[{'epsilon':e,'p_raw':float(p),'p_holm':float(a)} for e,p,a in zip(eps,ps,adj)]},
            'sources_sha256':sources,
            'limitations':['Checkpoint training provenance is incomplete. Data checks do not prove historical training scope.',
                           'Direct-OD and Partial-OD historical results are not included.',
                           'Noise source summary uses two-sided benefit tests. Manuscript retains its stated one-sided protocol using p-values recomputed here from the same city-level observations.',
                           'Noise and spatial re-evaluation differ from the main result by less than 1e-5 per city. Sources remain separate.']}

    (ROOT/'paper/interzonal_update_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['sources_sha256','limitations']},indent=2))


if __name__=='__main__':
    main()
