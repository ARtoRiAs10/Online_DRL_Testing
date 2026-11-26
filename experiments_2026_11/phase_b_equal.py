"""Phase B4: equal-cost comparison (Bmax = randomized realized cost)."""
from common import *
rows=[]
print("equal-cost synthetic")
for s in range(5):
    cat,tr=synthetic_world(N=40,T=8000,seed=s,imbalance=2.0,cap_mult=4)
    r=rand_drl(cat,tr,12,np.random.default_rng(s),K=4,repair_mode='stateless',gate_downloads=True,download_time_mode='ema')
    d=online_drl(cat,tr,12,theta=1.0,bmax=float(r.cost))
    rows.append(dict(family='synthetic',regime='raw',seed=s,rand_lat=float(r.latency),rand_cost=float(r.cost),det_lat=float(d.latency),det_cost=float(d.cost)))
    print('syn',s,'done',flush=True)
for regime,kw in (('raw',dict(D=1000.0,disk_scale=1.0)),('crossing',dict(D=1000.0,disk_scale=1e7))):
    cat,tr=load_google('google.csv',max_rows=10000,**kw); C=min(12,max(1,cat.N//2))
    for s in range(3):
        r=rand_drl(cat,tr,C,np.random.default_rng(s),K=4,repair_mode='stateless',gate_downloads=True,download_time_mode='ema')
        d=online_drl(cat,tr,C,theta=1.0,bmax=float(r.cost))
        rows.append(dict(family='google10k',regime=regime,seed=s,rand_lat=float(r.latency),rand_cost=float(r.cost),det_lat=float(d.latency),det_cost=float(d.cost)))
        print('g10k',regime,s,'done',flush=True)
summary={}
for fam in ('synthetic','google10k'):
    for regime in sorted(set(r['regime'] for r in rows if r['family']==fam)):
        fr=[r for r in rows if r['family']==fam and r['regime']==regime]
        if fr:
            summary[f'{fam}/{regime}']=dict(paired_lat=paired_ci([r['rand_lat'] for r in fr],[r['det_lat'] for r in fr]),paired_cost=paired_ci([r['rand_cost'] for r in fr],[r['det_cost'] for r in fr]),rand_cost=ms([r['rand_cost'] for r in fr]),det_cost=ms([r['det_cost'] for r in fr]))
save('phase_b_equal',dict(rows=rows,summary=summary))
for k,s in summary.items(): print(k,s)
