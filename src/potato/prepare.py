"""Reproduce exclusions, inferred related-capture groups and frozen partitions."""
import argparse,collections,datetime as dt,hashlib,io,json,zipfile
import numpy as np
from PIL import Image
from .common import ROOT,CLASSES,readcsv,writecsv,savejson,sha

class Union:
    def __init__(self,n):self.p=list(range(n))
    def find(self,a):
        while self.p[a]!=a:self.p[a]=self.p[self.p[a]];a=self.p[a]
        return a
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[max(a,b)]=min(a,b)
def prepare():
    out=ROOT/'data/manifests/candidate_a';out.mkdir(parents=True,exist_ok=True)
    if (out/'freeze.json').exists():
        verify();materialize_images();print('Existing frozen split verified; source images materialised without changing membership.');return
    archive=ROOT/'data/raw/mendeley_rgn2d2jb7f_v1/potato_desease_dataset.zip'
    assert sha(archive)=='0d6ff2f249993a404d3b533b0247063e0f442e0c6267c24b23d7463645492662'
    allrows=readcsv(ROOT/'outputs/audit/image_manifest.csv');rows=[r for r in allrows if r['primary_candidate']=='True'];assert len(rows)==6337
    index={r['archive_member']:i for i,r in enumerate(rows)};uf=Union(len(rows));edges=[];quarantine={};arrays={}
    conflicts=['IMG20251108123556','IMG20251108124413']
    for i,r in enumerate(rows):
        if any(p in r['filename'] for p in conflicts):quarantine[i]='Previously visually verified contradictory burst sequence'
    # Fixed anchor windows avoid chaining an entire continuous session. All labels share the rule.
    sessions=collections.defaultdict(list)
    for i,r in enumerate(rows):
        t=dt.datetime.fromisoformat(r['filename_capture_time']);sessions[(r['variety'],t.date(),r['filename'].startswith('IMG_'))].append((t,i))
    for values in sessions.values():
        values.sort();anchor=None;first=None
        for t,i in values:
            if anchor is None or (t-anchor).total_seconds()>60:anchor=t;first=i
            elif first!=i:
                uf.join(first,i);edges.append(dict(a=rows[first]['archive_member'],b=rows[i]['archive_member'],reason='same variety/camera filename pattern; fixed-anchor <=60-second capture window',phash='',dhash='',rgb_mae='',centre_mae=''))
    with zipfile.ZipFile(archive) as z:
        def pixels(i):
            if i not in arrays:
                with Image.open(io.BytesIO(z.read(rows[i]['archive_member']))) as im:arrays[i]=np.asarray(im.resize((64,64)),dtype=np.float32)/255
            return arrays[i]
        pair_review=[]
        for pair in readcsv(ROOT/'outputs/audit/near_duplicate_candidates.csv'):
            if pair['image_a'] not in index or pair['image_b'] not in index:continue
            a,b=index[pair['image_a']],index[pair['image_b']];ra,rb=rows[a],rows[b];ph=int(pair['phash_hamming']);dh=int(pair['dhash_hamming'])
            delta=abs((dt.datetime.fromisoformat(ra['filename_capture_time'])-dt.datetime.fromisoformat(rb['filename_capture_time'])).total_seconds())
            diff=np.abs(pixels(a)-pixels(b));mae=float(diff.mean());centre=float(diff[12:52,12:52].mean())
            temporal=ra['variety']==rb['variety'] and delta<=120
            visual=ph<=4 and dh<=8 and mae<=.04 and centre<=.06
            accepted=temporal or visual
            reason='perceptual candidate corroborated by <=120-second same-variety timing' if temporal else 'strict perceptual and full/centre RGB agreement' if visual else 'uncorroborated perceptual candidate; not assumed related'
            record=dict(a=ra['archive_member'],b=rb['archive_member'],reason=reason,phash=ph,dhash=dh,rgb_mae=round(mae,6),centre_mae=round(centre,6))
            pair_review.append(dict(**record,accepted=accepted))
            if accepted:uf.join(a,b);edges.append(record)
            if ph<=2 and dh<=4 and mae<=.025 and centre<=.035 and ra['proposed_class']!=rb['proposed_class']:
                quarantine[a]='Highly similar same-view candidate with contradictory severity label';quarantine[b]=quarantine[a]
        # Quarantine complete inferred components linked to a contradictory image; never relabel.
        bad_roots={uf.find(i) for i in quarantine};groups=collections.defaultdict(list)
        for i in range(len(rows)):groups[uf.find(i)].append(i)
        exclusions=[];usable=[]
        for root,indices in sorted(groups.items()):
            gid='inferred_'+hashlib.sha256('\n'.join(sorted(rows[i]['archive_member'] for i in indices)).encode()).hexdigest()[:16]
            for i in indices:
                r=rows[i];rec=dict(image_id=r['sha256'],archive_member=r['archive_member'],source_category=r['source_category'],variety=r['variety'],class_name=r['proposed_class'],label=CLASSES.index(r['proposed_class']),filename_capture_time=r['filename_capture_time'],sha256=r['sha256'],pixel_sha256=r['pixel_sha256'],inferred_group=gid,group_kind='inferred related captures; NOT verified physical-tuber ID',path='data/processed/candidate_a/images/'+r['sha256']+'.jpg')
                if root in bad_roots:exclusions.append(dict(**rec,reason=quarantine.get(i,'Same inferred component as a quarantined contradictory capture')))
                else:
                    dest=ROOT/rec['path'];dest.parent.mkdir(parents=True,exist_ok=True);raw=z.read(r['archive_member']);assert hashlib.sha256(raw).hexdigest()==r['sha256']
                    if not dest.exists():dest.write_bytes(raw)
                    usable.append(rec)
    writecsv(out/'exclusions.csv',exclusions);writecsv(out/'group_edges.csv',edges);writecsv(out/'perceptual_candidate_review.csv',pair_review)
    # Deterministic repeated grouped allocations; objective uses counts only, never images/scores.
    grouped=collections.defaultdict(list)
    for r in usable:grouped[r['inferred_group']].append(r)
    gids=sorted(grouped);counts=np.asarray([[sum(r['label']==c for r in grouped[g]) for c in range(4)] for g in gids]);total=counts.sum(0);ratios=np.array([.7,.15,.15]);target=ratios[:,None]*total
    rng=np.random.default_rng(42);best=None;bestloss=float('inf')
    for _ in range(1000):
        assignments=rng.choice(3,len(gids),p=ratios);actual=np.array([counts[assignments==s].sum(0) for s in range(3)])
        if (actual==0).any():continue
        loss=float(np.mean(((actual-target)/np.maximum(target,1))**2))
        if loss<bestloss:bestloss=loss;best=assignments.copy()
    assert best is not None
    allocated={g:['train','validation','test'][int(s)] for g,s in zip(gids,best)}
    for r in usable:r['split']=allocated[r['inferred_group']]
    usable.sort(key=lambda r:r['image_id']);writecsv(out/'all.csv',usable)
    for split in ['train','validation','test']:writecsv(out/(split+'.csv'),[r for r in usable if r['split']==split])
    stats=dict(original_primary_count=6337,excluded_primary_images=len(exclusions),usable_images=len(usable),inferred_groups=len(grouped),largest_inferred_group=max(map(len,grouped.values())),verified_physical_tuber_count='UNKNOWN',seed=42,grouping=dict(anchor_window_seconds=60,perceptual_time_seconds=120,strict_phash_max=4,strict_dhash_max=8,strict_rgb_mae_max=.04,strict_centre_mae_max=.06),split_method='1000 seed-42 grouped candidate allocations; minimum squared relative deviation from per-class 70/15/15 targets; no model information',class_counts={s:{c:sum(r['split']==s and r['class_name']==c for r in usable) for c in CLASSES} for s in ['train','validation','test']},training_class_weights=(total.sum()/(4*np.asarray([sum(r['split']=='train' and r['label']==c for r in usable) for c in range(4)]))).tolist())
    # Correct conventional weights N_train/(C*n_train), not N_total.
    nt=sum(r['split']=='train' for r in usable);stats['training_class_weights']=[nt/(4*sum(r['split']=='train' and r['label']==c for r in usable)) for c in range(4)]
    stats['uncorroborated_perceptual_candidates_cross_split']=sum(not p['accepted'] and allocated.get(rows[index[p['a']]]['sha256'],'')=='unused' for p in []) # populated explicitly below
    membership={r['archive_member']:r['split'] for r in usable}
    stats['uncorroborated_perceptual_candidates_cross_split']=sum(not p['accepted'] and p['a'] in membership and p['b'] in membership and membership[p['a']]!=membership[p['b']] for p in pair_review)
    savejson(out/'preparation_summary.json',stats)
    names=['all.csv','train.csv','validation.csv','test.csv','exclusions.csv','group_edges.csv','perceptual_candidate_review.csv','preparation_summary.json']
    savejson(out/'freeze.json',dict(seed=42,classes=CLASSES,source_archive_sha256=sha(archive),files={n:sha(out/n) for n in names},group_identity='inferred; not verified physical tuber identity'))
    verify();print(json.dumps(stats,indent=2))

def materialize_images():
    """Restore ignored image files from the verified archive using frozen membership."""
    archive=ROOT/'data/raw/mendeley_rgn2d2jb7f_v1/potato_desease_dataset.zip'
    assert sha(archive)=='0d6ff2f249993a404d3b533b0247063e0f442e0c6267c24b23d7463645492662'
    rows=readcsv(ROOT/'data/manifests/candidate_a/all.csv');missing=0
    with zipfile.ZipFile(archive) as z:
        for r in rows:
            dest=ROOT/r['path']
            if dest.exists():
                if sha(dest)!=r['sha256']:raise ValueError(f'Existing image hash mismatch: {dest}')
                continue
            raw=z.read(r['archive_member'])
            if hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ValueError(f'Source image hash mismatch: {r["archive_member"]}')
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);missing+=1
    print(f'Materialised {missing} missing source images from the verified archive.',flush=True)
def verify():
    out=ROOT/'data/manifests/candidate_a';freeze=json.loads((out/'freeze.json').read_text())
    for name,h in freeze['files'].items():assert sha(out/name)==h, name+' changed after freeze'
    rows=readcsv(out/'all.csv');assert len({r['image_id'] for r in rows})==len(rows)
    for key in ['inferred_group','sha256','pixel_sha256']:
        d=collections.defaultdict(set)
        for r in rows:d[r[key]].add(r['split'])
        assert all(len(v)==1 for v in d.values()),key+' crosses partitions'
    for split in ['train','validation','test']:
        part=readcsv(out/(split+'.csv'));assert {r['image_id'] for r in part}=={r['image_id'] for r in rows if r['split']==split}
        assert {int(r['label']) for r in part}==set(range(4))
    return freeze
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--materialize-only',action='store_true');a=p.parse_args()
    if a.verify:verify()
    elif a.materialize_only:verify();materialize_images()
    else:prepare()
