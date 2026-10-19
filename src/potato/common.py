"""Shared manifest, metric and reproducibility utilities."""
import csv,hashlib,json,os,random
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CLASSES=['Healthy','DryRot-Mild','DryRot-Moderate','DryRot-Severe']
def readcsv(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows,fields=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def savejson(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def metrics(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=int);cm=np.zeros((4,4),dtype=int)
    np.add.at(cm,(y,p),1);tp=np.diag(cm);support=cm.sum(1);pred=cm.sum(0)
    precision=np.divide(tp,pred,out=np.zeros(4),where=pred!=0);recall=np.divide(tp,support,out=np.zeros(4),where=support!=0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(4),where=precision+recall!=0)
    return dict(accuracy=float(tp.sum()/max(1,cm.sum())),precision_macro=float(precision.mean()),recall_macro=float(recall.mean()),macro_f1=float(f1.mean()),weighted_f1=float((f1*support).sum()/max(1,support.sum())),confusion_matrix=cm.tolist(),per_class=[dict(class_name=CLASSES[i],precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i]),support=int(support[i])) for i in range(4)])
def seed_all(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
