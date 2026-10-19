import argparse, csv, json, os, platform, random, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .common import ROOT, CLASSES, readcsv, savejson, metrics, seed_all
from .data import ManifestDataset
from .models import build_model, freeze_backbone, unfreeze


def run_epoch(model, loader, loss_fn, optimizer, device, scaler=None):
    training = optimizer is not None
    model.train(training)
    if training and getattr(model, "_head_only", False):
        # Keep running normalization statistics stable during linear probing.
        model.eval()
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)) and any(p.requires_grad for p in m.parameters()):
                m.train()
    total_loss = 0.; ys=[]; ps=[]
    for x, y, _ in loader:
        x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", enabled=(scaler is not None)):
                logits=model(x); loss=loss_fn(logits,y)
            if training:
                if scaler:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
        total_loss += loss.item()*len(y); ys.extend(y.cpu().tolist()); ps.extend(logits.argmax(1).cpu().tolist())
    return total_loss/len(loader.dataset), metrics(ys,ps)


def train_one(name, cfg, device, counts, class_weights):
    out=ROOT/"outputs/experiments"/name;out.mkdir(parents=True,exist_ok=True)
    seed_all(cfg["seed"])
    train_ds=ManifestDataset("train",True,cfg["image_size"]); val_ds=ManifestDataset("validation",False,cfg["image_size"])
    # Small batches fit the RTX 2070 SUPER; deterministic order and worker seeds are fixed.
    generator=torch.Generator().manual_seed(cfg["seed"])
    def worker_init(worker_id):
        s=cfg["seed"]+worker_id; random.seed(s);np.random.seed(s);torch.manual_seed(s)
    train_dl=DataLoader(train_ds,batch_size=cfg["batch_size"],shuffle=True,num_workers=cfg["num_workers"],pin_memory=device.type=="cuda",worker_init_fn=worker_init,generator=generator)
    val_dl=DataLoader(val_ds,batch_size=cfg["batch_size"],shuffle=False,num_workers=cfg["num_workers"],pin_memory=device.type=="cuda")
    model,weights=build_model(name,pretrained=True)
    model=model.to(device)
    head_only=name!="custom_cnn"
    if head_only: freeze_backbone(model,name)
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(class_weights,dtype=torch.float32,device=device))
    history=[];best=-1.;bad=0;start=time.time();epochs=0
    scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda" and cfg["use_amp"]))
    phases=[("head",cfg["head_epochs"],cfg["head_lr"])] if head_only else []
    phases.append(("finetune",cfg["finetune_epochs"],cfg["cnn_lr"] if name=="custom_cnn" else cfg["backbone_lr"]))
    for phase,phase_epochs,lr in phases:
        if phase=="finetune" and head_only: unfreeze(model);model._head_only=False
        elif phase=="head": model._head_only=True
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=lr,weight_decay=cfg["weight_decay"])
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=.5,patience=2,min_lr=1e-7)
        for ep in range(phase_epochs):
            tl,_=run_epoch(model,train_dl,loss_fn,optimizer,device,scaler if scaler.is_enabled() else None)
            vl,vm=run_epoch(model,val_dl,loss_fn,None,device)
            epochs+=1;scheduler.step(vm["macro_f1"])
            row={"epoch":epochs,"phase":phase,"train_loss":tl,"validation_loss":vl,"validation_macro_f1":vm["macro_f1"],"validation_accuracy":vm["accuracy"],"learning_rate":optimizer.param_groups[0]["lr"]}
            history.append(row);print(name,row,flush=True)
            if vm["macro_f1"]>best:
                best=vm["macro_f1"];bad=0
                torch.save({"model":model.state_dict(),"model_name":name,"classes":CLASSES,"epoch":epochs,"validation_macro_f1":best},out/"best.pt")
            else: bad+=1
            if phase=="finetune" and bad>=cfg["patience"]: break
        if phase=="finetune" and bad>=cfg["patience"]: break
    with (out/"history.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]),lineterminator='\n');w.writeheader();w.writerows(history)
    props=torch.cuda.get_device_properties(0) if device.type=="cuda" else None
    runinfo={"model":name,"pretrained_weights":weights,"pretrained_weights_source":("https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth" if name=="efficientnet_v2_s" else "https://download.pytorch.org/models/vit_b_16-c867db91.pth" if name=="vit_b_16" else None),"device":str(device),"python":platform.python_version(),"torch":torch.__version__,"torchvision":__import__("torchvision").__version__,"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),"gpu":torch.cuda.get_device_name(0) if props else None,"gpu_memory_bytes":props.total_memory if props else None,"batch_size":cfg["batch_size"],"train_counts":counts,"class_weights":class_weights,"optimizer":"AdamW","scheduler":"ReduceLROnPlateau(mode=max, factor=0.5, patience=2, min_lr=1e-7)","loss":"weighted CrossEntropyLoss; weights from training partition only","input_size":cfg["image_size"],"preprocessing":"RGB; resize shorter edge to 256; center crop 224 at validation/test; ImageNet mean/std normalization","training_augmentation":"RandomResizedCrop scale 0.82-1.0; horizontal and vertical flips; random rotation +/-10 degrees; training only","seed":cfg["seed"],"deterministic_algorithms":True,"config":cfg,"epochs_run":epochs,"best_validation_macro_f1":best,"runtime_seconds":time.time()-start}
    savejson(out/"run.json",runinfo)
    return runinfo


def evaluate_once(name,cfg,device):
    out=ROOT/"outputs/experiments"/name
    model,_=build_model(name,pretrained=False)
    model=model.to(device)
    ck=torch.load(out/"best.pt",map_location=device,weights_only=False);model.load_state_dict(ck["model"])
    ds=ManifestDataset("test",False,cfg["image_size"]);dl=DataLoader(ds,batch_size=cfg["batch_size"],shuffle=False,num_workers=cfg["num_workers"],pin_memory=device.type=="cuda")
    model.eval();ys=[];ps=[];saved=[]
    with torch.inference_mode():
        for x,y,ids in dl:
            logits=model(x.to(device));pred=logits.argmax(1).cpu().tolist();ys+=y.tolist();ps+=pred
            saved.extend({"image_id":i,"true":CLASSES[t],"predicted":CLASSES[p],"correct":t==p} for i,t,p in zip(ids,y.tolist(),pred))
    result=metrics(ys,ps);result.update({"model":name,"checkpoint_epoch":ck["epoch"],"checkpoint_validation_macro_f1":ck["validation_macro_f1"]})
    savejson(out/"test_metrics.json",result)
    with (out/"test_predictions.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(saved[0]),lineterminator='\n');w.writeheader();w.writerows(saved)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--model",choices=["custom_cnn","efficientnet_v2_s","vit_b_16"]);p.add_argument("--evaluate-only",action="store_true");a=p.parse_args()
    cfg=json.loads((ROOT/"configs/experiment.json").read_text());device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifests={s:readcsv(ROOT/f"data/manifests/candidate_a/{s}.csv") for s in ("train","validation","test")}
    counts=[sum(int(r["label"])==i for r in manifests["train"]) for i in range(4)];n=len(manifests["train"])
    weights=[n/(4*c) for c in counts]
    if not all(counts):raise RuntimeError("Empty training class")
    names=[a.model] if a.model else cfg["models"]
    print(json.dumps({"device":str(device),"train_counts":counts,"class_weights":weights},indent=2),flush=True)
    if not a.evaluate_only:
        for name in names:train_one(name,cfg,device,counts,weights)
    for name in names:
        result=evaluate_once(name,cfg,device);print("FINAL TEST",json.dumps(result),flush=True)

if __name__=="__main__":main()
