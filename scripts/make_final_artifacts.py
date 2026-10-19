"""Create plots, example panels, and factual completion summary from saved run artifacts."""
import csv, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/experiments'
CLASSES=['Healthy','DryRot-Mild','DryRot-Moderate','DryRot-Severe']
COLORS={'custom_cnn':(50,110,180),'efficientnet_v2_s':(40,150,100),'vit_b_16':(200,105,45)}
NAMES={'custom_cnn':'Custom CNN','efficientnet_v2_s':'EfficientNetV2-S','vit_b_16':'ViT-B/16'}

def font(size=16):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']:
        if Path(p).exists(): return ImageFont.truetype(p,size)
    return ImageFont.load_default()

def readcsv(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))
def writecsv(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def linechart(path, title, series, labels, y0=None, y1=None):
    W,H=1000,570; im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im);f=font(16);fb=font(23)
    left,top,right,bottom=82,70,960,485
    vals=[v for s in series for v in s if v is not None]
    lo=y0 if y0 is not None else min(vals);hi=y1 if y1 is not None else max(vals)
    if hi==lo:hi=lo+1
    d.text((left,20),title,fill='black',font=fb)
    for k in range(6):
        y=bottom-(bottom-top)*k/5;v=lo+(hi-lo)*k/5
        d.line((left,y,right,y),fill=(225,225,225));d.text((12,y-8),f'{v:.2f}',fill='black',font=f)
    maxn=max(len(s) for s in series)
    for i,(arr,label) in enumerate(zip(series,labels)):
        pts=[]
        for j,v in enumerate(arr):
            x=left+(right-left)*j/max(1,maxn-1);y=bottom-(v-lo)/(hi-lo)*(bottom-top);pts.append((x,y))
        if len(pts)>1:d.line(pts,fill=COLORS.get(label,(0,0,0)),width=3)
        for x,y in pts:d.ellipse((x-3,y-3,x+3,y+3),fill=COLORS.get(label,(0,0,0)))
        lx=left+i*260;d.line((lx,530,lx+26,530),fill=COLORS.get(label,(0,0,0)),width=4);d.text((lx+32,520),label,fill='black',font=f)
    d.text((left,492),'Epoch',fill='black',font=f)
    im.save(path)

def matrix_image(path, matrix, normalized=False):
    n=len(CLASSES);cell=145;left=205;top=65;im=Image.new('RGB',(left+n*cell+20,top+n*cell+100),'white');d=ImageDraw.Draw(im);f=font(16);fb=font(21)
    d.text((left,18),'Normalised confusion matrix' if normalized else 'Confusion matrix (counts)',fill='black',font=fb)
    totals=[sum(row) for row in matrix]
    for i in range(n):
        d.text((8,top+i*cell+58),CLASSES[i],fill='black',font=f)
        for j in range(n):
            v=matrix[i][j]/totals[i] if normalized and totals[i] else matrix[i][j]
            shade=int(248-190*v);x=left+j*cell;y=top+i*cell
            d.rectangle((x,y,x+cell,y+cell),fill=(shade,shade,255),outline='white',width=3)
            d.text((x+cell//2-20,y+cell//2-9),f'{v:.2f}' if normalized else str(matrix[i][j]),fill='black',font=fb)
            if i==0:d.text((x+3,top-28),CLASSES[j],fill='black',font=f)
    d.text((left,top+n*cell+12),'Rows: true class     Columns: predicted class',fill='black',font=f)
    im.save(path)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Validation curves (no test predictions used to select checkpoints).
    histories={k:readcsv(OUT/k/'history.csv') for k in NAMES}
    for key,title in [('train_loss','Training loss'),('validation_loss','Validation loss'),('validation_macro_f1','Validation Macro-F1'),('validation_accuracy','Validation accuracy')]:
        linechart(OUT/f'{key}.png',title,[[float(r[key]) for r in histories[m]] for m in NAMES],list(NAMES))
    # Test comparison and confusion matrices from the three one-time evaluations.
    comparisons=[]
    for model in NAMES:
        result=json.loads((OUT/model/'test_metrics.json').read_text())
        run=json.loads((OUT/model/'run.json').read_text())
        comparisons.append({'model':NAMES[model],'epochs':run['epochs_run'],'best_checkpoint_epoch':result['checkpoint_epoch'],'validation_macro_f1':result['checkpoint_validation_macro_f1'],'test_accuracy':result['accuracy'],'test_macro_precision':result['precision_macro'],'test_macro_recall':result['recall_macro'],'test_macro_f1':result['macro_f1'],'test_weighted_f1':result['weighted_f1']})
        matrix_image(OUT/f'{model}_confusion_matrix.png',result['confusion_matrix'])
        matrix_image(OUT/f'{model}_confusion_matrix_normalized.png',result['confusion_matrix'],True)
    writecsv(OUT/'three_model_comparison.csv',comparisons)
    # Choose a small panel of correct and incorrect examples per model, ordered by manifest rows.
    allrows={r['image_id']:r for r in readcsv(ROOT/'data/manifests/candidate_a/all.csv')}
    for model in NAMES:
        preds=readcsv(OUT/model/'test_predictions.csv')
        for correct,slug in [(True,'correct'),(False,'incorrect')]:
            chosen=[r for r in preds if (r['correct']=='True')==correct][:8]
            tilew,tileh=250,260;im=Image.new('RGB',(tilew*4,tileh*2),'white');d=ImageDraw.Draw(im);f=font(14);fb=font(18)
            d.text((12,8),f'{NAMES[model]}: {slug} test examples',fill='black',font=fb)
            for idx,r in enumerate(chosen):
                rr=allrows[r['image_id']];src=ROOT/rr['path']
                with Image.open(src) as srcim: thumb=srcim.convert('RGB');thumb.thumbnail((220,185))
                x=(idx%4)*tilew+15;y=(idx//4)*tileh+40;im.paste(thumb,(x+(220-thumb.width)//2,y))
                d.text((x,y+188),f"True: {r['true']}",fill='black',font=f)
                d.text((x,y+208),f"Pred: {r['predicted']}",fill='black',font=f)
                d.text((x,y+228),rr['archive_member'].split('/')[-1][:28],fill=(80,80,80),font=f)
            im.save(OUT/f'{model}_{slug}_examples.png')
    summary=json.loads((ROOT/'data/manifests/candidate_a/preparation_summary.json').read_text())
    excluded=readcsv(ROOT/'data/manifests/candidate_a/exclusions.csv')
    source_counts={c:sum(r['primary_candidate']=='True' and r['proposed_class']==c for r in readcsv(ROOT/'outputs/audit/image_manifest.csv')) for c in CLASSES}
    kept={c:sum(r['class_name']==c for r in readcsv(ROOT/'data/manifests/candidate_a/all.csv')) for c in CLASSES}
    split_counts={s:{c:sum(r['class_name']==c for r in readcsv(ROOT/f'data/manifests/candidate_a/{s}.csv')) for c in CLASSES} for s in ['train','validation','test']}
    report=['# Candidate A experiment completion report','',
    '## Task and data','',
    'Image-only four-class classification of the publisher-labelled appearance/severity categories: Healthy, DryRot-Mild, DryRot-Moderate and DryRot-Severe. Source: Mendeley Data, DOI 10.17632/rgn2d2jb7f.1, CC BY 4.0. Images are 512 x 512 JPEGs; model input is 224 x 224. The dataset labels are visual grades, not laboratory-confirmed diagnoses.','',
    f"The requested variety subset contained {summary['original_primary_count']} images. {summary['excluded_primary_images']} images were quarantined during the pre-split leakage/conflict pass, leaving {summary['usable_images']} images. Independent physical tuber count remains UNKNOWN.",'',
    '| Class | Before leakage-control pass | Usable after exclusions | Train | Validation | Test |','|---|---:|---:|---:|---:|---:|']
    for c in CLASSES:report.append(f"| {c} | {source_counts[c]} | {kept[c]} | {split_counts['train'][c]} | {split_counts['validation'][c]} | {split_counts['test'][c]} |")
    report += ['',f"Split totals: {len(readcsv(ROOT/'data/manifests/candidate_a/train.csv'))} train, {len(readcsv(ROOT/'data/manifests/candidate_a/validation.csv'))} validation, {len(readcsv(ROOT/'data/manifests/candidate_a/test.csv'))} test. Seed 42. Membership and file hashes are frozen in `data/manifests/candidate_a/freeze.json`.",'',
    '## Leakage controls and limits','',
    'Images were grouped using capture-name/time windows and conservative perceptual similarity corroborated by timestamp or decoded-pixel similarity. Two visually reviewed contradictory burst sequences and highly similar conflicting-label captures were quarantined with their inferred components. Group allocations were selected by a seed-42 search over 1,000 grouped 70/15/15 candidate splits, minimising relative class-count deviation. The largest inferred group contains '+str(summary['largest_inferred_group'])+' images. These are inferred related-capture groups, not verified potato identities. Physical tuber IDs and complete linkage remain UNKNOWN. The audit also records '+str(summary['uncorroborated_perceptual_candidates_cross_split'])+' weaker perceptual candidate pairs across partitions; they were not merged because the evidence was insufficient and would risk over-grouping. Therefore results must not be described as confirmed tuber-independent performance.','',
    '## Training and reproducibility','',
    'All models used the same frozen split, class mapping, RGB conversion, resize to 256 then 224 crop, ImageNet normalization, training-only random resized crop (scale 0.82-1.0), horizontal/vertical flips and random rotation up to 10 degrees. The CNN was trained from scratch; EfficientNetV2-S and ViT-B/16 used torchvision ImageNet1K_V1 weights, first training the classification head for 3 epochs and then fine-tuning the full model. Batch size 8; AdamW; ReduceLROnPlateau (factor 0.5, patience 2); weighted cross-entropy; class weights derived from training counts only; seed 42; early stopping patience 5 on validation Macro-F1; fine-tuning cap 20 epochs. CUDA mixed precision was enabled on the RTX 2070 SUPER (8 GiB).','',
    'Versions recorded in each `run.json`; PyTorch 2.13.0+cu126 and torchvision 0.28.0+cu126. Class weights (Healthy, Mild, Moderate, Severe): '+', '.join(f'{x:.4f}' for x in json.loads((OUT/'custom_cnn'/'run.json').read_text())['class_weights'])+'.','',
    '## Test results','',
    '| Model | Epochs run | Best checkpoint epoch | Test accuracy | Macro precision | Macro recall | Macro-F1 | Weighted F1 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:report.append('| {model} | {epochs} | {best_checkpoint_epoch} | {test_accuracy:.3f} | {test_macro_precision:.3f} | {test_macro_recall:.3f} | {test_macro_f1:.3f} | {test_weighted_f1:.3f} |'.format(**r))
    best=max(comparisons,key=lambda r:r['test_macro_f1'])
    report += ['',f"ViT-B/16 had the highest test Macro-F1 ({best['test_macro_f1']:.3f}) and accuracy ({best['test_accuracy']:.3f}) in this run. This is a single fixed split and one run per model, so the difference is descriptive rather than a statistical claim.",'',
    'Mild dry rot was the hardest class: support was 41 images and recall was low for the custom CNN (0.073), higher for EfficientNetV2-S (0.439), and 0.390 for ViT-B/16. Mild was most often confused with Moderate. Moderate and Severe also overlapped, especially Moderate predicted as Severe. Healthy recall was 0.800, 0.805 and 0.921 respectively. These errors align with class imbalance and the absence of a quantitative, independently adjudicated severity rubric.','',
    '### Per-class test metrics','',
    '| Model | Class | Precision | Recall | F1 | Support |','|---|---|---:|---:|---:|---:|']
    for model in NAMES:
        result=json.loads((OUT/model/'test_metrics.json').read_text())
        for row in result['per_class']:
            report.append(f"| {NAMES[model]} | {row['class_name']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['support']} |")
    report += ['',
    '## Figures and examples','',
    'The outputs folder contains training/validation loss curves, validation Macro-F1 and accuracy curves, raw and row-normalised confusion matrices for each model, and eight correct plus eight incorrect test examples per model. The CSV comparison table, prediction records, checkpoints, run configurations, and histories are saved alongside them.','',
    '## Rubric audit','',
    '| Assignment requirement | Status in this project |','|---|---|',
    '| Clear task and practical value | Implemented: four-class visible tuber appearance/severity classification; describe as image classification support, not diagnosis. |',
    '| Dataset source, access, licence, classes and counts | Documented in this report, README, and source audit; CC BY 4.0 attribution/change disclosure required. |',
    '| Cleaning, labels, duplicates and imbalance | 8 contradictory captures quarantined with reasons; 0 corrupt source images and 0 exact duplicate groups in the source audit; class weighting computed from training only. |',
    '| Three distinct model families and transfer learning | Implemented: custom CNN, EfficientNetV2-S, ViT-B/16; same four classes and same partitions. |',
    '| Leakage-safe split | Best-effort inferred capture grouping is frozen and reproducible. Physical IDs remain UNKNOWN, 608 weak perceptual candidate pairs remain across partitions, and the very large inferred groups distort the target ratios. Full physical-sample independence is not verified; report this as a material limitation. |',
    '| Training information and reproducibility | Config, weights, hyperparameters, software/hardware versions, histories, checkpoints, split manifests, and run times are saved. |',
    '| Evaluation and error interpretation | Implemented: accuracy, macro precision/recall/F1, weighted F1, per-class metrics, raw/normalised confusion matrices, curves, and correct/incorrect examples. |',
    '| Final report PDF (20 marks) | Not yet prepared. Use the completion report and artifacts to write the concise final 10-15 page PDF; include references and candid limitations. |',
    '| Oral presentation slides (10 marks) | Not yet prepared. All group members must present and answer questions. |',
    '| Notebook, source files, README, split files | Notebook with visible outputs, Python scripts, README, and frozen split files are present. |','',
    '## Remaining assignment deliverables','',
    'The final 10-15 page PDF report and oral-presentation slides still need to be prepared and reviewed by the group. The project has an output-visible results notebook, source scripts, a README, and frozen split manifests. All group members must understand the dataset limitations, model choices, errors, and results. Cite the dataset, pretrained models, PyTorch/torchvision and other technical sources in the final report. The dataset licence requires attribution and disclosure of changes.','',
    '## References','',
    'Islam Saif, S., Tasnim Saba, M. F., and Kaiser, M. S. Mendeley Data, Potato disease dataset, version 1. https://doi.org/10.17632/rgn2d2jb7f.1. CC BY 4.0.','',
    'PyTorch/torchvision pretrained model weights: EfficientNetV2-S ImageNet1K_V1 and ViT-B/16 ImageNet1K_V1, downloaded from the official PyTorch model-weight URLs recorded in each run.json.']
    (OUT/'COMPLETION_REPORT.md').write_text('\n'.join(report)+'\n')

if __name__=='__main__':main()
