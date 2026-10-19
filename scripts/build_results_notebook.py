"""Assemble an output-visible notebook from saved run artifacts; never runs test inference."""
import base64,csv,json
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/experiments'
def png_output(p):
    return {'output_type':'display_data','metadata':{},'data':{'image/png':base64.b64encode(p.read_bytes()).decode('ascii'),'text/plain':'<PIL.Image.Image>'}}
def main():
    summary=json.loads((ROOT/'data/manifests/candidate_a/preparation_summary.json').read_text())
    cells=[nbformat.v4.new_markdown_cell('''# Potato Tuber Dry-Rot Appearance and Severity Classification

This notebook presents the completed four-class experiment from saved artifacts. It does not retrain models or rerun test inference. Labels are the dataset's visual grades, not laboratory-confirmed disease diagnoses. The dataset has no verified physical-tuber IDs; results are based on the documented inferred capture groups and retain that limitation.'''),
    nbformat.v4.new_markdown_cell('''## Dataset, classes, and frozen split

Source: Mendeley Data, DOI 10.17632/rgn2d2jb7f.1, CC BY 4.0. The experiment uses Asterix, Diamant, and Lal Pakri. Rumana, Scab, and Fully-Rotten are excluded. Eight samples were quarantined with contradictory inferred capture components before splitting. The remaining 6,329 images use seed 42 and a frozen grouped split. The grouping does not establish independent physical tubers.'''),
    nbformat.v4.new_code_cell("""import csv, json
from pathlib import Path
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'data/manifests/candidate_a').exists()), Path.cwd())
OUT = ROOT / 'outputs/experiments'
summary = json.loads((ROOT/'data/manifests/candidate_a/preparation_summary.json').read_text())
print('Images:', summary['usable_images'], '| inferred groups:', summary['inferred_groups'], '| seed:', summary['seed'])
print('Verified physical tuber count:', summary['verified_physical_tuber_count'])
for split, counts in summary['class_counts'].items():
    print(split, sum(counts.values()), counts)"""),
    nbformat.v4.new_markdown_cell('## Model comparison — held-out test set, evaluated once'),
    nbformat.v4.new_code_cell("""with (OUT/'three_model_comparison.csv').open(newline='') as f:
    rows = list(csv.DictReader(f))
for r in rows:
    print(f\"{r['model']:18} accuracy={float(r['test_accuracy']):.3f} macro-F1={float(r['test_macro_f1']):.3f} weighted-F1={float(r['test_weighted_f1']):.3f}\")"""),
    nbformat.v4.new_markdown_cell('## Per-class test metrics'),
    nbformat.v4.new_code_cell("""from IPython.display import display, HTML
rows = []
for model in ['custom_cnn','efficientnet_v2_s','vit_b_16']:
    result = json.loads((OUT/model/'test_metrics.json').read_text())
    for r in result['per_class']:
        rows.append(f\"<tr><td>{model}</td><td>{r['class_name']}</td><td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td><td>{r['f1']:.3f}</td><td>{r['support']}</td></tr>\")
display(HTML('<table><tr><th>Model</th><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>'+''.join(rows)+'</table>'))"""),
    nbformat.v4.new_markdown_cell('## Learning curves'),
    nbformat.v4.new_code_cell("""from IPython.display import display, Image
for name in ['train_loss.png','validation_loss.png','validation_macro_f1.png','validation_accuracy.png']:
    print(name); display(Image(filename=str(OUT/name)))"""),
    nbformat.v4.new_markdown_cell('## Confusion matrices'),
    nbformat.v4.new_code_cell("""from IPython.display import display, Image
for name in ['custom_cnn_confusion_matrix.png','efficientnet_v2_s_confusion_matrix.png','vit_b_16_confusion_matrix.png',
             'custom_cnn_confusion_matrix_normalized.png','efficientnet_v2_s_confusion_matrix_normalized.png','vit_b_16_confusion_matrix_normalized.png']:
    print(name); display(Image(filename=str(OUT/name)))"""),
    nbformat.v4.new_markdown_cell('## Correct and incorrect ViT test examples'),
    nbformat.v4.new_code_cell("""all_rows = {r['image_id']: r for r in csv.DictReader((ROOT/'data/manifests/candidate_a/all.csv').open(newline=''))}
preds = list(csv.DictReader((OUT/'vit_b_16'/'test_predictions.csv').open(newline='')))
for correct in ['True','False']:
    print('Correct predictions' if correct == 'True' else 'Incorrect predictions')
    for r in [x for x in preds if x['correct'] == correct][:8]:
        print(all_rows[r['image_id']]['archive_member'], '| true:', r['true'], '| predicted:', r['predicted'])"""),
    nbformat.v4.new_markdown_cell('''## Main findings and limitations

ViT-B/16 had the strongest test Macro-F1 in this single run. Mild dry rot was the hardest class and was most often confused with Moderate. Performance is descriptive for this fixed split and one run per model. The source provides no verified potato IDs or quantitative severity rubric, and residual grouping uncertainty remains. Refer to `outputs/experiments/COMPLETION_REPORT.md` for complete metrics, classwise results, all-model matrices, reproducibility settings, and the rubric/deliverables audit.''')]
    # Populate outputs from recorded result files, never by recalculating model predictions.
    cells[2].outputs=[{'output_type':'stream','name':'stdout','text':f"Images: {summary['usable_images']} | inferred groups: {summary['inferred_groups']} | seed: {summary['seed']}\nVerified physical tuber count: UNKNOWN\n"+''.join(f"{s} {sum(c.values())} {c}\n" for s,c in summary['class_counts'].items())}];cells[2].execution_count=1
    table=csv.DictReader((OUT/'three_model_comparison.csv').open())
    rows=list(table)
    text=''.join(f"{r['model']:18} accuracy={float(r['test_accuracy']):.3f} macro-F1={float(r['test_macro_f1']):.3f} weighted-F1={float(r['test_weighted_f1']):.3f}\n" for r in rows)
    cells[4].outputs=[{'output_type':'stream','name':'stdout','text':text}];cells[4].execution_count=2
    html=''.join(f"<tr><td>{model}</td><td>{r['class_name']}</td><td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td><td>{r['f1']:.3f}</td><td>{r['support']}</td></tr>" for model in ['custom_cnn','efficientnet_v2_s','vit_b_16'] for r in json.loads((OUT/model/'test_metrics.json').read_text())['per_class'])
    fullhtml='<table><tr><th>Model</th><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>'+html+'</table>'
    cells[6].outputs=[{'output_type':'display_data','metadata':{},'data':{'text/html':fullhtml,'text/plain':'Per-class precision, recall, F1, and support for all models.'}}];cells[6].execution_count=3
    cells[8].outputs=[{'output_type':'stream','name':'stdout','text':n+'\n'} if i%2==0 else png_output(OUT/n) for i,n in enumerate(['train_loss.png','validation_loss.png','validation_macro_f1.png','validation_accuracy.png'])];cells[8].execution_count=4
    names=[f'{model}_confusion_matrix.png' for model in ['custom_cnn','efficientnet_v2_s','vit_b_16']]+[f'{model}_confusion_matrix_normalized.png' for model in ['custom_cnn','efficientnet_v2_s','vit_b_16']]
    cells[10].outputs=[]
    for n in names:cells[10].outputs.extend([{'output_type':'stream','name':'stdout','text':n+'\n'},png_output(OUT/n)])
    cells[10].execution_count=5
    allrows={r['image_id']:r for r in csv.DictReader((ROOT/'data/manifests/candidate_a/all.csv').open(newline=''))}
    preds=list(csv.DictReader((OUT/'vit_b_16'/'test_predictions.csv').open(newline='')))
    example_text=''
    for correct in ['True','False']:
        example_text+=('Correct predictions\n' if correct=='True' else 'Incorrect predictions\n')
        for r in [x for x in preds if x['correct']==correct][:8]:
            example_text+=f"{allrows[r['image_id']]['archive_member']} | true: {r['true']} | predicted: {r['predicted']}\n"
    cells[12].outputs=[{'output_type':'stream','name':'stdout','text':example_text}];cells[12].execution_count=6
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    nb=nbformat.from_dict(nb)
    nbformat.validate(nb)
    path=ROOT/'notebooks/experiment_results.ipynb';path.parent.mkdir(parents=True,exist_ok=True);nbformat.write(nb,path)
    print(path)
if __name__=='__main__':main()
