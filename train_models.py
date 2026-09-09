"""One-time export of the project's existing GAP/facility CatBoost specifications.

No training occurs when app.py is run. Each label is checkpointed and verified.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = Path('/Users/minah/Desktop/K-ds/예선/6. 한국산림복지진흥원')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def literal_assignment(path, name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f'{name} not found in {path}')


def training_frames(project, kind):
    an = project / '분석산출물'
    prep = an / ('catboost_tuning_results_raw515' if kind == 'gap_activity' else 'catboost_tuning_results_raw515_facility') / '_prep_artifacts'
    x, y = pd.read_parquet(prep/'X.parquet'), pd.read_parquet(prep/'Y.parquet')
    meta = json.loads((prep/'meta.json').read_text())
    sources = {str(p.relative_to(project)): digest(p) for p in [prep/'X.parquet', prep/'Y.parquet', prep/'meta.json']}
    if kind == 'gap_activity':
        source = an/'gap_target_results/gap_target_run.py'
        raw_path = project/'DATA_2024_산림휴양복지활동조사_공표용_결측열제거.parquet'
        sources.update({str(p.relative_to(project)): digest(p) for p in [source, raw_path]})
        mapping = literal_assignment(source, 'ACT2BROAD')
        x = x.drop(columns=[c for c in x if c.startswith(('Q20_', 'Q19_5')) or c in {'Q21','Q21_1'}])
        assert x.shape == (7626,409)
        raw = pd.read_parquet(raw_path)
        raw = raw[raw[[f'Q17A{i}' for i in range(1,30)]].notna().any(axis=1)].reset_index(drop=True)
        def selected(prefixes):
            cols = [f'{p}A{i}' for p in prefixes for i in range(1,30) if f'{p}A{i}' in raw]
            return raw[cols].apply(lambda row: {mapping[v] for v in row if pd.notna(v) and v in mapping}, axis=1)
        intent, experience = selected(['Q17']), selected(['Q10_1','Q10_2'])
        rebuilt = np.array([[int(label in row) for label in y.columns] for row in intent])
        assert np.array_equal(rebuilt, y.to_numpy()), 'Target row alignment failed'
        y = pd.DataFrame({label: [int(label in i and label not in e) for i,e in zip(intent,experience)] for label in y.columns})
        params = dict(depth=6, iterations=360, boosting_type='Ordered', random_seed=123,
                      eval_metric='TotalF1:average=Macro')
    else:
        assert x.shape == (11437,498)
        assert not any(c.startswith('Q20_5') for c in x)
        params = dict(depth=7, iterations=880, boosting_type='Plain', random_seed=42)
    cats = [c for c in meta['cat_cols'] if c in x]
    for c in cats:
        x[c] = x[c].astype(str)
    params.update(learning_rate=.05, l2_leaf_reg=3., random_strength=1., bagging_temperature=1.,
                  one_hot_max_size=10, auto_class_weights='Balanced', verbose=False,
                  allow_writing_files=False)
    return x, y, cats, params, sources


def export(project, kind, threads):
    x,y,cats,params,sources = training_frames(project, kind)
    schema = dict(kind=kind, features=list(x.columns), cat_features=cats,
                  numeric_features=[c for c in x if c not in cats], labels=list(y.columns),
                  missing_numeric=0, missing_categorical='미상', params=params, sources=sources,
                  n_train=len(x), target='Q17 AND NOT Q10' if kind=='gap_activity' else 'Q20_5',
                  category_tokens={c:sorted(x[c].unique().tolist()) for c in cats})
    signature = hashlib.sha256(json.dumps(schema,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    dest = HERE/'models'/kind
    dest.mkdir(parents=True,exist_ok=True)
    schema['signature']=signature
    (dest/'schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2))
    pool = Pool(x,cat_features=cats)
    manifest = dict(signature=signature, labels=[], model_kind=kind, n_train=len(x))
    for j,label in enumerate(y.columns):
        out=dest/f'label_{j:02d}.cbm'
        stamp=dest/f'label_{j:02d}.json'
        record=json.loads(stamp.read_text()) if stamp.exists() else {}
        if out.exists() and record.get('signature')==signature and record.get('sha256')==digest(out):
            print(f'{kind} {j+1}/{len(y.columns)} cached: {label}',flush=True)
        else:
            started=time.monotonic()
            model=CatBoostClassifier(**params,thread_count=threads)
            model.fit(Pool(x,y.iloc[:,j].to_numpy(),cat_features=cats))
            tmp=out.with_suffix('.tmp')
            model.save_model(str(tmp),format='cbm')
            saved=CatBoostClassifier();saved.load_model(str(tmp))
            check=Pool(x.iloc[:5],cat_features=cats)
            np.testing.assert_allclose(model.predict_proba(check),saved.predict_proba(check),rtol=0,atol=1e-12)
            tmp.replace(out)
            record=dict(label=label,file=out.name,signature=signature,sha256=digest(out))
            stamp.write_text(json.dumps(record,ensure_ascii=False))
            print(f'{kind} {j+1}/{len(y.columns)} saved: {label} ({time.monotonic()-started:.1f}s)',flush=True)
        manifest['labels'].append(record)
    (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(f'{kind}: all models saved and reload-verified',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=DEFAULT_PROJECT)
    p.add_argument('--kind',choices=['gap_activity','facility','all'],default='all')
    p.add_argument('--threads',type=int,default=8)
    args=p.parse_args()
    for kind in (['gap_activity','facility'] if args.kind=='all' else [args.kind]):
        export(args.project,kind,args.threads)
