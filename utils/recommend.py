"""Predict from saved CatBoost models, never from invented scores or cell averages."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier
from .preprocessing import ROOT,BROAD_LABELS,prepare_input,form_to_features,experienced_from_features


@st.cache_resource(show_spinner=False)
def load_models(kind):
    directory=ROOT/'models'/kind
    if not (directory/'manifest.json').exists():
        raise FileNotFoundError(f'{kind} 저장 모델이 없습니다. 앱 외부에서 train_models.py를 먼저 완료하세요.')
    manifest=json.loads((directory/'manifest.json').read_text())
    schema=json.loads((directory/'schema.json').read_text())
    if manifest['signature']!=schema['signature']:raise ValueError('모델과 입력 스키마 버전이 다릅니다.')
    if len(manifest['labels'])!=len(schema['labels']):raise ValueError('일부 모델 파일이 누락됐습니다.')
    models=[]
    for label,record in zip(schema['labels'],manifest['labels']):
        path=directory/record['file']
        if label!=record['label'] or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('모델 라벨 또는 파일 무결성을 확인하세요.')
        m=CatBoostClassifier();m.load_model(str(path))
        if m.feature_names_!=schema['features']:raise ValueError('모델의 입력 순서가 스키마와 다릅니다.')
        models.append(m)
    return models,schema


def _predict(kind,user_input):
    models,schema=load_models(kind)
    frame,info=prepare_input(user_input,schema)
    scores=np.array([m.predict_proba(frame)[0,1] for m in models],dtype=float)
    if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():raise RuntimeError('모델 점수 범위를 확인하세요.')
    return pd.DataFrame({'활동' if kind=='gap_activity' else '시설':schema['labels'],'score':scores}),info


def predict_new_activity(user_input, experienced_activities):
    unknown=set(experienced_activities)-set(BROAD_LABELS)
    if unknown:raise ValueError(f'알 수 없는 활동 유형: {sorted(unknown)}')
    known=set(experienced_from_features(form_to_features(user_input)))
    excluded=known|set(experienced_activities)
    scores,info=_predict('gap_activity',user_input)
    available=scores[~scores['활동'].isin(excluded)].sort_values('score',ascending=False,kind='stable')
    return dict(recommendations=available.head(3).reset_index(drop=True),scores=scores,
                excluded=[s for s in BROAD_LABELS if s in excluded],input_info=info)


def predict_facility(user_input):
    scores,info=_predict('facility',user_input)
    return dict(recommendations=scores.sort_values('score',ascending=False,kind='stable').head(3).reset_index(drop=True),
                scores=scores,input_info=info)
