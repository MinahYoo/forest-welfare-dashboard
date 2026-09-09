"""Match the existing feature order, categorical tokens and missing-value rules."""
from __future__ import annotations
import json
from pathlib import Path
import math

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SIDO_CODES={'서울':11,'부산':21,'대구':22,'인천':23,'광주':24,'대전':25,'울산':26,'세종':29,
            '경기':31,'강원':32,'충북':33,'충남':34,'전북':35,'전남':36,'경북':37,'경남':38,'제주':39}
BROAD_LABELS=['등산·트레킹형','자연감상·산책형','캠핑·야영형','체험·학습형','치유·웰니스형','레포츠·모험형']
SURVEY_YEAR=2025


def schema_for(kind):
    p=ROOT/'models'/kind/'schema.json'
    if not p.exists():raise FileNotFoundError('모델 입력 스키마가 없습니다. train_models.py를 먼저 실행하세요.')
    return json.loads(p.read_text())


def options_for(column):
    d=pd.read_csv(ROOT/'data/feature_dictionary.csv')
    desc=d.set_index('변수').loc[column,'값설명']
    return {int(part.split('=',1)[0].strip()):part.split('=',1)[1].strip() for part in str(desc).split(';') if '=' in part}


def form_to_features(user_input):
    if '_raw_features' in user_input:
        return dict(user_input['_raw_features'])
    age=int(user_input['age'])
    if not 15<=age<=100:raise ValueError('연령은 15~100세 범위로 입력해 주세요.')
    sido=user_input['sido']
    if sido not in SIDO_CODES:raise ValueError('시도를 확인해 주세요.')
    hh=int(user_input['household']);income=user_input.get('income');job=user_input.get('occupation')
    if hh not in [1,2,3,4]:raise ValueError('가구원 현황을 확인해 주세요.')
    features={'SQ7_1':SURVEY_YEAR-age,'D_SQ7':1 if age<20 else min(age//10,7),
              'SQ6':int(user_input['gender']),'SQ3':hh,'D_SQ3':min(hh,3),
              'DQ3':user_input.get('marital'),'DQ2':user_input.get('education'),
              'DQ5':income,'D_DQ5':min(int(income),8) if income is not None else None,
              'CO11':SIDO_CODES[sido],'D_CO11':list(SIDO_CODES).index(sido)+1,
              'CO12':user_input.get('sigungu_code'),'CO13':user_input.get('eup_code')}
    if job is not None:
        features.update(DQ1=1 if job<=10 else 2,DQ1_1=job if job<=10 else None,DQ1_2=job if job>10 else None)
    # A broad category alone does not identify the fine Q10 activity code.
    # Only explicitly provided day/overnight selections populate those original fields.
    for key,prefix in [('day_activity_codes','Q10_1'),('night_activity_codes','Q10_2')]:
        if key in user_input:
            selected={int(v) for v in user_input[key]}
            if not selected.issubset(set(range(1,30))):raise ValueError('세부활동 코드를 확인하세요.')
            for code in range(1,30):features[f'{prefix}A{code}']=float(code) if code in selected else None
    return features


def prepare_input(user_input, schema):
    supplied=form_to_features(user_input)
    cats=set(schema['cat_features']);values={};missing=0
    for col in schema['features']:
        value=supplied.get(col)
        absent=value is None or (not isinstance(value,(list,dict)) and pd.isna(value))
        if col in cats:
            token='미상' if absent else str(value)
            known=schema['category_tokens'][col]
            if token not in known and token!='미상':
                try:
                    numeric=float(token)
                    token=next((s for s in known if s!='미상' and float(s)==numeric),token)
                except ValueError:pass
            values[col]=token
            missing+=int(token=='미상')
        else:
            value=0. if absent else float(value)
            if not math.isfinite(value):raise ValueError(f'{col} 값은 유한한 숫자여야 합니다.')
            values[col]=value
            missing+=int(absent)
    frame=pd.DataFrame([values],columns=schema['features'])
    provided=len(set(supplied)&set(schema['features']))
    mode=('full_record' if provided==len(schema['features']) else 'partial_record') if '_raw_features' in user_input else 'partial_form'
    return frame,dict(provided_columns=provided,
                      total_columns=len(schema['features']),missing_values=missing,
                      mode=mode)


def experienced_from_features(features):
    mapping=json.loads((ROOT/'data/activity_labels.json').read_text())['broad_by_code']
    found=set()
    for col,value in features.items():
        if not col.startswith(('Q10_1A','Q10_2A')):continue
        try:label=mapping.get(str(int(float(value))))
        except (ValueError,TypeError,OverflowError):continue
        if label:found.add(label)
    return [label for label in BROAD_LABELS if label in found]
