"""Read the final supply-adjusted regional analysis and provenance."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

DATA=Path(__file__).resolve().parents[1]/'data'

@st.cache_data(show_spinner=False)
def load_data():
    required=['regional_index.csv','region_codes.csv','model_card.json','optimization_constants.json']
    absent=[p for p in required if not (DATA/p).exists()]
    if absent:raise FileNotFoundError('지역 데이터가 없습니다. extract_data.py 실행 필요: '+', '.join(absent))
    regional=pd.read_csv(DATA/'regional_index.csv')
    if len(regional)!=17 or regional.sido.duplicated().any():raise ValueError('시도 자료 정합성을 확인하세요.')
    return dict(regional=regional,codes=pd.read_csv(DATA/'region_codes.csv'),
                card=json.loads((DATA/'model_card.json').read_text()),
                constants=json.loads((DATA/'optimization_constants.json').read_text()))
