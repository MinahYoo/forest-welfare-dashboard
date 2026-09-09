"""Extract verified project assets once; no source notebooks are executed by the app."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from train_models import DEFAULT_PROJECT, HERE, literal_assignment


def dump(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))


def extract(project):
    out=HERE/'data';out.mkdir(exist_ok=True)
    an=project/'분석산출물';ext=project/'외부데이터'
    source=an/'pkgC_operations/pkgC_optimization_v2.py'
    # Execute only the existing input-building section, before optimisation/output.
    tree=ast.parse(source.read_text())
    end=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.FunctionDef) and n.name=='optimise')
    ns={'__file__':str(source),'__name__':'input_extraction'}
    exec(compile(ast.Module(body=tree.body[:end],type_ignores=[]),str(source),'exec'),ns)
    gap=ns['gap'].copy()
    reg=gap.rename_axis('sido').reset_index()
    reg['우선확충지수']=ns['priority100'].reindex(reg.sido).to_numpy()
    reg['z_가중치']=ns['_score_z'].reindex(reg.sido).to_numpy()
    reg['목적함수_가중치']=reg['z_가중치'].clip(lower=0)
    reg['기존참여_2023']=ns['existing'].reindex(reg.sido).to_numpy()
    reg['수요수준']=np.where(reg['미충족잠재수요율']>=reg['미충족잠재수요율'].median(),'상대적으로 높음','상대적으로 낮음')
    reg['공급수준']=np.where(reg['공급_10만명당']>=reg['공급_10만명당'].median(),'공급밀도 높음','공급밀도 낮음')
    reg['정책판단']=np.where((reg['수요수준']=='상대적으로 높음')&(reg['공급수준']=='공급밀도 높음'),
                            '기존시설 이용전환·접근 연계',np.where(reg['목적함수_가중치']>0,'우선 공급·운영 보완 검토','기존시설 활용·수요 관찰'))
    raw=pd.read_parquet(project/'DATA_2024_산림휴양복지활동조사_공표용_결측열제거.parquet')
    weighted=ns['_d'].copy()
    weighted['used']=ns['_used_n']>0
    used=weighted.groupby('sido').apply(lambda d:np.average(d.used,weights=d.WT)*100,include_groups=False)
    reg['시설이용경험률']=reg.sido.map(used)
    aux=pd.read_csv(project/'본선_준비/experiments/results/regional_supply_indicators.csv')
    aux=aux[['sido','최근접시설거리_km','인구10만명당_시설수']]
    reg=reg.merge(aux,on='sido',how='left',validate='one_to_one')
    reg.to_csv(out/'regional_index.csv',index=False)
    reg[['sido','인구','z_가중치','우선확충지수','기존참여_2023']].to_csv(out/'optimization_input.csv',index=False)
    const=dict(budget_won=float(ns['BUDGET']*10000),cost_per_run_won=float(ns['COST_PER_RUN']*10000),
               capacity=int(ns['CAPACITY_PER_RUN']),target_rate=float(ns['HEADLINE_RATE']),
               staff_per_run=int(ns['STAFF_PER_RUN']),participants_2023=float(ns['PART_2023']),
               staff_note='회당 2인 가정의 인력-회차이며 고유 인원/FTE가 아님',
               cost_note='2023 결산 ÷ 참여실적 × 300명으로 역산한 평균 단가',
               budget_note='ALIO 숲체험교육사업 2023 결산 6,321백만원',
               source=str(source.relative_to(project)))
    dump(out/'optimization_constants.json',const)
    shutil.copy2(an/'pkgC_operations/optimization_v2_scenarios.csv',out/'optimization_reference.csv')
    shutil.copy2(an/'pkgC_operations/optimization_v2_allocation.csv',out/'optimization_allocation_reference.csv')
    shutil.copy2(ext/'시군구_격차지수.csv',out/'sigungu_index.csv')
    shutil.copy2(an/'변수_전체목록.csv',out/'feature_dictionary.csv')
    # Facility coordinates and 시군구 centroids for the distance-sorted nearby list
    # (verified public data; same artifacts the 숲BTI web demo assembles).
    webapp=an/'personal_recommender/webapp/data'
    shutil.copy2(webapp/'facilities.json',out/'facilities.json')
    shutil.copy2(webapp/'sigungu_centroids.json',out/'sigungu_centroids.json')

    # Administrative codes are observed survey codes, never inferred from postal codes.
    codes=raw[['CO11','CO12','CO13']].drop_duplicates().astype(int)
    sido_map=ns['_SIDO']
    numeric_sido={11:'서울',21:'부산',22:'대구',23:'인천',24:'광주',25:'대전',26:'울산',29:'세종',31:'경기',32:'강원',33:'충북',34:'충남',35:'전북',36:'전남',37:'경북',38:'경남',39:'제주'}
    codes['sido']=codes.CO11.map(numeric_sido)
    names={}
    geo=Path('/private/tmp/claude-501/-Users-minah-K-ds/3dc4c565-f6e7-45f7-96e6-c77a53524d60/scratchpad/sgg_geo.json')
    if geo.exists():
        for f in json.loads(geo.read_text())['features']:
            names[int(f['properties']['code'])]=f['properties']['name']
    codes['sigungu']=codes.apply(lambda r:names.get(r.CO11*1000+r.CO12,f'시군구 조사코드 {r.CO12}'),axis=1)
    codes.to_csv(out/'region_codes.csv',index=False)

    ga=an/'gap_target_results'
    summary=pd.read_csv(ga/'gap_summary.csv',index_col=0)
    pooled=pd.read_csv(ga/'gap_perlabel_pooled.csv')
    labels=literal_assignment(ga/'gap_target_run.py','ACT2BROAD')
    activity_labels=literal_assignment(project/'본선_준비/experiments/config.py','ACTIVITY_LABELS')
    dump(out/'activity_labels.json',{'broad_by_code':labels,'detail_by_code':activity_labels})
    facet=pd.read_csv(an/'catboost_tuning_results_raw515_facility/ranking_metrics.csv',index_col=0)
    card={'gap_activity':{'metrics':summary['model_mean'].to_dict(),
             'pooled_macro_pr_auc':float(pooled[pooled['label'].str.contains('Macro')]['pr_auc'].iloc[0]),
             'baseline_p_at_1':float(summary.loc['P@1','histaware_pop_mean']),
             'naive_p_at_1':float(summary.loc['P@1','naive_pop_mean']),
             'n_train':7626,'n_positive_topk':5579,'n_excluded_topk':2047,
             'evaluation':'기존 가구그룹 5-fold OOF; 간편 입력 폼의 별도 성능 검증값이 아님'},
          'facility':{'metrics':facet.loc['mean'].to_dict()},
          'regional':'개인 GAP 예측 집계가 아니라 기존 Q20 설문 가중집계와 관내 공급 자료의 결합',
          'policy_rule':'수요·공급 수준은 17개 시도 중앙값 대비; 정책 문구는 기존 보고서 해석을 적용한 안내 규칙'}
    dump(out/'model_card.json',card)
    # Exact existing responses for an optional full-input demonstration; no invented records.
    prep_a=an/'catboost_tuning_results_raw515/_prep_artifacts'
    prep_f=an/'catboost_tuning_results_raw515_facility/_prep_artifacts'
    xa=pd.read_parquet(prep_a/'X.parquet');xf=pd.read_parquet(prep_f/'X.parquet')
    idx=[0,100,500,1000,2000]
    a_rows=np.flatnonzero(raw[[f'Q17A{i}' for i in range(1,30)]].notna().any(axis=1))
    fc=[c for c in raw if c.startswith('Q20_5A')]
    f_rows=np.flatnonzero((raw[fc].notna()&~raw[fc].isin([99.,999999999.])).any(axis=1))
    f_positions={int(raw_idx):pos for pos,raw_idx in enumerate(f_rows)}
    idx=[i for i in idx if int(a_rows[i]) in f_positions]
    paired=[f_positions[int(a_rows[i])] for i in idx]
    assert idx, 'No aligned demonstration respondents'
    xa.iloc[idx].drop(columns=[c for c in xa if c.startswith(('Q20_','Q19_5')) or c in {'Q21','Q21_1'}]).to_parquet(out/'demo_gap.parquet',index=False)
    xf.iloc[paired].to_parquet(out/'demo_facility.parquet',index=False)
    selected_sources=[source,ga/'gap_target_run.py',ga/'gap_summary.csv',
                      an/'personal_recommender/build_recommender_tuned.py',
                      an/'catboost_tuning_results_raw515/prep_raw515.py',
                      an/'catboost_tuning_results_raw515_facility/prep_raw515_facility.py',
                      an/'catboost_tuning_results_raw515_facility/ranking_metrics.csv',
                      an/'pkgC_operations/기관별_2023_참여인원.csv',
                      an/'pkgC_operations/param_reality_check.csv',
                      ext/'시군구_수급_자연휴양림.csv',
                      project/'본선_준비/experiments/results/regional_supply_indicators.csv',
                      ext/'(증빙) 수입지출 세부현황(2023년).xlsx',
                      webapp/'facilities.json',
                      webapp/'sigungu_centroids.json']
    dump(out/'provenance.json',{str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest() for p in selected_sources})
    from utils.optimize import run_optimization
    result=run_optimization(const['budget_won'],const['cost_per_run_won'],const['capacity'],const['target_rate'])
    result['allocation'].to_csv(out/'optimization_result.csv',index=False)
    print(f'Extracted: {len(reg)} regions, {len(codes)} observed administrative-code combinations')
    print(f'Budget {const["budget_won"]:,.0f} won; cost {const["cost_per_run_won"]:,.2f} won/run')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=DEFAULT_PROJECT)
    extract(p.parse_args().project)
