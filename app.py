"""Offline hackathon dashboard: existing GAP recommendations and resource allocation."""
from __future__ import annotations
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from utils.preprocessing import BROAD_LABELS,SIDO_CODES,ROOT,options_for,experienced_from_features
from utils.recommend import predict_new_activity,predict_facility
from utils.regional import load_data
from utils.optimize import run_optimization

st.set_page_config(page_title='숲BTI | 산림복지 의사결정',page_icon='🌲',layout='wide')
st.markdown('''<style>
html,body,[class*="css"],.stApp{font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",sans-serif}
.stApp{background:#f7f5ef;color:#18382d}
[data-testid="stHeader"]{display:none}
.block-container{max-width:1380px;padding-top:2rem;padding-bottom:4rem}
h1,h2,h3{letter-spacing:-.045em;color:#173f30}
h1{font-size:2.3rem!important;font-weight:750!important}
[data-testid="stMetric"]{background:#fffdf7;border:1px solid #e0e3d8;border-radius:10px;padding:15px 18px}
[data-testid="stMetricLabel"]{color:#526358}
[data-testid="stMetricValue"]{color:#205d42;font-variant-numeric:tabular-nums}
button[kind="primary"]{background:#205d42;border-color:#205d42}
.stTabs [data-baseweb="tab-list"]{gap:28px;border-bottom:1px solid #d6ddce}
.stTabs [data-baseweb="tab"]{height:58px;font-size:16px}
.stTabs [aria-selected="true"]{color:#205d42;font-weight:700}
.brand{font-size:12px;letter-spacing:.15em;color:#617567;font-weight:600}
.flow{border-top:1px solid #cfdbcd;border-bottom:1px solid #cfdbcd;padding:14px 0;margin:16px 0 20px;color:#526358;line-height:1.8}
.decision{background:#e8eee1;border-left:4px solid #205d42;padding:18px 22px;border-radius:0 8px 8px 0;font-size:17px;line-height:1.8}
.small-note{font-size:12px;color:#667369;line-height:1.7}
footer{visibility:hidden}
</style>''',unsafe_allow_html=True)

st.markdown('<div class="brand">FOREST WELFARE · DECISION LAB</div>',unsafe_allow_html=True)
st.title('숲BTI · 새로운 경험에서 지역의 기회로')
st.write('아직 해보지 않은 산림활동을 찾고, 실제 사업예산을 바탕으로 지역별 운영 배분을 설계합니다.')
st.markdown('<div class="flow">01 개인의 새로운 활동　 →　 02 예산과 운영 배분</div>',unsafe_allow_html=True)

try:
    assets=load_data()
except (FileNotFoundError,ValueError) as exc:
    st.error(str(exc));st.code('python extract_data.py');st.stop()
codes=assets['codes'];card=assets['card'];const=assets['constants']
labels=json.loads((ROOT/'data/activity_labels.json').read_text())
detail_options={int(k):v for k,v in labels['detail_by_code'].items()}


def score_chart(frame,label_col):
    chart=frame.copy();chart['모델 점수']=chart['score']*100
    return alt.Chart(chart).mark_bar(color='#2c6a4b',cornerRadiusEnd=4).encode(
        x=alt.X('모델 점수:Q',scale=alt.Scale(domain=[0,100]),axis=alt.Axis(tickCount=6),title='모델 점수 (0–100)'),
        y=alt.Y(f'{label_col}:N',sort='-x',title=None,scale=alt.Scale(paddingInner=.35,paddingOuter=.15),
                axis=alt.Axis(labelOverlap=False,labelLimit=220)),
        tooltip=[label_col,alt.Tooltip('모델 점수:Q',format='.1f')]).properties(height=max(180,60*len(chart)+40))


def horizontal(frame,name,value,color='#2c6a4b',height=480):
    return alt.Chart(frame).mark_bar(color=color,cornerRadiusEnd=3).encode(
        x=alt.X(f'{value}:Q',title=value,axis=alt.Axis(tickCount=6)),
        y=alt.Y(f'{name}:N',sort='-x',title=None,scale=alt.Scale(paddingInner=.25),
                axis=alt.Axis(labelOverlap=False,labelLimit=250)),
        tooltip=[name,alt.Tooltip(f'{value}:Q',format=',.1f')]).properties(height=height)


def metric_row(items):
    for column,(label,value) in zip(st.columns(len(items)),items):column.metric(label,value)


@st.cache_data(show_spinner=False)
def demo_rows(kind):
    return pd.read_parquet(ROOT/'data'/f'demo_{kind}.parquet')


def field(label,code,key,default=None):
    opts=options_for(code)
    values=list(opts)
    return st.selectbox(label,values,index=values.index(default) if default in values else 0,
                        format_func=lambda x:opts[x],key=key)


def reset_person():
    st.session_state['experienced']=[]
    st.session_state.pop('recommendation_result',None)


tab_person,tab_opt=st.tabs(['개인 맞춤 추천','운영 최적화'])
with tab_person:
    st.header('나에게 맞는 새로운 산림활동 추천')
    left,right=st.columns([.43,.57],gap='large')
    with left:
        mode=st.radio('입력 방식',['내 정보 입력','실제 조사 응답으로 시연','전체 설문 CSV'],horizontal=True,key='input_mode',on_change=reset_person)
        user=None;facility_user=None;natural_experience=[];form_sido='서울'
        if mode=='내 정보 입력':
            st.caption('연령은 학습 조사와 같은 2025년 기준입니다. 간편 입력 결과는 참고 추천으로 확인해 주세요.')
            a,b=st.columns(2)
            with a:
                age=st.number_input('연령',15,100,40,key='age')
                household=field('가구원 현황','SQ3','household',4)
                income=field('월평균 가구소득','DQ5','income',5)
                education=field('학력','DQ2','education',5)
            with b:
                gender=field('성별','SQ6','gender',2)
                marital=field('혼인상태','DQ3','marital',2)
                occupation_options={**options_for('DQ1_1'),**options_for('DQ1_2')}
                occupation=st.selectbox('직업',list(occupation_options),format_func=lambda x:occupation_options[x],key='occupation')
                form_sido=st.selectbox('시도',list(SIDO_CODES),key='sido')
            region_codes=codes[codes.sido==form_sido]
            district=region_codes[['CO12','sigungu']].drop_duplicates().set_index('CO12').sigungu.to_dict()
            sigungu=st.selectbox('시군구',[None]+list(district),format_func=lambda x:'선택 안 함' if x is None else f'{district[x]} · 조사코드 {x}',key='sigungu')
            eup_options=region_codes[region_codes.CO12==sigungu].CO13.unique().tolist() if sigungu is not None else []
            eup=st.selectbox('읍면동 (선택)',[None]+eup_options,format_func=lambda x:'선택 안 함' if x is None else f'읍면동 조사코드 {x}',key='eup')
            user=dict(age=age,gender=gender,household=household,income=income,education=education,
                      marital=marital,occupation=occupation,sido=form_sido,sigungu_code=sigungu,eup_code=eup)
            st.caption('읍면동 명칭표는 현재 프로젝트에서 발견되지 않아 실제 관측된 조사코드만 제공합니다.')
            with st.expander('세부 활동 경험도 입력하기'):
                st.caption('직접 경험한 세부활동만 선택하세요. 이 응답은 모델의 원래 Q10 입력에 반영됩니다.')
                day=st.multiselect('당일형으로 경험한 활동',list(detail_options),format_func=lambda x:detail_options[x],key='day_details',placeholder='경험한 활동 선택')
                night=st.multiselect('숙박형으로 경험한 활동',list(detail_options),format_func=lambda x:detail_options[x],key='night_details',placeholder='경험한 활동 선택')
                if day or night:
                    user.update(day_activity_codes=day,night_activity_codes=night)
                    natural_experience=list({labels['broad_by_code'][str(c)] for c in day+night})
            facility_user=user
        elif mode=='실제 조사 응답으로 시연':
            idx=st.selectbox('익명 조사 응답',list(range(len(demo_rows('gap')))),format_func=lambda i:f'실제 응답 {i+1}',key='demo_row',on_change=reset_person)
            row=demo_rows('gap').iloc[idx].to_dict()
            user={'_raw_features':row};facility_user={'_raw_features':demo_rows('facility').iloc[idx].to_dict()}
            natural_experience=experienced_from_features(row)
            inverse={v:k for k,v in SIDO_CODES.items()}
            form_sido=inverse[int(float(row['CO11']))]
            st.write(f"**{form_sido} · {2025-int(row['SQ7_1'])}세 · {'남성' if str(row['SQ6'])=='1' else '여성'}**")
            st.caption('같은 실제 응답자의 전체 입력으로 모델을 실행합니다. 학습 자료에 포함된 시연 사례이며 검증용 테스트 표본이 아닙니다.')
            st.write('조사에 기록된 경험: '+(' · '.join(natural_experience) or '없음'))
        else:
            uploaded=st.file_uploader('원래 전처리된 설문 1행 CSV',type=['csv'],key='full_csv')
            st.caption('모델 스키마의 원래 열 이름을 사용합니다. GAP 409열 이상이 필요하며 시설 입력은 해당 스키마를 따릅니다.')
            if uploaded is not None:
                try:
                    frame=pd.read_csv(uploaded)
                    if len(frame)!=1:raise ValueError('한 사람의 설문 1행만 넣어 주세요.')
                    schema=json.loads((ROOT/'models/gap_activity/schema.json').read_text())
                    absent=set(schema['features'])-set(frame.columns)
                    if absent:raise ValueError(f'GAP 입력 {len(absent)}열이 누락됐습니다.')
                    row=frame.iloc[0].to_dict();user={'_raw_features':row};facility_user=user
                    natural_experience=experienced_from_features(row)
                except (ValueError,FileNotFoundError,pd.errors.ParserError) as exc:st.error(str(exc))
        done=st.multiselect('지금까지 경험한 산림활동을 선택해주세요.',BROAD_LABELS,
                            default=[],key='experienced',placeholder='경험한 활동 선택 (복수 선택 가능)')
        if natural_experience:
            st.caption('실제 세부 응답에 기록된 경험도 함께 제외합니다: '+' · '.join(natural_experience))
        clicked=st.button('새로운 활동 추천 받기',type='primary',use_container_width=True,disabled=user is None,key='recommend')
    with right:
        signature=json.dumps({'user':user,'facility':facility_user,'done':done},ensure_ascii=False,default=str,sort_keys=True)
        if clicked:
            try:
                with st.spinner('저장된 GAP 모델로 추천을 계산하고 있습니다…'):
                    result=predict_new_activity(user,done)
                    fac=predict_facility(facility_user)
                st.session_state['recommendation_result']=(signature,result,fac)
            except (FileNotFoundError,ValueError,RuntimeError) as exc:
                st.error(str(exc));st.session_state.pop('recommendation_result',None)
        saved=st.session_state.get('recommendation_result')
        if saved and saved[0]==signature:
            result,fac=saved[1:]
            st.subheader('아직 경험하지 않은 활동 중 추천')
            recommended=result['recommendations']
            if recommended.empty:
                st.info('모든 활동 유형이 경험 목록에 있습니다. 새로 추천할 유형이 없습니다.')
            else:
                for rank,row in enumerate(recommended.itertuples(index=False),1):
                    st.markdown(f'**{rank}위　{row.활동}**　{row.score*100:.1f}점')
                st.altair_chart(score_chart(recommended,'활동'),use_container_width=True)
                st.caption('점수는 저장된 GAP 모델의 predict_proba × 100입니다. 개인의 실제 참여확률로 보정된 값은 아닙니다.')
            st.caption('추천에서 제외된 기존 경험 활동')
            st.write(' · '.join(result['excluded']) or '선택된 활동 없음')
            st.divider();st.subheader('함께 살펴볼 시설 유형')
            st.altair_chart(score_chart(fac['recommendations'],'시설'),use_container_width=True)
            if result['input_info']['mode']!='full_record' or fac['input_info']['mode']!='full_record':
                st.info('간편 입력 추천입니다. 입력하지 않은 설문 항목은 학습 때와 같은 결측 규칙으로 처리했습니다.')
        elif saved:
            st.info('입력이 바뀌었습니다. 추천 버튼을 눌러 새 결과를 확인하세요.')
        else:
            st.subheader('새로운 숲 경험을 찾아보세요')
            st.write('내 정보를 입력하고 이미 해본 활동을 선택하면, 남은 활동 중 GAP 모델이 추천합니다.')
            st.caption('실제 조사 응답 시연을 선택하면 전체 설문 입력으로 바로 확인할 수 있습니다.')
    with st.expander('모델 정보 · 입력 범위와 검증 결과'):
        g=card['gap_activity'];m=g['metrics']
        metric_row([('기존 GAP P@1',f"{m['P@1']:.3f}"),('P@2',f"{m['P@2']:.3f}"),('LRAP',f"{m['LRAP']:.3f}"),('Macro-F1',f"{m['Macro_F1']:.3f}")])
        st.write(f"GAP: Q17 향후 의향 AND NOT Q10 경험. CatBoost 6개 · depth 6 · 360회 · Ordered · raw409. 시설: 13개 · depth 7 · 880회 · Plain · raw498.")
        st.write(f"GAP PR-AUC: fold 평균 {m['PR_AUC_macro']:.3f}, OOF 통합 Macro {g['pooled_macro_pr_auc']:.3f}. 인기순 P@1 {g['naive_p_at_1']:.3f}, 경험 제외 인기순 {g['baseline_p_at_1']:.3f}.")
        st.caption('GAP P@k는 정답 활동이 있는 5,579명만 평가하며 정답 없는 2,047명은 제외합니다. 간편 입력 화면의 성능은 별도 미검증입니다. 6대분류 단위로 경험을 제외합니다.')
        st.caption('활동 입력에서 Q17 및 Q20·Q19_5·Q21 계열 제외. 시설 입력에서 해당 타깃 Q20_5 제외. 원래 모델의 입력 순서·문자형 범주·결측 규칙을 보존합니다.')

with tab_opt:
    st.header('AI 기반 산림복지 자원배분 최적화')
    left,right=st.columns([.3,.7],gap='large')
    with left:
        budget_eok=st.number_input('총 사업예산 (억 원)',0.,1000.,const['budget_won']/1e8,step=1.,format='%.2f',key='budget')
        cost_man=st.number_input('회당 운영비 (만원)',.01,100000.,const['cost_per_run_won']/1e4,step=10.,format='%.4f',key='cost')
        capacity=st.number_input('회당 수용인원 (명)',1,100000,const['capacity'],key='capacity')
        rate=st.number_input('정책 목표율 (%)',0.,100.,const['target_rate']*100,step=.1,key='target_rate')
        execute=st.button('최적 배분 실행',type='primary',use_container_width=True,key='optimize')
        st.caption('63.21억 원은 ALIO 2023년 숲체험교육사업 결산입니다. 회당 약 332만원은 참여실적에서 역산한 평균 단가입니다.')
        st.caption('운영횟수의 임의 지역 상한과 총인력 상한을 두지 않습니다. 필요 인력은 회당 2인 가정으로 산출합니다.')
    parameters=dict(budget=budget_eok*1e8,cost_per_run=cost_man*1e4,capacity=int(capacity),target_rate=rate/100)
    with right:
        if execute:
            try:
                with st.spinner('예산과 지역 목표에 맞춰 배분을 계산하고 있습니다…'):
                    st.session_state['optimization']=run_optimization(**parameters)
            except (ValueError,RuntimeError,FileNotFoundError) as exc:st.error(str(exc))
        result=st.session_state.get('optimization')
        if result and result['assumptions']==parameters:
            pct=lambda v:'비교 불가' if v is None else f'{v:+.1f}%'
            metric_row([('예산 사용률',f"{result['budget_utilization']:.1f}%"),('인구비례 대비',pct(result['population_improvement'])),
                        ('기존 참여비례 대비',pct(result['status_quo_improvement'])),('필요 인력-회차',f"{result['staff_run_units']:,}")])
            st.caption(f"총 {result['total_runs']:,}회 · 배정 수용인원 {result['total_served']:,.0f}명 · 사용예산 {result['budget_used']/1e8:.2f}억 원")
            st.subheader('지역별 최적 프로그램 배정 횟수')
            st.altair_chart(horizontal(result['allocation'],'sido','운영횟수',height=400),use_container_width=True)
            st.subheader('배분 방식별 격차가중 커버리지')
            compare=result['comparison'].copy()
            if result['coverage']>0:
                compare['최적화 대비 지수']=compare['격차가중 커버리지']/result['coverage']*100
                st.altair_chart(horizontal(compare,'방식','최적화 대비 지수',color='#a96641',height=150),use_container_width=True)
            else:st.info('현재 입력에서 배정 가능한 운영횟수 또는 목표가 0입니다.')
            st.caption('세 방식에 같은 총 운영횟수를 배분하고, 동일한 0–100 우선확충지수로 평가합니다. 최적화 목적함수의 양수 z 가중치와 평가 지표는 구분됩니다.')
            if result['budget_utilization']<95 and result['total_runs']>0:
                message='이 시나리오에서는 예산 증액보다 필요한 지역으로의 재배분과 이를 운영할 인력 확보가 중요한 실행 과제입니다.'
            elif result['total_runs']>0:
                message='현재 시나리오는 가용 예산을 대부분 사용합니다. 운영 목표·단가·수용력을 함께 검토하세요.'
            else:message='운영을 배정하려면 1회 이상 실행 가능한 예산과 양수의 목표율이 필요합니다.'
            st.markdown(f'<div class="decision">{message}</div>',unsafe_allow_html=True)
            st.caption('배정 인원은 모형상 수용 가능한 인원입니다. 실제 참여·성과의 관측값이 아닙니다. 인력-회차는 고유 직원 수가 아니며 실제 인력 병목은 전담 인력자료로 추가 확인해야 합니다.')
            with st.expander('지역별 배분 상세'):
                st.dataframe(result['allocation'],hide_index=True,use_container_width=True)
                st.dataframe(compare,hide_index=True,use_container_width=True)
            st.download_button('최적 배분 CSV 내려받기',result['allocation'].to_csv(index=False).encode('utf-8-sig'),'optimization_result.csv','text/csv')
        elif result:st.info('설정이 바뀌었습니다. 최적 배분 실행을 눌러 다시 계산하세요.')
        else:
            st.subheader('같은 자원을, 필요한 지역에')
            st.write('실제 사업예산을 기준으로 지역별 운영횟수를 계산하고 인구비례·기존 참여비례 배분과 비교합니다.')
            st.info('왼쪽의 최적 배분 실행 버튼으로 시연을 시작하세요.')

st.divider()
st.caption('자료: 산림휴양·복지활동 조사 · 지역 공급 산출물 · ALIO 2023 결산 · 진흥원 2023 참여실적 | 모든 예측·최적화는 로컬에서 실행')
