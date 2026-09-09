"""Regression tests of real stored models, original allocation results and the UI."""
import itertools
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.optimize import run_optimization
from utils.preprocessing import BROAD_LABELS,prepare_input,experienced_from_features
from utils.recommend import load_models,predict_new_activity,predict_facility


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.const=json.loads((ROOT/'data/optimization_constants.json').read_text())
        cls.user=dict(age=40,gender=2,household=4,income=5,education=5,marital=2,occupation=3,sido='서울')

    def test_saved_model_round_trip_inputs(self):
        for kind,name in [('gap_activity','gap'),('facility','facility')]:
            models,schema=load_models(kind)
            demos=pd.read_parquet(ROOT/'data'/f'demo_{name}.parquet')
            for _,row in demos.iterrows():
                frame,info=prepare_input({'_raw_features':row.to_dict()},schema)
                self.assertEqual(info['mode'],'full_record')
                for col in schema['cat_features']:self.assertEqual(frame.iloc[0][col],row[col])
                np.testing.assert_allclose(frame[schema['numeric_features']].iloc[0].to_numpy(float),row[schema['numeric_features']].to_numpy(float))
                for model in models:
                    np.testing.assert_allclose(model.predict_proba(frame),model.predict_proba(pd.DataFrame([row])),atol=1e-12)
        schema=load_models('gap_activity')[1]
        self.assertFalse(any(c.startswith(('Q17','Q20_','Q19_5')) or c in {'Q21','Q21_1'} for c in schema['features']))
        self.assertFalse(any(c.startswith('Q20_5') for c in load_models('facility')[1]['features']))

    def test_all_experience_masks(self):
        for bits in itertools.product([False,True],repeat=6):
            excluded=[label for label,flag in zip(BROAD_LABELS,bits) if flag]
            r=predict_new_activity(self.user,excluded)
            rec=r['recommendations']
            self.assertTrue(set(rec['활동']).isdisjoint(excluded))
            self.assertEqual(len(rec),min(3,6-len(excluded)))
            self.assertTrue(rec.score.is_monotonic_decreasing)
            self.assertTrue(rec.score.between(0,1).all())
        row=pd.read_parquet(ROOT/'data/demo_gap.parquet').iloc[0].to_dict()
        r=predict_new_activity({'_raw_features':row},[])
        self.assertTrue(set(r['recommendations']['활동']).isdisjoint(experienced_from_features(row)))
        self.assertEqual(len(predict_facility(self.user)['scores']),13)

    def test_precise_reference_and_constraints(self):
        c=self.const
        reference=pd.read_csv(ROOT/'data/optimization_reference.csv')
        for rate in [.003,.005,.01]:
            r=run_optimization(c['budget_won'],c['cost_per_run_won'],c['capacity'],rate)
            row=reference[reference['목표_pop비']==f'{rate*100:.1f}%'].iloc[0]
            self.assertEqual(r['total_runs'],row['배정회차'])
            self.assertEqual(r['staff_run_units'],row['필요인력_회차'])
            self.assertAlmostEqual(r['budget_utilization'],row['예산소진_pct'],delta=.051)
            self.assertAlmostEqual(r['population_improvement'],row['vs_인구비례_pct'],delta=.051)
            self.assertAlmostEqual(r['status_quo_improvement'],row['vs_기존공급비례_pct'],delta=.051)
        for budget,cost,capacity,rate in [(0,1e6,300,.01),(1e7,3e6,50,.01),(6.321e9,4e6,120,.03),(6.321e9,3e6,300,0)]:
            r=run_optimization(budget,cost,capacity,rate);a=r['allocation']
            self.assertLessEqual(r['budget_used'],budget+1e-5)
            self.assertTrue((a['충족인원']<=a['목표인원']+1e-5).all())
            self.assertTrue((a['충족인원']<=a['운영횟수']*capacity+1e-5).all())
            self.assertEqual(a['운영횟수'].sum(),a['인구비례_횟수'].sum())
            self.assertEqual(a['운영횟수'].sum(),a['기존참여비례_횟수'].sum())
        for args in [(-1,1,1,.01),(1,0,1,.01),(1,1,0,.01),(1,1,1,1.1)]:
            with self.assertRaises(ValueError):run_optimization(*args)

    def test_streamlit_demo_flow(self):
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=40).run()
        self.assertFalse(at.exception)
        self.assertEqual([t.label for t in at.tabs],['개인 맞춤 추천','지역 수요 분석','운영 최적화'])
        at.button(key='recommend').click().run()
        self.assertFalse(at.exception);self.assertFalse(at.error)
        at.multiselect(key='experienced').set_value(BROAD_LABELS).run()
        at.button(key='recommend').click().run()
        self.assertTrue(any('새로 추천할 유형이 없습니다' in v.value for v in at.info))
        at.selectbox(key='sido').set_value('강원').run()
        self.assertFalse(at.exception)
        at.radio(key='input_mode').set_value('실제 조사 응답으로 시연').run()
        at.button(key='recommend').click().run()
        self.assertFalse(at.exception);self.assertFalse(at.error)
        at.button(key='optimize').click().run()
        self.assertFalse(at.exception);self.assertFalse(at.error)
        self.assertTrue(any(m.label=='필요 인력-회차' and m.value=='2,416' for m in at.metric))
        at.number_input(key='budget').set_value(0.).run()
        self.assertTrue(any('설정이 바뀌었습니다' in v.value for v in at.info))
        at.button(key='optimize').click().run()
        self.assertFalse(at.exception);self.assertFalse(at.error)


if __name__=='__main__':unittest.main(verbosity=2)
