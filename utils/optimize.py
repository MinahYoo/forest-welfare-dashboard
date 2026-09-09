"""Final 63.21억원 project formulation, with all currency inputs in won.

Budget is actually 6.321 billion won (63.21억원). Uses original input precision,
positive z weights, population caps, and largest-remainder reference baselines.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

DATA=Path(__file__).resolve().parents[1]/'data'


def proportional(total_runs, key):
    key=pd.Series(key,dtype=float)
    if total_runs==0:return key*0
    if key.sum()<=0:raise ValueError('비례 배분 기준의 합이 0입니다.')
    raw=key/key.sum()*total_runs
    allocation=np.floor(raw)
    remainder=int(round(total_runs-allocation.sum()))
    # Preserve the original pandas tie order: several zero-participation regions
    # have equal remainders, so changing the sorting algorithm changes the baseline.
    allocation.loc[(raw-allocation).sort_values(ascending=False).index[:remainder]]+=1
    return allocation


def run_optimization(budget, cost_per_run, capacity, target_rate):
    """Run existing allocation problem. Currency in KRW; target_rate is e.g. .01."""
    vals=[budget,cost_per_run,capacity,target_rate]
    if not all(np.isfinite(v) for v in vals):raise ValueError('입력값은 유한한 숫자여야 합니다.')
    if budget<0 or cost_per_run<=0 or capacity<=0 or capacity!=int(capacity) or not 0<=target_rate<=1:
        raise ValueError('예산은 0 이상, 단가·정원은 양수, 목표율은 0~100%여야 합니다.')
    d=pd.read_csv(DATA/'optimization_input.csv').set_index('sido')
    const=json.loads((DATA/'optimization_constants.json').read_text())
    if len(d)!=17 or d.index.duplicated().any():raise ValueError('17개 시도 입력 자료를 확인하세요.')
    goal=d['인구']*target_rate;weight=d['z_가중치'].clip(lower=0)
    max_runs=int(np.floor(budget/cost_per_run))
    served=pd.Series(0.,index=d.index)
    if max_runs and target_rate and weight.max()>0:
        problem=pulp.LpProblem('forest_allocation_v2',pulp.LpMaximize)
        x={r:pulp.LpVariable(f'x_{i}',lowBound=0,cat='Integer') for i,r in enumerate(d.index)}
        y={r:pulp.LpVariable(f'y_{i}',lowBound=0) for i,r in enumerate(d.index)}
        for r in d.index:
            problem+=y[r]<=float(goal[r]);problem+=y[r]<=capacity*x[r]
        # Uniform cost permits this exact integer equivalent of the budget constraint.
        problem+=pulp.lpSum(x.values())<=max_runs
        objective=pulp.lpSum(float(weight[r])*y[r] for r in d.index)
        problem+=objective
        solver=pulp.PULP_CBC_CMD(msg=False,timeLimit=20)
        def solve():
            try:
                problem.solve(solver)
            except pulp.PulpSolverError as exc:
                raise RuntimeError('로컬 CBC 최적화 엔진을 실행할 수 없습니다. README의 CBC 설치 점검을 확인하세요.') from exc
        solve()
        if pulp.LpStatus[problem.status]!='Optimal':raise RuntimeError('최적해를 찾지 못했습니다. 예산과 입력 자료를 확인하세요.')
        # Resolve zero-weight degeneracy by minimizing runs at the same primary optimum.
        optimum=float(pulp.value(objective))
        problem+=objective>=optimum-1e-7
        problem.sense=pulp.LpMinimize
        problem.setObjective(pulp.lpSum(x.values()))
        solve()
        if pulp.LpStatus[problem.status]!='Optimal':raise RuntimeError('최적 배분의 운영횟수 정리에 실패했습니다.')
        served=pd.Series({r:max(0.,float(y[r].value() or 0)) for r in d.index})
        served=served.clip(upper=goal)
    allocation=np.ceil(np.maximum(served-1e-7,0)/capacity).astype(int)
    total=int(allocation.sum())
    pop_alloc=proportional(total,d['인구'])
    existing=d['기존참여_2023'];positive=existing[existing>0]
    if positive.empty:raise ValueError('기존 참여실적 자료가 없습니다.')
    status_key=existing.replace(0,positive.min()*.1)  # original reference rule
    status_alloc=proportional(total,status_key)
    pop_served=np.minimum(goal,capacity*pop_alloc);status_served=np.minimum(goal,capacity*status_alloc)
    coverage=lambda s:float((d['우선확충지수']*s).sum())
    coverage_ai,coverage_pop,coverage_status=map(coverage,[served,pop_served,status_served])
    improvement=lambda b:None if b<=0 else (coverage_ai/b-1)*100
    table=d.copy();table['목표인원']=goal;table['운영횟수']=allocation;table['충족인원']=served
    table['사용예산_원']=allocation*cost_per_run
    table['인구비례_횟수']=pop_alloc;table['기존참여비례_횟수']=status_alloc
    used=float(total*cost_per_run)
    if used>budget+1e-5:raise RuntimeError('배분 예산 검증에 실패했습니다.')
    if not ((served<=goal+1e-5)&(served<=capacity*allocation+1e-5)).all():raise RuntimeError('지역 수용력 검증에 실패했습니다.')
    return dict(allocation=table.reset_index(),budget_used=used,budget_utilization=used/budget*100 if budget else 0,
                total_runs=total,staff_run_units=total*const['staff_per_run'],total_served=float(served.sum()),
                target_people=float(goal.sum()),weighted_objective=float((weight*served).sum()),
                coverage=coverage_ai,population_improvement=improvement(coverage_pop),status_quo_improvement=improvement(coverage_status),
                comparison=pd.DataFrame({'방식':['AI Need-based Optimization','단순 인구비례','기존 참여비례 Status Quo'],
                    '격차가중 커버리지':[coverage_ai,coverage_pop,coverage_status],
                    '운영횟수':[total,total,total]}),
                assumptions=dict(budget=budget,cost_per_run=cost_per_run,capacity=capacity,target_rate=target_rate))
