# 숲BTI · 해커톤 시연 대시보드

기존 GAP 활동추천 모델의 실제 추론, 지역 수요·공급 분석, ALIO 2023년 결산 63.21억원 기준 운영 최적화를 세 탭으로 제공합니다. 모델과 데이터는 로컬 파일입니다. 앱 실행 시 학습하거나 외부 API를 호출하지 않습니다.

기존 `/Users/minah/K-ds/forest_dashboard`에서 병행 편집 중인 파일이 발견되어 이 구현은 `hackathon/forest_dashboard`에 분리했습니다.

## 다른 컴퓨터에서 접속하기: 클라우드 배포

Streamlit Community Cloud에서 이 저장소를 배포하면 노트북을 꺼도 공유 주소로 접속할 수 있습니다. GitHub 저장소 업로드만으로 앱이 실행되지는 않으므로 처음 한 번은 Cloud에서 앱을 생성해야 합니다.

[배포 화면 바로 열기 — 저장소·브랜치·app.py 자동 입력](https://share.streamlit.io/deploy?repository=MinahYoo%2Fforest-welfare-dashboard&branch=main&mainModule=app.py)

1. [Streamlit Community Cloud](https://share.streamlit.io)에서 `MinahYoo` GitHub 계정으로 로그인합니다.
2. **Create app → Yup, I have an app**에서 아래 값을 지정합니다.

   | 항목 | 값 |
   |---|---|
   | Repository | `MinahYoo/forest-welfare-dashboard` |
   | Branch | `main` |
   | Main file path | `app.py` |
   | Python version (Advanced settings) | `3.12` |
   | App URL (사용 가능하면) | `forest-welfare-minah-2026` |

3. **Deploy**를 누릅니다. 학습 없이 패키지 설치 후 저장된 19개 모델을 불러옵니다. Secrets는 필요하지 않습니다.
4. 저장소는 비공개로 보관합니다. 다른 사람이 로그인 없이 이용하려면 Cloud 앱의 **Settings → Sharing → This app is public and searchable**를 선택합니다.
5. 배포가 끝난 뒤 Cloud가 표시한 실제 `https://….streamlit.app` 주소를 공유합니다. 위 App URL은 희망 이름이며 배포 전에는 접속 주소로 보장되지 않습니다.

모델 약 77MB와 실행 데이터가 저장소에 포함돼 있습니다. 실행 시 원본 Desktop 경로가 필요하지 않습니다. `.streamlit/config.toml`은 클라우드 프록시가 접속할 수 있게 `0.0.0.0`에 바인딩하고 설문 CSV 업로드를 5MB로 제한합니다. 기본 인증·XSRF 설정은 유지합니다. GitHub Actions는 Linux/Python 3.12에서 모델 로딩, 원본 수치 재현, 세 탭의 입력과 클릭을 검사합니다.

설정 근거: [공식 배포 안내](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [공개 앱 공유 설정](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app).

## 설치 및 실행

Python 3.10–3.12를 사용하세요. 이 컴퓨터에서는 `/opt/anaconda3/bin/python`의 설치된 패키지로 검증했습니다.

```bash
cd /Users/minah/K-ds/hackathon/forest_dashboard
python -m pip install -r requirements.txt
streamlit run app.py
```

이 컴퓨터에서 실행 환경을 명확히 지정하려면:

```bash
/opt/anaconda3/bin/python -m streamlit run app.py --server.port 8502
```

브라우저에서 `http://127.0.0.1:8502`를 엽니다. 다른 프로그램이 해당 포트를 사용하면 다른 포트 번호를 지정하세요.

대회 전에 의존성을 설치하고 `models/`와 `data/`를 포함한 폴더 전체를 복사하세요. 인터넷이 없는 현장에서는 패키지를 새로 설치하지 않아도 실행할 수 있습니다. PuLP의 CBC 실행 파일도 설치 환경에 포함돼 있어야 합니다. `.streamlit/config.toml`은 사용 통계 전송을 끕니다. 로컬 컴퓨터 내부에서만 열려면 실행 옵션에 `--server.address 127.0.0.1`을 추가하세요.

## 모델과 데이터 생성

이 작업은 앱 실행과 별개의 일회성 준비입니다. 제공된 모델·데이터가 있으면 다시 실행할 필요가 없습니다.

```bash
python extract_data.py --project '/Users/minah/Desktop/K-ds/예선/6. 한국산림복지진흥원'
python train_models.py --project '/Users/minah/Desktop/K-ds/예선/6. 한국산림복지진흥원' --threads 8
```

`train_models.py`는 기존 GAP 실험의 타깃·409열·depth6/360/Ordered/seed123과 기존 시설 서빙 빌더의 498열·depth7/880/Plain/seed42 설정을 재사용합니다. 두 모델 모두 원래의 `auto_class_weights=Balanced`, learning rate 0.05를 유지합니다. 새로운 모델 설계나 하이퍼파라미터 탐색을 하지 않습니다. 가구그룹 교차검증 성능은 기존 결과를 인용하고, 앱 배포용 모델만 전체 기존 학습자료에 한 번 적합합니다.

GAP 6개와 시설 13개의 `.cbm`은 `models/gap_activity/`, `models/facility/`에 저장됩니다. `schema.json`은 열 순서·범주 토큰·결측 규칙·학습 설정·입력 파일 해시를 보관합니다. `manifest.json`은 라벨 순서와 모델 해시를 기록합니다. 라벨별 저장·재로딩 예측 일치 검사를 수행하며 중단 후 재실행하면 검증된 체크포인트를 재사용합니다. `--kind gap_activity` 또는 `--kind facility`로 한 종류만 준비할 수도 있습니다.

`extract_data.py`는 원본 최적화 코드의 **입력 계산 부분만** 실행합니다. 원본 출력 파일이나 노트북을 변경하지 않습니다. 예산 증빙, 참여실적, 원자료, 공급 CSV로부터 반올림 전 최적화 가중치를 재현합니다. `optimization_reference.csv`는 검증용 기존 결과이며 런타임 최적해로 대체 출력하지 않습니다.

시군구 이름은 기존 `sgg_geo.json`이 있을 때 조사코드로 매핑합니다. 지도가 없어도 관측된 조사코드로 실행되며, 이름을 추측해 채우지 않습니다. 읍면동 명칭표는 프로젝트에서 발견되지 않아 관측 코드만 제공합니다.

## 입력과 점수의 의미

- **내 정보 입력:** 연령·성별·가구·혼인·소득·학력·직업·지역을 원래 설문 코드로 변환합니다. 연령은 기존 전처리와 같은 2025년 조사 기준입니다. 선택한 세부 당일/숙박 경험은 실제 Q10 코드로 반영합니다.
- 입력하지 않은 수치형은 0, 범주형은 `미상`으로 처리합니다. 이는 원 학습 전처리 규칙입니다. 간편 입력은 409/498개 설문 응답을 모두 확보한 것과 다르므로 **간편 입력 성능은 미검증**입니다. 답하지 않은 행동을 임의의 평균 응답으로 만들지 않습니다.
- 6대분류 경험 선택만으로 등산/트레킹 등 세부 문항의 답을 추측하지 않습니다. 이 선택은 최종 후보 제외에 사용하고, 세부 경험을 입력하면 Q10 모델 입력에도 반영합니다.
- **실제 조사 응답 시연:** 동일 응답자의 GAP·시설 전처리 행을 정렬해 로컬로 제공합니다. 실제 기존 자료를 사용하며 학습 표본 내 시연이므로 독립 테스트 정확도를 나타내지 않습니다.
- **전체 설문 CSV:** 전처리된 코드 이름의 1행을 불러옵니다. GAP 409열을 요구하며, 시설 498열까지 있으면 해당 모델에도 전체 입력이 제공됩니다. 원시 타깃 열이 들어 있어도 모델 스키마 밖의 열은 추론에 전달하지 않습니다.
- 경험한 활동은 입력 선택과 실제 Q10에 기록된 경험의 합집합으로 제외합니다. 6개 모두 경험했으면 결과를 억지로 채우지 않고 빈 후보 안내를 표시합니다.
- 차트의 0–100점은 `.cbm` 모델의 `predict_proba` 값에 100을 곱한 것입니다. 개인의 실제 참여확률로 보정된 값은 아닙니다. 점수나 순위를 임의 생성하지 않습니다.

GAP 참고 성능은 기존 가구그룹 5-fold OOF입니다. P@1 0.681, P@2 0.555, LRAP 0.843, Macro-F1 0.591, Macro PR-AUC는 fold 평균 0.585 / OOF 통합 약 0.582입니다. P@k는 GAP 양성 활동이 있는 5,579명만 포함하며 양성이 없는 2,047명을 제외합니다. 시설 Macro-F1 0.6604, LRAP 0.8270, P@2 0.7201입니다.

## 지역 분석과 운영 최적화

개인 GAP 예측, 지역 수요 집계, 예산 배분은 **연결된 의사결정 흐름**입니다. 지역 수요는 기존 Q20 설문 가중집계입니다. 앱에서 한 사람을 입력할 때마다 전국 수요가 바뀌거나 개인 GAP 예측을 전국 합계로 환산하는 구조가 아닙니다.

최종 원점수는 `z(미충족잠재수요율) + z(-log1p(자연휴양림 수용인원 / 인구 × 100000))`입니다. 차트와 평가에는 원점수를 0–100으로 환산한 지수를 쓰고 목적함수에는 다시 표준화한 z의 양수 부분을 씁니다. 원본과 같은 표본 표준편차(ddof=1)를 사용합니다. 접근성·실제 이용 수준은 보조 지표로 함께 표시합니다. 자연휴양림 공급밀도가 모든 시설 유형의 충분도를 의미하지는 않습니다.

`run_optimization(budget, cost_per_run, capacity, target_rate)`의 금액 단위는 **원**, 목표율은 비율(`0.01` = 1%)입니다.

```text
maximize  sum(max(z_r,0) * y_r)
subject to
  y_r <= population_r * target_rate
  y_r <= capacity * x_r
  sum(cost_per_run * x_r) <= budget
  x_r >= 0 integer, y_r >= 0
```

균일 회당 비용을 이용해 예산 제약을 총 운영횟수의 정수 상한으로 정확히 변환합니다. 주 목적함수 최적값을 유지하면서 운영횟수를 최소화하는 2차 계산으로 0 가중치 지역의 임의 배정을 방지합니다. 예산 사용률은 실제 반환 횟수에서 다시 계산합니다. 공급시설 신설 위치나 직원 배치 일정을 최적화하는 모형은 아닙니다.

인구비례·기존 참여비례 기준선은 **같은 총 운영횟수**를 최대잔여법으로 배분합니다. 기존 참여 0인 지역에는 원본 코드와 같은 `최소 양수 참여 × 0.1`의 작은 배분 기준을 줍니다. 세 방식 모두 동일한 0–100 우선확충지수와 목표 상한으로 격차가중 커버리지를 계산합니다.

동점 처리의 플랫폼 차이를 방지하기 위해 17개 잔여값을 Python float 객체로 정렬합니다. 수치와 최대잔여법은 유지하면서 Mac/ARM 원본의 일반 quicksort 순서를 Linux/x86에서도 사용합니다. [NumPy 정렬 구현](https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/npysort/quicksort.cpp)의 CPU별 수치 정렬 분기를 피하기 위한 처리입니다.

기본값은 63.21억원, 회당 3,321,271.69원(약 332만원), 정원 300명, 목표 인구 1%입니다. 1,208회·예산 63.47%·인구비례 +25.80%·기존 참여비례 +646.24%·인력-회차 2,416을 실제 계산으로 재현합니다. **인력-회차는 회당 2인 가정의 투입량으로 고유 직원 수나 FTE가 아닙니다.** 직원 수 제한을 두지 않으므로 실제 인력 병목을 입증한 결과로 해석하지 않습니다. 배정 충족인원도 실제 관측된 참여자 수와 다릅니다.

## 2–3분 시연 순서

1. 개인 맞춤 추천에서 정보를 입력하고 등산·트레킹형과 자연감상·산책형을 경험 목록에 선택합니다. 필요한 경우 세부 경험도 입력합니다.
2. 새로운 활동 추천 받기를 누르고 경험한 활동이 Top 3에 없음을 확인합니다. 전체 설문 추론은 실제 조사 응답 시연 모드에서 확인합니다.
3. 지역 수요 분석에서 서울과 강원을 비교합니다. 수요가 높아도 기존 공급이 많은 지역의 정책 판단은 다르다는 점을 설명합니다.
4. 운영 최적화에서 기본 63.21억원을 확인하고 최적 배분 실행을 누릅니다.
5. 지역별 운영횟수, 같은 총량의 기준선 비교, 예산 사용률과 필요 인력-회차를 설명합니다. 예산/목표율을 바꿔 다시 실행합니다.

## 검증 및 문제 해결

```bash
python -m unittest discover -s tests -v
```

실제 저장 모델의 입력 복원과 예측 일치, 64가지 경험 제외 조합, 0.3/0.5/1% 원본 최적화 결과, 예산·수용력 제약과 0 예산, Streamlit 클릭 시연 흐름을 검사합니다. 이 검사는 기능 검증이며 추천 정확도를 새로 측정하는 검증은 아닙니다.

2026-09-10 검증: 테스트 4개 그룹 모두 통과했습니다. 실제 Chrome에서도 세 탭의 추천·지역 분석·최적화 결과와 기본 KPI를 확인했으며 로컬 서버 health 응답은 `ok`였습니다. 확인 화면은 `screenshots/01_personal.png`, `02_regional.png`, `03_optimization.png`에 있습니다.

- 모델 누락/해시 불일치: `models/` 전체 복사 여부를 확인하고 일회성 저장 스크립트를 실행합니다. 앱이 임의 모델로 대체하지 않습니다.
- 데이터 누락: `extract_data.py --project ...`의 정본 경로와 예산 증빙·원자료 파일을 확인합니다.
- CBC 실행 오류: `python -c "import pulp; print(pulp.listSolvers(onlyAvailable=True))"`로 CBC를 확인합니다. 대회 전에 해당 OS의 PuLP 패키지와 solver를 준비하세요.
- 입력을 바꿨는데 결과가 숨겨짐: 이전 입력의 결과를 현재 결과처럼 보여주지 않기 위한 동작입니다. 추천/최적화 버튼을 다시 누릅니다.
- 시군구 명칭 미매칭: 명칭을 추정하지 않고 조사코드로 표시합니다. 지역 명칭표가 확보되면 `region_codes.csv`의 명칭만 보완할 수 있습니다.
- 초기 6.2억원·77회·+39.7% 결과는 이 앱의 입력/성과로 사용하지 않습니다.

## 파일 구성

`app.py`, `utils/{preprocessing,recommend,regional,optimize}.py`, `train_models.py`, `extract_data.py`, `models/{gap_activity,facility}/`, `data/`, `tests/`, `SOURCE_AUDIT.md`가 핵심입니다. `data/provenance.json`에 원본 파일 SHA-256을 저장합니다. `data/optimization_result.csv`는 준비 시 기본값으로 실제 계산한 결과이며 앱에서는 매번 요청 조건으로 다시 계산합니다.

Streamlit의 [탭](https://docs.streamlit.io/develop/api-reference/layout/st.tabs)으로 세 화면을 구성하고, [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)로 입력과 클릭 흐름을 검증합니다. 탭 변경만으로 최적화나 재학습이 실행되지 않도록 계산은 버튼 안에 둡니다.
