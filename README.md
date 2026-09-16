# 🛡️ KcELECTRA 기반 한국어 스미싱 문자 분류 모델(KcELECTRA vs RoBERTa)

> 이 프로젝트는 문자 메시지의 문맥을 이해하여 정상 문자(광고/일상)와 스미싱 문자를 정교하게 분류하는 Transformer 기반 Sequence Classification 프로젝트입니다.

---

## 1. 개요 (Overview)

### 1.1 스미싱 분류 태스크의 특성

* **문맥 기반 판단의 필요성**: 스미싱 문자는 단순히 특정 키워드(`대출`, `국외발신`, `URL` 등)의 존재 여부만으로 판별하기 어렵고 전체적인 문맥 흐름을 파악해야 합니다.
* **모델 선정 이유**: Sequence Classification에 최적화된 Transformer Encoder 구조인 **KcELECTRA**를 채택하였습니다.

### 1.2 모델 비교 (Model Comparison)

| 비교 항목 | KcELECTRA-base (최종 채택) | KLUE/RoBERTa-base |
| --- | --- | --- |
| **주요 학습 데이터** | 네이버 뉴스 댓글/대댓글 (1.7억 건) | 뉴스, 위키, 나무위키, 청와대 국민청원 등 |
| **특화 분야** | 구어체, 비정형, 오탈자, 변형어, 신조어 | 정형체, 표준어, 긴 문장, 정교한 문법 |
| **스미싱 적합성** | **높음** (스미싱 특유의 변형 텍스트 대응 우수) | **높음** (문맥 이해 능력 우수) |

---

## 2. 데이터셋 구축 및 증강 (Dataset)

### 2.1 원천 데이터 수집

* **`jmjmjm3/kor-smishing-message`** (12,258개)
* 정상 메시지(`label 0`) 데이터만 추출하여 사용 (광고성 및 일상 문자 포함)

* **`meal-bbang/Korean_message`** (5,778개)
* 한국어 스팸 메시지 분류 데이터셋 중 정상 및 스미싱 레이블 활용
  
* 실제 스미싱 데이터(67,308개)

### 2.2 더미(Dummy) 데이터 생성 (Data Augmentation)

실제 수신 문자, 법률 상담 사례, 블로그 등에서 25개의 기본 정상 템플릿(택배, 금융, 경찰/검찰 등)을 수집한 뒤 변수를 생성하여 **25,000개**의 데이터를 확장 증강하였습니다.

#### 템플릿 예시

> 딩동♬
> **{user}**고객님!진심을 다하는 롯데택배입니다.**{user}**님께서 기다리시던 상품을 가지고 출발합니다.
> ■ 보내는 분(곳) : **{retail_shop}**
> ■ 상품명 : **{product_name}** - {rand1_1}개
> ■ 운송장번호 : **{shipment_number}**
> ■ 배송지 : **{address}**
> ■ 배송예정시간 : {time_hour}시
> ■ 배송점소 : **{delivery_location}**
> ■ 배송기사정보 : **{driver_name}** ☎010-{rand4_1}-{rand4_2}
> ▶ 수령장소 선정 및 실시간 배송정보
> `[http://mdm.alps.llogis.com:8200/openui/mdm/pages/mo/MoBef?inv=](http://mdm.alps.llogis.com:8200/openui/mdm/pages/mo/MoBef?inv=){shipment_number}&empno={lotte_empno}`
> 불편하신 점이나 추가 문의사항은 고객센터 ☎1588-2121 또는 롯데택배 챗봇 LODA로 연락주세요.

#### 변수 생성 규칙

* **개인정보 보호**: 실제 개인정보를 일절 사용하지 않고 `random`, `secrets`, 사전 정의 목록, 온톨로지 규칙에 따라 무작위 생성
* **다양성 확보**: 전화번호 형식(- 유무), 날짜 표현, 주소, 범죄 혐의, 관서명 등 표현 방식을 다변화하여 생성
* **생성 규모**: 템플릿당 1,000개씩 생성하여 **총 25,000개** 추가

### 2.3 Train 데이터 분포 (총 115,673개)

#### Label별 분포

* **Label 0 (정상)**: 48,365개 (41.81%)
* **Label 1 (스미싱)**: 67,308개 (58.19%)

#### Label 0 출처별 세부 비율

* `dummy` (증강 데이터): 25,000개
* `jmjmjm3/kor-smishing-message`: 12,258개
* `meal-bbang/Korean_message`: 5,778개
* `기타`: 5,329개

---

## 3. 실험 환경 및 학습 (Experiments & Training)

### 3.1 환경 설정

```yaml
Task Type: Binary Classification
Target Device: cuda:0 (cuda ver: 13.0)
Random Seed: 42
Output Directory: ./models/{base-model}/

```

### 3.2 학습 파라미터 (Fixed Options)

* **Batch Size**: 128 (Train / Eval per device)
* **Max Token Length**: 512
* **Evaluation & Save Strategy**: epoch

### 3.3 Optuna 하이퍼파라미터 튜닝
두 모델 모두 동일한 파라미터 탐색 공간 및 30회의 Optuna Trial 조건을 동일하게 적용하였습니다.

* **탐색 공간**:
* `num_train_epochs`: [2, 5]
* `learning_rate`: 1e-6 ~ 5e-5
* `weight_decay`: 0.001 ~ 0.1
* `warmup_ratio`: 0.001 ~ 0.1


* **시도 횟수**: 30 Trials (Pruning 적용)
* **Best Model 선정 기준**: Validation F1-Score

#### Best Trial (#13) Hyperparameters

* **Learning Rate**: `4.6945e-05`
* **Epochs**: `4`
* **Weight Decay**: `0.08187`
* **Warmup Ratio**: `0.00127`
* **Val F1**: `0.997619` | **Val ACC**: `0.995594`

---

## 4. 최종 테스트 결과 및 분석 (Evaluation & Analysis)

### 4.1 Best Model Test Metrics (`max=1`)

| Metric |	KcELECTRA-base (Best)	| KLUE/RoBERTa-base (Best)	| 성능 차이 (Diff) |
| --- | --- | --- | --- |
| Accuracy	| 0.9960	| 0.9950	| +0.0010 |
| F1-Score	| 0.9978	| 0.9973 |	+0.0005 |
| Precision	| 0.9982	| 0.9974 |	+0.0008 |
| Recall	| 0.9975	| 0.9973 |	+0.0002 |
| AUROC	| 0.9875	| 0.9821 |	+0.0054 |

### 4.2 오탐/미탐 분석 및 향후 개선 방향

* **오차 원인 분석**:
  * **FN (False Negative, 미탐)**: FN으로 탐지된 데이터 중 1개는 데이터셋 자체의 **Label 오류**로 확인.
  * **URL 단독 존재 취약성**: 레이블 오류를 제외한 35개 오탐 데이터 중 **23개가 URL만 단독으로 존재**하는 케이스.
  * **오픈채팅 유도**: 카카오톡 오픈채팅 URL 자체는 정상 도메인이므로 모델 단독으로 탐지하는 데 어려움 존재.


* **개선 방안 (Post-processing Logic Proposal)**:
  * 보안 특성상 FN(스미싱을 정상으로 오탐)을 최소화하는 것이 핵심 목표입니다.
  * 1차적으로 모델이 `0(정상)`으로 예측하더라도, **문자 내 URL 존재 여부를 후처리 로직에서 검증**하고, 화이트리스트 기반의 **정상 도메인 일치 여부를 교차 체크**하는 파이프라인 결합을 권장합니다.
