## 신용카드 이탈 예측: 결측치 처리 방법과 모델 성능 비교를 통한 서비스 개선 (Predicting Credit Card Churn: Handling Missing Data and Comparing Model Performances for Service Improvement)

2024.04.13 ~ 2024.04.30

## **프로젝트 요약**

### **목표**

이 프로젝트의 목표는 신용카드 고객 이탈 예측 모델을 개발하여, 고객의 이탈 가능성을 사전에 예측하고 이를 방지하기 위한 맞춤형 전략을 수립하는 것입니다.

### **데이터 및 전처리**

- 이 데이터셋은 신용카드 회사의 고객 정보를 포함하고 있으며, 각 행에는 고객 ID와 나이, 연봉, 결혼 여부, 신용카드 한도, 신용카드 카테고리 등과 같은 다양한 특성이 포함되어 있습니다.
- “Unknown” 데이터 처리를 위해 다섯 가지 방법을 적용하여 데이터를 전처리하고, 각각의 전처리된 데이터에 대해 모델을 학습시켜 성능을 비교하였습니다.

<img src="https://github.com/PARKYUNSU/chun/assets/125172299/b8ca27ff-5e9e-4c4d-9478-c6f28d18c5ea" width="500">




### **전처리 방법**

다섯 가지 결측치 처리 방법을 적용하여 다섯 개의 데이터프레임(df1 ~ df5)을 생성하였습니다.

| 데이터프레임 | 방법 | 데이터 수 |
| --- | --- | --- |
| df1 | 최빈값 대체 | 10,127 |
| df2 | 완전삭제 | 7,081 |
| df3 | Hot-Deck 방법 사용 | 10,127 |
| df4 | KNN 기법 사용 | 10,127 |
| df5 | 결측값 사용 | 10,127 |

### 데이터 준비 및 스케일링

**다중공선성 평가**

VIF가 높고 공차가 낮은 컬럼 제거: 'Customer_Age', 'Months_on_book', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Trans_Ct'

| Feature | 최대 VIF | 최소 VIF | 최대 공차 | 최소 공차 |
| --- | --- | --- | --- | --- |
| Credit_Limit | inf | inf | 0.00 | 0.00 |
| Total_Revolving_Bal | inf | inf | 0.00 | 0.00 |
| Avg_Open_To_Buy | inf | inf | 0.00 | 0.00 |
| Customer_Age | 80.17 | 78.09 | 0.01 | 0.01 |
| Months_on_book | 56.97 | 56.45 | 0.02 | 0.02 |
| Total_Trans_Ct | 23.86 | 23.68 | 0.04 | 0.04 |
| Card_Category | 14.95 | 14.50 | 0.07 | 0.07 |
| Total_Amt_Chng_Q4_Q1 | 14.46 | 14.34 | 0.07 | 0.07 |
| Income_Category * | 12.70 | 8.62 | 0.12 | 0.08 |
| Total_Ct_Chng_Q4_Q1 | 11.84 | 11.82 | 0.08 | 0.08 |

**PCA 평가**

<div style="display:flex;">
    <div style="text-align:center; margin-right:20px;">
        <img src="https://github.com/PARKYUNSU/chun/assets/125172299/9fcb48f3-b31c-48a0-ae46-0a379007cfbe" width="400">
        <p>df1 (최빈값 대체)</p>
    </div>
    <div style="text-align:center; margin-right:20px;">
        <img src="https://github.com/PARKYUNSU/chun/assets/125172299/22230ff9-244f-4c79-b23f-9672012201c9" width="400">
        <p>완전삭제</p>
    </div>
    <div style="text-align:center;">
        <img src="https://github.com/PARKYUNSU/chun/assets/125172299/32af11dd-8ce3-44bf-b1c8-b7094161f4af" width="400">
        <p>Hot-Deck 방법 사용</p>
    </div>
</div>
<div style="display:flex;">
    <div style="text-align:center; margin-right:20px;">
        <img src="https://github.com/PARKYUNSU/chun/assets/125172299/8d75e08b-9597-4f9a-acb6-602728c40343" width="400">
        <p>KNN 기법 사용</p>
    </div>
    <div style="text-align:center;">
        <img src="https://github.com/PARKYUNSU/chun/assets/125172299/5a434f52-6645-492e-8786-3ae0d677c1af" width="400">
        <p>결측값 사용</p>
    </div>
</div>

각 곡선은 df1 부터 df5 까지 13개 차원 분산 중 첫 번째 N개의 구성 요소에 얼마나 많은 분산이 포함되어 있는지 나타냅니다.

1. PCA Standard Scaler - 첫 10개의 구성 요소가 전체 분산의 90%를 포함하고 있습니다.
2. PCA Min-Max - MinMax를 사용하여 스케일링한 후 PCA를 수행하여, 첫 8개의 구성 요소가 전체 분산의 90%를 포함하고 있습니다.
3. PCA Power Transformer - Power Transformer를 사용하여 스케일링한 후 PCA를 수행하여, 첫 10개의 구성 요소가 전체 분산의 90%를 포함하고 있습니다.

### **모델링 및 성능 비교**

**데이터 불균형 해소**

SMOTE (Synthetic Minority Over-sampling Technique)
다수 클래스에 치우쳐 학습되는 현상인 불균형 문제를 해결하기 위한 오버샘플링 기법 모델 SMOTE 사용

각 데이터프레임에 대해 다섯 가지 모델(KNN, RFC, XGBC, LR, SVC)을 학습시키고, 성능을 평가하였습니다.

**모델 성능 평가 지표**

- Accuracy(정확도)
- Precision(클래스 1에 대한 정밀도)
- Recall(클래스 1에 대한 재현율)
- F1-score(평균 F1 점수)
- AUC(Area Under the ROC Curve)

### **최빈값 대체 (df1)**

| 모델 | 정확도 | 정밀도 | 재현율 | F1 점수 | AUC |
| --- | --- | --- | --- | --- | --- |
| KNN | 0.85 | 0.80 | 0.88 | 0.84 | 0.91 |
| RFC | 0.91 | 0.88 | 0.91 | 0.89 | 0.94 |
| XGBC | 0.90 | 0.87 | 0.90 | 0.88 | 0.93 |
| LR | 0.84 | 0.79 | 0.86 | 0.82 | 0.89 |
| SVC | 0.87 | 0.83 | 0.89 | 0.86 | 0.92 |

### **완전 삭제 (df2)**

| 모델 | 정확도 | 정밀도 | 재현율 | F1 점수 | AUC |
| --- | --- | --- | --- | --- | --- |
| KNN | 0.88 | 0.84 | 0.89 | 0.86 | 0.92 |
| RFC | 0.90 | 0.87 | 0.90 | 0.88 | 0.93 |
| XGBC | 0.89 | 0.86 | 0.89 | 0.87 | 0.92 |
| LR | 0.86 | 0.81 | 0.88 | 0.84 | 0.90 |
| SVC | 0.87 | 0.83 | 0.89 | 0.86 | 0.92 |

### **Hot-Deck 방법 사용 (df3)**

| 모델 | 정확도 | 정밀도 | 재현율 | F1 점수 | AUC |
| --- | --- | --- | --- | --- | --- |
| KNN | 0.89 | 0.85 | 0.90 | 0.87 | 0.92 |
| RFC | 0.91 | 0.88 | 0.91 | 0.89 | 0.94 |
| XGBC | 0.90 | 0.87 | 0.90 | 0.88 | 0.93 |
| LR | 0.85 | 0.80 | 0.86 | 0.83 | 0.90 |
| SVC | 0.87 | 0.83 | 0.89 | 0.86 | 0.92 |

### **KNN 기법 사용 (df4)**

| 모델 | 정확도 | 정밀도 | 재현율 | F1 점수 | AUC |
| --- | --- | --- | --- | --- | --- |
| KNN | 0.89 | 0.85 | 0.90 | 0.87 | 0.92 |
| RFC | 0.91 | 0.88 | 0.91 | 0.89 | 0.94 |
| XGBC | 0.90 | 0.87 | 0.90 | 0.88 | 0.93 |
| LR | 0.84 | 0.79 | 0.86 | 0.82 | 0.89 |
| SVC | 0.87 | 0.83 | 0.89 | 0.86 | 0.92 |

### **결측값 사용 (df5)**

| 모델 | 정확도 | 정밀도 | 재현율 | F1 점수 | AUC |
| --- | --- | --- | --- | --- | --- |
| KNN | 0.89 | 0.85 | 0.90 | 0.87 | 0.92 |
| RFC | 0.91 | 0.88 | 0.91 | 0.89 | 0.94 |
| XGBC | 0.90 | 0.87 | 0.90 | 0.88 | 0.93 |
| LR | 0.84 | 0.79 | 0.86 | 0.82 | 0.89 |
| SVC | 0.87 | 0.83 | 0.89 | 0.86 | 0.92 |
- KNN 모델

df2 에서 가장 높은 Accuracy(0.78)와 F1-score(0.87)를 보임

Precision 및 Recall 면에서는 일관된 결과

- RFC 모델

df2 에서 가장 높은 Accuracy(0.98)와 F1-score(0.99)를 보임

Precision 및 Recall 면에서도 df2 에서 가장 높은 성능

- XGBC 모델

df2 에서 가장 높은 Accuracy(0.97)와 F1-score(0.98)를 보임

Precision 및 Recall 면에서도 df2 에서 가장 높은 성능

- Logistic Regression 모델

df5 에서 가장 높은 Accuracy(0.74)를 보임, 그러나 전체적으로 성능이 다소 낮음

다른 모델에 비해 Precision 및 Recall 면에서도 상대적으로 성능이 낮음

- SupportVectorMachine 모델

df5에서 가장 높은 Accuracy(0.81)와 F1-score(0.88)를 df5 보임

Precision 및 Recall 면에서도 df5 데이터 프레임에서 가장 높은 성능

**ROC 커브**


<img src="https://github.com/PARKYUNSU/chun/assets/125172299/3a961d26-f10b-40d5-9a57-afe3699fc9e2" alt="image_12" style="width:30%">
<img src="https://github.com/PARKYUNSU/chun/assets/125172299/192eb2db-f3a6-4cf7-a443-737d317eb0bc" alt="image_13" style="width:30%">

<img src="https://github.com/PARKYUNSU/chun/assets/125172299/314cba59-63b9-4cc5-9413-688d3faada9f" alt="image_14" style="width:30%">
<img src="https://github.com/PARKYUNSU/chun/assets/125172299/d8c1cda6-4aa0-4b7c-a45d-03e3b582badb" alt="image_15" style="width:30%">

<img src="https://github.com/PARKYUNSU/chun/assets/125172299/1d06d065-0c30-476c-a74f-bb534618be1e" alt="image_16" style="width:30%">


### **정리**

Random Forest Classifier(RFC)와 Extreme Gradient Boosting Classifier(XGBC) 모델이 df2 데이터 프레임에서 가장 좋은 성능을 보임

### **제공 서비스**

위의 방법으로 선택된 df2 (완전삭제) 데이터와 XGBoost 모델을 활용하여 어떤 요소가 고객의 이탈률에 얼마나 영향을 끼쳤는지를 확인하여 신용카드 회사의 서비스를 개선하고자 합니다.

XGBoost 특성 중요도 TOP10

<img src="https://github.com/PARKYUNSU/chun/assets/125172299/9543222b-c75b-4d92-bb7d-95595bc3d0ca" alt="image_16" style="width:60%">


SHAP로 본 특성 별 예측에 미치는 관계

<img src="https://github.com/PARKYUNSU/chun/assets/125172299/9c516a71-2563-48ed-8023-5a1de1247748" alt="image_16" style="width:60%">


### **해석**

**1. Total_Trans_Amt (총 거래 금액):**

가장 높은 중요도를 가진 특성으로, 고객의 총 거래 금액이 신용카드 이탈률에 가장 큰 영향을 미칩니다. 거래 금액이 높을수록 이탈 가능성이 낮아질 수 있습니다.

**2. Total_Amt_Chng_04_Q1 (1분기 대비 총 거래 금액 변동):**

1분기 대비 총 거래 금액의 변동이 두 번째로 높은 중요도를 가집니다. 이것은 최근 거래 활동의 변화가 이탈 가능성에 미치는 영향을 나타냅니다.

**3. Total_Ct_Chng_Q4_Q1 (4분기 대비 총 거래 건수 변동):**

4분기 대비 총 거래 건수의 변동이 세 번째로 높은 중요도를 가집니다. 이것은 최근 거래 활동의 변화가 이탈 가능성에 미치는 영향을 나타냅니다.

**4. Avg_Utilization_Ratio (평균 카드 이용률):**

평균 카드 이용률로 예를들어 카드 사용금액이 1,000 , 카드 한도 5,000 이면 이용률은 20%로 이다(계산방법 Total_Revolving_Bal / Credit_Limit *100)

신용카드의 평균 이용률이 이탈률에 미치는 영향을 나타냅니다. 이용률이 높을 수록 이탈 가능성이 낮아질 수 있습니다.

**5. Total_Relationship_Count (카드 보유 수):**

고객의 카드 보유 수가 많을수록 이탈할 가능성이 낮아질 수 있습니다.

### 서비스 기획

### 1. 고객에게 맞춤형 혜택 제공:

고객의 총 거래 금액이 매우 중요한 요소임을 고려할 때, 총 거래 금액에 따라 적립되는 포인트나 현금 환급, 할인 혜택 등을 제공하여 고객들에게 맞춤형 혜택을 제공할 수 있습니다.

2. 거래 활동에 따른 개인화된 마케팅 전략 구성:
거래 활동 변동 및 총 거래 건수 변동이 이탈 가능성에 큰 영향을 미친다는 것을 고려하여, 최근 거래 활동이 감소하는 고객에게는 특별한 혜택이나 할인을 제공하여 활동을 유도할 수 있습니다.

3. 고객들 신용 한도를 증가시키기:
카드 이용률 관리 및 교육 프로그램 제공:
평균 카드 이용률이 이탈 가능성에 큰 영향을 미친다는 것을 고려하여, 카드 이용률을 적절히 관리할 수 있는 교육 프로그램을 제공하거나, 카드 이용률이 높은 고객에게는 더 높은 신용 한도를 부여함으로써 해당 고객 그룹이 조직을 떠나는 가능성을 낮출 수 있을 것입니다.

4. 신규 고객 유치를 위한 마케팅 전략:
카드 보유 수가 이탈 가능성에 영향을 미친다는 것을 고려하여, 새로운 고객을 유치하기 위해 혜택이나 프로모션을 제공하고, 이를 통해 고객들을 계속 유지하도록 유도할 수 있습니다.

### **결과 및 해석**

- Random Forest Classifier(RFC)와 Extreme Gradient Boosting Classifier(XGBC) 모델이 모든 데이터프레임에서 가장 높은 성능을 보였습니다.
- 고객의 총 거래 금액, 거래 활동 변동, 카드 이용률 등이 이탈 가능성에 큰 영향을 미침을 확인하였습니다.
- 제공 서비스를 통해 고객에게 맞춤형 혜택을 제공하고, 개인화된 마케팅 전략을 수립하여 고객 이탈을 방지할 수 있습니다.
