import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ─── 모델 로딩 (한 번만 수행) ─────────────────────────────────────────────────
model_name = "ProsusAI/finbert"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(model_name)
# pipeline에 batch_size를 지정해 놓으면, 내부에서 자동 배치 처리가 됩니다.
finbert    = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=0,         # GPU가 있으면 0, 없으면 -1로 설정
    batch_size=16     # 한 번에 16개씩 처리
)

# ─── 배치+진행 바 버전 감성 분석 함수 ────────────────────────────────────────
def analyze_sentiment_with_progress(df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """
    df의 'title' 컬럼을 batch_size 단위로 묶어서 finbert pipeline에 넘기고,
    tqdm으로 진행 상태를 표시합니다. 반환되는 DataFrame은
    ['date','sentiment','confidence','sentiment_score'] 네 개 칼럼을 가집니다.
    """
    # 1) date, title만 남긴 복사본 만들기
    df2 = df[['date', 'title']].copy()
    n = len(df2)
    
    # 2) 결과를 담을 리스트 미리 생성
    all_labels = [None] * n
    all_scores = [None] * n
    
    # 3) 총 배치 개수 계산
    total_batches = int(np.ceil(n / batch_size))
    
    # 4) 인덱스별로 batch 처리하며 tqdm으로 진행 상황 표시
    for i in tqdm(range(total_batches), desc="감성 분석 진행중", unit="batch"):
        start_idx = i * batch_size
        end_idx   = min(start_idx + batch_size, n)
        
        texts = df2['title'].iloc[start_idx:end_idx].astype(str).tolist()
        results = finbert(texts)  # 리스트: [{ 'label': ..., 'score': ... }, ...]
        
        # 배치 결과를 미리 만든 리스트에 저장
        for j, res in enumerate(results, start=start_idx):
            all_labels[j] = res['label']
            all_scores[j] = round(res['score'], 4)
    
    # 5) 리스트를 칼럼에 할당
    df2['sentiment']  = all_labels
    df2['confidence'] = all_scores
    
    return df2

def add_integer_column(df):
    # 6) 정수형 점수 컬럼 추가
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_score'] = df['sentiment'].map(sentiment_map)
    
    # 7) date 인덱스로 설정하거나, 원한다면 리턴 전에 reset_index() 해도 됩니다
    df.set_index('date', inplace=True)
    return df
    
def sums_sentiment_score_for_7_days(df):
    weights = np.linspace(1.0, 0.1, 7)  # 1.0 → 0.1까지 7단계로 줄어듦
    df['sentiment_influence'] = 0.0
    
    for date, row in tqdm(df.iterrows(), total=len(df)):
        # neutral 또는 NaN은 건너뜀
        if pd.isna(row['sentiment_score']) or pd.isna(row['confidence']):
            continue

        base = row['sentiment_score'] * row['confidence']

        # 7일 동안 가중치 적용해서 누적
        for i in range(7):
            apply_date = date + pd.Timedelta(days=i)
            if apply_date in df.index:
                df.at[apply_date, 'sentiment_influence'] += base * weights[i]