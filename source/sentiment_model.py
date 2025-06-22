import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from . import exceptions

# ─── 모델 로딩 (한 번만 수행) ─────────────────────────────────────────────────
try:
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    finbert = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0,  # GPU가 있으면 0, 없으면 -1로 설정
        batch_size=16,
    )
except Exception as e:
    # 모델 로딩은 파이프라인의 필수 요소이므로, 실패 시 ModelLoadError를 발생시킵니다.
    raise exceptions.ModelLoadError(
        f"FinBERT 모델 로딩에 실패했습니다. (모델명: {model_name}): {e}"
    )


def filtering_none_score_date(df):
    return df[df["sentiment_score"].isnull()].copy()


def analyze_sentiment_with_progress(
    df: pd.DataFrame, batch_size: int = 32
) -> pd.DataFrame:
    try:  # [추가] 감성 분석 전체 과정을 try-except로 감쌉니다.
        df2 = df[["news_id", "date", "title"]].copy()
        n = len(df2)

        all_labels = [None] * n
        all_scores = [None] * n

        total_batches = int(np.ceil(n / batch_size))

        for i in tqdm(range(total_batches), desc="감성 분석 진행중", unit="batch"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n)

            texts = df2["title"].iloc[start_idx:end_idx].astype(str).tolist()
            results = finbert(texts)

            for j, res in enumerate(results, start=start_idx):
                all_labels[j] = res["label"]
                all_scores[j] = round(res["score"], 4)

        df2["sentiment"] = all_labels
        df2["confidence"] = all_scores

        return df2
    except Exception as e:
        raise exceptions.SentimentAnalysisError(
            f"FinBERT 파이프라인 실행 중 오류 발생: {e}"
        ) from e


# ─── 배치+진행 바 버전 감성 분석 함수 ────────────────────────────────────────
def analyze_sentiment_with_progress(
    df: pd.DataFrame, batch_size: int = 32
) -> pd.DataFrame:
    """
    df의 'title' 컬럼을 batch_size 단위로 묶어서 finbert pipeline에 넘기고,
    tqdm으로 진행 상태를 표시합니다. 반환되는 DataFrame은
    ['date','sentiment','confidence','sentiment_score'] 네 개 칼럼을 가집니다.
    """
    # filtering_df = filtering_none_score_date(df)

    # 1) date, title만 남긴 복사본 만들기
    df2 = df[["news_id", "date", "title"]].copy()
    n = len(df2)

    # 2) 결과를 담을 리스트 미리 생성
    all_labels = [None] * n
    all_scores = [None] * n

    # 3) 총 배치 개수 계산
    total_batches = int(np.ceil(n / batch_size))

    # 4) 인덱스별로 batch 처리하며 tqdm으로 진행 상황 표시
    for i in tqdm(range(total_batches), desc="감성 분석 진행중", unit="batch"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n)

        texts = df2["title"].iloc[start_idx:end_idx].astype(str).tolist()
        results = finbert(texts)  # 리스트: [{ 'label': ..., 'score': ... }, ...]

        # 배치 결과를 미리 만든 리스트에 저장
        for j, res in enumerate(results, start=start_idx):
            all_labels[j] = res["label"]
            all_scores[j] = round(res["score"], 4)

    # 5) 리스트를 칼럼에 할당
    df2["sentiment"] = all_labels
    df2["confidence"] = all_scores

    return df2


def add_integer_column(df):
    # 6) 정수형 점수 컬럼 추가
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["sentiment_score"] = df["sentiment"].map(sentiment_map)

    # 7) date 인덱스로 설정하거나, 원한다면 리턴 전에 reset_index() 해도 됩니다
    df.set_index("date", inplace=True)
    return df


def sums_sentiment_score_for_7_days(df):
    weights = np.linspace(1.0, 0.1, 7)  # 1.0 → 0.1까지 7단계로 줄어듦
    df["sentiment_influence"] = 0.0

    for date, row in tqdm(df.iterrows(), total=len(df)):
        # neutral 또는 NaN은 건너뜀
        if pd.isna(row["sentiment_score"]) or pd.isna(row["confidence"]):
            continue

        base = row["sentiment_score"] * row["confidence"]

        # 7일 동안 가중치 적용해서 누적
        for i in range(7):
            apply_date = date + pd.Timedelta(days=i)
            if apply_date in df.index:
                df.at[apply_date, "sentiment_influence"] += base * weights[i]


def update_sentiment_score_in_db(supabase_client, df_to_update):
    if df_to_update.empty:
        print("ℹ️ 감성 점수: DB에 업데이트할 데이터가 없습니다.")
        return

    if (
        "news_id" not in df_to_update.columns
        or "sentiment_score" not in df_to_update.columns
    ):
        # [수정] print 대신 예외를 발생시켜 잘못된 데이터가 넘어왔음을 알립니다.
        raise ValueError(
            "업데이트를 위해 DataFrame에 'news_id'와 'sentiment_score' 컬럼이 필요합니다."
        )

    print(f"🔄 {len(df_to_update)}개의 감성 점수를 DB에 업데이트합니다...")

    records_to_update = df_to_update[["news_id", "sentiment_score"]].dropna()

    if records_to_update.empty:
        print("ℹ️ 감성 점수: 유효한 업데이트 데이터가 없습니다.")
        return

    update_strings = []
    for index, row in records_to_update.iterrows():
        record_literal = f"({row['news_id']}, {row['sentiment_score']})"
        update_strings.append(record_literal)

    try:
        # [수정] "조용한 실패"를 막기 위해 예외를 상위로 던집니다.
        response = supabase_client.rpc(
            "update_batch_sentiment_scores", {"updates": update_strings}
        ).execute()

        if hasattr(response, "error") and response.error:
            raise exceptions.DatabaseError(f"DB 업데이트 실패: {response.error}")
        else:
            print("✅ 감성 점수: DB 업데이트가 성공적으로 완료되었습니다.")

    except Exception as e:
        if not isinstance(e, exceptions.DatabaseError):
            raise exceptions.DatabaseError(f"DB 업데이트 중 예외 발생: {e}") from e
        else:
            raise e
