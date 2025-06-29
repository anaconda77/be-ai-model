import os
import pickle
from datetime import datetime, timedelta

import finnhub
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from dotenv import load_dotenv
from supabase import Client, create_client
from tensorflow.keras.models import load_model
from tqdm import tqdm

from source import (
    exceptions,
    input_processing,
    lstm_model,
    market_capitalization,
    sentiment_model,
)
from source.config import STOCK_LIST
from source.logging_config import setup_logging

logger = setup_logging()

SEQUENCE_LENGTH = 7
SENTIMENT_WINDOW_DAYS = 7
FETCH_DAYS = 13
pandas_ts = pd.Timestamp.now(tz="Asia/Seoul")


def get_price_data_from_db(supabase, stock_id):
    try:
        prices_latest_date_response = (
            supabase.table("stock_prices")
            .select("price_date")
            .eq("stock_id", stock_id)
            .order("price_date", desc=True)
            .limit(1)
            .single()
            .execute()
        )
    except Exception as e:
        # [수정] .single() 에러 발생 시 DataFetchError를 발생시켜 파이프라인 중단
        raise exceptions.DataFetchError(
            f"'{stock_id}'에 대한 최근 주가 데이터 조회 실패 (DB에 데이터가 없거나 에러 발생): {e}"
        )

    prices_latest_date = prices_latest_date_response.data["price_date"]

    prices_response = (
        supabase.table("stock_prices")
        .select("*")
        .eq("stock_id", stock_id)
        .lte("price_date", prices_latest_date)
        .order("price_date", desc=True)
        .limit(FETCH_DAYS)
        .execute()
    )

    if not prices_response.data:
        raise exceptions.DataFetchError(
            f"'{stock_id}'에 대한 주가 상세 데이터가 DB에 없습니다."
        )

    return prices_response.data


def get_news_data_from_db(supabase, start_date, end_date, stock_id):

    news_response = (
        supabase.table("news")
        .select("*")
        .eq("stock_id", stock_id)
        .gte("published_date", start_date.strftime("%Y-%m-%d"))
        .lte("published_date", end_date.strftime("%Y-%m-%d"))
        .execute()
    )

    if not news_response.data:
        raise exceptions.DataFetchError(
            f"'{stock_id}'에 대한 뉴스 데이터가 DB에 없습니다. (기간: {start_date} ~ {end_date})"
        )

    return news_response.data


def get_existing_model(stock_code):
    """모델 파일을 불러옵니다. 파일이 없으면 에러를 발생시킵니다."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    load_path = os.path.join(base_dir, "models", f"{stock_code}_finn_model.keras")
    if not os.path.exists(load_path):
        # [수정] FileNotFoundError 대신 더 구체적인 ModelLoadError 사용
        raise exceptions.ModelLoadError(f"모델 파일을 찾을 수 없습니다: {load_path}")
    return load_model(load_path)


def get_existing_scaler(stock_code):
    """스케일러 파일을 불러옵니다. 파일이 없으면 에러를 발생시킵니다."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    load_path = os.path.join(base_dir, "models", f"{stock_code}_finn_scaler.pkl")
    if not os.path.exists(load_path):
        # [수정] FileNotFoundError 대신 더 구체적인 ModelLoadError 사용
        raise exceptions.ModelLoadError(
            f"스케일러 파일을 찾을 수 없습니다: {load_path}"
        )
    with open(load_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def get_change_rate(prev_price, today_price):
    change_rate = ((today_price - prev_price) / prev_price) * 100
    return change_rate.round(2)


def get_closely_prev_close_price(df):
    # 2. 'close_price'가 NaN(비어있지 않은)이 아닌 행만 필터링합니다.
    valid_data_df = df.dropna(subset=["close_price"])

    # 3. 인덱스(date)를 기준으로 내림차순 정렬하여 가장 최신 데이터가 맨 위로 오게 합니다.
    sorted_df = valid_data_df.sort_index(ascending=False)

    latest_valid_row = sorted_df.iloc[0]

    # 인덱스가 날짜이므로, .name 속성으로 날짜를 가져옵니다.
    latest_close_price = latest_valid_row["close_price"]

    return latest_close_price

def get_next_us_trading_day():
    """
    pandas-market-calendars를 사용해 KST 오늘을 기준으로
    가장 가까운 미국 증시(NYSE) 개장일을 'YYYY-MM-DD' 형식으로 반환합니다.
    """
    # NYSE 캘린더를 가져옵니다.
    nyse = mcal.get_calendar('NYSE')
    
    # 한국 시간 기준 오늘 날짜를 구합니다.
    today = datetime.now()
    
    # 오늘부터 향후 10일간의 개장일 스케줄을 조회합니다.
    schedule = nyse.schedule(start_date=today.strftime('%Y-%m-%d'), 
                             end_date=(today + timedelta(days=10)).strftime('%Y-%m-%d'))
    
    # 조회된 스케줄의 첫 번째 날짜가 바로 '가장 가까운 개장일'입니다.
    if not schedule.empty:
        next_trading_day = schedule.index[0].date()
        return next_trading_day.strftime('%Y-%m-%d')
    else:
        # 10일 안에 개장일이 없는 극단적인 경우에 대한 예외 처리
        raise exceptions.TradingDayFoundError("향후 10일 내에 개장일을 찾을 수 없습니다.")
    
# Predictions row를 만들고, db에 저장(시가총액 정보도 호출하여 저장)
def save_predictions_in_db(
    supabase,
    stock_id,
    stock_code,
    company_name,
    prediction_price,
    change_rate,
    capitalization,
):
    try:
        prediction_date = get_next_us_trading_day() # 가장 가까운 개장일 찾아옴
        response = (
            supabase.table("predictions")
            .upsert(
                {
                    "stock_id": stock_id,
                    "prediction_date": prediction_date,
                    "stock_code": stock_code,
                    "company_name": company_name,
                    "prediction_price": prediction_price,
                    "change_rate": change_rate,
                    "capitalization": capitalization,
                    "created_at": pandas_ts.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
            )
            .execute()
        )

        if hasattr(response, "error") and response.error is not None:
            logger.error(f"DB 업데이트 중 에러가 발생했습니다: {response.error}")
        else:
            logger.info("✅ DB 업데이트가 성공적으로 완료되었습니다.")
    except Exception as e:
        raise exceptions.DatabaseError(f"DB 업데이트 중 예외 발생: {e}") from e


def run_prediction_for_stock(supabase, finnhub_client, stock_code: str):
    """단일 주식 코드에 대한 전체 예측 파이프라인을 실행합니다."""
    logger.info("=" * 50)
    logger.info(f"🚀 {stock_code} 예측 프로세스 시작")
    logger.info("=" * 50)

    # 1. 모델 및 스케일러 로드
    model = get_existing_model(stock_code)
    scaler = get_existing_scaler(stock_code)

    # 2. DB에서 데이터 조회
    try:
        stock_response = (
            supabase.table("stocks")
            .select("id,stock_code,company_name")
            .eq("stock_code", stock_code)
            .single()
            .execute()
        )
    except Exception as e:
        # [추가] .single() 에러 처리
        raise exceptions.DataFetchError(
            f"'{stock_code}' 정보를 'stocks' 테이블에서 조회 실패: {e}"
        )

    if not stock_response.data:
        raise exceptions.DataFetchError(
            f"'{stock_code}' 정보를 'stocks' 테이블에서 찾을 수 없습니다."
        )

    stock_id = stock_response.data["id"]
    company_name = stock_response.data["company_name"]

    prices_data = get_price_data_from_db(supabase, stock_id)
    stock_prices_df = pd.DataFrame(prices_data)

    # 3. 데이터 검증 (길이 확인)
    if len(stock_prices_df) < FETCH_DAYS:
        raise exceptions.InsufficientDataError(
            f"주가 데이터 부족: 예측에 필요한 {FETCH_DAYS}일치 데이터 중 {len(stock_prices_df)}일치만 조회됨."
        )

    # ... (이후 데이터 처리 로직은 기존과 동일) ...
    stock_prices_df = stock_prices_df.rename(
        columns={"price_date": "date", "id": "stock_price_id"}
    )
    start_date = pd.to_datetime(stock_prices_df["date"]).min()
    end_date = pd.Timestamp.now(tz='Asia/Seoul') + timedelta(days=1)

    news_response = get_news_data_from_db(supabase, start_date, end_date, stock_id)
    news_df = pd.DataFrame(news_response)
    news_df = news_df.rename(columns={"published_date": "date", "id": "news_id"})
    # 감정평가 수행 후, sentiment_score를 db news 테이블에 새롭게 업데이트한다.
    news_df = sentiment_model.analyze_sentiment_with_progress(news_df)

    merged_df = input_processing.merge_data(news_df, stock_prices_df)
    merged_df["sentiment_influence"] = 0.0
    merged_df = sentiment_model.add_integer_column(merged_df)
    sentiment_model.sums_sentiment_score_for_7_days(merged_df)
    sentiment_model.update_sentiment_score_in_db(supabase, merged_df)

    features = ["sentiment_influence", "open", "high", "low", "adjClose", "volume"]
    target = "close"
    all_cols = features + [target]
    dropped_merged_df = merged_df.rename(
        columns={
            "open_price": "open",
            "high_price": "high",
            "low_price": "low",
            "close_price": "close",
            "adj_close_price": "adjClose",
        }
    )
    dropped_merged_df = dropped_merged_df[features + [target]].dropna()

    # 4. 최종 입력 데이터 검증
    if len(dropped_merged_df) < SEQUENCE_LENGTH:
        raise exceptions.InsufficientDataError(
            f"최종 데이터 부족: 모델 입력에 필요한 {SEQUENCE_LENGTH}일치 데이터 생성 실패 ({len(dropped_merged_df)}일치만 생성됨)."
        )

    # 5. 스케일링 및 예측
    scaled = lstm_model.get_scale_data(scaler, dropped_merged_df)
    X = lstm_model.create_sequences_for_prod(scaled)

    y_pred_scaled = lstm_model.predict_prices(model, X)
    all_cols = [
        "sentiment_influence",
        "open",
        "high",
        "low",
        "adjClose",
        "volume",
        "close",
    ]
    features = all_cols[:-1]  # 'close'를 제외한 모든 컬럼
    target = "close"
    target_col_index = all_cols.index(target)

    dummy_array = np.zeros((len(y_pred_scaled), len(all_cols)))

    # 'close' 위치(6번 인덱스)에 예측된 값을 삽입
    dummy_array[:, target_col_index] = y_pred_scaled.ravel()
    # Scaler를 이용해 전체 배열을 역변환하고, 'close' 컬럼만 추출
    y_pred_actual = scaler.inverse_transform(dummy_array)[:, target_col_index]

    next_day_predicted_close = y_pred_actual[-1].round(4)
    closely_prev_price = get_closely_prev_close_price(merged_df)
    change_rate = get_change_rate(closely_prev_price, next_day_predicted_close)
    logger.info(
        f"[{stock_code}] 모델 예측 완료. 예측 종가: ${next_day_predicted_close:.4f}"
    )

    logger.info(f"[{stock_code}] 시가총액 정보 조회 및 최종 결과 DB 저장 시작...")
    capitalization = market_capitalization.get_capitalization(
        finnhub_client, stock_code
    )
    save_predictions_in_db(
        supabase,
        stock_id,
        stock_code,
        company_name,
        next_day_predicted_close,
        change_rate,
        capitalization,
    )
    logger.info(f"[{stock_code}] 예측 프로세스 성공적으로 종료.")


def main():
    load_dotenv()

    try:
        SUPABASE_URL = os.environ.get("SUPABASE_URL")
        SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
        FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")

        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

        if not all([SUPABASE_URL, SUPABASE_KEY, FINNHUB_API_KEY]):
            raise exceptions.ConfigError(
                "필수 환경 변수(SUPABASE_URL, SUPABASE_KEY, FINNHUB_API_KEY)가 설정되지 않았습니다."
            )

        """STOCK_LIST에 있는 모든 주식에 대해 예측을 순차적으로 실행합니다."""
        logger.info("===== 전체 주가 예측 작업을 시작합니다 =====")

        for stock_code in STOCK_LIST:
            try:
                # 각 주식에 대한 예측 파이프라인 실행
                run_prediction_for_stock(supabase, finnhub_client, stock_code)
                tqdm.write(f"✅ {stock_code} 예측 및 저장 작업 완료.")

            except (
                exceptions.ModelLoadError,
                exceptions.DataFetchError,
                exceptions.InsufficientDataError,
                exceptions.DatabaseError,
            ) as e:
                tqdm.write(
                    f"🚨 [{e.__class__.__name__}] {stock_code}: 처리 건너뜀 (원인: {e})"
                )

            except Exception as e:
                # ⭐ [수정] traceback.print_exc() 대신 logger 사용
                tqdm.write(
                    f"🚨 [심각] {stock_code}: 처리 중 알 수 없는 에러 발생. 다음 종목으로 넘어갑니다. (원인: {e})"
                )
                logger.error(f"[{stock_code}] 예측 불가 오류 발생", exc_info=True)

    except Exception as e:
        logger.critical(
            f"메인 작업 실행 중 예측하지 못한 심각한 오류 발생: {e}", exc_info=True
        )

    logger.info("\n===== 모든 주가 예측 작업이 완료되었습니다 =====")


if __name__ == "__main__":
    main()
