import sys, os
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from supabase import create_client, Client
import finnhub
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
import pickle
from config import STOCK_LIST

base_dir = os.path.dirname(__file__)
parent_path = os.path.join(base_dir, '..')
sys.path.append(parent_path)

from source.input_processing import get_csv_data, merge_data
from source.sentiment_model import analyze_sentiment_with_progress, add_integer_column, sums_sentiment_score_for_7_days, update_sentiment_score_in_db
from source.lstm_model import get_scale_data, get_scale_data_with_fit, create_sequences_for_train, compile_model, train_model, predict_prices, create_sequences_for_prod
from source.output_processing import compare_prices_with_graph
from source.market_capitalization import get_capitalization

SEQUENCE_LENGTH = 7
SENTIMENT_WINDOW_DAYS = 7
FETCH_DAYS = 13

load_dotenv()

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_API_KEY = os.environ.get('SUPABASE_API_KEY')
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY')

supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

def get_price_data_from_db(stock_id):
  prices_latest_date_response = supabase.table('stock_prices').select('price_date') \
          .eq('stock_id', stock_id) \
          .order('price_date', desc=True) \
          .limit(1) \
          .single() \
          .execute()

  if not prices_latest_date_response.data:
      print(f"🚨 '{stock_id}'에 대한 최근의 주가 데이터가 DB에 없습니다.")

  prices_latest_date = prices_latest_date_response.data['price_date']

  prices_response = supabase.table('stock_prices').select('*') \
          .eq('stock_id', stock_id) \
          .lte('price_date', prices_latest_date) \
          .order('price_date', desc=True) \
          .limit(FETCH_DAYS) \
          .execute()

  return prices_response.data

def get_news_data_from_db(start_date, end_date, stock_id):

  news_response = supabase.table('news').select('*').eq('stock_id', stock_id) \
  .gte('created_date', start_date.strftime('%Y-%m-%d')) \
  .lte('created_date', end_date.strftime('%Y-%m-%d')) \
  .execute()

  if not news_response.data:
      print(f"🚨 '{stock_id}'에 대한 최근의 뉴스 데이터가 DB에 없습니다.")

  return news_response.data

def get_existing_model(stock_code):
    """모델 파일을 불러옵니다. 파일이 없으면 에러를 발생시킵니다."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    load_path = os.path.join(base_dir, 'models', f'{stock_code}_finn_model.keras')
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {load_path}")
    return load_model(load_path)

def get_existing_scaler(stock_code):
    """스케일러 파일을 불러옵니다. 파일이 없으면 에러를 발생시킵니다."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    load_path = os.path.join(base_dir, 'models', f'{stock_code}_finn_scaler.pkl')
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"스케일러 파일을 찾을 수 없습니다: {load_path}")
    with open(load_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def get_change_rate(prev_price, today_price):
    change_rate = ((today_price - prev_price) / prev_price) * 100
    return change_rate.round(2)

def get_closely_prev_close_price(df):
    # 2. 'close_price'가 NaN(비어있지 않은)이 아닌 행만 필터링합니다.
    valid_data_df = df.dropna(subset=['close_price'])

    # 3. 인덱스(date)를 기준으로 내림차순 정렬하여 가장 최신 데이터가 맨 위로 오게 합니다.
    sorted_df = valid_data_df.sort_index(ascending=False)

    latest_valid_row = sorted_df.iloc[0]
    
    # 인덱스가 날짜이므로, .name 속성으로 날짜를 가져옵니다.
    latest_date = latest_valid_row.name.strftime('%Y-%m-%d')
    latest_close_price = latest_valid_row['close_price']

    
    return latest_close_price
# Predictions row를 만들고, db에 저장(시가총액 정보도 호출하여 저장)
def save_predictions_in_db(stock_id, stock_code, company_name, prediction_price, prediction_date, change_rate, capitalization):
    try :
        response = supabase.table('predictions') \
        .upsert({"stock_id" : stock_id, "prediction_date": prediction_date, "stock_code" : stock_code,
                "company_name" : company_name, "prediction_price" : prediction_price, "change_rate" : change_rate,
                "capitalization" : capitalization}) \
        .execute()
        
        if hasattr(response, 'error') and response.error is not None:
                print(f"🚨 DB 업데이트 중 에러가 발생했습니다: {response.error}")
        else:
            print("✅ DB 업데이트가 성공적으로 완료되었습니다.")
    except Exception as e:
        print(f"🚨 DB 업데이트 중 예외 발생: {e}")

def run_prediction_for_stock(stock_code: str):
    """단일 주식 코드에 대한 전체 예측 파이프라인을 실행합니다."""
    print("\n" + "="*50)
    print(f"🚀 {stock_code} 예측 프로세스 시작")
    print("="*50)

    # 1. 모델 및 스케일러 로드
    model = get_existing_model(stock_code)
    scaler = get_existing_scaler(stock_code)

    # 2. DB에서 데이터 조회
    stock_response = supabase.table('stocks').select('id,stock_code,company_name').eq('stock_code', stock_code).single().execute()
    if not stock_response.data:
        raise ValueError(f"'{stock_code}' 정보를 'stocks' 테이블에서 찾을 수 없습니다.")
    
    stock_id = stock_response.data['id']
    company_name = stock_response.data['company_name']

    prices_data = get_price_data_from_db(stock_id)
    if not prices_data:
        raise ValueError(f"'{stock_code}'의 주가 데이터가 DB에 없습니다.")
    stock_prices_df = pd.DataFrame(prices_data)

    # 3. 데이터 검증 (길이 확인)
    if len(stock_prices_df) < FETCH_DAYS:
        raise ValueError(f"주가 데이터 부족: 예측에 필요한 {FETCH_DAYS}일치 데이터 중 {len(stock_prices_df)}일치만 조회됨.")
    
    # ... (이후 데이터 처리 로직은 기존과 동일) ...
    stock_prices_df = stock_prices_df.rename(columns={'price_date':'date', 'id' : 'stock_price_id'})
    start_date = pd.to_datetime(stock_prices_df['date']).min()
    end_date = pd.to_datetime(stock_prices_df['date']).max()

    news_response = get_news_data_from_db(start_date, end_date, stock_id)
    news_df = pd.DataFrame(news_response)
    news_df = news_df.rename(columns={'created_date':'date', 'id' : 'news_id'})
    # 감정평가 수행 후, sentiment_score를 db news 테이블에 새롭게 업데이트한다.
    news_df = analyze_sentiment_with_progress(news_df)

    merged_df = merge_data(news_df, stock_prices_df)
    merged_df['sentiment_influence'] = 0.0
    merged_df = add_integer_column(merged_df)

    sums_sentiment_score_for_7_days(merged_df)
    update_sentiment_score_in_db(supabase, merged_df)
    
    features = ['sentiment_influence', 'open', 'high', 'low', 'adjClose', 'volume']
    target = 'close'
    all_cols = features + [target]

    dropped_merged_df = merged_df.rename(columns={'open_price':'open', 'high_price' : 'high', 'low_price' : 'low', 'close_price' : 'close', 'adj_close_price' : 'adjClose'})
    dropped_merged_df = dropped_merged_df[ features + [target] ].dropna()
    
    # 4. 최종 입력 데이터 검증
    # dropped_merged_df 생성...
    if len(dropped_merged_df) < SEQUENCE_LENGTH:
        raise ValueError(f"최종 데이터 부족: 모델 입력에 필요한 {SEQUENCE_LENGTH}일치 데이터 생성 실패 ({len(dropped_merged_df)}일치만 생성됨).")
        
    # 5. 스케일링 및 예측
    scaled = get_scale_data(scaler, dropped_merged_df)
    X = create_sequences_for_prod(scaled)
    
    y_pred_scaled = predict_prices(model, X)
    all_cols = ['sentiment_influence', 'open', 'high', 'low', 'adjClose', 'volume', 'close']
    features = all_cols[:-1] # 'close'를 제외한 모든 컬럼
    target = 'close'
    target_col_index = all_cols.index(target)

    num_features = len(features)
    dummy_array = np.zeros((len(y_pred_scaled), len(all_cols)))

    # 'close' 위치(6번 인덱스)에 예측된 값을 삽입
    dummy_array[:, target_col_index] = y_pred_scaled.ravel()
    # Scaler를 이용해 전체 배열을 역변환하고, 'close' 컬럼만 추출
    y_pred_actual = scaler.inverse_transform(dummy_array)[:, target_col_index]

    next_day_predicted_close = y_pred_actual[-1].round(4)
    closely_prev_price = get_closely_prev_close_price(merged_df)
    change_rate = get_change_rate(closely_prev_price, next_day_predicted_close)

    print(f"예측된 실제 종가: ${next_day_predicted_close:.4f}")
    capitalization = get_capitalization(finnhub_client, stock_code)
    today_date = datetime.now().strftime("%Y-%m-%d")
    save_predictions_in_db(stock_id, stock_code, company_name, next_day_predicted_close, today_date, change_rate, capitalization)
    print(f"✅ {stock_code} 예측 및 저장 완료.")


def main():
    """STOCK_LIST에 있는 모든 주식에 대해 예측을 순차적으로 실행합니다."""
    print("===== 전체 주가 예측 작업을 시작합니다 =====")
    
    # config.py 에 정의된 주식 리스트
    # STOCK_LIST = ['AAPL', 'TSLA', 'GOOGL', 'NON_EXISTENT_STOCK'] 
    
    for stock_code in STOCK_LIST:
        try:
            # 각 주식에 대한 예측 파이프라인 실행
            run_prediction_for_stock(stock_code)
        
        except FileNotFoundError as e:
            # 모델이나 스케일러 파일이 없는 경우
            print(f"🚨 [에러] {stock_code}: 처리 건너뜀 (원인: {e})")
        
        except ValueError as e:
            # 데이터가 부족하거나 찾을 수 없는 경우
            print(f"🚨 [에러] {stock_code}: 처리 건너뜀 (원인: {e})")
        
        except Exception as e:
            # 그 외 예상치 못한 모든 에러
            import traceback
            print(f"🚨 [심각] {stock_code}: 처리 중 알 수 없는 에러 발생. 다음 종목으로 넘어갑니다.")
            # 상세한 에러 로그를 보고 싶을 때 아래 주석 해제
            # traceback.print_exc()

    print("\n===== 모든 주가 예측 작업이 완료되었습니다 =====")


if __name__ == '__main__':
    main()