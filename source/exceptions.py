class PredictionError(Exception):
    """예측 파이프라인 관련 최상위 예외 클래스"""

    pass


class ConfigError(PredictionError):
    """환경 변수 등 설정 관련 오류"""

    pass


class ModelLoadError(PredictionError):
    """모델 또는 스케일러 파일 로딩 실패 오류"""

    pass


class DataFetchError(PredictionError):
    """DB 또는 API로부터 데이터 조회 실패 오류"""

    pass


class InsufficientDataError(PredictionError):
    """학습 또는 예측에 필요한 데이터가 부족할 때의 오류"""

    pass


class DatabaseError(PredictionError):
    """DB에 데이터 저장/수정 실패 오류"""

    pass


class SentimentAnalysisError(PredictionError):
    """감성 분석 모델 실행 중 오류"""

    pass
