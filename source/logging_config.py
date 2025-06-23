import logging
import sys


def setup_logging():
    """
    표준 출력(stdout)으로 로그를 보내도록 로거를 설정합니다.
    systemd 저널이 이를 자동으로 캡처합니다.
    """
    # 이전에 설정된 핸들러가 있다면 중복 추가를 방지합니다.
    # getLogger()에 이름을 부여하여 다른 라이브러리의 로거와 분리합니다.
    logger = logging.getLogger("FINN_AI_MODEL")
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)  # INFO 레벨 이상의 로그만 출력

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
