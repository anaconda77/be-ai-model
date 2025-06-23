import os
import subprocess  # 모델 실행 스크립트를 별도 프로세스로 실행하기 위함
import time

import oci
from oci.auth.signers import InstancePrincipalsSecurityTokenSigner

from source.logging_config import setup_logging

logger = setup_logging()


def run_model_pipeline():
    """AI 모델 실행 스크립트를 호출하는 함수"""
    logger.info("AI 모델 실행 파이프라인(main.py)을 서브프로세스로 시작합니다...")

    try:
        script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "main.py"
        )
        result = subprocess.run(
            ["/usr/bin/python3", script_path],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("모델 실행 파이프라인 서브프로세스가 성공적으로 완료되었습니다.")
        logger.info(f"실행 결과:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error("모델 실행 파이프라인 서브프로세스 실패:", exc_info=False)
        logger.error(f"Error Code: {e.returncode}")
        logger.error(f"Stderr:\n{e.stderr}")
        raise e
    except FileNotFoundError:
        logger.error("오류: poetry 또는 main.py 스크립트를 찾을 수 없습니다.")
        raise


def main():
    logger.info("리스너 서비스가 시작되었습니다. 큐에서 새 메시지를 기다립니다...")

    signer = InstancePrincipalsSecurityTokenSigner()

    queue_id = os.environ.get("QUEUE_ID")
    messages_endpoint = os.environ.get("QUEUE_ENDPOINT")

    queue_client = oci.queue.QueueClient(
        config={}, 
        signer=signer,
        service_endpoint=messages_endpoint
    )
    
    while True:
        try:
            # 큐에서 메시지를 가져오려고 시도 (메시지가 없으면 여기서 잠든 상태로 대기)
            get_messages_response = queue_client.get_messages(
                queue_id=queue_id,
                visibility_in_seconds=300  # 5분간 이 메시지를 다른 리스너가 못 가져가게 함
            )

            messages = get_messages_response.data.messages

            if messages:
                message = messages[0]
                logger.info(f"작업 신호 수신: {message.content}")

                # 모델 실행 파이프라인 호출
                run_model_pipeline()

                logger.info("작업 완료. 큐에서 메시지를 삭제합니다.")
                # 작업이 성공적으로 끝나면, 큐에서 메시지를 삭제하여 중복 실행 방지
                queue_client.delete_message(
                    queue_id=queue_id,
                    message_receipt=message.receipt
                )
            else:
                # 큐가 비어있으면 5초 대기 (네트워크 이슈 등 대비)
                time.sleep(5)

        except Exception as e:
            logger.error(f"리스너 메인 루프에서 치명적인 오류 발생: {e}", exc_info=True)
            time.sleep(60)  # 오류 발생 시 1분 대기 후 재시도


if __name__ == "__main__":
    main()
