import oci
import json
import time
import os
import subprocess # 모델 실행 스크립트를 별도 프로세스로 실행하기 위함

def run_model_pipeline():
    """AI 모델 실행 스크립트를 호출하는 함수"""
    print("AI 모델 실행 파이프라인을 시작합니다...")
    try:
        # run_model.py 스크립트가 같은 폴더에 있다고 가정
        # python3의 전체 경로를 사용하고, 스크립트의 전체 경로를 사용하는 것이 안정적
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_model.py')
        result = subprocess.run(['/usr/bin/python3', script_path], check=True, capture_output=True, text=True)
        print("모델 실행 성공:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("모델 실행 실패:")
        print(e.stderr)
        raise e
    except FileNotFoundError:
        print("오류: run_model.py 스크립트를 찾을 수 없습니다.")
        raise

def main():
    print("리스너 서비스 시작. OCI Queue에서 새 메시지를 기다립니다...")
    
    signer = oci.auth.signers.get_instance_principals_signer()
    queue_client = oci.queue.QueueClient(config={}, signer=signer)

    queue_id = os.environ.get("QUEUE_ID")
    messages_endpoint = os.environ.get("QUEUE_ENDPOINT")

    while True:
        try:
            # 큐에서 메시지를 가져오려고 시도 (메시지가 없으면 여기서 잠든 상태로 대기)
            get_messages_response = queue_client.get_messages(
                queue_id=queue_id,
                visibility_in_seconds=300, # 5분간 이 메시지를 다른 리스너가 못 가져가게 함
                endpoint=messages_endpoint
            )
            
            messages = get_messages_response.data.messages
            
            if messages:
                message = messages[0]
                print(f"작업 신호 수신: {message.content}")
                
                # 모델 실행 파이프라인 호출
                run_model_pipeline()

                print("작업 완료. 큐에서 메시지를 삭제합니다.")
                # 작업이 성공적으로 끝나면, 큐에서 메시지를 삭제하여 중복 실행 방지
                queue_client.delete_message(
                    queue_id=queue_id, 
                    message_receipt=message.receipt,
                    endpoint=messages_endpoint
                )
            else:
                # 큐가 비어있으면 5초 대기 (네트워크 이슈 등 대비)
                time.sleep(5)

        except Exception as e:
            print(f"리스너 실행 중 오류 발생: {e}")
            time.sleep(60) # 오류 발생 시 1분 대기 후 재시도

if __name__ == "__main__":
    # VM 환경 변수 설정 (이전 답변 참고하여 ~/.bashrc에 추가)
    # export QUEUE_ID=...
    # export QUEUE_ENDPOINT=...
    main()