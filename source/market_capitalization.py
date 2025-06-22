# 무료 플랜 정책 상, 분 당 60회 호출 제한
# 현재 종목 수는 60개 미만이므로 상관없지만, 나중에 60개보다 종목이 많아지면 비동기 호출을 고려할 필요가 있어보임


def get_capitalization(finnhub_client, stock_code):
    response = finnhub_client.company_profile2(symbol=stock_code)
    return int(response["marketCapitalization"])
