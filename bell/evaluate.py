import requests

from config import TEACHER_BASE_URL, HEADERS


def main():
    url = f"{TEACHER_BASE_URL}/competition/evaluate"

    payload = {
        "document_received": False
    }

    print("[EVALUATE FIRST] url =", url)
    print("[EVALUATE FIRST] payload =", payload)

    try:
        res = requests.post(
            url,
            headers=HEADERS,
            json=payload,
            timeout=300
        )

        print("[EVALUATE FIRST] status_code =", res.status_code)
        print("[EVALUATE FIRST] response =", res.text)

    except requests.exceptions.ReadTimeout:
        print("[EVALUATE FIRST] timeout, teacher có thể vẫn đang gửi document/chấm.")

    except Exception as e:
        print("[EVALUATE FIRST] error =", e)


if __name__ == "__main__":
    main()
