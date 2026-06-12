import requests

from config import TEACHER_BASE_URL, HEADERS


def main():
    url = f"{TEACHER_BASE_URL}/competition/evaluate"

    payload = {
        "document_received": True
    }

    print("[EVALUATE AGAIN] url =", url)
    print("[EVALUATE AGAIN] payload =", payload)

    try:
        res = requests.post(
            url,
            headers=HEADERS,
            json=payload,
            timeout=300
        )

        print("[EVALUATE AGAIN] status_code =", res.status_code)
        print("[EVALUATE AGAIN] response =", res.text)

    except requests.exceptions.ReadTimeout:
        print("[EVALUATE AGAIN] timeout, teacher có thể vẫn đang chấm.")

    except Exception as e:
        print("[EVALUATE AGAIN] error =", e)


if __name__ == "__main__":
    main()
