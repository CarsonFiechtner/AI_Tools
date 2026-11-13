from settings.network import MAX_RETRIES, TIMEOUT


def fetch(url):
    for attempt in range(MAX_RETRIES):
        print(f"fetching {url}, attempt {attempt + 1}, timeout {TIMEOUT}")
