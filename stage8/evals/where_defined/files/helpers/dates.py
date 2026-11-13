import datetime


def parse_date(text):
    return datetime.date.fromisoformat(text)
