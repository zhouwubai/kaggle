from datetime import datetime


def timedelta_str(str_a, str_b, formatter='%Y-%m-%d %H:%M:%S'):
    """
    calcuate the time difference between two times representing
    by str_a, str_b
    """
    datetime1 = datetime.strptime(str_a, formatter)
    datetime2 = datetime.strptime(str_b, formatter)
    timedelta = datetime2 - datetime1
    return timedelta.seconds


