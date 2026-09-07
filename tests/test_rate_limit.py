from src.api.rate_limit import is_allowed, reset, WINDOW_SECONDS, MAX_REQUESTS_PER_WINDOW


def setup_function():
    reset()


def test_allows_requests_up_to_the_limit():
    for _ in range(MAX_REQUESTS_PER_WINDOW):
        assert is_allowed("1.2.3.4") is True


def test_blocks_the_request_past_the_limit():
    for _ in range(MAX_REQUESTS_PER_WINDOW):
        is_allowed("1.2.3.4")
    assert is_allowed("1.2.3.4") is False


def test_tracks_clients_independently():
    for _ in range(MAX_REQUESTS_PER_WINDOW):
        is_allowed("1.2.3.4")
    assert is_allowed("5.6.7.8") is True


def test_old_requests_age_out_of_the_window():
    now = 1_000_000.0
    for _ in range(MAX_REQUESTS_PER_WINDOW):
        is_allowed("1.2.3.4", now=now)
    assert is_allowed("1.2.3.4", now=now + 1) is False
    assert is_allowed("1.2.3.4", now=now + WINDOW_SECONDS + 1) is True
