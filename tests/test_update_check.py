"""The update check: version comparison, feed validation, and that it stays silent and offline
when no feed is configured or anything goes wrong."""
import io
import json
import os
import unittest
from unittest import mock

from pose_craft import update_check


def _opener(payload, calls=None):
    def opener(request, timeout):
        if calls is not None:
            calls.append((request, timeout))
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        return io.BytesIO(data)
    return opener


class UpdateCheckTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {update_check.ENV_VAR: ""})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_versions_compare_numerically(self):
        self.assertTrue(update_check.is_newer("0.10.0", "0.9.9"))
        self.assertTrue(update_check.is_newer("v1.2", "1.1.9"))
        self.assertFalse(update_check.is_newer("1.2.0", "1.2"))
        self.assertFalse(update_check.is_newer("1.2.0-beta", "1.2.0"))
        with self.assertRaises(ValueError):
            update_check.parse_version("latest")

    def test_no_feed_means_no_request_and_no_thread(self):
        with mock.patch.object(update_check, "UPDATE_FEED_URL", ""):
            calls = []
            self.assertIsNone(update_check.check_for_update("1.0.0", opener=_opener({}, calls)))
            self.assertEqual(calls, [])
            self.assertIsNone(update_check.check_in_background("1.0.0", lambda info: None))

    def test_a_newer_release_is_reported(self):
        calls = []
        feed = {"version": "9.0.0", "url": "https://example.com/get", "notes": "New things"}
        info = update_check.check_for_update("1.0.0", url="https://example.com/feed.json", opener=_opener(feed, calls))
        self.assertEqual(info, update_check.UpdateInfo("9.0.0", "https://example.com/get", "New things"))
        request, timeout = calls[0]
        self.assertEqual(timeout, update_check.TIMEOUT_SECONDS)
        self.assertTrue(request.get_header("User-agent").endswith("/1.0.0"))

    def test_same_or_older_release_is_not_reported(self):
        feed = {"version": "1.0.0", "url": "https://example.com/get"}
        self.assertIsNone(update_check.check_for_update("1.0.0", url="https://e.com/f", opener=_opener(feed)))

    def test_bad_feeds_and_failures_are_silent(self):
        url = "https://example.com/feed.json"
        for payload in (b"not json", b"[]", {"version": "2.0"}, {"version": "2.0", "url": "http://insecure"},
                        {"version": "soon", "url": "https://e.com"}, b"{" + b" " * update_check.MAX_FEED_BYTES + b"}"):
            result = update_check.check_for_update("1.0.0", url=url, opener=_opener(payload))
            self.assertIsNone(result, repr(payload)[:40])

        def offline(request, timeout):
            raise OSError("offline")
        self.assertIsNone(update_check.check_for_update("1.0.0", url=url, opener=offline))

    def test_only_https_feeds_are_fetched(self):
        calls = []
        self.assertIsNone(update_check.check_for_update("1.0.0", url="http://example.com/f", opener=_opener({}, calls)))
        self.assertEqual(calls, [])

    def test_off_turns_the_check_off(self):
        with mock.patch.dict(os.environ, {update_check.ENV_VAR: "OFF"}):
            self.assertEqual(update_check.feed_url(), "")
            self.assertIsNone(update_check.check_in_background("1.0.0", lambda info: None))
        with mock.patch.dict(os.environ, {update_check.ENV_VAR: ""}):
            self.assertTrue(update_check.feed_url().startswith("https://github.com/ackermanl77/"))

    def test_environment_variable_overrides_the_built_in_feed(self):
        with mock.patch.dict(os.environ, {update_check.ENV_VAR: " https://example.com/test.json "}):
            self.assertEqual(update_check.feed_url(), "https://example.com/test.json")

    def test_background_check_calls_back_with_the_update(self):
        found = []
        info = update_check.UpdateInfo("9.0.0", "https://example.com/get")
        with mock.patch.dict(os.environ, {update_check.ENV_VAR: "https://example.com/feed.json"}), \
                mock.patch.object(update_check, "check_for_update", return_value=info):
            thread = update_check.check_in_background("1.0.0", found.append)
            thread.join(5)
        self.assertEqual(found, [info])


if __name__ == "__main__":
    unittest.main()
