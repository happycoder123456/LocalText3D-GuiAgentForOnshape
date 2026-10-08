from __future__ import annotations

import io
import unittest

from agent.loopback import (
    host_header_ok,
    is_loopback_host,
    origin_header_ok,
    read_http_body,
    request_is_local,
    require_loopback_bind,
    require_loopback_port,
    server_class_for,
)


class BindTests(unittest.TestCase):
    def test_loopback_bind_accepted(self):
        self.assertEqual(require_loopback_bind("127.0.0.1"), "127.0.0.1")
        self.assertEqual(require_loopback_bind("localhost"), "localhost")
        self.assertEqual(require_loopback_bind("::1"), "::1")
        self.assertEqual(require_loopback_bind("[::1]"), "::1")

    def test_non_loopback_bind_refused(self):
        for host in ("0.0.0.0", "192.168.1.10", "example.com", "10.0.0.5"):
            with self.assertRaises(ValueError):
                require_loopback_bind(host)

    def test_empty_bind_defaults_to_127(self):
        self.assertEqual(require_loopback_bind(""), "127.0.0.1")

    def test_port_zero_allowed(self):
        self.assertEqual(require_loopback_port(0), 0)
        self.assertEqual(require_loopback_port(8767), 8767)

    def test_bad_ports_refused(self):
        for port in (-1, 65536, "abc", None):
            with self.assertRaises(ValueError):
                require_loopback_port(port)  # type: ignore[arg-type]


class HeaderTests(unittest.TestCase):
    def test_is_loopback_host(self):
        self.assertTrue(is_loopback_host("127.0.0.1"))
        self.assertTrue(is_loopback_host("LOCALHOST."))
        self.assertTrue(is_loopback_host("[::1]"))
        self.assertFalse(is_loopback_host("evil.com"))
        self.assertFalse(is_loopback_host("127.0.0.1.evil.com"))

    def test_host_header(self):
        self.assertTrue(host_header_ok("127.0.0.1:8767"))
        self.assertTrue(host_header_ok("localhost:8767"))
        self.assertTrue(host_header_ok("[::1]:8767"))
        self.assertTrue(host_header_ok(""))  # HTTP/1.0
        self.assertFalse(host_header_ok("evil.com:8767"))
        self.assertFalse(host_header_ok("127.0.0.1.evil.com:8767"))

    def test_origin_header(self):
        self.assertTrue(origin_header_ok(""))
        self.assertTrue(origin_header_ok("http://127.0.0.1:8767"))
        self.assertTrue(origin_header_ok("http://localhost:8000"))
        self.assertFalse(origin_header_ok("https://evil.com"))
        self.assertFalse(origin_header_ok("http://evil.com"))

    def test_request_is_local(self):
        class H(dict):
            pass

        ok = H({"Host": "127.0.0.1:8767"})
        self.assertTrue(request_is_local(ok))
        bad = H({"Host": "evil.com"})
        self.assertFalse(request_is_local(bad))


class ServerClassTests(unittest.TestCase):
    def test_ipv4_unchanged(self):
        class Base:
            pass

        self.assertIs(server_class_for("127.0.0.1", Base), Base)

    def test_ipv6_gets_subclass(self):
        import socket

        class Base:
            address_family = socket.AF_INET

        cls = server_class_for("::1", Base)
        self.assertIsNot(cls, Base)
        self.assertEqual(cls.address_family, socket.AF_INET6)


class _Resp:
    def __init__(self, data: bytes, content_length: str | None = None):
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = content_length
        self._data = data

    def read(self, n: int) -> bytes:
        return self._data[:n]


class BodyCapTests(unittest.TestCase):
    def _resp(self, data: bytes, content_length: str | None = None):
        return _Resp(data, content_length)

    def test_reads_within_cap(self):
        body = read_http_body(self._resp(b"hello", "5"), 100)
        self.assertEqual(body, b"hello")

    def test_rejects_oversized_header(self):
        with self.assertRaises(ValueError):
            read_http_body(self._resp(b"x", "99999"), 10)

    def test_rejects_oversized_stream(self):
        with self.assertRaises(ValueError):
            read_http_body(self._resp(b"x" * 50, "50"), 10)

    def test_negative_cap_refused(self):
        with self.assertRaises(ValueError):
            read_http_body(self._resp(b""), -1)


if __name__ == "__main__":
    unittest.main()
