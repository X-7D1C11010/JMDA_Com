"""Network failures are simulated; no real weights are fetched."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from tools import download_official_uda_weights as weights


class Response(io.BytesIO):
    def __init__(self, body, status=200, **headers):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body)), **headers}

    def getcode(self):
        return self.status


class WeightRetryChecks(unittest.TestCase):
    def spec(self, body=b"complete fixture"):
        return {"id": "fixture", "method": "fixture", "url": "https://example.invalid/fixture.pth",
                "relative_path": "fixture.pth", "sha256": hashlib.sha256(body).hexdigest()}

    def test_offline_verification_rejects_corruption_without_network_access(self):
        spec = self.spec()
        with tempfile.TemporaryDirectory() as name, patch.dict(weights.WEIGHTS_BY_ID, fixture=spec), \
                patch.object(weights.urllib.request, "urlopen") as network:
            root = Path(name)
            with self.assertRaises(FileNotFoundError):
                weights.require_weight("fixture", root)
            (root/"fixture.pth").write_bytes(b"complete fixture")
            self.assertEqual(Path(weights.require_weight("fixture", root)), root/"fixture.pth")
            (root/"fixture.pth").write_bytes(b"corrupt")
            with self.assertRaises(FileNotFoundError):
                weights.require_weight("fixture", root)
            network.assert_not_called()

    def test_HTTP_range_resumes_the_existing_prefix(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/"file.part"; path.write_bytes(b"abc")
            response = Response(b"def", 206, **{"Content-Range": "bytes 3-5/6"})
            with patch.object(weights, "certifi_ssl_context", return_value=object()), \
                    patch.object(weights.urllib.request, "urlopen", return_value=response) as network:
                weights.download_https("https://example.invalid/x", path, timeout=7)
            request = network.call_args.args[0]
            self.assertEqual(request.get_header("Range"), "bytes=3-")
            self.assertEqual(network.call_args.kwargs["timeout"], 7)
            self.assertEqual(path.read_bytes(), b"abcdef")

    def test_server_ignoring_Range_restarts_instead_of_corrupting_file(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/"file.part"; path.write_bytes(b"old partial")
            with patch.object(weights, "certifi_ssl_context", return_value=object()), \
                    patch.object(weights.urllib.request, "urlopen", return_value=Response(b"abcdef")):
                weights.download_https("https://example.invalid/x", path)
            self.assertEqual(path.read_bytes(), b"abcdef")

    def test_invalid_Content_Range_does_not_modify_partial_file(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/"file.part"; path.write_bytes(b"abc")
            response = Response(b"def", 206, **{"Content-Range": "bytes 0-2/6"})
            with patch.object(weights, "certifi_ssl_context", return_value=object()), \
                    patch.object(weights.urllib.request, "urlopen", return_value=response):
                with self.assertRaisesRegex(OSError, "Content-Range"):
                    weights.download_https("https://example.invalid/x", path)
            self.assertEqual(path.read_bytes(), b"abc")

    def test_truncated_response_is_not_treated_as_success(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/"file.part"
            with patch.object(weights, "certifi_ssl_context", return_value=object()), \
                    patch.object(weights.urllib.request, "urlopen", return_value=Response(b"abc", **{"Content-Length": "6"})):
                with self.assertRaisesRegex(OSError, "Incomplete download"):
                    weights.download_https("https://example.invalid/x", path)
            self.assertEqual(path.read_bytes(), b"abc")

    def test_BrokenPipe_retries_preserves_partial_and_only_publishes_verified_file(self):
        calls = []
        def transfer(url, destination, timeout):
            path = Path(destination); calls.append(1)
            if len(calls) == 1:
                path.write_bytes(b"complete ")
                raise BrokenPipeError("simulated disconnect")
            self.assertEqual(path.read_bytes(), b"complete ")
            self.assertFalse(path.with_suffix("").exists())
            path.write_bytes(b"complete fixture")
        with tempfile.TemporaryDirectory() as name, \
                patch.object(weights, "cached_weight_candidates", return_value=[]), \
                patch.object(weights, "download_https", side_effect=transfer), \
                patch.object(weights.time, "sleep"):
            path = weights.download(self.spec(), name, retries=2)
            self.assertEqual(Path(path).read_bytes(), b"complete fixture")
            self.assertFalse(Path(path+".part").exists())
            self.assertEqual(len(calls), 2)

    def test_exhaustion_keeps_partial_for_the_next_invocation(self):
        def transfer(url, destination, timeout):
            Path(destination).write_bytes(b"complete ")
            raise TimeoutError("simulated read timeout")
        with tempfile.TemporaryDirectory() as name, \
                patch.object(weights, "cached_weight_candidates", return_value=[]), \
                patch.object(weights, "download_https", side_effect=transfer), patch.object(weights.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "2 attempts"):
                weights.download(self.spec(), name, retries=2)
            self.assertFalse((Path(name)/"fixture.pth").exists())
            self.assertEqual((Path(name)/"fixture.pth.part").read_bytes(), b"complete ")

    def test_complete_valid_partial_is_published_without_network(self):
        with tempfile.TemporaryDirectory() as name, \
                patch.object(weights, "cached_weight_candidates", return_value=[]), \
                patch.object(weights, "download_https") as network:
            (Path(name)/"fixture.pth.part").write_bytes(b"complete fixture")
            path = weights.download(self.spec(), name)
            self.assertTrue(weights.is_valid(path, self.spec()))
            network.assert_not_called()

    def test_bad_checksum_and_HTTP416_restart_from_zero(self):
        for failure in ("checksum", "416"):
            calls = []
            def transfer(url, destination, timeout):
                calls.append(1); path = Path(destination)
                if len(calls) == 1:
                    path.write_bytes(b"bad bytes")
                    if failure == "416":
                        raise urllib.error.HTTPError(url, 416, "bad range", {}, None)
                    return
                self.assertFalse(path.exists())
                path.write_bytes(b"complete fixture")
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as name, \
                    patch.object(weights, "cached_weight_candidates", return_value=[]), \
                    patch.object(weights, "download_https", side_effect=transfer), patch.object(weights.time, "sleep"):
                self.assertTrue(weights.is_valid(weights.download(self.spec(), name, retries=2), self.spec()))


if __name__ == "__main__":
    unittest.main()
