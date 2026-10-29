"""Opt-in --insecure-skip-tls-verify: private addresses only, pinned, loudly warned."""

import socket
import ssl

import pytest
from aiohttp import web

from src.auth import login_helper
from src.auth.session_bundle import SessionError
from tests.test_helper_ca_file import needs_openssl, run_cli, self_signed, tls_instance


class FakeResolver:
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    async def resolve(self, host, port=0, family=socket.AF_INET):
        self.calls.append(host)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return [{"hostname": host, "host": value, "port": port, "family": socket.AF_INET6 if ":" in value
                 else socket.AF_INET, "proto": 0, "flags": 0} for value in answer]

    async def close(self):
        pass


def private_resolver(*answers, url="https://tdm.home.arpa:8443"):
    fake = FakeResolver(*answers)
    return login_helper.PrivateResolver(login_helper.HelperDestination(url), fake), fake


@pytest.mark.parametrize("address", [
    "10.1.2.3", "172.16.0.1", "172.31.255.255", "192.168.1.10", "127.0.0.1",
    "::1", "fd12:3456::1", "fe80::1", "fe80::1%eth0", "::ffff:192.168.1.2",
])
def test_private_and_loopback_addresses_are_allowed(address):
    assert login_helper.PrivateResolver.allowed(address)


@pytest.mark.parametrize("address", [
    "8.8.8.8", "172.32.0.1", "100.64.0.1", "169.254.1.1", "0.0.0.0", "2001:db8::1",
    "2606:4700::1111", "::ffff:8.8.8.8", "::", "not-an-address",
])
def test_public_and_other_addresses_are_refused(address):
    assert not login_helper.PrivateResolver.allowed(address)


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [["192.168.1.10", "8.8.8.8"], ["fd00::1", "2001:4860::8888"], []])
async def test_mixed_or_empty_answers_are_refused(answer):
    resolver, _fake = private_resolver(answer)
    with pytest.raises(SessionError, match="HELPER_INSECURE_TLS") as error:
        await resolver.vet()
    assert error.value.diagnostic.stage == "helper_configuration"
    assert "tdm.home.arpa" not in repr(error.value) + str(error.value)


@pytest.mark.asyncio
async def test_resolution_failure_is_a_network_error():
    resolver, _fake = private_resolver(OSError("no such host"))
    with pytest.raises(SessionError, match="HELPER_NETWORK") as error:
        await resolver.vet()
    assert error.value.diagnostic.reason == "dns"


@pytest.mark.asyncio
async def test_connections_are_pinned_to_vetted_addresses():
    # A later DNS answer cannot redirect the connection to a public address.
    resolver, fake = private_resolver(["192.168.1.10", "fd00::10"], ["8.8.8.8"])
    await resolver.vet()
    ipv4 = await resolver.resolve("tdm.home.arpa", 8443, socket.AF_INET)
    assert [result["host"] for result in ipv4] == ["192.168.1.10"]
    both = await resolver.resolve("tdm.home.arpa", 8443, socket.AF_UNSPEC)
    assert [result["host"] for result in both] == ["192.168.1.10", "fd00::10"]
    assert fake.calls == ["tdm.home.arpa"]
    with pytest.raises(OSError):
        await resolver.resolve("other.example.test", 443, socket.AF_UNSPEC)


@pytest.mark.asyncio
async def test_unvetted_resolver_refuses_every_host():
    resolver, _fake = private_resolver(["192.168.1.10"])
    with pytest.raises(OSError):
        await resolver.resolve("tdm.home.arpa", 8443, socket.AF_UNSPEC)


def test_default_helper_keeps_verification_and_system_resolution():
    helper = login_helper.NativeLoginHelper("https://tdm.example.test")
    assert helper.resolver is None
    assert helper.ssl_context.verify_mode == ssl.CERT_REQUIRED and helper.ssl_context.check_hostname


def test_flag_disables_verification_only_with_private_pinning():
    helper = login_helper.NativeLoginHelper("https://tdm.example.test", insecure_skip_tls_verify=True)
    assert isinstance(helper.resolver, login_helper.PrivateResolver)
    assert helper.ssl_context.verify_mode == ssl.CERT_NONE and not helper.ssl_context.check_hostname


def test_flag_requires_https_and_excludes_ca_file(tmp_path):
    with pytest.raises(SessionError, match="HELPER_INSECURE_TLS"):
        login_helper.NativeLoginHelper("http://192.168.1.10:8080", insecure_skip_tls_verify=True)
    with pytest.raises(ValueError):
        login_helper.NativeLoginHelper("https://192.168.1.10:8443", insecure_skip_tls_verify=True,
                                       ca_file=tmp_path / "ca.pem")


async def insecure_connect(destination):
    async with login_helper.HelperHTTP(destination, ssl_context=login_helper.HelperHTTP.unverified(),
            resolver=login_helper.PrivateResolver(destination), clock=lambda: 1000) as client:
        return await client.connect()


@pytest.mark.asyncio
async def test_public_literal_is_refused_before_connecting():
    with pytest.raises(SessionError, match="HELPER_INSECURE_TLS"):
        await insecure_connect(login_helper.HelperDestination("https://8.8.8.8:443"))


@needs_openssl
@pytest.mark.asyncio
async def test_flag_accepts_untrusted_certificate_on_loopback(tmp_path):
    # Wrong host name and no trust anchor: only disabled verification can connect.
    cert, key = self_signed(tmp_path, "untrusted", san="DNS:other.example.test")
    async with tls_instance(cert, key) as destination:
        with pytest.raises(SessionError, match="HELPER_NETWORK"):
            async with login_helper.HelperHTTP(destination, clock=lambda: 1000) as client:
                await client.connect()
        assert (await insecure_connect(destination)).expires_at == 1600
        port = destination.url.rsplit(":", 1)[1]
        named = login_helper.HelperDestination(f"https://localhost:{port}")
        assert (await insecure_connect(named)).expires_at == 1600


@needs_openssl
@pytest.mark.asyncio
async def test_redirects_are_refused_with_the_flag(tmp_path):
    cert, key = self_signed(tmp_path, "redirect")
    contacted = []

    async def other(request):
        contacted.append(request.path)
        return web.json_response({})

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", other)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    target = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/api/helper/connect"
    try:
        async with tls_instance(cert, key, redirect=target) as destination:
            with pytest.raises(SessionError, match="HELPER_REDIRECT"):
                await insecure_connect(destination)
    finally:
        await runner.cleanup()
    assert contacted == []


def test_cli_warns_and_refuses_public_address_without_echoing_it():
    result = run_cli("--tdm", "https://8.8.8.8:443", "--insecure-skip-tls-verify")
    assert result.returncode == 1
    assert "TLS certificate verification for TDM is OFF" in result.stderr
    assert "--ca-file" in result.stderr
    assert "SESSION_HELPER_INSECURE_TLS" in result.stderr
    assert "8.8.8.8" not in result.stderr


def test_cli_rejects_flag_with_ca_file(tmp_path):
    result = run_cli("--tdm", "https://192.168.1.10:8443", "--insecure-skip-tls-verify",
                     "--ca-file", str(tmp_path / "ca.pem"))
    assert result.returncode == 2
    assert "not allowed with" in result.stderr
    assert "TLS certificate verification for TDM is OFF" not in result.stderr


def test_cli_without_flag_prints_no_warning():
    result = run_cli("--tdm", "https://user:secret@example.test")
    assert "TLS certificate verification" not in result.stdout + result.stderr
