import time

import jwt
import pytest
from fastapi.testclient import TestClient

from backend import app as app_module
from backend.caremind.auth import AuthError, authenticate, validate_auth_config, workspace_for_user
from backend.caremind.config import Settings

SECRET = "x" * 40


def make_settings(**overrides) -> Settings:
    base = dict(
        CAREMIND_ENV="local",
        CAREMIND_USERNAME=None,
        CAREMIND_PASSWORD=None,
        SUPABASE_JWT_SECRET=None,
        SUPABASE_JWKS_URL=None,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def token(sub: str, secret: str = SECRET, exp_in: int = 3600, aud: str = "authenticated") -> str:
    return jwt.encode({"sub": sub, "aud": aud, "exp": int(time.time()) + exp_in}, secret, algorithm="HS256")


class FakeStore:
    def list_documents(self, workspace_id):
        return [{"workspace": workspace_id}]


class FakeServices:
    store = FakeStore()


@pytest.fixture
def client_factory():
    def build(settings: Settings) -> TestClient:
        app_module.app.dependency_overrides[app_module.get_settings] = lambda: settings
        app_module.app.dependency_overrides[app_module.get_services] = lambda: FakeServices()
        return TestClient(app_module.app)

    yield build
    app_module.app.dependency_overrides.clear()


def test_workspace_comes_from_jwt_not_client(client_factory):
    client = client_factory(make_settings(SUPABASE_JWT_SECRET=SECRET))
    a = client.get("/documents", params={"workspace_id": "user-bob"}, headers={"Authorization": f"Bearer {token('alice')}"})
    b = client.get("/documents", params={"workspace_id": "user-alice"}, headers={"Authorization": f"Bearer {token('bob')}"})
    assert a.json() == [{"workspace": workspace_for_user("alice")}]
    assert b.json() == [{"workspace": workspace_for_user("bob")}]


def test_bad_tokens_rejected(client_factory):
    client = client_factory(make_settings(SUPABASE_JWT_SECRET=SECRET))
    for bad in (token("a", secret="y" * 40), token("a", exp_in=-10), token("a", aud="other")):
        assert client.get("/documents", headers={"Authorization": f"Bearer {bad}"}).status_code == 401
    assert client.get("/documents").status_code == 401


def test_basic_auth_uses_legacy_workspace(client_factory):
    client = client_factory(make_settings(CAREMIND_USERNAME="u", CAREMIND_PASSWORD="p"))
    assert client.get("/documents", auth=("u", "p")).json() == [{"workspace": "default"}]
    assert client.get("/documents", auth=("u", "wrong")).status_code == 401
    assert client.get("/documents").status_code == 401


def test_cache_delete_cannot_target_other_workspace(client_factory):
    client = client_factory(make_settings(SUPABASE_JWT_SECRET=SECRET))
    resp = client.delete("/cache/user-bob", headers={"Authorization": f"Bearer {token('alice')}"})
    assert resp.status_code == 403


def test_alg_none_and_unconfigured_secret_rejected():
    settings = make_settings(CAREMIND_USERNAME="u", CAREMIND_PASSWORD="p", SUPABASE_JWKS_URL="https://example.invalid/jwks")
    with pytest.raises(AuthError):
        authenticate(settings, bearer=token("a"), basic=None)  # HS256 but no secret configured


def test_fail_closed_outside_local():
    with pytest.raises(RuntimeError):
        validate_auth_config(make_settings(CAREMIND_ENV="production"))
    validate_auth_config(make_settings(CAREMIND_ENV="production", CAREMIND_USERNAME="u", CAREMIND_PASSWORD="p"))
    validate_auth_config(make_settings(CAREMIND_ENV="local"))
