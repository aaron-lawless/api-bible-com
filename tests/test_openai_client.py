from unittest.mock import MagicMock, patch

from app.services.llm.openai_client import create_openai_client


def test_create_openai_client_uses_default_constructor_when_available():
    mock_client = MagicMock()

    with patch("app.services.openai_client.openai.OpenAI", return_value=mock_client):
        client = create_openai_client("sk-test", timeout_seconds=45)

    assert client is mock_client


def test_create_openai_client_passes_timeout_to_openai():
    mock_client = MagicMock()

    with patch("app.services.openai_client.openai.OpenAI", return_value=mock_client) as mock_openai:
        create_openai_client("sk-test", timeout_seconds=45)

    _, kwargs = mock_openai.call_args
    assert kwargs["timeout"] == 45


def test_create_openai_client_falls_back_to_explicit_httpx_client():
    fallback_client = MagicMock()
    constructors = [
        TypeError("Client.__init__() got an unexpected keyword argument 'proxies'"),
        fallback_client,
    ]

    with (
        patch("app.services.openai_client.httpx.Client", return_value=MagicMock()) as mock_httpx_client,
        patch("app.services.openai_client.openai.OpenAI", side_effect=constructors) as mock_openai,
    ):
        client = create_openai_client("sk-test", timeout_seconds=45)

    assert client is fallback_client
    assert mock_openai.call_count == 2
    mock_httpx_client.assert_called_once_with(timeout=45)