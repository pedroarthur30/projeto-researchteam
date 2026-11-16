import sys
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from projeto_researchteam.cli import _executar_cli, main
from projeto_researchteam.schemas import AnaliseFinanceira, AnaliseReputacao, DossieRequest


def test_dossie_request_valida_min_length_e_strip() -> None:
    req = DossieRequest(empresa="  Nubank  ")
    assert req.empresa == "Nubank"

    with pytest.raises(ValidationError):
        DossieRequest(empresa="A")

    with pytest.raises(ValidationError):
        DossieRequest(empresa="   ")


def test_analise_financeira_e_reputacao_defaults_e_literal() -> None:
    fin = AnaliseFinanceira(
        empresa="NexaTech",
        modelo_monetizacao="SaaS B2B",
        receita_ou_valuation_estimado="R$ 50M",
    )
    assert fin.principais_concorrentes == []
    assert fin.fontes == []

    rep = AnaliseReputacao(sentimento_geral="Positivo")
    assert rep.resumo_noticias_recentes == []
    assert rep.parcerias_ou_lancamentos == []
    assert rep.riscos_ou_polemicas == []

    with pytest.raises(ValidationError):
        AnaliseReputacao(sentimento_geral="Excelente")  # type: ignore[arg-type]


def test_cli_main_sem_argumentos_nao_executa_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(sys, "argv", ["projeto-researchteam"])
    with patch("projeto_researchteam.cli.asyncio.run") as mock_run:
        main()

    mock_run.assert_not_called()
    assert "Nenhuma empresa informada" in caplog.text


def test_cli_main_com_argumentos_chama_executar_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["projeto-researchteam", "NexaTech", "Solutions"])
    fake_coro = MagicMock()
    mock_exec = MagicMock(return_value=fake_coro)
    with (
        patch("projeto_researchteam.cli._executar_cli", new=mock_exec),
        patch("projeto_researchteam.cli.asyncio.run") as mock_run,
    ):
        main()

    mock_exec.assert_called_once_with("NexaTech Solutions")
    mock_run.assert_called_once_with(fake_coro)


@pytest.mark.asyncio
async def test_executar_cli_faz_streaming_para_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    class FakeStreamPipeline:
        def __init__(self) -> None:
            self.last_input: dict[str, Any] | None = None
            self.last_config: dict[str, Any] | None = None

        async def astream(
            self,
            input_data: dict[str, str],
            config: dict[str, Any] | None = None,
        ) -> AsyncGenerator[str, None]:
            self.last_input = input_data
            self.last_config = config
            for token in ["# Dossiê ", "NexaTech"]:
                yield token

    fake_pipe = FakeStreamPipeline()
    with (
        patch("projeto_researchteam.cli.init_vectorstore", return_value=MagicMock()),
        patch("projeto_researchteam.cli.TavilyClient", return_value=MagicMock()),
        patch("projeto_researchteam.cli.build_research_pipeline", return_value=fake_pipe),
    ):
        await _executar_cli("NexaTech")

    captured = capsys.readouterr()
    assert captured.out == "# Dossiê NexaTech\n"
    assert fake_pipe.last_input == {"empresa": "NexaTech"}
    assert fake_pipe.last_config == {"configurable": {"tenant_id": "default_tenant"}}
