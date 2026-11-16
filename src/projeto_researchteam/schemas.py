from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AnaliseFinanceira(BaseModel):
    empresa: str = Field(description="Nome oficial da empresa analisada.")
    modelo_monetizacao: str = Field(description="Como a empresa gera receita.")
    receita_ou_valuation_estimado: str = Field(description="Dados públicos de receita.")
    principais_concorrentes: list[str] = Field(default_factory=list)
    fontes: list[str] = Field(default_factory=list)


class AnaliseReputacao(BaseModel):
    sentimento_geral: Literal["Positivo", "Neutro", "Negativo", "Misto"]
    resumo_noticias_recentes: list[str] = Field(default_factory=list)
    parcerias_ou_lancamentos: list[str] = Field(default_factory=list)
    riscos_ou_polemicas: list[str] = Field(default_factory=list)


class DossieRequest(BaseModel):
    """Payload de entrada da API com validação estrita."""

    empresa: str = Field(
        min_length=2,
        description="Nome da empresa alvo da pesquisa de inteligência competitiva.",
        examples=["Nubank"],
    )

    @field_validator("empresa")
    @classmethod
    def validar_empresa(cls, value: str) -> str:
        v_limpo = value.strip()
        if not v_limpo:
            raise ValueError("O nome da empresa não pode ser vazio ou conter apenas espaços.")
        return v_limpo


class DossieResponse(BaseModel):
    empresa: str
    dossie_markdown: str