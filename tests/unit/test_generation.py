from __future__ import annotations

import json

import httpx
import pytest
import respx

from ragtr.generation import (
    ABSTAIN,
    SYSTEM_PROMPT,
    ExtractiveGenerator,
    OllamaGenerator,
    build_messages,
    is_abstention,
    make_generator,
    parse_citations,
)
from ragtr.models import Chunk, ScoredChunk


def ctx(text: str, i: int = 0, title: str = "Akçaburç Belediyesi Su Aboneliği Rehberi") -> ScoredChunk:
    return ScoredChunk(Chunk(f"d#{i:04d}", "d", i, text, text, 0, len(text), title, "Ücretler"), 1.0)


CONTEXTS = [
    ctx("Sayaç en geç 4 iş günü içinde takılır.", 0),
    ctx("Yeni konut aboneliklerinde 1.450 TL güvence bedeli alınır. Bedel iade edilir.", 1),
]


def test_parse_citations_keeps_order_dedupes_and_drops_out_of_range() -> None:
    assert parse_citations("Bedel 1.450 TL [2]. Sayaç [1, 2] ve [7] [2;1].", 2) == [2, 1]
    assert parse_citations("kaynak yok", 2) == []


@pytest.mark.parametrize(
    "text", [ABSTAIN, "belgelerde bu sorunun YANITI YOK.", "Üzgünüm. Belgelerde bu sorunun yanıtı yok"]
)
def test_abstention_detection_is_case_and_diacritic_insensitive(text: str) -> None:
    assert is_abstention(text)


def test_extractive_answers_with_the_best_sentence_and_its_citation() -> None:
    answer = ExtractiveGenerator().generate("Su aboneliği güvence bedeli ne kadar?", CONTEXTS)
    assert answer.text == "Yeni konut aboneliklerinde 1.450 TL güvence bedeli alınır. [2]"
    assert answer.citations == [2]
    assert not answer.abstained


def test_extractive_abstains_when_coverage_is_low() -> None:
    answer = ExtractiveGenerator().generate("Otopark aylık abonman ücreti nedir?", CONTEXTS)
    assert answer.abstained
    assert answer.text == ABSTAIN
    assert ExtractiveGenerator().generate("bir şey", []).abstained


def test_prompt_puts_documents_in_a_delimited_block_with_numbered_sources() -> None:
    messages = build_messages("Bedel ne kadar?", CONTEXTS)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = messages[1]["content"]
    assert user.startswith("<belgeler>\n[1] (Akçaburç Belediyesi Su Aboneliği Rehberi > Ücretler)\n")
    assert "\n</belgeler>\n\nSoru: Bedel ne kadar?" in user


@respx.mock
def test_ollama_request_is_deterministic_and_answer_is_parsed() -> None:
    route = respx.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "Güvence bedeli 1.450 TL'dir [2]."},
                "prompt_eval_count": 311,
                "eval_count": 14,
            },
        )
    )
    gen = OllamaGenerator("http://ollama:11434", "gemma3:4b")
    answer = gen.generate("Bedel ne kadar?", CONTEXTS)
    sent = json.loads(route.calls.last.request.content)
    assert sent["options"]["temperature"] == 0
    assert sent["options"]["seed"] == 42
    assert sent["stream"] is False
    assert answer.citations == [2]
    assert answer.usage == {"prompt_tokens": 311, "completion_tokens": 14}
    assert answer.model == "ollama:gemma3:4b"


@respx.mock
def test_ollama_abstention_has_no_citations() -> None:
    respx.post("http://ollama:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": f"{ABSTAIN} [1]"}})
    )
    answer = OllamaGenerator("http://ollama:11434").generate("Otopark?", CONTEXTS)
    assert answer.abstained
    assert answer.citations == []


@respx.mock
def test_ollama_http_errors_propagate() -> None:
    respx.post("http://ollama:11434/api/chat").mock(return_value=httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        OllamaGenerator("http://ollama:11434").generate("x", CONTEXTS)


def test_ollama_skips_the_call_without_contexts() -> None:
    assert OllamaGenerator("http://127.0.0.1:9").generate("x", []).abstained


def test_make_generator_rejects_unknown_names() -> None:
    assert make_generator("extractive", "", "").name == "extractive"
    with pytest.raises(ValueError, match="unknown generator"):
        make_generator("gpt", "", "")
