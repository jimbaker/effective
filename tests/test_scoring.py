"""Official HotpotQA answer scorers (EM + token-F1) with the canonical normalization."""

from agent.scoring import exact_match, f1_score, normalize_answer


def test_normalize_drops_articles_punct_case() -> None:
    assert normalize_answer("The Astoria!") == "astoria"
    assert normalize_answer("A B  C") == "b c"


def test_exact_match_is_normalized() -> None:
    assert exact_match("Astoria", "  astoria.") == 1.0
    assert exact_match("Sefton", "Astoria") == 0.0


def test_f1_token_overlap() -> None:
    # full overlap -> 1.0; partial -> between 0 and 1; none -> 0.0
    assert f1_score("Scott Derrickson", "scott derrickson") == 1.0
    assert 0.0 < f1_score("Scott Derrickson Jr", "Scott Derrickson") < 1.0
    assert f1_score("Ed Wood", "Tim Burton") == 0.0


def test_f1_yes_no_scored_by_polarity() -> None:
    assert f1_score("yes", "yes") == 1.0
    assert f1_score("no", "yes") == 0.0  # wrong polarity earns 0, not token overlap
    assert f1_score("yes, definitely", "yes") == 0.0  # a yes/no gold needs exact yes/no
