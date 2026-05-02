from ai_company_insights.utils import is_probable_ico, normalize_company_name


def test_normalize_company_name_removes_czech_legal_form_and_diacritics() -> None:
    assert normalize_company_name("ČEZ, a. s.") == "cez"
    assert normalize_company_name("Example s.r.o.") == "example"


def test_is_probable_ico() -> None:
    assert is_probable_ico("45274649")
    assert not is_probable_ico("CEZ")
