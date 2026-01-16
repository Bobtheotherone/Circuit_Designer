import pytest

from fidp.dsl import format_program, parse_dsl
from fidp.errors import DSLParseError


def test_parse_roundtrip_stable():
    dsl = """
    circuit RoundTrip {
      ports: (P);
      let cell = series(R(sym(R0,100,min=10,max=1000,snap=E12)),C(1e-6));
      body: series(repeat(cell,2),gen domino_ladder(c=2e-6,r=50,stages=2));
    }
    """
    program = parse_dsl(dsl)
    formatted = format_program(program)
    reparsed = parse_dsl(formatted)
    assert program == reparsed


def test_parse_error_includes_location():
    dsl = """
    circuit Bad {
      ports: (P);
      body: series(R(1),;
    }
    """
    with pytest.raises(DSLParseError) as excinfo:
        parse_dsl(dsl)
    assert excinfo.value.line is not None
    assert excinfo.value.column is not None
    assert "line" in str(excinfo.value).lower()
