import pytest

from fidp.dsl import compile_dsl, format_program, parse_dsl
from fidp.errors import DSLParseError, DSLValidationError


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


def test_generator_aliases_for_domino_ladder():
    dsl = """
    circuit DominoAlias {
      ports: (P);
      body: gen domino_ladder(stages=2, r=50, c=2e-6);
    }
    """
    program = parse_dsl(dsl)
    circuit = compile_dsl(program, seed=1)
    assert circuit.metadata["generator"] == "domino_ladder"
    assert len(circuit.components) == 4
    r_values = [comp.value.nominal for comp in circuit.components if comp.kind == "R"]
    c_values = [comp.value.nominal for comp in circuit.components if comp.kind == "C"]
    assert r_values == [50.0, 50.0]
    assert c_values == [2e-06, 2e-06]


def test_generator_argument_error_is_structured():
    dsl = """
    circuit BadGen {
      ports: (P);
      body: gen domino_ladder(stages=2, r=50, c=2e-6, nope=1);
    }
    """
    program = parse_dsl(dsl)
    with pytest.raises(DSLValidationError) as excinfo:
        compile_dsl(program, seed=0)
    message = str(excinfo.value)
    assert "domino_ladder" in message
    assert "expected" in message.lower()
    assert "r_value" in message
    assert "c_value" in message
