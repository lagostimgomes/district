"""
state_configs.py

Dataclass and registry for all 50 US states used in the congressional
redistricting pipeline.

Each entry contains:
    fips         — 2-digit zero-padded FIPS code
    abbr         — 2-letter USPS abbreviation
    name         — full state name
    k            — number of congressional districts (118th Congress, 2023)
    crs          — projection string (EPSG:5070 for CONUS, EPSG:3338 for AK,
                   EPSG:26904 for HI)
    admin_notes  — one-line note on special administrative geography

118th Congress seat counts sum to 435.

STRICTLY GEOGRAPHY-ONLY — ZERO PARTISAN OR DEMOGRAPHIC DATA
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StateConfig:
    fips: str          # "01", "02", ... "56" (zero-padded)
    abbr: str          # "AL", "AK", ...
    name: str          # "Alabama", "Alaska", ...
    k: int             # number of congressional districts
    crs: str           # projection string
    admin_notes: str   # one-line administrative geography note


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_NEW_ENGLAND = (
    "No county government; towns/cities are primary unit"
    " — county weight is weaker signal"
)
_STANDARD = "Standard county structure"

ALL_STATES: dict[str, StateConfig] = {cfg.fips: cfg for cfg in [
    StateConfig(
        fips="01", abbr="AL", name="Alabama", k=7,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="02", abbr="AK", name="Alaska", k=1,
        crs="EPSG:3338",
        admin_notes=(
            "Boroughs and census areas replace counties;"
            " unorganized borough has no government"
        ),
    ),
    StateConfig(
        fips="04", abbr="AZ", name="Arizona", k=9,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="05", abbr="AR", name="Arkansas", k=4,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="06", abbr="CA", name="California", k=52,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="08", abbr="CO", name="Colorado", k=8,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="09", abbr="CT", name="Connecticut", k=5,
        crs="EPSG:5070",
        admin_notes=_NEW_ENGLAND,
    ),
    StateConfig(
        fips="10", abbr="DE", name="Delaware", k=1,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="12", abbr="FL", name="Florida", k=28,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="13", abbr="GA", name="Georgia", k=14,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="15", abbr="HI", name="Hawaii", k=2,
        crs="EPSG:26904",
        admin_notes=(
            "4 counties cover entire state;"
            " no incorporated places or townships"
        ),
    ),
    StateConfig(
        fips="16", abbr="ID", name="Idaho", k=2,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="17", abbr="IL", name="Illinois", k=17,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="18", abbr="IN", name="Indiana", k=9,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="19", abbr="IA", name="Iowa", k=4,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="20", abbr="KS", name="Kansas", k=4,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="21", abbr="KY", name="Kentucky", k=6,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="22", abbr="LA", name="Louisiana", k=6,
        crs="EPSG:5070",
        admin_notes=(
            "Parishes are county-equivalents in TIGER;"
            " functionally identical to counties"
        ),
    ),
    StateConfig(
        fips="23", abbr="ME", name="Maine", k=2,
        crs="EPSG:5070",
        admin_notes=_NEW_ENGLAND,
    ),
    StateConfig(
        fips="24", abbr="MD", name="Maryland", k=8,
        crs="EPSG:5070",
        admin_notes=(
            "Baltimore City (FIPS 510) is an independent city,"
            " county-equivalent"
        ),
    ),
    StateConfig(
        fips="25", abbr="MA", name="Massachusetts", k=9,
        crs="EPSG:5070",
        admin_notes=_NEW_ENGLAND,
    ),
    StateConfig(
        fips="26", abbr="MI", name="Michigan", k=13,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="27", abbr="MN", name="Minnesota", k=8,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="28", abbr="MS", name="Mississippi", k=4,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="29", abbr="MO", name="Missouri", k=8,
        crs="EPSG:5070",
        admin_notes=(
            "St. Louis City (FIPS 510) is an independent city,"
            " county-equivalent"
        ),
    ),
    StateConfig(
        fips="30", abbr="MT", name="Montana", k=2,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="31", abbr="NE", name="Nebraska", k=3,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="32", abbr="NV", name="Nevada", k=4,
        crs="EPSG:5070",
        admin_notes=(
            "Carson City (FIPS 510) is an independent city,"
            " county-equivalent"
        ),
    ),
    StateConfig(
        fips="33", abbr="NH", name="New Hampshire", k=2,
        crs="EPSG:5070",
        admin_notes=_NEW_ENGLAND,
    ),
    StateConfig(
        fips="34", abbr="NJ", name="New Jersey", k=12,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="35", abbr="NM", name="New Mexico", k=3,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="36", abbr="NY", name="New York", k=26,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="37", abbr="NC", name="North Carolina", k=14,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="38", abbr="ND", name="North Dakota", k=1,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="39", abbr="OH", name="Ohio", k=15,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="40", abbr="OK", name="Oklahoma", k=5,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="41", abbr="OR", name="Oregon", k=6,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="42", abbr="PA", name="Pennsylvania", k=17,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="44", abbr="RI", name="Rhode Island", k=2,
        crs="EPSG:5070",
        admin_notes=_NEW_ENGLAND,
    ),
    StateConfig(
        fips="45", abbr="SC", name="South Carolina", k=7,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="46", abbr="SD", name="South Dakota", k=1,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="47", abbr="TN", name="Tennessee", k=9,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="48", abbr="TX", name="Texas", k=38,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="49", abbr="UT", name="Utah", k=4,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="50", abbr="VT", name="Vermont", k=1,
        crs="EPSG:5070",
        admin_notes=_NEW_ENGLAND,
    ),
    StateConfig(
        fips="51", abbr="VA", name="Virginia", k=11,
        crs="EPSG:5070",
        admin_notes=(
            "38 independent cities are county-equivalent"
            " (FIPS 510-840); treated as county-level units in graph"
        ),
    ),
    StateConfig(
        fips="53", abbr="WA", name="Washington", k=10,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="54", abbr="WV", name="West Virginia", k=2,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="55", abbr="WI", name="Wisconsin", k=8,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
    StateConfig(
        fips="56", abbr="WY", name="Wyoming", k=1,
        crs="EPSG:5070",
        admin_notes=_STANDARD,
    ),
]}

# Convenience lookup by abbreviation.
STATES_BY_ABBR: dict[str, StateConfig] = {cfg.abbr: cfg for cfg in ALL_STATES.values()}


def _validate_seat_counts() -> None:
    """Assert that the 50-state seat counts sum to 435."""
    total = sum(cfg.k for cfg in ALL_STATES.values())
    assert total == 435, (
        f"Congressional seat counts sum to {total}, expected 435. "
        "Check state_configs.py."
    )


_validate_seat_counts()
