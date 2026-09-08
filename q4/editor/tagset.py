"""Brown corpus tagset → Penn Treebank tagset reconciliation."""

from __future__ import annotations

# Compact many-to-one map from Brown tags (Q1) to PTB tags (PCFG lexicon).
# Documented in REPORT.md. Unknown tags fall back to NN.

BROWN_TO_PTB: dict[str, str] = {
    # determiners / articles
    "AT": "DT",
    "DT": "DT",
    "DTS": "DT",
    "DTI": "DT",
    "DTX": "DT",
    "AP": "DT",
    "APS": "DT",
    "ABL": "RB",
    "ABN": "DT",
    "ABX": "DT",
    # nouns
    "NN": "NN",
    "NN$": "NN",
    "NNS": "NNS",
    "NNS$": "NNS",
    "NP": "NNP",
    "NP$": "NNP",
    "NPS": "NNPS",
    "NPS$": "NNPS",
    "NR": "NN",
    "NRS": "NNS",
    # adjectives
    "JJ": "JJ",
    "JJR": "JJR",
    "JJT": "JJS",
    "JJS": "JJS",
    # verbs / be / have
    "VB": "VB",
    "VBD": "VBD",
    "VBG": "VBG",
    "VBN": "VBN",
    "VBZ": "VBZ",
    "DO": "VB",
    "DOD": "VBD",
    "DOZ": "VBZ",
    "HV": "VB",
    "HVD": "VBD",
    "HVG": "VBG",
    "HVN": "VBN",
    "HVZ": "VBZ",
    "BE": "VB",
    "BED": "VBD",
    "BEDZ": "VBD",
    "BEG": "VBG",
    "BEM": "VBP",
    "BEN": "VBN",
    "BER": "VBP",
    "BEZ": "VBZ",
    "MD": "MD",
    # pronouns
    "PP$": "PRP$",
    "PP$$": "PRP$",
    "PPL": "PRP",
    "PPLS": "PRP",
    "PPO": "PRP",
    "PPS": "PRP",
    "PPSS": "PRP",
    "PPO$": "PRP$",
    "PN": "NN",
    "PN$": "NN",
    # adverbs / particles
    "RB": "RB",
    "RBR": "RBR",
    "RBT": "RBS",
    "RN": "RB",
    "RP": "RP",
    "QL": "RB",
    "QLP": "RB",
    # prepositions / conjunctions / to
    "IN": "IN",
    "CS": "IN",
    "CC": "CC",
    "TO": "TO",
    # wh-
    "WDT": "WDT",
    "WP$": "WP$",
    "WPO": "WP",
    "WPS": "WP",
    "WQL": "WRB",
    "WRB": "WRB",
    # misc
    "CD": "CD",
    "OD": "JJ",
    "EX": "EX",
    "FW": "FW",
    "UH": "UH",
    "NEG": "RB",
    "*": "RB",
    "'": "''",
    "''": "''",
    "``": "``",
    ".": ".",
    ",": ",",
    ":": ":",
    "(": "(",
    ")": ")",
}


def brown_to_ptb(tag: str) -> str:
    """Map one Brown tag to the closest PTB tag."""
    if tag in BROWN_TO_PTB:
        return BROWN_TO_PTB[tag]
    # strip Brown hyphenated suffixes e.g. VB-HL
    base = tag.split("-")[0].split("+")[0]
    return BROWN_TO_PTB.get(base, "NN")


def map_tagged_sentence(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(w, brown_to_ptb(t)) for w, t in pairs]
