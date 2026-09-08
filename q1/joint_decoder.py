"""
Joint beam-search decoder for word segmentation + POS tagging.

Score for extending a hypothesis with (word, tag):

    α · log P_LM(word | prev2 words) + β · (
        log P_trans(tag | prev2 tags) + log P_emit(word | tag)
    )

This is the English entrypoint Q4 should call for [SEGMENT-ALERT].
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BeamHypothesis:
    position: int
    prev_prev_word: str
    prev_word: str
    prev_prev_tag: str
    prev_tag: str
    score: float
    path: tuple  # ((word, tag), ...)


class JointBeamDecoder:
    def __init__(
        self,
        language_model,
        pos_tagger,
        max_word_length: int = 20,
        alpha: float = 1.0,
        beta: float = 1.0,
        beam_width: int = 8,
    ):
        self.lm = language_model
        self.tagger = pos_tagger
        self.max_word_length = max_word_length
        self.alpha = alpha
        self.beta = beta
        self.beam_width = beam_width

    def _candidate_words(self, text: str, position: int):
        max_length = min(self.max_word_length, len(text) - position)
        for length in range(1, max_length + 1):
            candidate = text[position : position + length]
            if candidate in self.lm.vocabulary:
                yield candidate

    def decode(self, text: str):
        """
        Jointly segment and tag `text` (spaces removed).

        Returns list of (word, tag). Empty list if no complete path exists.
        """
        text = text.lower()
        if not text:
            return []

        start = BeamHypothesis(
            position=0,
            prev_prev_word="<START>",
            prev_word="<START>",
            prev_prev_tag="<START>",
            prev_tag="<START>",
            score=0.0,
            path=(),
        )
        beam = [start]

        for _ in range(len(text) + 1):
            incomplete = [h for h in beam if h.position < len(text)]
            if not incomplete:
                break

            candidates = []
            for hyp in incomplete:
                for word in self._candidate_words(text, hyp.position):
                    lm_score = self.lm.log_trigram_probability(
                        hyp.prev_prev_word, hyp.prev_word, word
                    )
                    for tag in self.tagger.possible_tags(word):
                        pos_score = (
                            self.tagger.log_transition_probability(
                                hyp.prev_prev_tag, hyp.prev_tag, tag
                            )
                            + self.tagger.log_emission_probability(word, tag)
                        )
                        new_score = hyp.score + self.alpha * lm_score + self.beta * pos_score
                        candidates.append(
                            BeamHypothesis(
                                position=hyp.position + len(word),
                                prev_prev_word=hyp.prev_word,
                                prev_word=word,
                                prev_prev_tag=hyp.prev_tag,
                                prev_tag=tag,
                                score=new_score,
                                path=hyp.path + ((word, tag),),
                            )
                        )

            # Keep finished hypotheses from previous beam; prune incomplete expansions.
            finished = [h for h in beam if h.position == len(text)]
            if not candidates and not finished:
                return []

            ranked = sorted(candidates + finished, key=lambda h: h.score, reverse=True)
            beam = ranked[: self.beam_width]

        complete = [h for h in beam if h.position == len(text)]
        if not complete:
            return []

        best = max(
            complete,
            key=lambda h: h.score
            + self.alpha
            * self.lm.log_trigram_probability(h.prev_prev_word, h.prev_word, "<END>"),
        )
        return list(best.path)

    def should_split(self, token: str):
        """
        Compare treating `token` as one word vs a multi-word segmentation.

        Returns (should_split, decoded_pairs).
        """
        token = token.lower()
        decoded = self.decode(token)
        if len(decoded) <= 1:
            return False, decoded

        # Single-word score if the whole token is in vocabulary.
        if token in self.lm.vocabulary:
            single_pairs, pos_score = self.tagger.tag_score([token])
            single_score = (
                self.alpha
                * (
                    self.lm.log_trigram_probability("<START>", "<START>", token)
                    + self.lm.log_trigram_probability("<START>", token, "<END>")
                )
                + self.beta * pos_score
            )
            multi_score = 0.0
            prev_prev_w, prev_w = "<START>", "<START>"
            prev_prev_t, prev_t = "<START>", "<START>"
            for word, tag in decoded:
                multi_score += self.alpha * self.lm.log_trigram_probability(
                    prev_prev_w, prev_w, word
                )
                multi_score += self.beta * (
                    self.tagger.log_transition_probability(prev_prev_t, prev_t, tag)
                    + self.tagger.log_emission_probability(word, tag)
                )
                prev_prev_w, prev_w = prev_w, word
                prev_prev_t, prev_t = prev_t, tag
            multi_score += self.alpha * self.lm.log_trigram_probability(
                prev_prev_w, prev_w, "<END>"
            )
            if multi_score > single_score:
                return True, decoded
            return False, single_pairs

        return True, decoded
