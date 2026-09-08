"""Viterbi word segmentation with a trigram language model."""

from __future__ import annotations


class ViterbiSegmenter:
    def __init__(self, language_model, max_word_length=20):
        self.lm = language_model
        self.max_word_length = max_word_length

    def segment(self, text):
        """
        Segment a string using trigram probabilities and dynamic programming.

        Returns a list of words for the best segmentation, or [] if none found.
        """
        text = text.lower()

        # DP: (position, prev_prev_word, prev_word) -> (score, prev_state, word)
        states = {(0, "<START>", "<START>"): (0.0, None, None)}

        for position in range(len(text)):
            current_states = [s for s in states if s[0] == position]

            for state in current_states:
                _, prev_prev_word, prev_word = state
                score, _, _ = states[state]
                max_length = min(self.max_word_length, len(text) - position)

                for length in range(1, max_length + 1):
                    candidate = text[position : position + length]
                    if candidate not in self.lm.vocabulary:
                        continue

                    new_score = score + self.lm.log_trigram_probability(
                        prev_prev_word, prev_word, candidate
                    )
                    new_state = (position + length, prev_word, candidate)

                    if new_state not in states or new_score > states[new_state][0]:
                        states[new_state] = (new_score, state, candidate)

        final_states = [s for s in states if s[0] == len(text)]
        if not final_states:
            return []

        best_state = max(
            final_states,
            key=lambda s: states[s][0]
            + self.lm.log_trigram_probability(s[1], s[2], "<END>"),
        )

        words = []
        current_state = best_state
        while current_state is not None:
            _score, previous_state, chosen_word = states[current_state]
            if chosen_word is not None:
                words.append(chosen_word)
            current_state = previous_state

        words.reverse()
        return words


if __name__ == "__main__":
    from corpus import load_brown, split_brown
    from language_model import TrigramLanguageModel

    print("Loading Brown corpus...")
    sentences = load_brown()
    train_sentences, _ = split_brown(sentences)
    train_words = [[word for word, _tag in sentence] for sentence in train_sentences]

    lm = TrigramLanguageModel()
    lm.train(train_words)
    segmenter = ViterbiSegmenter(lm, max_word_length=20)

    test_string = "thequickbrownfox"
    print("Input:", test_string)
    print("Segmentation:", segmenter.segment(test_string))
