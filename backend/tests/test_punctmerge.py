from app.punctmerge import merge_punctuation


def test_matched_words_take_v3_case_and_punctuation_but_keep_camoes_spelling():
    assert merge_punctuation(["bom dia o crescimento económico"],
                             "Bom dia. O crescimento econômico!") == ["Bom dia. O crescimento económico!"]


def test_disagreement_keeps_camoes_word_without_punctuation():
    # large-v3 misheard "receção" as "recessão,": CAMOES's word stays, bare
    assert merge_punctuation(["a receção foi boa"], "A recessão, foi boa.") == ["A receção foi boa."]


def test_insertions_and_deletions_and_segment_boundaries():
    out = merge_punctuation(["olá a todos", "hoje vamos falar de lisboa"],
                            "Olá todos. Hoje vamos, então, falar de Lisboa.")
    assert out == ["Olá a todos.", "Hoje vamos, falar de Lisboa."]


def test_acronym_and_empty():
    assert merge_punctuation(["a ue decidiu"], "A UE decidiu.") == ["A UE decidiu."]
    assert merge_punctuation([""], "Olá.") == [""]
