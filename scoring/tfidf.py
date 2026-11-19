"""Pairwise TF-IDF cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer

from .text import analyzer


@dataclass(frozen=True)
class TfidfResult:
    """Cosine similarity in [0, 1] and the terms that contributed most to it."""

    similarity: float
    top_terms: list[str]


def tfidf_similarity(job_description: str, resume: str, top_n: int = 10) -> TfidfResult:
    """Cosine similarity of the job description and one resume.

    The vectoriser is fitted on exactly these two documents, using unigrams
    and within-segment bigrams (see :func:`scoring.text.analyzer`), English
    stop words removed, ``sublinear_tf=True`` (1 + log tf) and L2
    normalisation. Because the fit only sees this pair, a candidate's score
    never depends on the other resumes in the batch.

    ``top_terms`` are the shared features with the largest product of the two
    TF-IDF weights, i.e. the terms that contribute most to the cosine. Ties are
    broken alphabetically so the output is deterministic.
    """
    jd_features = analyzer(job_description)
    resume_features = analyzer(resume)
    if not jd_features or not resume_features:
        return TfidfResult(0.0, [])
    vectorizer = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True, norm="l2")
    matrix = vectorizer.fit_transform([job_description, resume])
    jd_row = matrix[0]
    resume_row = matrix[1]
    contribution = jd_row.multiply(resume_row).tocoo()
    similarity = float(contribution.sum())
    names = vectorizer.get_feature_names_out()
    pairs = sorted(
        ((float(value), str(names[col])) for col, value in zip(contribution.col, contribution.data, strict=True) if value > 0),
        key=lambda item: (-round(item[0], 12), item[1]),
    )
    similarity = min(max(similarity, 0.0), 1.0)
    return TfidfResult(similarity, [term for _, term in pairs[:top_n]])
