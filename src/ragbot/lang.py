from lingua import LanguageDetectorBuilder

# Build the detector once at import time — it's relatively heavy to construct,
# so we reuse a single instance for every call.
# with_preloaded_language_models() loads them eagerly so the first real
# detection isn't slow.
_detector = (
    LanguageDetectorBuilder
    .from_all_languages()
    .with_preloaded_language_models()
    .build()
)


def detect_language(text: str) -> str:
    """Detect the language of the text and return its English name (e.g. 'Russian').

    Falls back to English when the text is too short/ambiguous to detect.
    """
    language = _detector.detect_language_of(text)
    if language is None:
        return "English"
    return language.name.capitalize()