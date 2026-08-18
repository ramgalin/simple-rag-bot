from lingua import Language, LanguageDetectorBuilder

from ragbot.config import settings


def parse_languages(spec: str) -> list[Language]:
    """Turn the DETECT_LANGUAGES setting into lingua languages.

    Names are lingua's own, case-insensitive: "English, Russian".
    """
    names = [part.strip() for part in spec.split(",") if part.strip()]
    if len(names) < 2:
        raise ValueError(
            f"DETECT_LANGUAGES needs at least two languages to choose between, got {spec!r}"
        )

    languages = []
    for name in names:
        try:
            languages.append(Language.from_str(name))
        except ValueError:
            raise ValueError(
                f"Unknown language {name!r} in DETECT_LANGUAGES. "
                f"Use lingua names such as 'English,Russian,Ukrainian'."
            ) from None
    return languages


_detector = None


def detector():
    """Built on first use and reused — construction is the expensive part.

    Deliberately NOT from_all_languages(). Across lingua's 75 languages a short
    question in Cyrillic is genuinely ambiguous: "кто основал стоицизм?" was
    detected as Serbian, so the bot answered a Russian user in Serbian. Choosing
    between the two or three languages actually in use removes that whole class
    of mistake — and builds faster.
    """
    global _detector
    if _detector is None:
        _detector = (
            LanguageDetectorBuilder
            .from_languages(*parse_languages(settings.detect_languages))
            .with_preloaded_language_models()
            .build()
        )
    return _detector


def detect_language(text: str) -> str:
    """Detect the language of the text and return its English name (e.g. 'Russian').

    Falls back to English when the text is too short/ambiguous to detect.
    """
    language = detector().detect_language_of(text)
    if language is None:
        return "English"
    return language.name.capitalize()
