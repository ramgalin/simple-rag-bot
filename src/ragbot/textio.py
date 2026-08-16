def strip_surrogates(text: str) -> str:
    """Replace lone surrogates (\\udc80-\\udcff) with '?'.

    Under a POSIX locale (LANG=C.UTF-8) Python decodes stdin with the
    surrogateescape handler, so a byte it cannot decode becomes a surrogate
    instead of raising. Those survive happily inside Python and then explode far
    away, when an HTTP client encodes the request body as strict UTF-8:

        UnicodeEncodeError: 'utf-8' codec can't encode character '\\udcd1'

    A split Cyrillic character is already unrecoverable at this point; '?' at
    least makes the damage visible instead of killing the process.
    """
    return text.encode("utf-8", "replace").decode("utf-8")
