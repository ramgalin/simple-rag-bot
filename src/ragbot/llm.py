from langchain.chat_models import init_chat_model

from ragbot.config import settings


def build_llm():
    provider = settings.llm_provider

    if provider == "langdock":
        # Langdock — OpenAI-совместимый шлюз: клиент openai, но свои ключ и адрес
        return init_chat_model(
            settings.llm_model,
            model_provider="openai",
            api_key=settings.langdock_api_key,
            base_url=settings.langdock_base_url,
        )

    # обычные openai / anthropic — берём нужный ключ из settings и передаём ЯВНО
    api_key = (
        settings.openai_api_key
        if provider == "openai"
        else settings.anthropic_api_key
    )
    return init_chat_model(
        settings.llm_model,
        model_provider=provider,
        api_key=api_key,
    )