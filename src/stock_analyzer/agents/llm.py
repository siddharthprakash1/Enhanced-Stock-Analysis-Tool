from ..config import Settings
from ..metrics.bundle import MetricsBundle


def make_llm(settings: Settings):
    # Provider-swappable. Gemini for dev (cheap), Claude for final runs.
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=settings.gemini_model,
                                      google_api_key=settings.google_api_key, temperature=0)
    from langchain_anthropic import ChatAnthropic  # Opus 4.8: adaptive thinking + effort
    return ChatAnthropic(model=settings.model, max_tokens=8000, anthropic_api_key=settings.anthropic_api_key,
                         model_kwargs={"thinking": {"type": "adaptive"},
                                       "output_config": {"effort": settings.effort}})


def structured(llm, schema):
    return llm.with_structured_output(schema)


def grounding_block(bundle: MetricsBundle) -> str:
    return ("You are grounded ONLY in the metrics below. You must only use these values; "
            "if a value is N/A, say so — never estimate or invent numbers.\n\n" + bundle.prompt_block())
