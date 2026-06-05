from ..config import Settings
from ..metrics.bundle import MetricsBundle


def make_llm(settings: Settings):
    # Provider-swappable. Gemini for dev (cheap), Claude for final runs.
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.rate_limiters import InMemoryRateLimiter
        # Gemini free tier caps gemini-2.5-flash at ~5 requests/min; throttle just under it
        # so the multi-agent pipeline (4 analysts + writer + verifier loop) doesn't hit 429.
        limiter = InMemoryRateLimiter(requests_per_second=0.075, check_every_n_seconds=0.1, max_bucket_size=1)
        return ChatGoogleGenerativeAI(model=settings.gemini_model, google_api_key=settings.google_api_key,
                                      temperature=0, rate_limiter=limiter, max_retries=3)
    from langchain_anthropic import ChatAnthropic  # Opus 4.8: adaptive thinking + effort
    return ChatAnthropic(model=settings.model, max_tokens=8000, anthropic_api_key=settings.anthropic_api_key,
                         model_kwargs={"thinking": {"type": "adaptive"},
                                       "output_config": {"effort": settings.effort}})


def structured(llm, schema):
    return llm.with_structured_output(schema)


def grounding_block(bundle: MetricsBundle) -> str:
    return ("You are grounded ONLY in the metrics below. You must only use these values; "
            "if a value is N/A, say so — never estimate or invent numbers.\n\n" + bundle.prompt_block())
