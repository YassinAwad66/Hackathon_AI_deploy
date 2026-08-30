import asyncio
import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError
import streamlit as st


logging.getLogger("google_genai").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Timing / retry configuration
# ---------------------------------------------------------------------------

# Hard ceiling for one analyze() call: search step + main call + all
# retries must fit inside this. Tune to whatever your frontend/gateway
# is willing to wait (leave some margin below that value).
TOTAL_BUDGET_SECONDS = 25.0

# Per-call ceilings — the actual timeout used is min(this, time left in budget).
REQUEST_TIMEOUT_SECONDS = 15.0
SEARCH_TIMEOUT_SECONDS = 8.0

MAX_RETRIES = 2
RETRY_BACKOFF_BASE_SECONDS = 1.5   # 1.5s, 3s, ...

# Caps how many Gemini calls (search + main, combined) run concurrently.
# Replaces the old ThreadPoolExecutor(max_workers=4) — but unlike that
# pool, a timed-out/cancelled call actually releases its slot.
MAX_CONCURRENT_REQUESTS = 8

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

SYSTEM_INSTRUCTION = """You are a climate and environmental data assistant for the Heat Guardian system.

You must be able to handle questions from ANY relevant domain/audience, inferred
from the user's question itself — not from a fixed category list. Examples of
domains you may need to answer for: general public heat safety, industrial /
worker safety, construction site safety, school administration, city planning
and urban heat management, building energy consumption, agriculture, insurance
risk pricing, public health, or general climate/environmental analysis. If a
question comes from a domain not listed here, still answer it as best you can
using the provided data and general climate/safety reasoning.

Answering policy — ALWAYS attempt a real answer:
- If the question concerns climate, weather, heat/cold, energy, industry,
  construction, buildings, agriculture, air quality, or related safety/risk/
  operational topics, you must attempt a substantive answer. Do not refuse
  or hedge into a non-answer just because the environmental_data provided is
  incomplete — supplementary web context is fetched specifically to fill
  those gaps, so use it.
- Only leave "answer" as a non-answer (and treat the question as out of
  scope) if it is genuinely unrelated to climate/weather/environment/energy/
  industry/buildings, e.g. a question with no connection to any of those
  topics.
- Do not set data_missing=true merely because environmental_data alone is
  sparse. Only set it true if BOTH the environmental_data AND the
  supplementary web context fail to provide what's needed to answer.

Instructions:
- Base every field primarily on the environmental data provided.
- You may also be given a block of "Supplementary web context" retrieved from
  a live web search. Treat the provided environmental_data as the primary,
  authoritative source for numeric readings (temperature, humidity, heat
  index, AQI, etc.) UNLESS the web context clearly indicates something more
  current or directly contradicts it (e.g. an active heat advisory, a
  forecast, or a news event) — in that case, note the discrepancy explicitly
  in "answer" rather than silently picking one. When environmental_data is
  missing a value the question needs, use the value found in the web context
  instead of leaving it out, and note in "answer" that it came from a live
  search rather than the sensor/API data.
- Do not invent or assume any environmental data not provided by either the
  environmental_data or the supplementary web context.
- Assume temperatures are in Celsius (°C) and humidity/AQI values are in
  their conventional units unless the data explicitly states otherwise.
- If the data includes a timestamp, treat that as "as of" the answer's
  reference point rather than assuming it reflects the current live moment.
- Identify "domain" yourself from the question's context (e.g. "industrial",
  "construction", "school", "city_planning", "energy", "agriculture",
  "insurance", "general_public"). Pick the closest fit; don't leave it blank.
- Identify "topic" as a short label for what the question is about
  (e.g. "heat_safety", "temperature_trend", "air_quality", "energy_load",
  "general_climate").
- Tailor "explanation" and "recommended_actions" to the identified domain.
  E.g. for "construction" talk about site scheduling and PPE; for "school"
  talk about recess/outdoor activity timing; for "city_planning" talk about
  cooling infrastructure or heat-island effects; for "energy" talk about
  cooling load and demand; for "general_public" keep it personal/practical.
- Only populate "risk_level" if the question concerns safety, exposure, or
  operational risk. Leave it null for purely informational/analytical questions.
- Only populate "recommended_actions" if there's a safety/risk/operational
  angle. Leave it as an empty list otherwise.
- Populate "key_metrics" with the specific numeric values relevant to the
  answer (e.g. temperature, humidity, heat index, AQI) so the frontend can
  display them as stat cards, including any values sourced from the web
  context when environmental_data didn't have them.
- Populate "external_sources" with short labels/titles for any web sources
  you actually relied on from the supplementary web context (leave empty if
  none were used or none were provided).
- Set "confidence" to "low", "medium", or "high" based on how complete and
  directly relevant the provided data (including any web context) is to the
  question. Sparse, indirect, or partially missing data should lower
  confidence; complete, directly relevant data should raise it. Filling a
  gap from web context rather than sensor data should not by itself force
  "low" — use "medium" if the web-sourced value is clear and specific.
- If, after considering BOTH environmental_data and the web context, the
  needed data genuinely isn't available anywhere, set data_missing=true and
  say so plainly in the answer rather than guessing.
- If the question is entirely unrelated to climate/weather/environment/
  energy/industry/buildings, say so in "answer" and leave the other fields
  empty/null.

Charts and comparisons:
- Use "chart_series" for anything plottable. Each entry in "chart_series" is
  a named series with its own list of points, so you can return ONE series
  for a simple trend, or MULTIPLE named series for a comparison question
  (e.g. "compare Alexandria vs Cairo", "today vs last week", "site A vs
  site B"). Name each series clearly (e.g. "Alexandria", "Last Week").
- Only populate "chart_series" if the data actually contains a time series
  or comparable set of points. Otherwise leave it empty.

Handling forecast / future questions:
- If the user asks about the future (e.g. "will it be safe tomorrow",
  "what will the heat index be this weekend", "is a heatwave coming"),
  set "is_forecast" to true.
- Base any forecast on patterns present in the provided data (e.g.
  rising/falling trend across the given time series, recent rate of
  change, seasonal pattern implied by the data) AND, if present, on any
  forecast/advisory information found in the supplementary web context.
- If neither the provided data nor the web context supports a forecast,
  set is_forecast=true, set data_missing=true, and explain in "answer"
  that the available data isn't sufficient to project forward — do not
  guess a number anyway. Lower "confidence" accordingly.
- When you do give a forecast, phrase it as an estimate/projection (e.g.
  "based on the current trend, expect..."), never as a certainty.
- If you provide forecasted points, label them clearly as projected (e.g.
  include "(projected)" in the point label) so they are visually
  distinguishable from historical points.
"""

SEARCH_SYSTEM_INSTRUCTION = """You are a research assistant supporting a climate/heat-risk
system called Heat Guardian. Given a user's question and some known environmental
data (location, current readings), use web search to find any CURRENT, directly
relevant supplementary information needed to fully answer the question.

Your job is to actively fill gaps, not just add color:
- First identify what specific data the question needs that is NOT already
  present in the given environmental_data (e.g. a missing metric like AQI,
  humidity, wind, a forecast for a specific day, an active advisory, energy
  prices/tariffs, industry-specific thresholds/regulations, etc.).
- Search specifically for each of those missing pieces, not just general
  news about the location.
- Also include any other CURRENT, directly relevant information even if not
  strictly "missing" — e.g. active heat advisories or warnings, official
  weather forecasts, recent local air quality reports, relevant news, energy
  or industry data — for that location and timeframe.

Rules:
- Be concise: a short paragraph (3-6 sentences) of factual findings, not a
  report. If you found specific numeric values to fill a gap, state them
  clearly and specifically (e.g. "Current AQI in X is 42 (moderate) per
  [source], as of [time]").
- Only include information you actually found via search; do not speculate
  or fabricate a value if search turns up nothing for it — say so instead.
- If search turns up nothing clearly relevant, say so in one sentence instead
  of padding with generic climate facts.
- Do not repeat back the environmental_data that was given to you; focus on
  NEW information not already present in it.
"""


class ChartPoint(BaseModel):
    label: str
    value: float


class ChartSeries(BaseModel):
    name: str
    points: list[ChartPoint] = []


class Metric(BaseModel):
    name: str
    value: float
    unit: Optional[str] = None


class ClimateAnalysisResult(BaseModel):
    answer: str
    domain: Optional[str] = None
    topic: Optional[str] = None
    risk_level: Optional[str] = None
    confidence: Optional[str] = None
    explanation: Optional[str] = None
    recommended_actions: list[str] = []
    key_metrics: list[Metric] = []
    chart_series: list[ChartSeries] = []
    chart_label: Optional[str] = None
    is_forecast: bool = False
    data_missing: bool = False
    external_sources: list[str] = []
    web_search_used: bool = False


def _is_retryable(exc: Exception) -> bool:
    """True for transient/server-side failures worth retrying (rate limits,
    5xx, model overload like 503 UNAVAILABLE); false for things a retry with
    the same prompt won't fix (bad request, auth, malformed schema)."""
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status_code in RETRYABLE_STATUS_CODES:
        return True
    if isinstance(exc, genai_errors.ServerError):
        return True
    message = str(exc).upper()
    return "UNAVAILABLE" in message or "OVERLOADED" in message or "RESOURCE_EXHAUSTED" in message


class ClimateAI:
    def __init__(self, model: str = "gemini-3.6-flash", enable_web_search: bool = True):
        self.client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        self.model = model
        self.enable_web_search = enable_web_search

        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=ClimateAnalysisResult,
        )

        self.search_config = types.GenerateContentConfig(
            system_instruction=SEARCH_SYSTEM_INSTRUCTION,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )

        # Bounds concurrent Gemini calls without leaking on timeout/cancel —
        # the old ThreadPoolExecutor's fixed workers stayed occupied by
        # abandoned (timed-out but still-running) tasks; an asyncio
        # Semaphore is released cleanly even if the caller cancels.
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # ---------- low-level async calls ----------

    async def _call_model(self, prompt: str):
        async with self._semaphore:
            return await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self.config,
            )

    async def _call_search_model(self, prompt: str):
        async with self._semaphore:
            return await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self.search_config,
            )

    # ---------- web search / grounding step ----------

    async def _fetch_web_context(
        self, user_question: str, environmental_data: dict, timeout: float
    ) -> tuple[str, list[str]]:
        """
        Best-effort supplementary context. On any failure or timeout,
        returns ("", []) so the caller silently falls back to data-only
        mode. `timeout` is whatever's left of the shared request budget —
        if that's <= 0, we skip the search entirely instead of firing a
        call we already know can't finish in time.
        """
        if timeout <= 0:
            return "", []

        location_hint = ""
        lat = environmental_data.get("latitude")
        lon = environmental_data.get("longitude")
        if lat is not None and lon is not None:
            location_hint = f"Location: approximately {lat}, {lon}.\n"

        search_prompt = f"""{location_hint}Known environmental data (do not repeat this back):
{environmental_data}

User question:
{user_question}

Identify what data this question needs that is NOT already present above
(e.g. a missing metric, a forecast, an advisory, energy/industry figures,
regulatory thresholds), and search specifically for each missing piece, in
addition to any other current, directly relevant supplementary information
(advisories, forecasts, air quality reports, relevant news) for this
location/question. Report specific values you find, not just general
commentary."""

        try:
            response = await asyncio.wait_for(
                self._call_search_model(search_prompt), timeout=timeout
            )
        except (asyncio.TimeoutError, Exception):
            return "", []

        text = (response.text or "").strip()
        sources: list[str] = []
        try:
            candidate = response.candidates[0]
            grounding_metadata = getattr(candidate, "grounding_metadata", None)
            chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if web and getattr(web, "title", None):
                    sources.append(web.title)
        except Exception:
            pass

        return text, sources

    # ---------- main entry point (async) ----------

    async def analyze_async(
        self, user_question: str, environmental_data: dict
    ) -> ClimateAnalysisResult:
        if not environmental_data:
            return ClimateAnalysisResult(
                answer="No environmental data available to answer this question.",
                data_missing=True,
            )

        deadline = time.monotonic() + TOTAL_BUDGET_SECONDS

        def remaining() -> float:
            return deadline - time.monotonic()

        web_context = ""
        web_sources: list[str] = []
        web_search_used = False

        if self.enable_web_search:
            search_timeout = min(SEARCH_TIMEOUT_SECONDS, remaining())
            web_context, web_sources = await self._fetch_web_context(
                user_question, environmental_data, search_timeout
            )
            web_search_used = bool(web_context)

        prompt = f"""Environmental data from FortyGuard:
{environmental_data}
"""
        if web_context:
            prompt += f"""
Supplementary web context (retrieved via live search, may be partial):
{web_context}
"""
            if web_sources:
                prompt += f"Sources: {', '.join(web_sources)}\n"

        prompt += f"""
User question:
{user_question}

Remember: if environmental_data above is missing something this question
needs, use the supplementary web context (if provided) to fill that gap
rather than declining to answer. Only say the data is missing if neither
source has what's needed.
"""

        last_error = None
        attempt = 0
        while True:
            time_left = remaining()
            if time_left <= 0.5:
                # Not enough budget left for a meaningful attempt.
                last_error = last_error or "Ran out of time budget before completing."
                break

            call_timeout = min(REQUEST_TIMEOUT_SECONDS, time_left)
            should_retry = False
            try:
                response = await asyncio.wait_for(
                    self._call_model(prompt), timeout=call_timeout
                )
                result = ClimateAnalysisResult.model_validate_json(response.text)
                result.web_search_used = web_search_used
                if web_sources and not result.external_sources:
                    result.external_sources = web_sources
                return result
            except asyncio.TimeoutError:
                last_error = "The request took too long to respond."
                should_retry = True
            except ValidationError:
                last_error = "The model returned an unexpected response format."
                should_retry = True
            except Exception as e:
                last_error = str(e)
                should_retry = _is_retryable(e)

            attempt += 1
            if attempt > MAX_RETRIES or not should_retry:
                break

            backoff = RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            # Don't sleep past the deadline — if there's no time left for
            # both the backoff AND a follow-up call, stop now instead of
            # burning the remaining budget on a guaranteed-to-fail sleep.
            time_left = remaining()
            if backoff + 1.0 >= time_left:
                break
            await asyncio.sleep(backoff)

        return ClimateAnalysisResult(
            answer=f"Sorry, I couldn't generate a response right now. ({last_error})",
            data_missing=True,
            web_search_used=web_search_used,
        )

    # ---------- sync convenience wrapper ----------

    def analyze(self, user_question: str, environmental_data: dict) -> ClimateAnalysisResult:
        """
        Sync wrapper for scripts/CLI use. Do NOT call this from inside an
        already-running event loop (e.g. a FastAPI async endpoint) — call
        analyze_async() directly there instead.
        """
        try:
            return asyncio.run(self.analyze_async(user_question, environmental_data))
        except RuntimeError as e:
            if "asyncio.run() cannot be called" in str(e):
                raise RuntimeError(
                    "analyze() can't be called from inside a running event loop "
                    "(e.g. an async web framework handler). Call "
                    "`await client.analyze_async(...)` instead."
                ) from e
            raise


async def _main():
    ai = ClimateAI()
    sample_data = {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "temperature_c": 30,
        "humidity_percent": 90,
        "heat_index_c": 30,
        "timestamp": "2026-08-22T14:00:00",
        "aqi": 65,
        "fountain": 34.18,
        "tree": 28.74,
        "sky": 17.17,
        "water": 11.03,
        "building": 5.38,
        "floor": 1.8,
        "grass": 0.9,
        "sidewalk": 0.52,
        "others": 0.28,
    }

    question = input("Enter your question: ")
    result = await ai.analyze_async(question, sample_data)

    print("\n" + result.answer)
    if result.domain:
        print(f"Domain: {result.domain}")
    if result.confidence:
        print(f"Confidence: {result.confidence}")
    if result.is_forecast:
        print("(This is a forecast/projection, not a guaranteed reading)")
    if result.risk_level:
        print(f"Risk level: {result.risk_level}")
    if result.recommended_actions:
        print("\nRecommended actions:")
        for action in result.recommended_actions:
            print(f"- {action}")
    if result.key_metrics:
        print("\nKey metrics:")
        for m in result.key_metrics:
            unit = f" {m.unit}" if m.unit else ""
            print(f"- {m.name}: {m.value}{unit}")
    if result.chart_series:
        print("\nChart series:")
        for series in result.chart_series:
            print(f"- {series.name}: {[(p.label, p.value) for p in series.points]}")
    if result.web_search_used:
        sources = f" — sources: {', '.join(result.external_sources)}" if result.external_sources else ""
        print(f"\n(Answer informed by live web search{sources})")


if __name__ == "__main__":
    asyncio.run(_main())