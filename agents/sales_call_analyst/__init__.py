"""Sales Call Analyst Agent.

Analyzes sales call recordings/transcripts for sentiment,
key moments, objections, competitor mentions, buying signals,
price sensitivity, and action items. Generates follow-up emails
and CRM activity logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SentimentSnapshot:
    """Sentiment at a point in the call."""
    speaker: str
    text: str
    sentiment: str  # "positive", "neutral", "negative"
    confidence: float = 0.0


@dataclass
class Objection:
    """A buyer objection detected in the transcript."""
    objection_text: str
    speaker: str
    timestamp_hint: str = ""
    category: str = ""  # "price", "timing", "trust", "competitor", "need", "authority"
    handled: bool | None = None  # None = not yet evaluated


@dataclass
class CompetitorMention:
    """A competitor mention in the call."""
    competitor_name: str
    context: str
    mentioned_by: str = ""  # "buyer" or "seller"


@dataclass
class BuyingSignal:
    """A buying signal detected in the transcript."""
    signal_text: str
    speaker: str
    signal_type: str = ""  # "urgency", "budget", "authority", "need", "timeline"


@dataclass
class CallAnalysis:
    """Structured result of a sales call analysis."""
    overall_sentiment_buyer: str = "neutral"
    overall_sentiment_seller: str = "neutral"
    sentiment_timeline: list[SentimentSnapshot] = field(default_factory=list)
    objections: list[Objection] = field(default_factory=list)
    competitor_mentions: list[CompetitorMention] = field(default_factory=list)
    buying_signals: list[BuyingSignal] = field(default_factory=list)
    price_sensitivity: str = "unknown"  # "high", "medium", "low", "unknown"
    key_moments: list[str] = field(default_factory=list)
    next_steps_agreed: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    call_quality_score: int = 0
    call_quality_reasoning: str = ""
    follow_up_email: str = ""
    crm_activity_log: str = ""
    coaching_notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sentiment detection helpers
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = {
    "interested", "excited", "great", "love", "perfect", "sounds good",
    "let's do it", "sounds great", "absolutely", "definitely", "yes",
    "agree", "looking forward", "impressed", "valuable", "helpful",
    "makes sense", "good fit", "move forward", "sign", "deal",
}

_NEGATIVE_WORDS = {
    "not interested", "too expensive", "no budget", "not now", "maybe later",
    "need to think", "not sure", "concerned", "worried", "problem",
    "issue", "doesn't work", "not a good pass", "pass", "no thanks",
    "too costly", "can't afford", "not the right time", "busy",
}

_OBJECTION_CATEGORIES = {
    "price": {"expensive", "cost", "budget", "price", "afford", "cheap", "investment"},
    "timing": {"not now", "later", "next quarter", "busy", "not the right time", "wait"},
    "trust": {"not sure", "concern", "risk", "guarantee", "proof", "reference", "track record"},
    "competitor": {"already using", "competitor", "other vendor", "alternative", "comparing"},
    "need": {"don't need", "not necessary", "already have", "sufficient", "fine as is"},
    "authority": {"need to ask", "check with", "boss", "team", "decision maker", "approve"},
}

_BUYING_SIGNAL_TYPES = {
    "urgency": {"asap", "soon", "quickly", "fast", "immediately", "this month", "this quarter"},
    "budget": {"budget approved", "allocated", "funding", "approved", "spend"},
    "authority": {"i decide", "my call", "i'm the decision", "sign off", "approve"},
    "need": {"we need", "problem we're facing", "challenge", "pain point", "struggling with"},
    "timeline": {"by end of", "launch date", "go live", "implementation", "roll out"},
}


def detect_sentiment(text: str) -> tuple[str, float]:
    """Simple keyword-based sentiment detection.

    Returns (sentiment_label, confidence).
    """
    text_lower = text.lower()
    pos_count = sum(1 for w in _POSITIVE_WORDS if w in text_lower)
    neg_count = sum(1 for w in _NEGATIVE_WORDS if w in text_lower)

    total = pos_count + neg_count
    if total == 0:
        return "neutral", 0.5
    if pos_count > neg_count:
        return "positive", min(pos_count / total, 1.0)
    if neg_count > pos_count:
        return "negative", min(neg_count / total, 1.0)
    return "neutral", 0.5


def detect_objections(text: str, speaker: str = "buyer") -> list[Objection]:
    """Detect objections in text segments."""
    objections: list[Objection] = []
    text_lower = text.lower()
    for category, keywords in _OBJECTION_CATEGORIES.items():
        for kw in keywords:
            if kw in text_lower:
                objections.append(Objection(
                    objection_text=kw,
                    speaker=speaker,
                    category=category,
                ))
    return objections


def detect_competitor_mentions(text: str) -> list[CompetitorMention]:
    """Detect competitor mentions in text.

    Looks for patterns like "using <name>", "competitor <name>",
    "compared to <name>", etc.
    """
    import re
    mentions: list[CompetitorMention] = []
    patterns = [
        r"(?:using|with|from|at|switching from|currently on)\s+([A-Z][A-Za-z0-9_\-]+)",
        r"(?:competitor|alternative|other vendor)\s+(?:is|was)?\s*([A-Z][A-Za-z0-9_\-]+)",
        r"(?:compared to|versus|vs\.?)\s+([A-Z][A-Za-z0-9_\-]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            name = match.group(1)
            if name.lower() not in {"i", "we", "the", "our", "my", "this", "that", "it"}:
                mentions.append(CompetitorMention(
                    competitor_name=name,
                    context=match.group(0),
                ))
    return mentions


def detect_buying_signals(text: str, speaker: str = "buyer") -> list[BuyingSignal]:
    """Detect buying signals in text segments."""
    signals: list[BuyingSignal] = []
    text_lower = text.lower()
    for signal_type, keywords in _BUYING_SIGNAL_TYPES.items():
        for kw in keywords:
            if kw in text_lower:
                signals.append(BuyingSignal(
                    signal_text=kw,
                    speaker=speaker,
                    signal_type=signal_type,
                ))
    return signals


def assess_price_sensitivity(objections: list[Objection]) -> str:
    """Assess price sensitivity based on objections."""
    price_objections = [o for o in objections if o.category == "price"]
    if len(price_objections) >= 3:
        return "high"
    if len(price_objections) >= 1:
        return "medium"
    return "low"


def score_call_quality(analysis: CallAnalysis) -> tuple[int, str]:
    """Score call quality 1-10 based on analysis results."""
    score = 5  # baseline
    reasons: list[str] = []

    # Positive signals
    if analysis.buying_signals:
        score += min(len(analysis.buying_signals), 3)
        reasons.append(f"{len(analysis.buying_signals)} buying signal(s) detected")

    if analysis.next_steps_agreed:
        score += 1
        reasons.append("Next steps agreed")

    if analysis.overall_sentiment_buyer == "positive":
        score += 1
        reasons.append("Positive buyer sentiment")

    # Negative signals
    if analysis.objections:
        unhandled = [o for o in analysis.objections if o.handled is False]
        score -= min(len(unhandled), 3)
        reasons.append(f"{len(unhandled)} unhandled objection(s)")

    if analysis.overall_sentiment_buyer == "negative":
        score -= 1
        reasons.append("Negative buyer sentiment")

    if analysis.price_sensitivity == "high":
        score -= 1
        reasons.append("High price sensitivity")

    score = max(1, min(score, 10))
    reasoning = "; ".join(reasons) if reasons else "Average call with no strong signals"
    return score, reasoning


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

def parse_transcript(transcript: str) -> list[dict[str, str]]:
    """Parse a transcript into structured speaker/text pairs.

    Supports formats:
    - "Speaker: text"
    - "[Speaker]: text"
    - "Speaker (role): text"
    """
    import re
    lines: list[dict[str, str]] = []
    for line in transcript.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Match "Name: text" or "[Name]: text" or "Name (role): text"
        m = re.match(r"^\[?([^\]\n:]+?)\]?(?:\s*\([^)]*\))?\s*:\s*(.+)$", line)
        if m:
            lines.append({"speaker": m.group(1).strip(), "text": m.group(2).strip()})
        else:
            lines.append({"speaker": "unknown", "text": line})
    return lines


def analyze_transcript_locally(transcript: str) -> CallAnalysis:
    """Perform local (non-LLM) analysis of a transcript.

    This provides structured data that the LLM can then enrich.
    """
    analysis = CallAnalysis()
    segments = parse_transcript(transcript)

    all_objections: list[Objection] = []
    all_signals: list[BuyingSignal] = []
    all_competitors: list[CompetitorMention] = []
    sentiment_scores: list[str] = []

    for seg in segments:
        speaker = seg["speaker"]
        text = seg["text"]
        sentiment, confidence = detect_sentiment(text)
        analysis.sentiment_timeline.append(SentimentSnapshot(
            speaker=speaker,
            text=text,
            sentiment=sentiment,
            confidence=confidence,
        ))

        # Track buyer sentiment separately
        if speaker.lower() in {"buyer", "prospect", "customer", "client"}:
            sentiment_scores.append(sentiment)
            all_objections.extend(detect_objections(text, speaker="buyer"))
            all_signals.extend(detect_buying_signals(text, speaker="buyer"))

        all_competitors.extend(detect_competitor_mentions(text))

    analysis.objections = all_objections
    analysis.buying_signals = all_signals
    analysis.competitor_mentions = all_competitors
    analysis.price_sensitivity = assess_price_sensitivity(all_objections)

    # Overall buyer sentiment
    if sentiment_scores:
        pos = sentiment_scores.count("positive")
        neg = sentiment_scores.count("negative")
        if pos > neg:
            analysis.overall_sentiment_buyer = "positive"
        elif neg > pos:
            analysis.overall_sentiment_buyer = "negative"
        else:
            analysis.overall_sentiment_buyer = "neutral"

    analysis.call_quality_score, analysis.call_quality_reasoning = score_call_quality(analysis)
    return analysis


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_sales_analyst() -> Agent:
    return Agent(
        role="Sales Call Analyst",
        goal="Extract maximum insight from sales conversations to improve close rates",
        backstory=(
            "You are a VP of Sales turned AI analyst. You've listened to thousands "
            "of sales calls and know exactly what wins deals. You analyze sentiment, "
            "identify objections, track competitor mentions, extract buying signals, "
            "and generate actionable coaching notes for sales reps."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


# ---------------------------------------------------------------------------
# Task factories
# ---------------------------------------------------------------------------

def create_transcript_analysis_task(transcript: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Analyze the following sales call transcript:\n\n{transcript}\n\n"
            "Extract: overall sentiment (buyer and seller), key conversation moments, "
            "objections raised, competitor mentions, buying signals, "
            "price sensitivity indicators, next steps agreed, and action items. "
            "Rate the call quality 1-10 with reasoning."
        ),
        expected_output="Call analysis report with sentiment timeline, objections, buying signals, and scored action items",
        agent=agent,
    )


def create_sentiment_deep_dive_task(transcript: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Perform a deep sentiment analysis of this sales call transcript:\n\n{transcript}\n\n"
            "For each speaker turn, classify sentiment as positive/neutral/negative. "
            "Identify sentiment shifts and their triggers. "
            "Map the emotional arc of the conversation. "
            "Highlight moments where the buyer opened up or shut down."
        ),
        expected_output="Sentiment arc map with per-turn classifications, shift triggers, and emotional turning points",
        agent=agent,
    )


def create_objection_tracking_task(transcript: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Analyze this sales call transcript for objections:\n\n{transcript}\n\n"
            "Identify every objection raised by the buyer. "
            "Categorize each: price, timing, trust, competitor, need, authority. "
            "Evaluate whether the seller handled each objection effectively. "
            "Suggest better responses for any unhandled objections."
        ),
        expected_output="Objection log with categories, handling effectiveness scores, and suggested responses",
        agent=agent,
    )


def create_competitor_analysis_task(transcript: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Analyze this sales call transcript for competitive intelligence:\n\n{transcript}\n\n"
            "Identify all competitor mentions. "
            "Extract what the buyer said about each competitor (positive/negative). "
            "Identify competitive advantages to emphasize. "
            "Suggest competitive positioning strategies."
        ),
        expected_output="Competitive intelligence report with mentions, buyer perceptions, and positioning recommendations",
        agent=agent,
    )


def create_follow_up_task(agent: Agent) -> Task:
    return Task(
        description=(
            "Based on the call analysis, generate: a personalized follow-up email "
            "that addresses objections raised, key value propositions to reinforce, "
            "a risk of inaction statement, and a clear next step with timing. "
            "Also generate a CRM activity log entry."
        ),
        expected_output="Follow-up email draft, value reinforcement points, next step recommendation, CRM activity log",
        agent=agent,
    )


def create_coaching_task(transcript: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Review this sales call transcript and generate coaching notes:\n\n{transcript}\n\n"
            "Identify 3 things the seller did well. "
            "Identify 3 areas for improvement with specific suggestions. "
            "Provide a call quality score 1-10 with detailed reasoning. "
            "Suggest specific techniques or frameworks the seller should practice."
        ),
        expected_output="Coaching report with strengths, improvement areas, score, and practice recommendations",
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class SalesCallWorkflow:
    """Sales call analysis workflow."""

    def __init__(self):
        self.agent = create_sales_analyst()

    def analyze(self, transcript: str) -> str:
        """Full analysis: transcript + follow-up generation."""
        analysis_task = create_transcript_analysis_task(transcript, self.agent)
        followup_task = create_follow_up_task(self.agent)

        crew = Crew(
            agents=[self.agent],
            tasks=[analysis_task, followup_task],
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff()

    def analyze_sentiment(self, transcript: str) -> str:
        """Deep sentiment analysis only."""
        task = create_sentiment_deep_dive_task(transcript, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def track_objections(self, transcript: str) -> str:
        """Objection tracking and handling analysis."""
        task = create_objection_tracking_task(transcript, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def analyze_competitors(self, transcript: str) -> str:
        """Competitive intelligence extraction."""
        task = create_competitor_analysis_task(transcript, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def generate_coaching(self, transcript: str) -> str:
        """Generate coaching notes for the sales rep."""
        task = create_coaching_task(transcript, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def full_analysis_pipeline(self, transcript: str) -> str:
        """Run all analysis tasks in sequence."""
        tasks = [
            create_transcript_analysis_task(transcript, self.agent),
            create_sentiment_deep_dive_task(transcript, self.agent),
            create_objection_tracking_task(transcript, self.agent),
            create_competitor_analysis_task(transcript, self.agent),
            create_coaching_task(transcript, self.agent),
            create_follow_up_task(self.agent),
        ]
        crew = Crew(
            agents=[self.agent],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff()
