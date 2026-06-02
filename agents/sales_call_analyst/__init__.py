"""Sales Call Analyst Agent.

Analyzes sales call recordings/transcripts for sentiment,
key moments, objections, competitor mentions, and action items.

Provides:
    - Transcript parsing and speaker identification
    - Sentiment analysis (per-speaker and per-segment)
    - Objection detection and categorization
    - Competitor mention tracking
    - Buying signal identification
    - Call quality scoring
    - Follow-up email generation
    - CRM activity log generation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class ObjectionCategory(str, Enum):
    """Categories of sales objections."""
    PRICE = "price"
    TIMING = "timing"
    AUTHORITY = "authority"
    NEED = "need"
    TRUST = "trust"
    COMPETITOR = "competitor"
    FEATURE = "feature"
    OTHER = "other"


class BuyingSignal(str, Enum):
    """Types of buying signals detected in transcripts."""
    URGENCY = "urgency"
    BUDGET_CONFIRMED = "budget_confirmed"
    DECISION_MAKER_PRESENT = "decision_maker_present"
    EXPLICIT_INTEREST = "explicit_interest"
    REFERRAL_MENTION = "referral_mention"
    COMPARISON_REQUEST = "comparison_request"
    TRIAL_REQUEST = "trial_request"
    PRICING_DISCUSSION = "pricing_discussion"


@dataclass
class TranscriptSegment:
    """A single segment of a sales call transcript."""
    speaker: str
    text: str
    timestamp: Optional[str] = None
    sentiment: Optional[SentimentLabel] = None
    sentiment_score: float = 0.0  # -1.0 to 1.0

    def __str__(self) -> str:
        ts = f"[{self.timestamp}] " if self.timestamp else ""
        return f"{ts}{self.speaker}: {self.text}"


@dataclass
class Objection:
    """A detected objection with context."""
    category: ObjectionCategory
    text: str
    resolved: bool = False
    timestamp: Optional[str] = None
    response: str = ""

    def __str__(self) -> str:
        status = "resolved" if self.resolved else "unresolved"
        return f"[{self.category.value.upper()}] {self.text} ({status})"


@dataclass
class CompetitorMention:
    """A mention of a competitor in the transcript."""
    competitor_name: str
    context: str
    sentiment: SentimentLabel = SentimentLabel.NEUTRAL
    timestamp: Optional[str] = None


@dataclass
class CallAnalysis:
    """Complete analysis result for a sales call."""
    # Metadata
    transcript_raw: str = ""
    segments: list[TranscriptSegment] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    duration_minutes: float = 0.0

    # Sentiment
    overall_sentiment: SentimentLabel = SentimentLabel.NEUTRAL
    overall_sentiment_score: float = 0.0
    buyer_sentiment: SentimentLabel = SentimentLabel.NEUTRAL
    seller_sentiment: SentimentLabel = SentimentLabel.NEUTRAL
    sentiment_timeline: list[dict] = field(default_factory=list)

    # Objections
    objections: list[Objection] = field(default_factory=list)
    objections_resolved: int = 0
    objections_unresolved: int = 0

    # Competitors
    competitor_mentions: list[CompetitorMention] = field(default_factory=list)

    # Buying signals
    buying_signals: list[dict] = field(default_factory=list)

    # Scoring
    call_quality_score: int = 0  # 1-10
    call_quality_reasoning: str = ""

    # Action items
    next_steps: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)

    # Follow-up
    follow_up_email: str = ""
    crm_activity_log: str = ""


@dataclass
class FollowUpContent:
    """Generated follow-up content from call analysis."""
    email_subject: str = ""
    email_body: str = ""
    crm_activity_log: str = ""
    key_value_props: list[str] = field(default_factory=list)
    risk_of_inaction: str = ""
    next_step: str = ""
    next_step_timing: str = ""

    def __str__(self) -> str:
        parts = [f"Subject: {self.email_subject}", "", self.email_body, ""]
        if self.key_value_props:
            parts.append("Key Value Propositions:")
            for v in self.key_value_props:
                parts.append(f"  - {v}")
        if self.risk_of_inaction:
            parts.append(f"\nRisk of Inaction: {self.risk_of_inaction}")
        if self.next_step:
            parts.append(f"Next Step: {self.next_step}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

# Regex patterns for common transcript formats
_TRANSCRIPT_PATTERNS = [
    # Pattern: "[HH:MM:SS] Speaker: text" — must come before plain "Speaker: text"
    re.compile(r"^\s*\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*([^:]+):\s*(.+)$"),
    # Pattern: "Speaker: text"
    re.compile(r"^\s*([^:]+):\s*(.+)$"),
    # Pattern: "Speaker (HH:MM): text"
    re.compile(r"^\s*([^(]+)\s*\((\d{1,2}:\d{2}(?::\d{2})?)\):\s*(.+)$"),
]


def parse_transcript(transcript: str) -> list[TranscriptSegment]:
    """Parse a raw transcript string into a list of TranscriptSegments.

    Supports formats:
        - "Speaker: text"
        - "[HH:MM:SS] Speaker: text"
        - "Speaker (HH:MM): text"

    Lines that don't match any pattern are appended to the last segment.
    Returns an empty list if transcript is empty.
    """
    if not transcript or not transcript.strip():
        return []

    segments: list[TranscriptSegment] = []
    lines = transcript.strip().splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        matched = False
        for pat in _TRANSCRIPT_PATTERNS:
            m = pat.match(line)
            if m:
                groups = m.groups()
                if len(groups) == 3:
                    timestamp, speaker, text = groups
                elif len(groups) == 2:
                    timestamp = None
                    speaker, text = groups
                else:
                    continue
                segments.append(TranscriptSegment(
                    speaker=speaker.strip(),
                    text=text.strip(),
                    timestamp=timestamp.strip() if timestamp else None,
                ))
                matched = True
                break

        if not matched and segments:
            # Append continuation text to the last segment
            segments[-1].text += " " + line

    return segments


def extract_speakers(segments: list[TranscriptSegment]) -> list[str]:
    """Extract unique speaker names in order of appearance."""
    seen: set[str] = set()
    speakers: list[str] = []
    for seg in segments:
        if seg.speaker not in seen:
            seen.add(seg.speaker)
            speakers.append(seg.speaker)
    return speakers


# ---------------------------------------------------------------------------
# Keyword-based heuristic analysis (works without LLM)
# ---------------------------------------------------------------------------

# Sentiment keyword weights (simplified heuristic)
_POSITIVE_WORDS = {
    "great": 0.6, "love": 0.7, "excellent": 0.8, "perfect": 0.9,
    "amazing": 0.8, "fantastic": 0.8, "wonderful": 0.7, "awesome": 0.7,
    "impressive": 0.6, "interested": 0.5, "excited": 0.6, "looking forward": 0.6,
    "good": 0.4, "nice": 0.4, "helpful": 0.5, "valuable": 0.6,
    "benefit": 0.4, "agree": 0.4, "yes": 0.4, "absolutely": 0.6,
    "definitely": 0.6, "sure": 0.4, "sounds good": 0.6, "makes sense": 0.5,
    "like": 0.3, "happy": 0.6, "pleased": 0.5, "thank": 0.3,
    "thanks": 0.3, "appreciate": 0.5, "convinced": 0.6, "ready": 0.5,
    "move forward": 0.6, "sign": 0.7, "deal": 0.5, "sold": 0.7,
}

_NEGATIVE_WORDS = {
    "expensive": -0.6, "costly": -0.6, "overpriced": -0.7, "too much": -0.5,
    "concern": -0.4, "concerned": -0.5, "worry": -0.5, "worried": -0.5,
    "problem": -0.5, "issue": -0.4, "difficult": -0.4, "struggle": -0.5,
    "not sure": -0.4, "hesitant": -0.5, "doubt": -0.5, "doubtful": -0.6,
    "no": -0.3, "can't": -0.4, "cannot": -0.4, "won't": -0.4, "refuse": -0.6,
    "unfortunately": -0.4, "disappointed": -0.6, "frustrated": -0.6,
    "bad": -0.5, "terrible": -0.7, "horrible": -0.8, "awful": -0.7,
    "wrong": -0.4, "fail": -0.6, "failed": -0.6, "missing": -0.3,
    "lacking": -0.4, "doesn't work": -0.6, "not working": -0.6,
    "not interested": -0.6, "pass": -0.5, "walk away": -0.6,
}

_OBJECTION_KEYWORDS: dict[ObjectionCategory, list[str]] = {
    ObjectionCategory.PRICE: [
        "too expensive", "overpriced", "can't afford", "budget",
        "too much", "pricey", "costly", "cheaper", "discount",
        "better price", "lower price", "competitor pricing",
        "expensive",
    ],
    ObjectionCategory.TIMING: [
        "not now", "too early", "too late", "not the right time",
        "maybe later", "too busy", "no time", "later",
        "next quarter", "next year", "postpone", "delay",
    ],
    ObjectionCategory.AUTHORITY: [
        "need to check", "talk to my", "boss", "manager",
        "decision maker", "committee", "approval", "my team",
        "not my decision", "need sign-off", "need to ask",
    ],
    ObjectionCategory.NEED: [
        "don't need", "not necessary", "already have",
        "satisfied with", "happy with current", "no need",
        "doesn't fit", "not a priority", "not relevant",
    ],
    ObjectionCategory.TRUST: [
        "not sure", "uncertain", "risky", "concerned about",
        "heard bad", "reviews", "track record", "guarantee",
        "proof", "case study", "reference", "trust",
    ],
    ObjectionCategory.COMPETITOR: [
        "competitor", "already using", "switching from",
        "other provider", "compared to", "alternative",
        "competing", "other option", "different vendor",
    ],
    ObjectionCategory.FEATURE: [
        "missing feature", "doesn't have", "need to integrate",
        "compatibility", "limitation", "not supported",
        "customization", "not flexible",
    ],
}

_BUYING_SIGNALS: dict[BuyingSignal, list[str]] = {
    BuyingSignal.URGENCY: [
        "asap", "urgently", "right away", "immediately",
        "this week", "this month", "deadline", "running out",
        "need it now", "time-sensitive",
    ],
    BuyingSignal.BUDGET_CONFIRMED: [
        "budget approved", "have budget", "funding allocated",
        "financially ready", "pre-approved", "allocated budget",
    ],
    BuyingSignal.DECISION_MAKER_PRESENT: [
        "i decide", "i'm the decision maker", "i can approve",
        "my call", "i'll sign", "authorize", "i have authority",
    ],
    BuyingSignal.EXPLICIT_INTEREST: [
        "very interested", "want to proceed", "sign up",
        "get started", "move forward", "let's do it",
        "count me in", "i'm in", "ready to buy",
    ],
    BuyingSignal.REFERRAL_MENTION: [
        "referred by", "heard from", "colleague recommended",
        "friend told", "existing customer recommended",
    ],
    BuyingSignal.COMPARISON_REQUEST: [
        "compare", "vs", "versus", "difference between",
        "how are you different", "why should i choose",
    ],
    BuyingSignal.TRIAL_REQUEST: [
        "free trial", "try before", "demo", "pilot",
        "proof of concept", "test drive", "evaluate",
    ],
    BuyingSignal.PRICING_DISCUSSION: [
        "what's the cost", "how much", "pricing",
        "payment plan", "roi", "return on investment",
        "total cost", "subscription", "license fee",
    ],
}


def _text_to_lowercase(text: str) -> str:
    return text.lower()


def detect_sentiment(text: str) -> tuple[SentimentLabel, float]:
    """Detect sentiment of a text using keyword heuristics.

    Returns a tuple of (label, score) where score is -1.0 to 1.0.
    """
    text_lower = _text_to_lowercase(text)

    score = 0.0
    matches = 0

    for word, weight in _POSITIVE_WORDS.items():
        if word in text_lower:
            score += weight
            matches += 1

    for word, weight in _NEGATIVE_WORDS.items():
        if word in text_lower:
            score += weight  # weight is already negative
            matches += 1

    if matches == 0:
        return SentimentLabel.NEUTRAL, 0.0

    # Normalize to [-1, 1]
    normalized = max(-1.0, min(1.0, score / max(matches * 0.5, 1.0)))

    if normalized >= 0.6:
        label = SentimentLabel.VERY_POSITIVE
    elif normalized >= 0.2:
        label = SentimentLabel.POSITIVE
    elif normalized <= -0.6:
        label = SentimentLabel.VERY_NEGATIVE
    elif normalized <= -0.2:
        label = SentimentLabel.NEGATIVE
    else:
        label = SentimentLabel.NEUTRAL

    return label, round(normalized, 3)


def detect_objections(text: str, timestamp: Optional[str] = None) -> list[Objection]:
    """Detect objections in text using keyword heuristics.

    Returns a list of Objection objects.
    """
    text_lower = _text_to_lowercase(text)
    objections: list[Objection] = []

    for category, keywords in _OBJECTION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                objections.append(Objection(
                    category=category,
                    text=text,
                    resolved=False,
                    timestamp=timestamp,
                ))
                break  # One match per category per segment

    return objections


def detect_buying_signals(text: str, timestamp: Optional[str] = None) -> list[dict]:
    """Detect buying signals in text using keyword heuristics.

    Returns a list of dicts with signal type and text.
    """
    text_lower = _text_to_lowercase(text)
    signals: list[dict] = []

    for signal_type, keywords in _BUYING_SIGNALS.items():
        for keyword in keywords:
            if keyword in text_lower:
                signals.append({
                    "type": signal_type.value,
                    "text": text,
                    "keyword": keyword,
                    "timestamp": timestamp,
                })
                break

    return signals


def detect_competitor_mentions(text: str) -> list[str]:
    """Detect potential competitor mentions using capitalization heuristics.

    Looks for capitalized proper nouns not at the start of sentences
    that are likely company/product names.
    """
    # Simple heuristic: capitalized words (2+ chars) in the middle of text
    # Exclude common sentence starters
    text_lower = _text_to_lowercase(text)
    # Known common words to exclude
    common_words = {
        "i", "we", "they", "he", "she", "it", "you", "the", "a", "an",
        "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "shall", "can", "need", "dare", "ought", "used",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "not", "no", "nor", "as", "but", "or", "and", "if", "then",
        "than", "that", "this", "these", "those", "what", "which",
    }

    # Find capitalized words (potential proper nouns)
    words = re.findall(r'\b[A-Z][a-z]{1,}\b', text)
    mentions: list[str] = []
    seen: set[str] = set()

    for w in words:
        lower_w = w.lower()
        if lower_w not in common_words and lower_w not in _text_to_lowercase(w) or w[0].isupper():
            if lower_w not in common_words and w not in seen:
                seen.add(w)
                mentions.append(w)

    return mentions[:5]  # Cap to reduce noise


# ---------------------------------------------------------------------------
# Analysis engine
# ---------------------------------------------------------------------------


def analyze_call(
    transcript: str,
    buyer_speaker: Optional[str] = None,
    seller_speaker: Optional[str] = None,
) -> CallAnalysis:
    """Perform a full heuristic analysis of a sales call transcript.

    This works entirely offline using keyword heuristics — no LLM required.
    For deeper analysis, use the LLM-powered SalesCallWorkflow instead.

    Args:
        transcript: Raw transcript text.
        buyer_speaker: Name of the buyer speaker (if known).
        seller_speaker: Name of the seller speaker (if known).

    Returns:
        CallAnalysis with all detected signals, objections, and scores.
    """
    analysis = CallAnalysis(transcript_raw=transcript)

    # Parse
    segments = parse_transcript(transcript)
    analysis.segments = segments
    analysis.speakers = extract_speakers(segments)

    if not segments:
        return analysis

    # Auto-detect buyer/seller if not provided
    speakers = analysis.speakers
    if buyer_speaker is None and len(speakers) >= 2:
        buyer_speaker = speakers[1]  # Assume second speaker is buyer
    if seller_speaker is None and speakers:
        seller_speaker = speakers[0]  # Assume first speaker is seller

    # Per-segment analysis
    buyer_scores: list[float] = []
    seller_scores: list[float] = []

    for seg in segments:
        # Sentiment
        label, score = detect_sentiment(seg.text)
        seg.sentiment = label
        seg.sentiment_score = score

        if seg.speaker == buyer_speaker:
            buyer_scores.append(score)
        elif seg.speaker == seller_speaker:
            seller_scores.append(score)

        analysis.sentiment_timeline.append({
            "speaker": seg.speaker,
            "sentiment": label.value,
            "score": score,
            "timestamp": seg.timestamp,
        })

        # Objections
        objections = detect_objections(seg.text, seg.timestamp)
        analysis.objections.extend(objections)

        # Buying signals
        signals = detect_buying_signals(seg.text, seg.timestamp)
        analysis.buying_signals.extend(signals)

        # Competitor mentions
        mentions = detect_competitor_mentions(seg.text)
        for mention in mentions:
            analysis.competitor_mentions.append(CompetitorMention(
                competitor_name=mention,
                context=seg.text,
                timestamp=seg.timestamp,
            ))

    # Aggregate sentiment
    if buyer_scores:
        avg_buyer = sum(buyer_scores) / len(buyer_scores)
        analysis.buyer_sentiment = _score_to_label(avg_buyer)

    if seller_scores:
        avg_seller = sum(seller_scores) / len(seller_scores)
        analysis.seller_sentiment = _score_to_label(avg_seller)

    all_scores = buyer_scores + seller_scores
    if all_scores:
        avg_all = sum(all_scores) / len(all_scores)
        analysis.overall_sentiment_score = round(avg_all, 3)
        analysis.overall_sentiment = _score_to_label(avg_all)

    # Objection summary
    analysis.objections_resolved = sum(1 for o in analysis.objections if o.resolved)
    analysis.objections_unresolved = sum(1 for o in analysis.objections if not o.resolved)

    # Call quality score
    analysis.call_quality_score, analysis.call_quality_reasoning = _score_call_quality(
        analysis, buyer_scores, seller_scores,
    )

    return analysis


def _score_to_label(score: float) -> SentimentLabel:
    if score >= 0.6:
        return SentimentLabel.VERY_POSITIVE
    elif score >= 0.2:
        return SentimentLabel.POSITIVE
    elif score <= -0.6:
        return SentimentLabel.VERY_NEGATIVE
    elif score <= -0.2:
        return SentimentLabel.NEGATIVE
    else:
        return SentimentLabel.NEUTRAL


def _score_call_quality(
    analysis: CallAnalysis,
    buyer_scores: list[float],
    seller_scores: list[float],
) -> tuple[int, str]:
    """Score call quality from 1-10 based on multiple factors."""
    score = 5.0  # Start at neutral

    # Factor 1: Overall sentiment
    if analysis.overall_sentiment_score > 0.3:
        score += 1.5
    elif analysis.overall_sentiment_score > 0:
        score += 0.5
    elif analysis.overall_sentiment_score < -0.3:
        score -= 1.5
    elif analysis.overall_sentiment_score < 0:
        score -= 0.5

    # Factor 2: Objections (some objections = engagement, too many = bad)
    if analysis.objections:
        resolution_rate = (
            analysis.objections_resolved / len(analysis.objections)
            if analysis.objections else 0.5
        )
        if resolution_rate >= 0.7:
            score += 1.5
        elif resolution_rate >= 0.4:
            score += 0.5
        else:
            score -= 1.0
    else:
        score += 0.5  # No objections is slightly positive

    # Factor 3: Buying signals
    signal_count = len(analysis.buying_signals)
    if signal_count >= 3:
        score += 2.0
    elif signal_count >= 1:
        score += 1.0

    # Factor 4: Call depth (more segments = more engagement)
    if len(analysis.segments) >= 10:
        score += 0.5

    # Clamp to 1-10
    score = max(1.0, min(10.0, score))
    final = round(score)

    # Build reasoning
    reasons: list[str] = []
    reasons.append(f"Overall sentiment: {analysis.overall_sentiment.value} ({analysis.overall_sentiment_score:+.2f})")
    reasons.append(f"Objections: {len(analysis.objections)} total, {analysis.objections_resolved} resolved")
    reasons.append(f"Buying signals detected: {signal_count}")
    reasons.append(f"Conversation depth: {len(analysis.segments)} segments")
    reasoning = "; ".join(reasons)

    return final, reasoning


# ---------------------------------------------------------------------------
# Follow-up generation (heuristic)
# ---------------------------------------------------------------------------


def generate_follow_up(analysis: CallAnalysis) -> FollowUpContent:
    """Generate follow-up content from a call analysis.

    Uses heuristic rules; for richer output, use the LLM-powered
    SalesCallWorkflow.follow_up() method.
    """
    follow_up = FollowUpContent()

    # Subject
    if analysis.overall_sentiment_score > 0.2:
        follow_up.email_subject = "Great speaking with you — next steps"
    elif analysis.overall_sentiment_score < -0.2:
        follow_up.email_subject = "Following up on our conversation"
    else:
        follow_up.email_subject = "Following up on our call"

    # Body
    body_parts: list[str] = []
    body_parts.append("Hi,")
    body_parts.append("")
    body_parts.append("Thank you for taking the time to speak with me today.")
    body_parts.append("")

    # Address objections
    unresolved = [o for o in analysis.objections if not o.resolved]
    if unresolved:
        body_parts.append("I wanted to address the concerns you raised:")
        body_parts.append("")
        for obj in unresolved[:3]:  # Address top 3
            body_parts.append(f"  • Regarding your {obj.category.value} concern: "
                             f"[We need to address: '{obj.text[:80]}...']")
        body_parts.append("")

    # Key value propositions based on buying signals
    if analysis.buying_signals:
        follow_up.key_value_props = [
            f"Signal: {s['type']} — {s['keyword']}" for s in analysis.buying_signals[:5]
        ]
        body_parts.append("Based on our discussion, here are the key benefits we can provide:")
        body_parts.append("")
        for prop in follow_up.key_value_props:
            body_parts.append(f"  • {prop}")
        body_parts.append("")

    # Risk of inaction
    if unresolved:
        follow_up.risk_of_inaction = (
            "Without addressing these concerns, you may miss out on "
            "the benefits we discussed."
        )
        body_parts.append(follow_up.risk_of_inaction)

    # Next step
    if analysis.next_steps:
        follow_up.next_step = analysis.next_steps[0]
        follow_up.next_step_timing = "within 3 business days"
    else:
        follow_up.next_step = "Schedule a follow-up call"
        follow_up.next_step_timing = "this week"

    body_parts.append("")
    body_parts.append(f"Next step: {follow_up.next_step}")
    body_parts.append(f"Timing: {follow_up.next_step_timing}")
    body_parts.append("")
    body_parts.append("Best regards,")
    body_parts.append("[Your Name]")

    follow_up.email_body = "\n".join(body_parts)

    # CRM log
    follow_up.crm_activity_log = _generate_crm_log(analysis)

    return follow_up


def _generate_crm_log(analysis: CallAnalysis) -> str:
    """Generate a CRM activity log entry from analysis."""
    lines: list[str] = []
    lines.append(f"Call Quality Score: {analysis.call_quality_score}/10")
    lines.append(f"Overall Sentiment: {analysis.overall_sentiment.value} ({analysis.overall_sentiment_score:+.2f})")
    lines.append(f"Buyer Sentiment: {analysis.buyer_sentiment.value}")
    lines.append(f"Seller Sentiment: {analysis.seller_sentiment.value}")
    lines.append(f"Objections: {len(analysis.objections)} "
                 f"({analysis.objections_resolved} resolved, {analysis.objections_unresolved} unresolved)")
    if analysis.objections:
        for obj in analysis.objections:
            lines.append(f"  - {obj}")
    if analysis.competitor_mentions:
        lines.append(f"Competitor Mentions: {', '.join(m.competitor_name for m in analysis.competitor_mentions)}")
    if analysis.buying_signals:
        lines.append(f"Buying Signals: {', '.join(s['type'] for s in analysis.buying_signals)}")
    if analysis.next_steps:
        lines.append(f"Next Steps: {'; '.join(analysis.next_steps)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CrewAI Agent & Tasks
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


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class SalesCallWorkflow:
    """Sales call analysis workflow.

    Provides both heuristic analysis (works offline) and LLM-powered
    analysis (richer, requires API key).
    """

    def __init__(self):
        self.agent = create_sales_analyst()

    def analyze(self, transcript: str) -> str:
        """Run the full CrewAI analysis workflow (LLM-powered)."""
        analysis_task = create_transcript_analysis_task(transcript, self.agent)
        followup_task = create_follow_up_task(self.agent)

        crew = Crew(
            agents=[self.agent],
            tasks=[analysis_task, followup_task],
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff()

    @staticmethod
    def analyze_heuristic(
        transcript: str,
        buyer_speaker: Optional[str] = None,
        seller_speaker: Optional[str] = None,
    ) -> CallAnalysis:
        """Run heuristic analysis (no LLM required).

        Returns a CallAnalysis object with all detected signals.
        """
        return analyze_call(transcript, buyer_speaker, seller_speaker)

    @staticmethod
    def generate_follow_up(analysis: CallAnalysis) -> FollowUpContent:
        """Generate follow-up content from a call analysis."""
        return generate_follow_up(analysis)

    @staticmethod
    def parse_transcript(transcript: str) -> list[TranscriptSegment]:
        """Parse a raw transcript into segments."""
        return parse_transcript(transcript)

    @staticmethod
    def detect_sentiments(transcript: str) -> list[tuple[str, str, float]]:
        """Detect sentiment per segment. Returns (speaker, label, score) tuples."""
        segments = parse_transcript(transcript)
        results: list[tuple[str, str, float]] = []
        for seg in segments:
            label, score = detect_sentiment(seg.text)
            results.append((seg.speaker, label.value, score))
        return results

    @staticmethod
    def detect_objections_in_transcript(transcript: str) -> list[Objection]:
        """Detect all objections in a transcript."""
        segments = parse_transcript(transcript)
        all_objections: list[Objection] = []
        for seg in segments:
            objs = detect_objections(seg.text, seg.timestamp)
            all_objections.extend(objs)
        return all_objections

    @staticmethod
    def detect_buying_signals_in_transcript(transcript: str) -> list[dict]:
        """Detect all buying signals in a transcript."""
        segments = parse_transcript(transcript)
        all_signals: list[dict] = []
        for seg in segments:
            signals = detect_buying_signals(seg.text, seg.timestamp)
            all_signals.extend(signals)
        return all_signals
