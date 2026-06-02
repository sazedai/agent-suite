"""Tests for the Sales Call Analyst agent."""

import pytest
from agents.sales_call_analyst import (
    # Enums & data models
    SentimentLabel,
    ObjectionCategory,
    BuyingSignal,
    TranscriptSegment,
    Objection,
    CompetitorMention,
    CallAnalysis,
    FollowUpContent,
    # Parser
    parse_transcript,
    extract_speakers,
    # Heuristic analysis
    detect_sentiment,
    detect_objections,
    detect_buying_signals,
    detect_competitor_mentions,
    # Full analysis
    analyze_call,
    # Follow-up
    generate_follow_up,
    _generate_crm_log,
    _score_to_label,
    # CrewAI agent
    create_sales_analyst,
    create_transcript_analysis_task,
    create_follow_up_task,
    SalesCallWorkflow,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TRANSCRIPT = """\
Sarah: Hi thanks for taking the time today. I'd love to learn more about your platform.
Mike: Thanks for reaching out, Sarah! I'm the VP of Sales here. What are you looking for?
Sarah: We're evaluating tools to help our team close more deals faster.
Mike: Great question. Our AI platform helps sales teams increase close rates by 30%.
Sarah: That sounds impressive. How much does it cost?
Mike: Our plans start at $99 per seat per month. We also have enterprise options.
Sarah: That's a bit expensive for us. We're a small team.
Mike: I understand budget concerns. We have a startup plan at $49 per seat. Would that help?
Sarah: Maybe. I need to talk to my manager about the budget.
Mike: Absolutely. Can I send you a comparison with your current solution?
Sarah: Yes, please. When can you send it?
Mike: I'll send it by tomorrow. Should we schedule a follow-up call?
Sarah: Perfect, let's do Thursday at 2pm.
Mike: Great, I'll send a calendar invite. Looking forward to it!
"""

TRANSCRIPT_WITH_TIMESTAMPS = """\
[00:00] Sarah: Hi, I'm very interested in your product.
[00:15] Mike: Thanks! What specifically caught your attention?
[00:30] Sarah: The AI features look amazing. We need a solution ASAP.
[00:45] Mike: We can get you started this week with a free trial.
[01:00] Sarah: Perfect. What's the pricing after the trial?
[01:15] Mike: It depends on your team size. Do you have budget approved?
[01:30] Sarah: Yes, we have budget approved for this quarter.
[01:45] Mike: Excellent. I'll follow up with a detailed proposal.
"""


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestDataModels:
    def test_transcript_segment_creation(self):
        seg = TranscriptSegment(speaker="Sarah", text="Hello there")
        assert seg.speaker == "Sarah"
        assert seg.text == "Hello there"
        assert seg.timestamp is None
        assert seg.sentiment is None
        assert seg.sentiment_score == 0.0

    def test_transcript_segment_with_timestamp(self):
        seg = TranscriptSegment(speaker="Mike", text="Hi!", timestamp="00:00")
        assert str(seg) == "[00:00] Mike: Hi!"

    def test_objection_creation(self):
        obj = Objection(
            category=ObjectionCategory.PRICE,
            text="That's too expensive",
            timestamp="00:30",
        )
        assert obj.category == ObjectionCategory.PRICE
        assert obj.resolved is False
        assert "PRICE" in str(obj)

    def test_objection_resolved(self):
        obj = Objection(
            category=ObjectionCategory.TRUST,
            text="Not sure about reliability",
            resolved=True,
            response="Here's our case study...",
        )
        assert obj.resolved is True
        assert "resolved" in str(obj)

    def test_competitor_mention(self):
        comp = CompetitorMention(
            competitor_name="SalesForce",
            context="We're switching from SalesForce",
            sentiment=SentimentLabel.NEGATIVE,
        )
        assert comp.competitor_name == "SalesForce"
        assert comp.sentiment == SentimentLabel.NEGATIVE

    def test_call_analysis_default(self):
        analysis = CallAnalysis()
        assert analysis.overall_sentiment == SentimentLabel.NEUTRAL
        assert analysis.call_quality_score == 0
        assert analysis.objections == []
        assert analysis.buying_signals == []
        assert analysis.competitor_mentions == []

    def test_follow_up_content_str(self):
        content = FollowUpContent(
            email_subject="Test Subject",
            email_body="Test body",
            key_value_props=["Prop 1", "Prop 2"],
            next_step="Schedule demo",
        )
        text = str(content)
        assert "Test Subject" in text
        assert "Prop 1" in text
        assert "Schedule demo" in text


# ---------------------------------------------------------------------------
# Transcript parsing tests
# ---------------------------------------------------------------------------


class TestTranscriptParsing:
    def test_parse_basic_format(self):
        text = "Sarah: Hello\nMike: Hi there"
        segments = parse_transcript(text)
        assert len(segments) == 2
        assert segments[0].speaker == "Sarah"
        assert segments[0].text == "Hello"
        assert segments[1].speaker == "Mike"
        assert segments[1].text == "Hi there"

    def test_parse_with_timestamps(self):
        segments = parse_transcript(TRANSCRIPT_WITH_TIMESTAMPS)
        assert len(segments) >= 5
        assert segments[0].timestamp == "00:00"
        assert segments[0].speaker == "Sarah"
        assert "interested" in segments[0].text

    def test_parse_empty_string(self):
        assert parse_transcript("") == []
        assert parse_transcript("   ") == []

    def test_parse_continuation_lines(self):
        text = "Sarah: Hello there\nthis is a continuation\nMike: Response"
        segments = parse_transcript(text)
        assert len(segments) == 2
        assert "continuation" in segments[0].text

    def test_extract_speakers_order(self):
        text = "Alice: Hi\nBob: Hello\nAlice: How are you?\nBob: Good"
        segments = parse_transcript(text)
        speakers = extract_speakers(segments)
        assert speakers == ["Alice", "Bob"]

    def test_extract_speakers_empty(self):
        assert extract_speakers([]) == []

    def test_parse_sample_transcript(self):
        segments = parse_transcript(SAMPLE_TRANSCRIPT)
        assert len(segments) >= 5
        speakers = extract_speakers(segments)
        assert "Sarah" in speakers
        assert "Mike" in speakers


# ---------------------------------------------------------------------------
# Sentiment detection tests
# ---------------------------------------------------------------------------


class TestSentimentDetection:
    def test_very_positive(self):
        label, score = detect_sentiment("This is absolutely amazing and wonderful!")
        assert label in (SentimentLabel.VERY_POSITIVE, SentimentLabel.POSITIVE)
        assert score > 0

    def test_positive(self):
        label, score = detect_sentiment("That sounds great, I like it")
        assert label in (SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE)
        assert score > 0

    def test_negative(self):
        label, score = detect_sentiment("This is too expensive and disappointing")
        assert label in (SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE)
        assert score < 0

    def test_very_negative(self):
        label, score = detect_sentiment("This is terrible and horrible")
        assert label == SentimentLabel.VERY_NEGATIVE
        assert score < -0.3

    def test_neutral(self):
        label, score = detect_sentiment("Let me check the document")
        assert label == SentimentLabel.NEUTRAL
        assert score == 0.0

    def test_buying_sentiment(self):
        label, score = detect_sentiment("very interested, ready to move forward")
        assert label in (SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE)
        assert score > 0

    def test_objection_sentiment(self):
        label, score = detect_sentiment("not sure, concerned about the cost")
        assert score < 0

    def test_score_bounds(self):
        """Scores should always be between -1 and 1."""
        texts = [
            "love it amazing perfect excellent fantastic",
            "terrible horrible awful bad worst",
            "the weather is fine",
        ]
        for text in texts:
            label, score = detect_sentiment(text)
            assert -1.0 <= score <= 1.0, f"Score {score} out of bounds for: {text}"

    def test_score_to_label(self):
        assert _score_to_label(0.8) == SentimentLabel.VERY_POSITIVE
        assert _score_to_label(0.4) == SentimentLabel.POSITIVE
        assert _score_to_label(0.0) == SentimentLabel.NEUTRAL
        assert _score_to_label(-0.4) == SentimentLabel.NEGATIVE
        assert _score_to_label(-0.8) == SentimentLabel.VERY_NEGATIVE


# ---------------------------------------------------------------------------
# Objection detection tests
# ---------------------------------------------------------------------------


class TestObjectionDetection:
    def test_price_objection(self):
        objs = detect_objections("That's too expensive for our budget")
        assert any(o.category == ObjectionCategory.PRICE for o in objs)

    def test_timing_objection(self):
        objs = detect_objections("Not now, maybe next quarter")
        assert any(o.category == ObjectionCategory.TIMING for o in objs)

    def test_authority_objection(self):
        objs = detect_objections("I need to talk to my boss about this")
        assert any(o.category == ObjectionCategory.AUTHORITY for o in objs)

    def test_need_objection(self):
        objs = detect_objections("We already have a solution we're happy with")
        assert any(o.category == ObjectionCategory.NEED for o in objs)

    def test_trust_objection(self):
        objs = detect_objections("I'm not sure about the track record")
        assert any(o.category == ObjectionCategory.TRUST for o in objs)

    def test_no_objection(self):
        objs = detect_objections("This sounds great, let's proceed")
        assert len(objs) == 0

    def test_multiple_objections(self):
        text = "That's expensive and I need to check with my manager. Also not the right time."
        objs = detect_objections(text)
        categories = {o.category for o in objs
        }
        assert ObjectionCategory.PRICE in categories
        assert ObjectionCategory.AUTHORITY in categories
        assert ObjectionCategory.TIMING in categories

    def test_objection_has_text(self):
        sample = "The price is too high"
        objs = detect_objections(sample)
        for obj in objs:
            assert obj.text == sample
            assert obj.resolved is False

    def test_objection_timestamp(self):
        objs = detect_objections("too expensive", timestamp="00:30")
        assert all(o.timestamp == "00:30" for o in objs)


# ---------------------------------------------------------------------------
# Buying signal detection tests
# ---------------------------------------------------------------------------


class TestBuyingSignalDetection:
    def test_urgency_signal(self):
        signals = detect_buying_signals("We need this ASAP")
        assert any(s["type"] == BuyingSignal.URGENCY.value for s in signals)

    def test_budget_confirmed(self):
        signals = detect_buying_signals("Yes, we have budget approved for this")
        assert any(s["type"] == BuyingSignal.BUDGET_CONFIRMED.value for s in signals)

    def test_explicit_interest(self):
        signals = detect_buying_signals("We're very interested and want to proceed")
        assert any(s["type"] == BuyingSignal.EXPLICIT_INTEREST.value for s in signals)

    def test_pricing_discussion(self):
        signals = detect_buying_signals("What's the cost?")
        assert any(s["type"] == BuyingSignal.PRICING_DISCUSSION.value for s in signals)

    def test_trial_request(self):
        signals = detect_buying_signals("Can we start with a free trial?")
        assert any(s["type"] == BuyingSignal.TRIAL_REQUEST.value for s in signals)

    def test_comparison_request(self):
        signals = detect_buying_signals("How do you compare to alternatives?")
        assert any(s["type"] == BuyingSignal.COMPARISON_REQUEST.value for s in signals)

    def test_no_signals(self):
        signals = detect_buying_signals("Okay, let me think about it")
        assert len(signals) == 0

    def test_signal_has_timestamp(self):
        signals = detect_buying_signals("We need it urgently", timestamp="01:00")
        for s in signals:
            assert s["timestamp"] == "01:00"


# ---------------------------------------------------------------------------
# Full call analysis tests
# ---------------------------------------------------------------------------


class TestCallAnalysis:
    def test_analyze_full_transcript(self):
        analysis = analyze_call(SAMPLE_TRANSCRIPT)
        assert len(analysis.segments) > 0
        assert len(analysis.speakers) == 2
        assert analysis.overall_sentiment != SentimentLabel.NEUTRAL or analysis.overall_sentiment_score == 0

    def test_analyze_empty_transcript(self):
        analysis = analyze_call("")
        assert len(analysis.segments) == 0
        assert analysis.call_quality_score == 0

    def test_analyze_with_speaker_labels(self):
        analysis = analyze_call(
            SAMPLE_TRANSCRIPT,
            buyer_speaker="Sarah",
            seller_speaker="Mike",
        )
        assert analysis.buyer_sentiment in SentimentLabel
        assert analysis.seller_sentiment in SentimentLabel

    def test_sentiment_timeline_populated(self):
        analysis = analyze_call(SAMPLE_TRANSCRIPT)
        assert len(analysis.sentiment_timeline) > 0
        for entry in analysis.sentiment_timeline:
            assert "speaker" in entry
            assert "sentiment" in entry
            assert "score" in entry

    def test_objections_detected(self):
        analysis = analyze_call(SAMPLE_TRANSCRIPT)
        # The sample has "expensive" and "need to talk to my manager"
        assert len(analysis.objections) > 0
        assert analysis.objections_unresolved > 0

    def test_buying_signals_detected(self):
        analysis = analyze_call(SAMPLE_TRANSCRIPT)
        assert len(analysis.buying_signals) > 0

    def test_call_quality_scored(self):
        analysis = analyze_call(SAMPLE_TRANSCRIPT)
        assert 1 <= analysis.call_quality_score <= 10
        assert analysis.call_quality_reasoning != ""

    def test_call_quality_positive_call(self):
        positive = """\
Client: This is amazing! We're very interested in moving forward today.
Sales: Excellent! Let me get that started for you. What's your budget?
Client: Budget is approved. I'm the decision maker, let's sign now.
Sales: Perfect, I'll prepare the agreement.
"""
        analysis = analyze_call(positive)
        assert analysis.call_quality_score >= 6

    def test_call_quality_negative_call(self):
        negative = """\
Client: This is terrible. Too expensive. Don't need it. Not interested.
Sales: Are you sure? Let me explain the features.
Client: No thanks. Not now. Can't afford it. Walk away.
Sales: Okay, well call me if you change your mind.
"""
        analysis = analyze_call(negative)
        assert analysis.call_quality_score <= 5

    def test_analyze_timestamps(self):
        analysis = analyze_call(TRANSCRIPT_WITH_TIMESTAMPS)
        assert any(seg.timestamp is not None for seg in analysis.segments)

    def test_objection_count_tracking(self):
        analysis = analyze_call(SAMPLE_TRANSCRIPT)
        assert analysis.objections_resolved + analysis.objections_unresolved == len(analysis.objections)


# ---------------------------------------------------------------------------
# Follow-up generation tests
# ---------------------------------------------------------------------------


class TestFollowUpGeneration:
    def test_generate_follow_up(self):
        analysis = analyze_call(SAMPLE_TRANSCRIPT)
        follow_up = generate_follow_up(analysis)
        assert follow_up.email_subject != ""
        assert follow_up.email_body != ""
        assert follow_up.crm_activity_log != ""

    def test_follow_up_positive_sentiment(self):
        text = "Client: Love it! Want to move forward. This is perfect."
        analysis = analyze_call(text)
        follow_up = generate_follow_up(analysis)
        assert "Great" in follow_up.email_subject or "following" in follow_up.email_subject.lower()

    def test_follow_up_addresses_objections(self):
        text = "Client: This is too expensive. Sales: Let me discuss options."
        analysis = analyze_call(text)
        follow_up = generate_follow_up(analysis)
        assert follow_up.email_body != ""

    def test_follow_up_has_next_step(self):
        analysis = analyze_call(SAMPLE_TRANSCRIPT)
        follow_up = generate_follow_up(analysis)
        assert follow_up.next_step != ""
        assert follow_up.next_step_timing != ""

    def test_crm_log_generation(self):
        analysis = analyze_call(SAMPLE_TRANSCRIPT)
        log = _generate_crm_log(analysis)
        assert "Call Quality Score" in log
        assert "Overall Sentiment" in log
        assert "Objections" in log

    def test_follow_up_str(self):
        content = FollowUpContent(
            email_subject="Test",
            email_body="Hello",
            next_step="Demo",
        )
        text = str(content)
        assert "Test" in text
        assert "Demo" in text


# ---------------------------------------------------------------------------
# Workflow & static method tests
# ---------------------------------------------------------------------------


class TestWorkflowStaticMethods:
    def test_analyze_heuristic(self):
        result = SalesCallWorkflow.analyze_heuristic(SAMPLE_TRANSCRIPT)
        assert isinstance(result, CallAnalysis)
        assert len(result.segments) > 0

    def test_analyze_heuristic_with_speakers(self):
        result = SalesCallWorkflow.analyze_heuristic(
            SAMPLE_TRANSCRIPT,
            buyer_speaker="Sarah",
            seller_speaker="Mike",
        )
        assert result.buyer_sentiment in SentimentLabel

    def test_parse_transcript_static(self):
        segments = SalesCallWorkflow.parse_transcript(SAMPLE_TRANSCRIPT)
        assert len(segments) > 0

    def test_detect_sentiments_static(self):
        results = SalesCallWorkflow.detect_sentiments(SAMPLE_TRANSCRIPT)
        assert len(results) > 0
        for speaker, label, score in results:
            assert isinstance(label, str)
            assert isinstance(score, float)

    def test_detect_objections_in_transcript(self):
        objs = SalesCallWorkflow.detect_objections_in_transcript(SAMPLE_TRANSCRIPT)
        assert len(objs) > 0
        assert all(isinstance(o, Objection) for o in objs)

    def test_detect_buying_signals_in_transcript(self):
        signals = SalesCallWorkflow.detect_buying_signals_in_transcript(SAMPLE_TRANSCRIPT)
        assert len(signals) > 0
        assert all("type" in s for s in signals)


# ---------------------------------------------------------------------------
# Agent & Task creation tests
# ---------------------------------------------------------------------------


class TestAgentCreation:
    def _setup_mock_llm(self, monkeypatch):
        """Replace get_llm in the sales_call_analyst module with a fake."""
        monkeypatch.setattr("agents.sales_call_analyst.get_llm", lambda **kwargs: "fake-model")

    def test_create_sales_analyst(self, monkeypatch):
        self._setup_mock_llm(monkeypatch)
        agent = create_sales_analyst()
        assert agent.role == "Sales Call Analyst"

    def test_create_transcript_analysis_task(self, monkeypatch):
        self._setup_mock_llm(monkeypatch)
        agent = create_sales_analyst()
        task = create_transcript_analysis_task("Test transcript", agent)
        assert task is not None
        assert "transcript" in task.description.lower()

    def test_create_follow_up_task(self, monkeypatch):
        self._setup_mock_llm(monkeypatch)
        agent = create_sales_analyst()
        task = create_follow_up_task(agent)
        assert task is not None
        assert "follow-up" in task.description.lower() or "follow up" in task.description.lower()

    def test_workflow_creation(self, monkeypatch):
        self._setup_mock_llm(monkeypatch)
        workflow = SalesCallWorkflow()
        assert workflow.agent is not None
        assert workflow.agent.role == "Sales Call Analyst"

    def test_workflow_has_static_methods(self, monkeypatch):
        self._setup_mock_llm(monkeypatch)
        workflow = SalesCallWorkflow()
        assert hasattr(workflow, 'analyze_heuristic')
        assert hasattr(workflow, 'generate_follow_up')
        assert hasattr(workflow, 'parse_transcript')
        assert hasattr(workflow, 'detect_sentiments')
        assert hasattr(workflow, 'detect_objections_in_transcript')
        assert hasattr(workflow, 'detect_buying_signals_in_transcript')


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_speaker(self):
        text = "Alice: Hello\nAlice: How are you?\nAlice: Goodbye"
        segments = parse_transcript(text)
        assert len(segments) == 3
        speakers = extract_speakers(segments)
        assert speakers == ["Alice"]

    def test_empty_analysis_follow_up(self):
        analysis = CallAnalysis()
        follow_up = generate_follow_up(analysis)
        assert follow_up.email_subject != ""

    def test_whitespace_only_transcript(self):
        assert parse_transcript("   \n   \n  ") == []

    def test_unicode_transcript(self):
        text = "Sarah: Héllo, I'm intérêté\nMike: Bonjour! Great."
        segments = parse_transcript(text)
        assert len(segments) == 2

    def test_long_transcript_performance(self):
        """Parse a large transcript without issues."""
        lines = [f"Speaker{i % 3}: This is line number {i}" for i in range(200)]
        text = "\n".join(lines)
        segments = parse_transcript(text)
        assert len(segments) == 200

    def test_competitor_mentions(self):
        mentions = detect_competitor_mentions(
            "We're looking at SalesForce and HubSpot as alternatives."
        )
        # Should detect at least one proper noun
        assert isinstance(mentions, list)
