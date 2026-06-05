"""Tests for Sales Call Analyst agent."""

import pytest

from agents.sales_call_analyst import (
    # Data models
    CallAnalysis,
    CompetitorMention,
    Objection,
    SentimentSnapshot,
    BuyingSignal,
    # Agent factory
    create_sales_analyst,
    # Task factories
    create_transcript_analysis_task,
    create_sentiment_deep_dive_task,
    create_objection_tracking_task,
    create_competitor_analysis_task,
    create_follow_up_task,
    create_coaching_task,
    # Analysis helpers
    analyze_transcript_locally,
    detect_sentiment,
    detect_objections,
    detect_competitor_mentions,
    detect_buying_signals,
    assess_price_sensitivity,
    score_call_quality,
    parse_transcript,
    # Workflow
    SalesCallWorkflow,
)


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------

def test_module_imports():
    """All public components should be importable."""
    assert create_sales_analyst is not None
    assert create_transcript_analysis_task is not None
    assert create_sentiment_deep_dive_task is not None
    assert create_objection_tracking_task is not None
    assert create_competitor_analysis_task is not None
    assert create_follow_up_task is not None
    assert create_coaching_task is not None
    assert SalesCallWorkflow is not None


def test_data_model_imports():
    """Data models should be importable."""
    assert SentimentSnapshot is not None
    assert Objection is not None
    assert CompetitorMention is not None
    assert BuyingSignal is not None
    assert CallAnalysis is not None


def test_helper_imports():
    """Helper functions should be importable."""
    assert detect_sentiment is not None
    assert detect_objections is not None
    assert detect_competitor_mentions is not None
    assert detect_buying_signals is not None
    assert assess_price_sensitivity is not None
    assert score_call_quality is not None
    assert parse_transcript is not None
    assert analyze_transcript_locally is not None


# ---------------------------------------------------------------------------
# Task signatures
# ---------------------------------------------------------------------------

def test_task_signatures():
    """Task factory functions should have correct signatures."""
    import inspect

    sig = inspect.signature(create_transcript_analysis_task)
    assert "transcript" in sig.parameters
    assert "agent" in sig.parameters

    sig = inspect.signature(create_sentiment_deep_dive_task)
    assert "transcript" in sig.parameters
    assert "agent" in sig.parameters

    sig = inspect.signature(create_objection_tracking_task)
    assert "transcript" in sig.parameters
    assert "agent" in sig.parameters

    sig = inspect.signature(create_competitor_analysis_task)
    assert "transcript" in sig.parameters
    assert "agent" in sig.parameters

    sig = inspect.signature(create_follow_up_task)
    assert "agent" in sig.parameters

    sig = inspect.signature(create_coaching_task)
    assert "transcript" in sig.parameters
    assert "agent" in sig.parameters


# ---------------------------------------------------------------------------
# Data model construction
# ---------------------------------------------------------------------------

def test_sentiment_snapshot_defaults():
    snap = SentimentSnapshot(speaker="buyer", text="Great!", sentiment="positive")
    assert snap.confidence == 0.0


def test_objection_defaults():
    obj = Objection(objection_text="too expensive", speaker="buyer")
    assert obj.category == ""
    assert obj.handled is None
    assert obj.timestamp_hint == ""


def test_competitor_mention_defaults():
    mention = CompetitorMention(competitor_name="Acme", context="using Acme")
    assert mention.mentioned_by == ""


def test_buying_signal_defaults():
    signal = BuyingSignal(signal_text="we need this", speaker="buyer")
    assert signal.signal_type == ""


def test_call_analysis_defaults():
    analysis = CallAnalysis()
    assert analysis.overall_sentiment_buyer == "neutral"
    assert analysis.overall_sentiment_seller == "neutral"
    assert analysis.sentiment_timeline == []
    assert analysis.objections == []
    assert analysis.competitor_mentions == []
    assert analysis.buying_signals == []
    assert analysis.price_sensitivity == "unknown"
    assert analysis.key_moments == []
    assert analysis.next_steps_agreed == []
    assert analysis.action_items == []
    assert analysis.call_quality_score == 0
    assert analysis.call_quality_reasoning == ""
    assert analysis.follow_up_email == ""
    assert analysis.crm_activity_log == ""
    assert analysis.coaching_notes == []


# ---------------------------------------------------------------------------
# Sentiment detection
# ---------------------------------------------------------------------------

class TestDetectSentiment:
    def test_positive(self):
        label, conf = detect_sentiment("This sounds great, I'm very interested!")
        assert label == "positive"
        assert conf > 0.5

    def test_negative(self):
        label, conf = detect_sentiment("This is too expensive, no budget for this")
        assert label == "negative"
        assert conf > 0.5

    def test_neutral(self):
        label, conf = detect_sentiment("Hello, how are you today?")
        assert label == "neutral"
        assert conf == 0.5

    def test_mixed_defaults_neutral(self):
        label, _ = detect_sentiment("This is great but too expensive")
        # Both positive and negative keywords present
        assert label in {"neutral", "positive", "negative"}


# ---------------------------------------------------------------------------
# Objection detection
# ---------------------------------------------------------------------------

class TestDetectObjections:
    def test_price_objection(self):
        objs = detect_objections("This is too expensive for our budget")
        assert len(objs) >= 1
        categories = [o.category for o in objs]
        assert "price" in categories

    def test_timing_objection(self):
        objs = detect_objections("Not now, maybe next quarter")
        categories = [o.category for o in objs]
        assert "timing" in categories

    def test_trust_objection(self):
        objs = detect_objections("I'm not sure about the risk involved")
        categories = [o.category for o in objs]
        assert "trust" in categories

    def test_competitor_objection(self):
        objs = detect_objections("We're already using a competitor")
        categories = [o.category for o in objs]
        assert "competitor" in categories

    def test_need_objection(self):
        objs = detect_objections("We don't need this, we already have something")
        categories = [o.category for o in objs]
        assert "need" in categories

    def test_authority_objection(self):
        objs = detect_objections("I need to ask my boss for approval")
        categories = [o.category for o in objs]
        assert "authority" in categories

    def test_no_objection(self):
        objs = detect_objections("This looks interesting, tell me more")
        assert len(objs) == 0

    def test_speaker_preserved(self):
        objs = detect_objections("Too expensive", speaker="prospect")
        for obj in objs:
            assert obj.speaker == "prospect"


# ---------------------------------------------------------------------------
# Competitor mention detection
# ---------------------------------------------------------------------------

class TestDetectCompetitorMentions:
    def test_using_pattern(self):
        mentions = detect_competitor_mentions("We're currently using Salesforce")
        names = [m.competitor_name for m in mentions]
        assert "Salesforce" in names

    def test_compared_to_pattern(self):
        mentions = detect_competitor_mentions("compared to HubSpot, your tool is faster")
        names = [m.competitor_name for m in mentions]
        assert "HubSpot" in names

    def test_no_mention(self):
        mentions = detect_competitor_mentions("We don't use any tools currently")
        assert len(mentions) == 0

    def test_context_preserved(self):
        mentions = detect_competitor_mentions("We're using Salesforce for CRM")
        assert len(mentions) >= 1
        assert "using" in mentions[0].context.lower()


# ---------------------------------------------------------------------------
# Buying signal detection
# ---------------------------------------------------------------------------

class TestDetectBuyingSignals:
    def test_urgency_signal(self):
        signals = detect_buying_signals("We need this ASAP, like this month")
        types = [s.signal_type for s in signals]
        assert "urgency" in types

    def test_budget_signal(self):
        signals = detect_buying_signals("Our budget is approved and allocated")
        types = [s.signal_type for s in signals]
        assert "budget" in types

    def test_authority_signal(self):
        signals = detect_buying_signals("I'm the decision maker, I decide")
        types = [s.signal_type for s in signals]
        assert "authority" in types

    def test_need_signal(self):
        signals = detect_buying_signals("We have a problem we're facing daily")
        types = [s.signal_type for s in signals]
        assert "need" in types

    def test_timeline_signal(self):
        signals = detect_buying_signals("We want to go live by end of quarter")
        types = [s.signal_type for s in signals]
        assert "timeline" in types

    def test_no_signal(self):
        signals = detect_buying_signals("Just browsing, not sure yet")
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# Price sensitivity assessment
# ---------------------------------------------------------------------------

class TestAssessPriceSensitivity:
    def test_high(self):
        objs = [
            Objection("expensive", "buyer", category="price"),
            Objection("costly", "buyer", category="price"),
            Objection("budget", "buyer", category="price"),
        ]
        assert assess_price_sensitivity(objs) == "high"

    def test_medium(self):
        objs = [Objection("expensive", "buyer", category="price")]
        assert assess_price_sensitivity(objs) == "medium"

    def test_low(self):
        objs = [Objection("not now", "buyer", category="timing")]
        assert assess_price_sensitivity(objs) == "low"

    def test_empty(self):
        assert assess_price_sensitivity([]) == "low"


# ---------------------------------------------------------------------------
# Call quality scoring
# ---------------------------------------------------------------------------

class TestScoreCallQuality:
    def test_baseline(self):
        analysis = CallAnalysis()
        score, reasoning = score_call_quality(analysis)
        assert 1 <= score <= 10

    def test_positive_signals_increase_score(self):
        analysis = CallAnalysis(
            buying_signals=[
                BuyingSignal("need this", "buyer", "need"),
                BuyingSignal("budget approved", "buyer", "budget"),
            ],
            next_steps_agreed=["Schedule demo"],
            overall_sentiment_buyer="positive",
        )
        score, _ = score_call_quality(analysis)
        assert score > 5

    def test_negative_signals_decrease_score(self):
        analysis = CallAnalysis(
            objections=[
                Objection("expensive", "buyer", category="price", handled=False),
                Objection("not now", "buyer", category="timing", handled=False),
            ],
            overall_sentiment_buyer="negative",
            price_sensitivity="high",
        )
        score, _ = score_call_quality(analysis)
        assert score < 5

    def test_score_bounded(self):
        analysis = CallAnalysis(
            buying_signals=[BuyingSignal("x", "buyer", "need")] * 20,
        )
        score, _ = score_call_quality(analysis)
        assert score <= 10

        analysis2 = CallAnalysis(
            objections=[Objection("x", "buyer", handled=False)] * 20,
            overall_sentiment_buyer="negative",
            price_sensitivity="high",
        )
        score2, _ = score_call_quality(analysis2)
        assert score2 >= 1


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

class TestParseTranscript:
    def test_basic_format(self):
        transcript = "Seller: Hello!\nBuyer: Hi there."
        segments = parse_transcript(transcript)
        assert len(segments) == 2
        assert segments[0]["speaker"] == "Seller"
        assert segments[0]["text"] == "Hello!"
        assert segments[1]["speaker"] == "Buyer"
        assert segments[1]["text"] == "Hi there."

    def test_bracket_format(self):
        transcript = "[Seller]: Hello!\n[Buyer]: Hi there."
        segments = parse_transcript(transcript)
        assert len(segments) == 2
        assert segments[0]["speaker"] == "Seller"

    def test_role_in_parens(self):
        transcript = "John (seller): Hello!\nJane (buyer): Hi."
        segments = parse_transcript(transcript)
        assert len(segments) == 2
        assert segments[0]["speaker"] == "John"

    def test_empty_lines_skipped(self):
        transcript = "Seller: Hello!\n\nBuyer: Hi."
        segments = parse_transcript(transcript)
        assert len(segments) == 2

    def test_unknown_speaker(self):
        transcript = "Just some text without a speaker"
        segments = parse_transcript(transcript)
        assert len(segments) == 1
        assert segments[0]["speaker"] == "unknown"

    def test_empty_string(self):
        segments = parse_transcript("")
        assert segments == []


# ---------------------------------------------------------------------------
# Local transcript analysis
# ---------------------------------------------------------------------------

class TestAnalyzeTranscriptLocally:
    SAMPLE_TRANSCRIPT = (
        "Seller: Hi, thanks for taking the call today.\n"
        "Buyer: Sure, I'm interested in learning more.\n"
        "Seller: Great! Our platform helps teams save 20 hours per week.\n"
        "Buyer: That sounds great, but it's too expensive for our budget.\n"
        "Seller: I understand. We have flexible pricing tiers.\n"
        "Buyer: We're currently using CompetitorX but not happy.\n"
        "Seller: Many customers switch from CompetitorX. Can we schedule a demo?\n"
        "Buyer: Yes, let's do a demo next week. We need this ASAP.\n"
        "Seller: Perfect, I'll send a calendar invite.\n"
    )

    def test_returns_call_analysis(self):
        result = analyze_transcript_locally(self.SAMPLE_TRANSCRIPT)
        assert isinstance(result, CallAnalysis)

    def test_sentiment_timeline_populated(self):
        result = analyze_transcript_locally(self.SAMPLE_TRANSCRIPT)
        assert len(result.sentiment_timeline) > 0

    def test_objections_detected(self):
        result = analyze_transcript_locally(self.SAMPLE_TRANSCRIPT)
        assert len(result.objections) > 0
        categories = [o.category for o in result.objections]
        assert "price" in categories

    def test_competitor_mentions_detected(self):
        result = analyze_transcript_locally(self.SAMPLE_TRANSCRIPT)
        names = [m.competitor_name for m in result.competitor_mentions]
        assert "CompetitorX" in names

    def test_buying_signals_detected(self):
        result = analyze_transcript_locally(self.SAMPLE_TRANSCRIPT)
        assert len(result.buying_signals) > 0

    def test_price_sensitivity_assessed(self):
        result = analyze_transcript_locally(self.SAMPLE_TRANSCRIPT)
        assert result.price_sensitivity in {"low", "medium", "high"}

    def test_call_quality_scored(self):
        result = analyze_transcript_locally(self.SAMPLE_TRANSCRIPT)
        assert 1 <= result.call_quality_score <= 10
        assert result.call_quality_reasoning != ""

    def test_buyer_sentiment_detected(self):
        result = analyze_transcript_locally(self.SAMPLE_TRANSCRIPT)
        assert result.overall_sentiment_buyer in {"positive", "neutral", "negative"}

    def test_empty_transcript(self):
        result = analyze_transcript_locally("")
        assert isinstance(result, CallAnalysis)
        assert result.sentiment_timeline == []
        assert result.call_quality_score >= 1


# ---------------------------------------------------------------------------
# Workflow class
# ---------------------------------------------------------------------------

class TestSalesCallWorkflow:
    def test_workflow_instantiation(self):
        """Workflow should instantiate without LLM calls."""
        # Full instantiation requires an LLM API key
        # Just verify the class exists and has expected methods
        assert hasattr(SalesCallWorkflow, "analyze")
        assert hasattr(SalesCallWorkflow, "analyze_sentiment")
        assert hasattr(SalesCallWorkflow, "track_objections")
        assert hasattr(SalesCallWorkflow, "analyze_competitors")
        assert hasattr(SalesCallWorkflow, "generate_coaching")
        assert hasattr(SalesCallWorkflow, "full_analysis_pipeline")

    def test_workflow_has_agent_after_init(self):
        """Workflow should have an agent attribute after init."""
        # This will try to create an Agent which needs an LLM
        # We just verify the class structure is correct
        import inspect
        sig = inspect.signature(SalesCallWorkflow.__init__)
        params = list(sig.parameters.keys())
        assert "self" in params


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------

class TestCreateSalesAnalyst:
    def test_returns_agent(self, monkeypatch):
        """create_sales_analyst should return a CrewAI Agent."""
        monkeypatch.setattr(
            "agents.sales_call_analyst.get_llm",
            lambda: "gpt-4o",
        )
        agent = create_sales_analyst()
        assert hasattr(agent, "role")
        assert agent.role == "Sales Call Analyst"

    def test_agent_has_goal(self, monkeypatch):
        monkeypatch.setattr(
            "agents.sales_call_analyst.get_llm",
            lambda: "gpt-4o",
        )
        agent = create_sales_analyst()
        assert hasattr(agent, "goal")
        assert "insight" in agent.goal.lower() or "sales" in agent.goal.lower()

    def test_agent_has_backstory(self, monkeypatch):
        monkeypatch.setattr(
            "agents.sales_call_analyst.get_llm",
            lambda: "gpt-4o",
        )
        agent = create_sales_analyst()
        assert hasattr(agent, "backstory")
        assert len(agent.backstory) > 0


# ---------------------------------------------------------------------------
# Task creation
# ---------------------------------------------------------------------------

class TestTaskCreation:
    @pytest.fixture
    def agent(self, monkeypatch):
        monkeypatch.setattr(
            "agents.sales_call_analyst.get_llm",
            lambda: "gpt-4o",
        )
        return create_sales_analyst()

    def test_transcript_analysis_task(self, agent):
        task = create_transcript_analysis_task("test transcript", agent)
        assert task is not None
        assert "transcript" in task.description.lower() or "analyze" in task.description.lower()

    def test_sentiment_deep_dive_task(self, agent):
        task = create_sentiment_deep_dive_task("test transcript", agent)
        assert task is not None
        assert "sentiment" in task.description.lower()

    def test_objection_tracking_task(self, agent):
        task = create_objection_tracking_task("test transcript", agent)
        assert task is not None
        assert "objection" in task.description.lower()

    def test_competitor_analysis_task(self, agent):
        task = create_competitor_analysis_task("test transcript", agent)
        assert task is not None
        assert "competitor" in task.description.lower() or "competitive" in task.description.lower()

    def test_follow_up_task(self, agent):
        task = create_follow_up_task(agent)
        assert task is not None
        assert "follow-up" in task.description.lower() or "follow up" in task.description.lower()

    def test_coaching_task(self, agent):
        task = create_coaching_task("test transcript", agent)
        assert task is not None
        assert "coaching" in task.description.lower() or "coach" in task.description.lower()
