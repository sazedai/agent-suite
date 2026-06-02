"""Tests for the Lead Scraping + Enrichment agent."""

import asyncio
import os
import tempfile

import pytest

from agents.lead_scraper import (
    ScrapedLead,
    LeadEnrichmentPipeline,
    deduplicate_leads,
    export_to_csv,
    score_lead,
    score_leads,
    store_leads,
    _extract_domain,
    _extract_email_from_text,
    _extract_phone_from_text,
)


# ---------------------------------------------------------------------------
# ScrapedLead dataclass
# ---------------------------------------------------------------------------


class TestScrapedLead:
    def test_create_blank_lead(self):
        lead = ScrapedLead()
        assert lead.email == ""
        assert lead.score == 0.0
        assert lead.enriched == "N"

    def test_dedup_key_auto_computed(self):
        lead = ScrapedLead(email="test@example.com", company="Acme Inc")
        assert len(lead.dedup_key) == 16

    def test_dedup_key_same_for_same_inputs(self):
        a = ScrapedLead(email="test@example.com", company="Acme")
        b = ScrapedLead(email="test@example.com", company="Acme")
        assert a.dedup_key == b.dedup_key

    def test_dedup_key_different_for_different_emails(self):
        a = ScrapedLead(email="a@test.com", company="Acme")
        b = ScrapedLead(email="b@test.com", company="Acme")
        assert a.dedup_key != b.dedup_key

    def test_basic_score_valid_email(self):
        lead = ScrapedLead(
            email="test@example.com",
            company="Acme",
            contact_name="John",
        )
        # email(30) + company(20) + contact_name(10) = 60
        assert lead.score == 60.0

    def test_basic_score_all_fields(self):
        lead = ScrapedLead(
            email="test@example.com",
            company="Acme",
            contact_name="John",
            linkedin_url="https://linkedin.com/in/john",
            phone="555-123-4567",
            website="https://acme.com",
        )
        # 30 + 20 + 10 + 15 + 15 + 10 = 100
        assert lead.score == 100.0

    def test_data_quality_empty(self):
        lead = ScrapedLead()
        assert lead.data_quality == 0.0

    def test_data_quality_partial(self):
        lead = ScrapedLead(email="a@b.com", company="Acme")
        # 2 of 8 fields = 25.0%
        assert lead.data_quality == 25.0

    def test_email_validation(self):
        assert ScrapedLead._validate_email("valid@example.com") is True
        assert ScrapedLead._validate_email("invalid") is False
        assert ScrapedLead._validate_email("user@domain.co.uk") is True
        assert ScrapedLead._validate_email("@nodomain.com") is False

    def test_to_dict(self):
        lead = ScrapedLead(email="a@b.com", company="Acme")
        d = lead.to_dict()
        assert isinstance(d, dict)
        assert d["email"] == "a@b.com"
        assert d["company"] == "Acme"

    def test_to_lead_model(self):
        lead = ScrapedLead(
            email="a@b.com",
            company="Acme",
            website="https://acme.com",
        )
        model = lead.to_lead_model()
        from core.storage import Lead
        assert isinstance(model, Lead)
        assert model.email == "a@b.com"
        assert model.website == "https://acme.com"

    def test_custom_dedup_key_preserved(self):
        lead = ScrapedLead(email="a@b.com", company="Acme", dedup_key="custom-key")
        assert lead.dedup_key == "custom-key"


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


class TestExtractors:
    def test_extract_domain_from_url(self):
        assert _extract_domain("https://www.example.com/path") == "example.com"
        assert _extract_domain("http://example.com") == "example.com"
        assert _extract_domain("www.example.com") == "example.com"

    def test_extract_email_from_text(self):
        text = "Contact us at hello@example.com or support@example.com"
        emails = _extract_email_from_text(text)
        assert "hello@example.com" in emails
        assert "support@example.com" in emails

    def test_extract_email_dedup(self):
        text = "Same email twice: ali@bar.com and ali@bar.com"
        emails = _extract_email_from_text(text)
        assert len(emails) == 1

    def test_extract_phone_from_text(self):
        text = "Call us at (555) 123-4567 or 555.987.6543"
        phones = _extract_phone_from_text(text)
        assert len(phones) == 2


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_no_duplicates(self):
        leads = [
            ScrapedLead(email="a@test.com", company="Acme"),
            ScrapedLead(email="b@test.com", company="Beta"),
        ]
        result = deduplicate_leads(leads)
        assert len(result) == 2

    def test_removes_duplicates(self):
        leads = [
            ScrapedLead(email="same@test.com", company="Acme"),
            ScrapedLead(email="same@test.com", company="Acme"),
        ]
        result = deduplicate_leads(leads)
        assert len(result) == 1

    def test_merges_duplicate_data(self):
        leads = [
            ScrapedLead(email="a@test.com", company="Acme", phone="555-0000"),
            ScrapedLead(email="a@test.com", company="Acme", website="https://acme.com"),
        ]
        result = deduplicate_leads(leads)
        assert len(result) == 1
        assert result[0].phone == "555-0000"
        assert result[0].website == "https://acme.com"

    def test_keeps_higher_score_on_dedup(self):
        leads = [
            ScrapedLead(email="a@test.com", score=20.0),
            ScrapedLead(email="a@test.com", score=80.0),
        ]
        result = deduplicate_leads(leads)
        assert len(result) == 1
        assert result[0].score == 80.0

    def test_keeps_higher_quality_on_dedup(self):
        leads = [
            ScrapedLead(email="a@test.com", data_quality=30.0),
            ScrapedLead(email="a@test.com", data_quality=70.0),
        ]
        result = deduplicate_leads(leads)
        assert result[0].data_quality == 70.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def test_empty_lead_score(self):
        lead = ScrapedLead()
        assert score_lead(lead) == 0.0

    def test_high_quality_lead(self):
        lead = ScrapedLead(
            email="ceo@bigco.com",
            contact_name="Jane CEO",
            company="BigCo",
            website="https://bigco.com",
            industry="SaaS",
        )
        score = score_lead(lead, "SaaS")
        assert score >= 60.0

    def test_industry_match_bonus(self):
        lead_match = ScrapedLead(
            email="a@test.com",
            industry="SaaS",
        )
        lead_no_match = ScrapedLead(
            email="a@test.com",
            industry="Agriculture",
        )
        score_match = score_lead(lead_match, "SaaS")
        score_no_match = score_lead(lead_no_match, "SaaS")
        assert score_match > score_no_match

    def test_score_cap_at_100(self):
        lead = ScrapedLead(
            email="a@b.com",
            contact_name="John",
            company="Acme",
            website="https://acme.com",
            phone="555-1234",
            linkedin_url="https://linkedin.com",
            industry="SaaS",
            company_size="50-100",
        )
        score = score_lead(lead, "SaaS")
        assert score <= 100.0

    def test_score_leads_function(self):
        leads = [
            ScrapedLead(email="a@test.com", contact_name="A"),
            ScrapedLead(email="b@test.com"),
        ]
        scored = score_leads(leads, "SaaS")
        # First lead has more fields, should score higher
        assert scored[0].score > scored[1].score


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------


class TestCSVExport:
    def test_export_creates_file(self):
        leads = [
            ScrapedLead(email="a@b.com", company="Acme"),
            ScrapedLead(email="c@d.com", company="Beta"),
        ]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        result = export_to_csv(leads, path)
        assert os.path.exists(result)
        with open(result) as f:
            content = f.read()
        # Header + 2 rows
        assert "contact_name" in content
        assert "a@b.com" in content
        os.unlink(path)

    def test_export_empty_list(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        result = export_to_csv([], path)
        # Should not crash
        assert result == path
        os.unlink(path)

    def test_export_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "leads.csv")
            leads = [ScrapedLead(email="a@b.com", company="Acme", score=50.0)]
            export_to_csv(leads, path)
            assert os.path.exists(path)

    def test_export_has_all_fields(self):
        leads = [
            ScrapedLead(
                contact_name="John",
                email="john@acme.com",
                company="Acme",
                website="https://acme.com",
                phone="555-1234",
                linkedin_url="https://linkedin.com/in/john",
                source="crunchbase",
                industry="SaaS",
                company_size="50-100",
                score=85.0,
                dedup_key="abc123",
                data_quality=87.5,
                tags="saas,fintech",
            ),
        ]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        export_to_csv(leads, path)
        with open(path) as f:
            content = f.read()
        for field in [
            "contact_name", "email", "company", "website",
            "score", "dedup_key", "data_quality",
        ]:
            assert field in content
        os.unlink(path)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TestLeadEnrichmentPipeline:
    def test_enrich_sets_enriched_flag(self):
        leads = [
            ScrapedLead(email="a@test.com", company="Acme"),
            ScrapedLead(email="b@test.com", company="Beta"),
        ]
        pipe = LeadEnrichmentPipeline("SaaS")
        result = pipe.enrich(leads)
        assert all(l.enriched == "Y" for l in result)

    def test_enrich_deduplicates(self):
        leads = [
            ScrapedLead(email="same@test.com", company="Acme"),
            ScrapedLead(email="same@test.com", company="Acme"),
        ]
        pipe = LeadEnrichmentPipeline()
        result = pipe.enrich(leads)
        assert len(result) == 1

    def test_enrich_sorts_by_score(self):
        leads = [
            ScrapedLead(email="low@test.com"),
            ScrapedLead(
                email="high@test.com", company="BigCo",
                contact_name="CEO", website="https://bigco.com",
                linkedin_url="https://linkedin.com",
            ),
        ]
        pipe = LeadEnrichmentPipeline()
        result = pipe.enrich(leads)
        scores = [l.score for l in result]
        assert scores == sorted(scores, reverse=True)

    def test_enrich_with_industry_boost(self):
        leads = [
            ScrapedLead(email="a@test.com", industry="SaaS"),
            ScrapedLead(email="b@test.com", industry="Agriculture"),
        ]
        pipe = LeadEnrichmentPipeline("SaaS")
        result = pipe.enrich(leads)
        saas_lead = next(l for l in result if l.industry == "SaaS")
        agri_lead = next(l for l in result if l.industry == "Agriculture")
        assert saas_lead.score >= agri_lead.score

    def test_empty_list(self):
        pipe = LeadEnrichmentPipeline()
        result = pipe.enrich([])
        assert result == []


# ---------------------------------------------------------------------------
# Database storage
# ---------------------------------------------------------------------------


class TestDatabaseStorage:
    @pytest.fixture(autouse=True)
    async def setup_db(self):
        from core.storage.database import init_db, engine, Base
        await init_db()
        yield
        # Cleanup: drop all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    @pytest.mark.asyncio
    async def test_store_and_deduplicate(self):
        """Storing the same leads twice should deduplicate by dedup_key."""
        leads = [
            ScrapedLead(email="store@test.com", company="Acme"),
            ScrapedLead(email="new@test.com", company="Beta"),
        ]
        stored1 = await store_leads(leads)
        assert len(stored1) == 2

        # Second store should skip duplicates
        leads2 = [
            ScrapedLead(email="store@test.com", company="Acme"),
            ScrapedLead(email="another@test.com", company="Gamma"),
        ]
        stored2 = await store_leads(leads2)
        assert len(stored2) == 1  # only Gamma is new
        assert stored2[0].company == "Gamma"

    @pytest.mark.asyncio
    async def test_store_empty_list(self):
        stored = await store_leads([])
        assert stored == []

    @pytest.mark.asyncio
    async def test_stored_fields_preserved(self):
        """Verify all ScrapedLead fields are preserved in the DB model."""
        leads = [
            ScrapedLead(
                contact_name="Jane Doe",
                email="jane@acme.com",
                company="Acme Corp",
                website="https://acme.com",
                phone="555-9876",
                linkedin_url="https://linkedin.com/in/jane",
                source="crunchbase",
                industry="SaaS",
                company_size="100-500",
                score=95.0,
                data_quality=100.0,
                tags="saas,b2b",
            ),
        ]
        stored = await store_leads(leads)
        assert len(stored) == 1
        model = stored[0]
        assert model.contact_name == "Jane Doe"
        assert model.email == "jane@acme.com"
        assert model.company == "Acme Corp"
        assert model.website == "https://acme.com"
        assert model.phone == "555-9876"
        assert model.linkedin_url == "https://linkedin.com/in/jane"
        assert model.source == "crunchbase"
        assert model.industry == "SaaS"
        assert model.company_size == "100-500"
        assert model.score == 95.0
        assert model.data_quality == 100.0
        assert model.tags == "saas,b2b"
        assert model.enriched == "N"


# ---------------------------------------------------------------------------
# Import / module structure tests (no external deps)
# ---------------------------------------------------------------------------


class TestModuleStructure:
    def test_scraped_lead_importable(self):
        from agents.lead_scraper import ScrapedLead
        assert ScrapedLead is not None

    def test_pipeline_importable(self):
        from agents.lead_scraper import LeadEnrichmentPipeline
        assert LeadEnrichmentPipeline is not None

    def test_functions_importable(self):
        from agents.lead_scraper import (
            deduplicate_leads, export_to_csv,
            score_lead, score_leads, store_leads,
        )
        assert callable(deduplicate_leads)
        assert callable(export_to_csv)
        assert callable(score_lead)
        assert callable(score_leads)
        assert callable(store_leads)

    def test_workflow_class_importable(self):
        from agents.lead_scraper import LeadScraperWorkflow
        assert LeadScraperWorkflow is not None

    def test_agent_class_importable(self):
        from agents.lead_scraper import LeadScraperAgent
        assert LeadScraperAgent is not None

    def test_agent_importable_from_package(self):
        from agents import lead_scraper
        assert lead_scraper is not None
