"""Tests for Operations Employee Agent."""

from unittest.mock import patch

import pytest

from agents.operations import (
    TaskItem,
    TaskQueue,
    SOP,
    SOPTracker,
    ProcessStep,
    AutomatedProcess,
    VendorRecord,
    VendorRegistry,
    create_operations_agent,
    create_task_management_task,
    create_sop_tracker_task,
    create_vendor_evaluation_task,
    create_process_automation_task,
    OperationsWorkflow,
)


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


def test_operations_module_imports():
    """Operations module should import all components."""
    assert create_operations_agent is not None
    assert create_task_management_task is not None
    assert create_sop_tracker_task is not None
    assert create_vendor_evaluation_task is not None
    assert create_process_automation_task is not None
    assert OperationsWorkflow is not None
    assert TaskItem is not None
    assert TaskQueue is not None
    assert SOP is not None
    assert SOPTracker is not None
    assert ProcessStep is not None
    assert AutomatedProcess is not None
    assert VendorRecord is not None
    assert VendorRegistry is not None


# ---------------------------------------------------------------------------
# TaskItem tests
# ---------------------------------------------------------------------------


class TestTaskItem:
    def test_create_defaults(self):
        t = TaskItem(id="t1", title="Deploy API")
        assert t.id == "t1"
        assert t.title == "Deploy API"
        assert t.status == "backlog"
        assert t.priority == "P2"
        assert t.effort_hours == 0.0
        assert t.depends_on == []
        assert t.tags == []

    def test_create_full(self):
        t = TaskItem(
            id="t2",
            title="Write docs",
            description="Document the new API",
            priority="P1",
            status="todo",
            effort_hours=4.0,
            depends_on=["t1"],
            assignee_role="Tech Writer",
            tags=["docs", "api"],
            batch_group="doc-batch",
        )
        assert t.priority == "P1"
        assert t.status == "todo"
        assert t.effort_hours == 4.0
        assert t.depends_on == ["t1"]
        assert t.assignee_role == "Tech Writer"
        assert t.tags == ["docs", "api"]
        assert t.batch_group == "doc-batch"

    def test_is_blocked_all_done(self):
        t = TaskItem(id="t2", title="Task 2", depends_on=["t1"])
        assert t.is_blocked({"t1"}) is False

    def test_is_blocked_some_missing(self):
        t = TaskItem(id="t3", title="Task 3", depends_on=["t1", "t2"])
        assert t.is_blocked({"t1"}) is True

    def test_is_blocked_none_completed(self):
        t = TaskItem(id="t2", title="Task 2", depends_on=["t1"])
        assert t.is_blocked(set()) is True

    def test_is_blocked_no_deps(self):
        t = TaskItem(id="t1", title="Task 1")
        assert t.is_blocked(set()) is False

    def test_to_dict(self):
        t = TaskItem(id="t1", title="Task", priority="P0")
        d = t.to_dict()
        assert d["id"] == "t1"
        assert d["title"] == "Task"
        assert d["priority"] == "P0"
        assert "depends_on" in d
        assert "tags" in d


# ---------------------------------------------------------------------------
# TaskQueue tests
# ---------------------------------------------------------------------------


class TestTaskQueue:
    def _sample_queue(self) -> TaskQueue:
        q = TaskQueue()
        q.add(TaskItem(id="t1", title="Foundation", priority="P0"))
        q.add(TaskItem(id="t2", title="API", priority="P1", depends_on=["t1"]))
        q.add(TaskItem(id="t3", title="Docs", priority="P3", depends_on=["t2"]))
        q.add(TaskItem(id="t4", title="Tests", priority="P2", depends_on=["t2"]))
        return q

    def test_add_and_get(self):
        q = TaskQueue()
        t = TaskItem(id="t1", title="Task")
        q.add(t)
        assert q.get("t1") is t

    def test_get_missing(self):
        q = TaskQueue()
        assert q.get("nonexistent") is None

    def test_remove(self):
        q = TaskQueue()
        q.add(TaskItem(id="t1", title="Task"))
        assert q.remove("t1") is True
        assert q.get("t1") is None

    def test_remove_missing(self):
        q = TaskQueue()
        assert q.remove("nope") is False

    def test_all(self):
        q = self._sample_queue()
        assert len(q.all()) == 4

    def test_by_status(self):
        q = TaskQueue()
        q.add(TaskItem(id="t1", title="A", status="done"))
        q.add(TaskItem(id="t2", title="B", status="todo"))
        q.add(TaskItem(id="t3", title="C", status="done"))
        done = q.by_status("done")
        assert len(done) == 2
        assert all(t.status == "done" for t in done)

    def test_by_priority(self):
        q = self._sample_queue()
        p0 = q.by_priority("P0")
        assert len(p0) == 1
        assert p0[0].id == "t1"

    def test_by_tag(self):
        q = TaskQueue()
        q.add(TaskItem(id="t1", title="A", tags=["urgent", "backend"]))
        q.add(TaskItem(id="t2", title="B", tags=["frontend"]))
        urgent = q.by_tag("urgent")
        assert len(urgent) == 1
        assert urgent[0].id == "t1"

    def test_blocked_tasks(self):
        q = self._sample_queue()
        q.get("t1").status = "done"
        blocked = q.blocked_tasks()
        # t3 depends on t2 (not done), t4 depends on t2 (not done)
        blocked_ids = {t.id for t in blocked}
        assert "t3" in blocked_ids
        assert "t4" in blocked_ids
        assert "t1" not in blocked_ids  # done

    def test_batchable_groups(self):
        q = TaskQueue()
        q.add(TaskItem(id="t1", title="A", batch_group="group1"))
        q.add(TaskItem(id="t2", title="B", batch_group="group1"))
        q.add(TaskItem(id="t3", title="C", batch_group="group2"))
        q.add(TaskItem(id="t4", title="D"))
        groups = q.batchable_groups()
        assert len(groups) == 2
        assert len(groups["group1"]) == 2
        assert len(groups["group2"]) == 1

    def test_prioritized(self):
        q = self._sample_queue()
        ordered = q.prioritized()
        priorities = [t.priority for t in ordered]
        # P0 < P1 < P2 < P3
        assert priorities == sorted(priorities, key=lambda p: {"P0": 0, "P1": 1, "P2": 2, "P3": 3}[p])

    def test_execution_order_respects_dependencies(self):
        q = self._sample_queue()
        order = q.execution_order()
        # t1 must come before t2, t2 before t3 and t4
        assert order.index("t1") < order.index("t2")
        assert order.index("t2") < order.index("t3")
        assert order.index("t2") < order.index("t4")

    def test_execution_order_all_ids_present(self):
        q = self._sample_queue()
        order = q.execution_order()
        assert set(order) == {"t1", "t2", "t3", "t4"}

    def test_execution_order_handles_circular_deps(self):
        q = TaskQueue()
        q.add(TaskItem(id="a", title="A", depends_on=["b"]))
        q.add(TaskItem(id="b", title="B", depends_on=["a"]))
        order = q.execution_order()
        assert set(order) == {"a", "b"}


# ---------------------------------------------------------------------------
# SOP tests
# ---------------------------------------------------------------------------


class TestSOP:
    def test_create_defaults(self):
        sop = SOP(id="sop-1", title="Onboarding")
        assert sop.id == "sop-1"
        assert sop.title == "Onboarding"
        assert sop.version == "1.0.0"
        assert sop.steps == []
        assert sop.tags == []

    def test_create_full(self):
        sop = SOP(
            id="sop-2",
            title="Incident Response",
            version="2.1.0",
            steps=["Detect", "Triage", "Resolve", "Post-mortem"],
            owner="Ops Team",
            review_date="2026-12-01",
            tags=["incident", "critical"],
        )
        assert sop.step_count() == 4
        assert sop.owner == "Ops Team"

    def test_has_empty_steps_with_blank(self):
        sop = SOP(id="s1", title="SOP", steps=["Step 1", "", "Step 3"])
        assert sop.has_empty_steps() is True

    def test_has_empty_steps_with_zero_steps(self):
        sop = SOP(id="s1", title="SOP")
        assert sop.has_empty_steps() is True

    def test_no_empty_steps(self):
        sop = SOP(id="s1", title="SOP", steps=["Step 1", "Step 2"])
        assert sop.has_empty_steps() is False

    def test_completeness_score_complete(self):
        sop = SOP(
            id="s1",
            title="Full SOP",
            version="1.0.0",
            steps=["a", "b", "c", "d", "e"],
            owner="Alice",
            review_date="2026-12-01",
        )
        assert sop.completeness_score() == pytest.approx(1.0)

    def test_completeness_score_minimal(self):
        """A minimal SOP (only id) should have a low but non-zero score."""
        sop = SOP(id="s1", title="")
        # version defaults to "1.0.0", so 1 of 4 fields filled + 0 steps
        # score = (1/4 * 0.6) + (0 * 0.4) = 0.15
        assert sop.completeness_score() == pytest.approx(0.15)

    def test_completeness_score_fully_empty(self):
        """A completely empty constructor SOP should score 0."""
        sop = SOP(id="", title="", version="", steps=[], owner="", review_date="")
        assert sop.completeness_score() == pytest.approx(0.0)

    def test_to_dict(self):
        sop = SOP(id="s1", title="SOP", steps=["Step 1"], owner="Bob")
        d = sop.to_dict()
        assert d["id"] == "s1"
        assert d["title"] == "SOP"
        assert d["steps"] == ["Step 1"]
        assert d["owner"] == "Bob"


# ---------------------------------------------------------------------------
# SOPTracker tests
# ---------------------------------------------------------------------------


class TestSOPTracker:
    def _sample_tracker(self) -> SOPTracker:
        t = SOPTracker()
        t.add(SOP(id="s1", title="Onboarding", steps=["a", "b"], owner="Alice", review_date="2026-06-01"))
        t.add(SOP(id="s2", title="Offboarding", steps=["x", "y", "z"], owner="Bob", review_date="2027-01-01"))
        t.add(SOP(id="s3", title="Incident", steps=[], owner="", review_date=""))
        return t

    def test_add_and_get(self):
        t = SOPTracker()
        sop = SOP(id="s1", title="SOP")
        t.add(sop)
        assert t.get("s1") is sop

    def test_get_missing(self):
        t = SOPTracker()
        assert t.get("nope") is None

    def test_remove(self):
        t = SOPTracker()
        t.add(SOP(id="s1", title="SOP"))
        assert t.remove("s1") is True
        assert t.get("s1") is None

    def test_all(self):
        t = self._sample_tracker()
        assert len(t.all()) == 3

    def test_by_owner(self):
        t = self._sample_tracker()
        alice_sops = t.by_owner("Alice")
        assert len(alice_sops) == 1
        assert alice_sops[0].id == "s1"

    def test_by_tag(self):
        t = SOPTracker()
        t.add(SOP(id="s1", title="A", tags=["incident"]))
        t.add(SOP(id="s2", title="B", tags=["onboarding"]))
        incident_sops = t.by_tag("incident")
        assert len(incident_sops) == 1

    def test_audit_total(self):
        t = self._sample_tracker()
        result = t.audit()
        assert result["total"] == 3

    def test_audit_catches_issues(self):
        t = self._sample_tracker()
        result = t.audit()
        # s3 has empty steps, no owner, no review date
        assert result["incomplete"] >= 1
        assert any("empty steps" in issue for issue in result["issues"])

    def test_audit_avg_completeness(self):
        t = self._sample_tracker()
        result = t.audit()
        assert 0.0 <= result["avg_completeness"] <= 1.0

    def test_needs_review(self):
        t = self._sample_tracker()
        due = t.needs_review("2026-07-01")
        # s1 has review_date 2026-06-01 <= 2026-07-01
        ids = {s.id for s in due}
        assert "s1" in ids

    def test_needs_review_includes_only_dated_sops(self):
        """Only SOPs with non-empty review_date should be returned."""
        t = SOPTracker()
        t.add(SOP(id="s1", title="A", review_date="2026-01-15"))
        t.add(SOP(id="s2", title="B", review_date="2027-01-01"))
        t.add(SOP(id="s3", title="C"))  # no review_date
        due = t.needs_review("2026-06-01")
        ids = {s.id for s in due}
        assert "s1" in ids
        assert "s2" not in ids
        assert "s3" not in ids  # empty review_date excluded by design

    def test_empty_tracker_audit(self):
        t = SOPTracker()
        result = t.audit()
        assert result["total"] == 0
        assert result["avg_completeness"] == 0.0
        assert result["issues"] == []


# ---------------------------------------------------------------------------
# ProcessStep tests
# ---------------------------------------------------------------------------


class TestProcessStep:
    def test_create_defaults(self):
        step = ProcessStep(id="ps1", name="Check", action="run_check")
        assert step.id == "ps1"
        assert step.name == "Check"
        assert step.action == "run_check"
        assert step.params == {}
        assert step.next_step_id == ""
        assert step.condition == ""

    def test_create_full(self):
        step = ProcessStep(
            id="ps2",
            name="Approve",
            action="auto_approve",
            params={"threshold": 500},
            next_step_id="ps3",
            condition="amount < 500",
        )
        assert step.params == {"threshold": 500}
        assert step.next_step_id == "ps3"

    def test_to_dict(self):
        step = ProcessStep(id="ps1", name="Check", action="run_check")
        d = step.to_dict()
        assert d["id"] == "ps1"
        assert d["action"] == "run_check"
        assert "next_step_id" in d
        assert "condition" in d


# ---------------------------------------------------------------------------
# AutomatedProcess tests
# ---------------------------------------------------------------------------


class TestAutomatedProcess:
    def test_create_defaults(self):
        proc = AutomatedProcess(id="p1", name="Onboarding")
        assert proc.id == "p1"
        assert proc.name == "Onboarding"
        assert proc.description == ""
        assert proc.trigger == ""

    def test_add_and_get_step(self):
        proc = AutomatedProcess(id="p1", name="Proc")
        step = ProcessStep(id="s1", name="Step 1", action="notify")
        proc.add_step(step)
        assert proc.get_step("s1") is step

    def test_all_steps(self):
        proc = AutomatedProcess(id="p1", name="Proc")
        proc.add_step(ProcessStep(id="s1", name="A", action="notify"))
        proc.add_step(ProcessStep(id="s2", name="B", action="assign"))
        assert len(proc.all_steps()) == 2

    def test_build_chain_linear(self):
        proc = AutomatedProcess(id="p1", name="Invoice")
        proc.add_step(ProcessStep(id="s1", name="Check", action="run_check", next_step_id="s2"))
        proc.add_step(ProcessStep(id="s2", name="Approve", action="auto_approve", next_step_id="s3"))
        proc.add_step(ProcessStep(id="s3", name="Notify", action="notify"))
        chain = proc.build_chain()
        ids = [s.id for s in chain]
        assert ids == ["s1", "s2", "s3"]

    def test_build_chain_single_step(self):
        proc = AutomatedProcess(id="p1", name="Simple")
        proc.add_step(ProcessStep(id="s1", name="Only", action="notify"))
        chain = proc.build_chain()
        assert len(chain) == 1
        assert chain[0].id == "s1"

    def test_build_chain_circular(self):
        proc = AutomatedProcess(id="p1", name="Loop")
        proc.add_step(ProcessStep(id="s1", name="A", action="notify", next_step_id="s2"))
        proc.add_step(ProcessStep(id="s2", name="B", action="notify", next_step_id="s1"))
        chain = proc.build_chain()
        assert len(chain) == 2  # should not infinite loop

    def test_to_dict(self):
        proc = AutomatedProcess(id="p1", name="Proc", trigger="on_new_order")
        proc.add_step(ProcessStep(id="s1", name="Step", action="notify"))
        d = proc.to_dict()
        assert d["id"] == "p1"
        assert d["trigger"] == "on_new_order"
        assert len(d["steps"]) == 1


# ---------------------------------------------------------------------------
# VendorRecord tests
# ---------------------------------------------------------------------------


class TestVendorRecord:
    def test_create_defaults(self):
        v = VendorRecord(id="v1", name="AWS")
        assert v.id == "v1"
        assert v.name == "AWS"
        assert v.category == ""
        assert v.sla_hours == 24
        assert v.status == "active"

    def test_create_full(self):
        v = VendorRecord(
            id="v2",
            name="Datadog",
            category="monitoring",
            contact_email="sales@datadog.com",
            sla_hours=4,
            status="active",
            notes="Primary monitoring vendor",
        )
        assert v.category == "monitoring"
        assert v.sla_hours == 4
        assert v.contact_email == "sales@datadog.com"

    def test_to_dict(self):
        v = VendorRecord(id="v1", name="AWS", category="cloud")
        d = v.to_dict()
        assert d["id"] == "v1"
        assert d["name"] == "AWS"
        assert d["category"] == "cloud"
        assert "sla_hours" in d
        assert "status" in d


# ---------------------------------------------------------------------------
# VendorRegistry tests
# ---------------------------------------------------------------------------


class TestVendorRegistry:
    def _sample_registry(self) -> VendorRegistry:
        r = VendorRegistry()
        r.add(VendorRecord(id="v1", name="AWS", category="cloud", status="active"))
        r.add(VendorRecord(id="v2", name="GCP", category="cloud", status="active"))
        r.add(VendorRecord(id="v3", name="OldVendor", category="logistics", status="inactive"))
        return r

    def test_add_and_get(self):
        r = VendorRegistry()
        v = VendorRecord(id="v1", name="AWS")
        r.add(v)
        assert r.get("v1") is v

    def test_get_missing(self):
        r = VendorRegistry()
        assert r.get("nope") is None

    def test_remove(self):
        r = VendorRegistry()
        r.add(VendorRecord(id="v1", name="AWS"))
        assert r.remove("v1") is True

    def test_all(self):
        r = self._sample_registry()
        assert len(r.all()) == 3

    def test_by_status(self):
        r = self._sample_registry()
        active = r.by_status("active")
        assert len(active) == 2
        inactive = r.by_status("inactive")
        assert len(inactive) == 1

    def test_by_category(self):
        r = self._sample_registry()
        cloud = r.by_category("cloud")
        assert len(cloud) == 2
        logistics = r.by_category("logistics")
        assert len(logistics) == 1


# ---------------------------------------------------------------------------
# OperationWorkflow integration tests (mock LLM to avoid API key requirement)
# ---------------------------------------------------------------------------


class TestOperationsWorkflow:
    @patch("agents.operations.get_llm", return_value="gpt-4o")
    def test_workflow_initialization(self, _mock_llm):
        wf = OperationsWorkflow()
        assert wf.agent is not None
        assert wf.task_queue is not None
        assert wf.sop_tracker is not None
        assert wf.vendor_registry is not None

    @patch("agents.operations.get_llm", return_value="gpt-4o")
    def test_workflow_task_queue_integration(self, _mock_llm):
        wf = OperationsWorkflow()
        wf.task_queue.add(TaskItem(id="t1", title="Task 1", priority="P0"))
        wf.task_queue.add(TaskItem(id="t2", title="Task 2", priority="P1"))
        assert len(wf.task_queue.all()) == 2
        assert len(wf.task_queue.by_priority("P0")) == 1

    @patch("agents.operations.get_llm", return_value="gpt-4o")
    def test_workflow_sop_tracker_integration(self, _mock_llm):
        wf = OperationsWorkflow()
        wf.sop_tracker.add(SOP(id="s1", title="Onboarding", steps=["Step 1"], owner="Alice"))
        audit = wf.sop_tracker.audit()
        assert audit["total"] == 1

    @patch("agents.operations.get_llm", return_value="gpt-4o")
    def test_workflow_vendor_registry_integration(self, _mock_llm):
        wf = OperationsWorkflow()
        wf.vendor_registry.add(VendorRecord(id="v1", name="AWS", category="cloud"))
        assert len(wf.vendor_registry.by_category("cloud")) == 1


# ---------------------------------------------------------------------------
# Function signature tests (no LLM needed)
# ---------------------------------------------------------------------------


class TestTaskFunctionSignatures:
    def test_vendor_evaluation_task_signature(self):
        import inspect
        sig = inspect.signature(create_vendor_evaluation_task)
        assert "vendor_info" in sig.parameters
        assert "agent" in sig.parameters

    def test_process_automation_task_signature(self):
        import inspect
        sig = inspect.signature(create_process_automation_task)
        assert "process_description" in sig.parameters
        assert "agent" in sig.parameters

    def test_task_management_task_signature(self):
        import inspect
        sig = inspect.signature(create_task_management_task)
        assert "tasks" in sig.parameters
        assert "agent" in sig.parameters

    def test_sop_tracker_task_signature(self):
        import inspect
        sig = inspect.signature(create_sop_tracker_task)
        assert "sop_list" in sig.parameters
        assert "agent" in sig.parameters
