"""Operations Employee Agent.

Manages tasks, tracks SOPs, handles vendors,
and automates operational processes.
"""

from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from core.llm import get_llm


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class TaskItem:
    """A single task with priority, status, and dependency tracking.

    Attributes:
        id: Unique task identifier.
        title: Short task description.
        description: Detailed task description.
        priority: Priority level — ``"P0"`` (critical) through ``"P3"`` (low).
        status: One of ``"backlog"``, ``"todo"``, ``"in_progress"``,
            ``"blocked"``, ``"done"``, ``"cancelled"``.
        effort_hours: Estimated effort in hours.
        depends_on: List of task IDs that must be completed first.
        assignee_role: Suggested role for the assignee.
        blocked_reason: Human-readable reason why the task is blocked.
        tags: Arbitrary tags for grouping and filtering.
        batch_group: Identifier for tasks that can be batched together.
    """

    def __init__(
        self,
        id: str,
        title: str,
        description: str = "",
        priority: str = "P2",
        status: str = "backlog",
        effort_hours: float = 0.0,
        depends_on: list[str] | None = None,
        assignee_role: str = "",
        blocked_reason: str = "",
        tags: list[str] | None = None,
        batch_group: str = "",
    ):
        self.id = id
        self.title = title
        self.description = description
        self.priority = priority
        self.status = status
        self.effort_hours = effort_hours
        self.depends_on = depends_on or []
        self.assignee_role = assignee_role
        self.blocked_reason = blocked_reason
        self.tags = tags or []
        self.batch_group = batch_group

    def is_blocked(self, completed_ids: set[str]) -> bool:
        """Return True if any dependency is not in *completed_ids*."""
        return any(dep_id not in completed_ids for dep_id in self.depends_on)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "status": self.status,
            "effort_hours": self.effort_hours,
            "depends_on": self.depends_on,
            "assignee_role": self.assignee_role,
            "blocked_reason": self.blocked_reason,
            "tags": self.tags,
            "batch_group": self.batch_group,
        }


class TaskQueue:
    """In-memory task queue with priority ordering and dependency resolution.

    Example::

        queue = TaskQueue()
        queue.add(TaskItem(id="t1", title="Deploy API", priority="P0"))
        queue.add(TaskItem(id="t2", title="Write docs", priority="P3", depends_on=["t1"]))
        order = queue.execution_order()  # ["t1", "t2"]
    """

    def __init__(self):
        self._tasks: dict[str, TaskItem] = {}

    # -- CRUD ---------------------------------------------------------------

    def add(self, task: TaskItem) -> None:
        """Add or replace a task."""
        self._tasks[task.id] = task

    def get(self, task_id: str) -> TaskItem | None:
        """Return a task by ID, or ``None`` if not found."""
        return self._tasks.get(task_id)

    def remove(self, task_id: str) -> bool:
        """Remove a task. Returns ``True`` if it existed."""
        return self._tasks.pop(task_id, None) is not None

    def all(self) -> list[TaskItem]:
        """Return all tasks."""
        return list(self._tasks.values())

    # -- Queries ------------------------------------------------------------

    def by_status(self, status: str) -> list[TaskItem]:
        """Return tasks filtered by *status*."""
        return [t for t in self._tasks.values() if t.status == status]

    def by_priority(self, priority: str) -> list[TaskItem]:
        """Return tasks filtered by *priority*."""
        return [t for t in self._tasks.values() if t.priority == priority]

    def by_tag(self, tag: str) -> list[TaskItem]:
        """Return tasks that contain *tag*."""
        return [t for t in self._tasks.values() if tag in t.tags]

    def blocked_tasks(self) -> list[TaskItem]:
        """Return tasks whose dependencies are not all satisfied."""
        completed = {t.id for t in self._tasks.values() if t.status == "done"}
        return [t for t in self._tasks.values() if t.status != "done" and t.is_blocked(completed)]

    def batchable_groups(self) -> dict[str, list[TaskItem]]:
        """Return non-empty batch groups keyed by *batch_group* name."""
        groups: dict[str, list[TaskItem]] = {}
        for t in self._tasks.values():
            if t.batch_group:
                groups.setdefault(t.batch_group, []).append(t)
        return groups

    # -- Ordering -----------------------------------------------------------

    _PRIORITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}

    def prioritized(self) -> list[TaskItem]:
        """Return non-done tasks sorted by priority (P0 first)."""
        active = [t for t in self._tasks.values() if t.status not in ("done", "cancelled")]
        return sorted(active, key=lambda t: self._PRIORITY_ORDER.get(t.priority, 99))

    def execution_order(self) -> list[str]:
        """Return task IDs in a dependency-respecting execution order.

        Tasks with fewer unresolved dependencies come first; ties are broken
        by priority.  Circular dependencies are broken arbitrarily.
        """
        completed: set[str] = set()
        order: list[str] = []
        remaining = list(self._tasks.values())

        # Protect against infinite loops from circular deps
        max_iterations = len(remaining) ** 2 + 1
        iterations = 0

        while remaining and iterations < max_iterations:
            iterations += 1
            ready = [t for t in remaining if not t.is_blocked(completed)]
            if not ready:
                # Circular dependency — break by adding remaining in priority order
                ready = remaining
            ready.sort(key=lambda t: self._PRIORITY_ORDER.get(t.priority, 99))
            for t in ready:
                order.append(t.id)
                completed.add(t.id)
                remaining.remove(t)

        return order


# ---------------------------------------------------------------------------
# SOP tracking
# ---------------------------------------------------------------------------

class SOP:
    """Standard Operating Procedure document.

    Attributes:
        id: Unique SOP identifier.
        title: SOP title.
        version: Semantic version string, e.g. ``"1.0.0"``.
        steps: Ordered list of step descriptions.
        owner: Person or team responsible for this SOP.
        review_date: ISO-8601 date string for the next review.
        tags: Arbitrary tags for categorisation.
    """

    def __init__(
        self,
        id: str,
        title: str,
        version: str = "1.0.0",
        steps: list[str] | None = None,
        owner: str = "",
        review_date: str = "",
        tags: list[str] | None = None,
    ):
        self.id = id
        self.title = title
        self.version = version
        self.steps = steps or []
        self.owner = owner
        self.review_date = review_date
        self.tags = tags or []

    def step_count(self) -> int:
        return len(self.steps)

    def has_empty_steps(self) -> bool:
        return len(self.steps) == 0 or any(not step.strip() for step in self.steps)

    def completeness_score(self) -> float:
        """Return a 0-1 completeness score based on field population."""
        fields = [self.title, self.version, self.owner, self.review_date]
        filled = sum(1 for f in fields if f)
        step_ratio = min(len(self.steps) / 5, 1.0)  # assume 5 steps = "complete"
        return (filled / len(fields) * 0.6) + (step_ratio * 0.4)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "version": self.version,
            "steps": self.steps,
            "owner": self.owner,
            "review_date": self.review_date,
            "tags": self.tags,
        }


class SOPTracker:
    """Manages a collection of SOP documents.

    Example::

        tracker = SOPTracker()
        tracker.add(SOP(id="sop-1", title="Onboarding", steps=["...", "..."]))
        audit = tracker.audit()   # compliance check across all SOPs
    """

    def __init__(self):
        self._sops: dict[str, SOP] = {}

    def add(self, sop: SOP) -> None:
        self._sops[sop.id] = sop

    def get(self, sop_id: str) -> SOP | None:
        return self._sops.get(sop_id)

    def remove(self, sop_id: str) -> bool:
        return self._sops.pop(sop_id, None) is not None

    def all(self) -> list[SOP]:
        return list(self._sops.values())

    def needs_review(self, before_date: str) -> list[SOP]:
        """Return SOPs whose *review_date* is on or before *before_date*."""
        return [s for s in self._sops.values() if s.review_date and s.review_date <= before_date]

    def by_owner(self, owner: str) -> list[SOP]:
        return [s for s in self._sops.values() if s.owner == owner]

    def by_tag(self, tag: str) -> list[SOP]:
        return [s for s in self._sops.values() if tag in s.tags]

    def audit(self) -> dict:
        """Run a compliance audit across all SOPs.

        Returns a dict with:
            - ``total``: total number of SOPs
            - ``needs_review``: count of SOPs past review date
            - ``incomplete``: count of SOPs with empty steps or missing fields
            - ``avg_completeness``: average completeness score (0-1)
            - ``issues``: list of human-readable issue strings
        """
        issues: list[str] = []
        total = len(self._sops)
        needs_review = 0
        incomplete = 0
        scores: list[float] = []

        for sop in self._sops.values():
            if sop.review_date:
                needs_review += 1  # simplified: all dated SOPs flagged

            if sop.has_empty_steps():
                incomplete += 1
                issues.append(f"SOP '{sop.id}' has empty steps.")

            if not sop.owner:
                issues.append(f"SOP '{sop.id}' has no owner assigned.")

            if not sop.review_date:
                issues.append(f"SOP '{sop.id}' has no review date.")

            score = sop.completeness_score()
            scores.append(score)
            if score < 0.5:
                incomplete += 1
                issues.append(f"SOP '{sop.id}' is incomplete (score: {score:.2f}).")

        avg = sum(scores) / len(scores) if scores else 0.0
        return {
            "total": total,
            "needs_review": needs_review,
            "incomplete": incomplete,
            "avg_completeness": round(avg, 2),
            "issues": issues,
        }


# ---------------------------------------------------------------------------
# Process automation
# ---------------------------------------------------------------------------

class ProcessStep:
    """A single step in an automated process.

    Attributes:
        id: Unique step identifier.
        name: Human-readable step name.
        action: Action type – ``"notify"``, ``"assign"``, ``"escalate"``,
            ``"auto_approve"``, ``"run_check"``.
        params: Action-specific parameters.
        next_step_id: ID of the next step, or ``""`` for terminal steps.
        condition: Optional condition expression (free-text for LLM evaluation).
    """

    def __init__(
        self,
        id: str,
        name: str,
        action: str,
        params: dict | None = None,
        next_step_id: str = "",
        condition: str = "",
    ):
        self.id = id
        self.name = name
        self.action = action
        self.params = params or {}
        self.next_step_id = next_step_id
        self.condition = condition

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "action": self.action,
            "params": self.params,
            "next_step_id": self.next_step_id,
            "condition": self.condition,
        }


class AutomatedProcess:
    """A multi-step automated workflow.

    Example::

        proc = AutomatedProcess(id="proc-1", name="Invoice Approval")
        proc.add_step(ProcessStep(id="s1", name="Check amount", action="run_check"))
        proc.add_step(ProcessStep(id="s2", name="Approve", action="auto_approve", next_step_id=""))
        chain = proc.build_chain()  # ordered list of steps
    """

    def __init__(self, id: str, name: str, description: str = "", trigger: str = ""):
        self.id = id
        self.name = name
        self.description = description
        self.trigger = trigger
        self._steps: dict[str, ProcessStep] = {}

    def add_step(self, step: ProcessStep) -> None:
        self._steps[step.id] = step

    def get_step(self, step_id: str) -> ProcessStep | None:
        return self._steps.get(step_id)

    def all_steps(self) -> list[ProcessStep]:
        return list(self._steps.values())

    def build_chain(self) -> list[ProcessStep]:
        """Return steps in execution order starting from the first step.

        The first step is the one that is not referenced as ``next_step_id``
        by any other step.  Falls back to insertion order if the graph is
        ambiguous.
        """
        referenced = {s.next_step_id for s in self._steps.values() if s.next_step_id}
        roots = [s for s in self._steps.values() if s.id not in referenced]

        if not roots:
            # Circular or ambiguous — return flat list
            return list(self._steps.values())

        ordered: list[ProcessStep] = []
        current = roots[0]
        visited: set[str] = set()
        while current and current.id not in visited:
            visited.add(current.id)
            ordered.append(current)
            current = self._steps.get(current.next_step_id) if current.next_step_id else None
        return ordered

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "trigger": self.trigger,
            "steps": [s.to_dict() for s in self.build_chain()],
        }


class VendorRecord:
    """Vendor / supplier information.

    Attributes:
        id: Unique vendor identifier.
        name: Vendor company name.
        category: Service category (e.g. ``"cloud"``, ``"logistics"``).
        contact_email: Primary contact email.
        sla_hours: SLA response-time commitment in hours.
        status: ``"active"``, ``"inactive"``, or ``"under_review"``.
        notes: Free-form notes.
    """

    def __init__(
        self,
        id: str,
        name: str,
        category: str = "",
        contact_email: str = "",
        sla_hours: int = 24,
        status: str = "active",
        notes: str = "",
    ):
        self.id = id
        self.name = name
        self.category = category
        self.contact_email = contact_email
        self.sla_hours = sla_hours
        self.status = status
        self.notes = notes

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "contact_email": self.contact_email,
            "sla_hours": self.sla_hours,
            "status": self.status,
            "notes": self.notes,
        }


class VendorRegistry:
    """Manages vendor records.

    Example::

        reg = VendorRegistry()
        reg.add(VendorRecord(id="v1", name="AWS", category="cloud"))
        active = reg.by_status("active")
    """

    def __init__(self):
        self._vendors: dict[str, VendorRecord] = {}

    def add(self, vendor: VendorRecord) -> None:
        self._vendors[vendor.id] = vendor

    def get(self, vendor_id: str) -> VendorRecord | None:
        return self._vendors.get(vendor_id)

    def remove(self, vendor_id: str) -> bool:
        return self._vendors.pop(vendor_id, None) is not None

    def all(self) -> list[VendorRecord]:
        return list(self._vendors.values())

    def by_status(self, status: str) -> list[VendorRecord]:
        return [v for v in self._vendors.values() if v.status == status]

    def by_category(self, category: str) -> list[VendorRecord]:
        return [v for v in self._vendors.values() if v.category == category]


# ---------------------------------------------------------------------------
# CrewAI agent & tasks
# ---------------------------------------------------------------------------


def create_operations_agent() -> Agent:
    return Agent(
        role="Operations Manager",
        goal="Keep business operations running smoothly through task management, SOPs, and process optimization",
        backstory=(
            "You are a detail-oriented operations manager who keeps everything "
            "running like clockwork. You manage task queues, standard operating "
            "procedures, vendor relationships, and process documentation. "
            "You identify bottlenecks before they become problems."
        ),
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_task_management_task(tasks: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Process and organize the following tasks:\n\n{tasks}\n\n"
            "For each task: assign priority (P0-P3), estimate effort, "
            "identify dependencies, suggest assignee role, "
            "flag blockers, and create a suggested execution order. "
            "Also identify tasks that can be batched or automated."
        ),
        expected_output="Prioritized task board with effort estimates, dependencies, and execution order",
        agent=agent,
    )


def create_sop_tracker_task(sop_list: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Review and update the following SOPs:\n\n{sop_list}\n\n"
            "For each SOP: check completeness, identify gaps, "
            "suggest improvements, flag outdated steps, "
            "and create a revision checklist. Also check "
            "cross-SOP consistency and identify duplicate processes."
        ),
        expected_output="SOP audit report with gap analysis, improvement suggestions, and revision priorities",
        agent=agent,
    )


def create_vendor_evaluation_task(vendor_info: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Evaluate the following vendors:\n\n{vendor_info}\n\n"
            "For each vendor: assess SLA compliance, categorize service quality, "
            "identify risks, recommend renewals or replacements, "
            "and suggest negotiation points. "
            "Also flag any vendors with overlapping services."
        ),
        expected_output="Vendor evaluation report with risk flags, recommendations, and negotiation points",
        agent=agent,
    )


def create_process_automation_task(process_description: str, agent: Agent) -> Task:
    return Task(
        description=(
            f"Design an automated process for:\n\n{process_description}\n\n"
            "Identify: trigger conditions, decision points, action steps, "
            "escalation paths, and success metrics. "
            "Flag any steps that require human approval. "
            "Also identify potential failure modes and recovery procedures."
        ),
        expected_output="Process automation blueprint with triggers, actions, escalation paths, and success metrics",
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class OperationsWorkflow:
    """Operations management workflow."""

    def __init__(self):
        self.agent = create_operations_agent()
        self.task_queue = TaskQueue()
        self.sop_tracker = SOPTracker()
        self.vendor_registry = VendorRegistry()

    def manage_tasks(self, tasks: str) -> str:
        task = create_task_management_task(tasks, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def audit_sops(self, sop_list: str) -> str:
        task = create_sop_tracker_task(sop_list, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def evaluate_vendors(self, vendor_info: str) -> str:
        task = create_vendor_evaluation_task(vendor_info, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()

    def automate_process(self, process_description: str) -> str:
        task = create_process_automation_task(process_description, self.agent)
        crew = Crew(agents=[self.agent], tasks=[task], verbose=True)
        return crew.kickoff()
