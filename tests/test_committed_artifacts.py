"""Every artifact committed under artifacts/ must still parse against the live
schema and be minimally well-formed. The pytest-test reframe of "add a CI pipeline
validating artifacts" a code review suggested for a different system: same
guarantee (a schema-incompatible or malformed artifact never lands undetected),
without standing up scaling infrastructure this project's own scope explicitly
doesn't need.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact.from_run import build_artifact_from_run
from artifact.schema import Artifact

ARTIFACT_PATHS = sorted(Path("artifacts").glob("*.json"))

# v1: byte-for-byte rebuildable from a single run's own, unmodified trace --
# (artifact path, evidence dir, template id, base url, entry url). This is
# what makes "derived from a real run, not hand-written" a checkable claim
# rather than an assertion in a README, and it catches the quieter failure
# too: a schema change that silently alters what a rebuild produces. This is
# the run that had the row-scoping bug (see the v2 test below) -- it is kept
# and still exercised so the fix is comparable against exactly what it fixed,
# not just asserted in prose.
DERIVED_ARTIFACTS = [
    ("artifacts/vb-bank-admin-balance.v1.json", "evidence/discovery-run-v1",
     "vb_bank_admin_balance", "https://vb-bank-demo.vercel.app/*", "https://vb-bank-demo.vercel.app/login"),
]

# v2: a second, independent recording (evidence/discovery-run-v2/), given a
# goal that added one explicit row-scoping instruction the v1 goal lacked --
# not a hand edit of v1's JSON. Its own declared_outcomes.json already carries
# the merge: checkpoint/output_extractors are this run's own declarations,
# business_outcomes is copied in verbatim from a third, separate run
# (evidence/discovery-run-notfound-probe/) whose goal was specifically to
# search a nonexistent account and declare what it observed. Rebuilding from
# that file reproduces every field except `version` and `revision_note`,
# which a discovery run has no way to author -- those two are asserted
# separately below.
DERIVED_ARTIFACTS_V2 = [
    ("artifacts/vb-bank-admin-balance.v2.json", "evidence/discovery-run-v2",
     "vb_bank_admin_balance", "https://vb-bank-demo.vercel.app/*", "https://vb-bank-demo.vercel.app/login"),
]

MERGED_BUSINESS_OUTCOMES = [
    ("evidence/discovery-run-v2/declared_outcomes.json", "evidence/discovery-run-notfound-probe/declared_outcomes.json"),
]


@pytest.mark.parametrize("path", ARTIFACT_PATHS, ids=[p.name for p in ARTIFACT_PATHS])
def test_committed_artifact_loads_and_is_well_formed(path: Path):
    artifact = Artifact.load(path)
    assert artifact.steps, f"{path} has no steps"
    assert artifact.checkpoint is not None, f"{path} declares no checkpoint"
    for step in artifact.steps:
        assert step.idempotency_note, f"{path} step {step.step} has an empty idempotency_note"


def test_at_least_one_artifact_is_actually_committed():
    # Guards against the parametrized test above silently collecting zero cases
    # (e.g. if artifacts/ is ever emptied or renamed) and reporting a false "pass".
    assert ARTIFACT_PATHS, "no artifacts found under artifacts/ -- test above would silently pass on nothing"


@pytest.mark.parametrize(
    "artifact_path,run_dir,template_id,base_url,entry_url",
    DERIVED_ARTIFACTS,
    ids=[Path(a).stem for a, *_ in DERIVED_ARTIFACTS],
)
def test_committed_artifact_still_rebuilds_from_its_evidence(artifact_path, run_dir, template_id, base_url, entry_url):
    rebuilt = build_artifact_from_run(
        run_log_path=f"{run_dir}/log.jsonl",
        declared_outcomes_path=f"{run_dir}/declared_outcomes.json",
        artifact_id=Path(artifact_path).name.split(".v")[0],
        template_id=template_id,
        base_url_pattern=base_url,
        entry_url=entry_url,
    )
    committed = Path(artifact_path).read_text()
    assert rebuilt.model_dump_json(indent=2) == committed, (
        f"{artifact_path} no longer matches what {run_dir} rebuilds -- either the trace, "
        f"the declared outcomes, or the schema changed. Re-run scripts/build_artifact.py "
        f"--force if the change is deliberate."
    )


def test_v2_rebuilds_from_its_evidence_except_version_and_revision_note():
    artifact_path, run_dir, template_id, base_url, entry_url = DERIVED_ARTIFACTS_V2[0]
    rebuilt = build_artifact_from_run(
        run_log_path=f"{run_dir}/log.jsonl",
        declared_outcomes_path=f"{run_dir}/declared_outcomes.json",
        artifact_id=Path(artifact_path).name.split(".v")[0],
        template_id=template_id,
        base_url_pattern=base_url,
        entry_url=entry_url,
    )
    committed = Artifact.load(artifact_path)
    rebuilt_dump = rebuilt.model_dump()
    committed_dump = committed.model_dump()
    for field in ("version", "revision_note"):
        del rebuilt_dump[field]
        del committed_dump[field]
    assert rebuilt_dump == committed_dump, (
        f"{artifact_path} no longer matches what {run_dir} rebuilds on any field other than "
        f"version/revision_note -- either the trace, the declared outcomes, or the schema changed"
    )
    # The two fields a discovery run genuinely cannot produce on its own --
    # this is the one place a human's own words (why this version exists,
    # what it changed relative to v1) are load-bearing rather than spliced
    # from another run's tool call.
    assert committed.version == 2
    assert committed.revision_note, f"{artifact_path} is a second version with no revision_note explaining why"


def test_no_artifact_declares_an_output_it_cannot_produce():
    # Artifact's own validator enforces this at load time; asserting it here
    # makes the contract visible as a property of the committed catalog rather
    # than only as a constructor invariant.
    for path in ARTIFACT_PATHS:
        artifact = Artifact.load(path)
        assert set(artifact.output_schema) == {e.name for e in artifact.output_extractors}, path


@pytest.mark.parametrize(
    "merged_path,source_path", MERGED_BUSINESS_OUTCOMES,
    ids=[Path(m).parent.name for m, _ in MERGED_BUSINESS_OUTCOMES],
)
def test_merged_business_outcomes_are_a_verbatim_copy_of_their_source_run(merged_path, source_path):
    merged = json.loads(Path(merged_path).read_text())
    source = json.loads(Path(source_path).read_text())
    assert merged["business_outcomes"] == source["business_outcomes"], (
        f"{merged_path}'s business_outcomes no longer match {source_path}'s own declaration -- "
        f"this field is supposed to be spliced in verbatim from a real recording, never hand-edited"
    )
    assert merged["business_outcomes"], f"{source_path} declared no business_outcomes to copy"


def test_v2_extractor_is_scoped_to_the_requested_row_unlike_v1():
    # The concrete, checkable version of the claim in REPORT.md §2: v1's
    # extractor has no reference to the account number at all (it reads
    # whatever currency-shaped text comes first in the results card); v2's
    # does (it's anchored on that exact text node). Asserted directly on the
    # committed files so the comparison can't silently drift out of sync with
    # the prose describing it.
    v1 = Artifact.load("artifacts/vb-bank-admin-balance.v1.json")
    v2 = Artifact.load("artifacts/vb-bank-admin-balance.v2.json")
    v1_target = v1.output_extractors[0].target.value
    v2_target = v2.output_extractors[0].target.value
    assert "{{account_number}}" not in v1_target, "this test's premise (v1 was unscoped) no longer holds"
    assert "{{account_number}}" in v2_target, "v2's extractor should be anchored on the requested account number"


def test_evidence_artifact_copies_match_the_catalog():
    # evidence/artifact/ holds copies for a reader of evidence/ alone; they
    # must never drift from the artifacts the scripts actually load.
    for path in ARTIFACT_PATHS:
        copy = Path("evidence/artifact") / path.name
        assert copy.exists(), f"{copy} missing"
        assert copy.read_text() == path.read_text(), f"{copy} differs from {path}"
