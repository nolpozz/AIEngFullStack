"""
Robust test suite for mock_scheduler.py

These tests target the *contract* every correct implementation must satisfy,
not a reference implementation — so they're meaningful the moment you finish
filling in the stubs. Because pytest runs each test independently, a partial
implementation still gets partial signal: finish BlockManager and the
TestBlockManager class goes green before the scheduler exists.

Run:
    pip install pytest
    pytest test_mock_scheduler.py -v
(Keep this file next to mock_scheduler.py.)

Two conventions make the suite deterministic and strict:

  * stop_prob=0.0 in scenario tests.  Disabling the random stop means a request
    finishes exactly when output_len >= max_tokens, so tick counts and generated
    counts are exact and assertable.  (Assumes _is_finished implements random
    stop as `random.random() < req.stop_prob`, which is 0 -> never.)

  * Invariants are checked EVERY tick, not just at the end.  The central one is
    a block-partition check: the free list and all per-sequence block tables must
    together form an exact, non-overlapping partition of the physical block pool.
    That one assertion catches leaks, double-frees, and aliasing simultaneously.
"""

from __future__ import annotations

import math
import random

import pytest

import mock_scheduler as ms
from mock_scheduler import (
    BlockManager,
    MockExecutor,
    Request,
    ScheduledSeq,
    Scheduler,
    SchedulerOutput,
    SchedulingBudget,
    SeqStatus,
)


# ============================================================================
# Helpers
# ============================================================================

def build(num_blocks=64, block_size=16, max_num_batched_tokens=4096, max_num_seqs=16):
    """Construct a scheduler + its block manager. Returns (scheduler, block_manager)."""
    bm = BlockManager(num_blocks=num_blocks, block_size=block_size)
    sch = Scheduler(bm, MockExecutor(), max_num_batched_tokens, max_num_seqs)
    return sch, bm


def add_requests(scheduler, specs):
    """specs: iterable of (prompt_len, max_tokens, stop_prob). Returns the Requests,
    which double as the full registry for invariant checks."""
    reqs = []
    for i, (plen, mtok, sprob) in enumerate(specs):
        r = Request(request_id=f"r{i}", prompt_len=plen, max_tokens=mtok, stop_prob=sprob)
        scheduler.add_request(r)
        reqs.append(r)
    return reqs


def spy_schedule(scheduler):
    """Wrap _schedule so we can inspect every SchedulerOutput it emits.
    Returns a list that accumulates outputs as steps run."""
    outputs = []
    original = scheduler._schedule

    def wrapped():
        out = original()
        outputs.append(out)
        return out

    scheduler._schedule = wrapped
    return outputs


# --- invariant assertions --------------------------------------------------

def assert_block_partition(bm, registry, num_blocks):
    """Free list + all block tables must be an exact, disjoint partition of the pool.
    Catches leaks (blocks vanish), double-free (block in free list twice or in
    free + a table), and aliasing (one physical block in two tables)."""
    free = list(bm.free_blocks)
    assert len(free) == len(set(free)), f"duplicate block in free list: {free}"

    allocated = []
    for r in registry:
        allocated.extend(r.block_ids)
    assert len(allocated) == len(set(allocated)), (
        "same physical block appears in two block tables (aliasing)"
    )

    overlap = set(free) & set(allocated)
    assert not overlap, f"block(s) both free and allocated: {sorted(overlap)}"

    assert set(free) | set(allocated) == set(range(num_blocks)), (
        "blocks lost or invented: partition != full pool"
    )


def assert_coverage(registry, block_size):
    """Between ticks, each request's block table must exactly cover its computed
    tokens, and non-running requests must hold nothing."""
    for r in registry:
        if r.status == SeqStatus.RUNNING:
            need = math.ceil(r.num_computed_tokens / block_size) if r.num_computed_tokens else 0
            assert len(r.block_ids) == need, (
                f"{r.request_id}: table has {len(r.block_ids)} blocks, "
                f"needs exactly {need} for {r.num_computed_tokens} tokens"
            )
        else:  # WAITING or FINISHED
            assert r.block_ids == [], (
                f"{r.request_id} is {r.status} but still holds blocks {r.block_ids}"
            )
        if r.status == SeqStatus.WAITING:
            assert r.num_computed_tokens == 0, (
                f"{r.request_id} is WAITING but num_computed_tokens={r.num_computed_tokens} "
                f"(recomputation preemption must reset it to 0)"
            )


def assert_budget(outputs, max_tokens, max_seqs):
    """No scheduled step may exceed the token or sequence budget."""
    for out in outputs:
        tok = sum(s.num_tokens for s in out.scheduled)
        assert tok <= max_tokens, f"step batched {tok} tokens > cap {max_tokens}"
        assert len(out.scheduled) <= max_seqs, (
            f"step scheduled {len(out.scheduled)} seqs > cap {max_seqs}"
        )


def run_to_completion(scheduler, bm, registry, num_blocks, block_size,
                      check=True, max_ticks=200_000):
    """Drive the scheduler until it drains, checking invariants each tick.
    Returns (finished_order, ticks, outputs). Fails loudly on livelock."""
    outputs = spy_schedule(scheduler)
    finished_order = []
    ticks = 0
    while scheduler.has_unfinished():
        if ticks > max_ticks:
            pytest.fail(f"did not drain within {max_ticks} ticks — livelock or infinite loop")
        done = scheduler.step()
        ticks += 1
        for r in done:
            finished_order.append((ticks, r.request_id))
        if check:
            assert_block_partition(bm, registry, num_blocks)
            assert_coverage(registry, block_size)
    if check:
        assert_budget(outputs, scheduler.max_num_batched_tokens, scheduler.max_num_seqs)
    return finished_order, ticks, outputs


# ============================================================================
# Component unit tests
# ============================================================================

class TestBlockManager:

    def test_blocks_for_is_ceiling(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        assert bm._blocks_for(0) == 0
        assert bm._blocks_for(1) == 1
        assert bm._blocks_for(16) == 1
        assert bm._blocks_for(17) == 2
        assert bm._blocks_for(32) == 2
        assert bm._blocks_for(33) == 3

    def test_initial_pool_is_all_free(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        assert bm.num_free_blocks() == 10

    def test_allocate_then_free_round_trips(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        r = Request("a", prompt_len=40, max_tokens=8)  # ceil(40/16) = 3 blocks
        assert bm.can_allocate(r)
        bm.allocate(r)
        assert len(r.block_ids) == 3
        assert bm.num_free_blocks() == 7
        bm.free(r)
        assert bm.num_free_blocks() == 10
        assert r.block_ids == []

    def test_can_allocate_false_when_insufficient(self):
        bm = BlockManager(num_blocks=2, block_size=16)
        r = Request("a", prompt_len=100, max_tokens=8)  # needs 7 blocks, only 2 exist
        assert not bm.can_allocate(r)

    def test_allocated_blocks_are_distinct(self):
        bm = BlockManager(num_blocks=10, block_size=16)
        r = Request("a", prompt_len=48, max_tokens=8)  # 3 blocks
        bm.allocate(r)
        assert len(set(r.block_ids)) == len(r.block_ids)

    def test_append_slot_grabs_block_on_boundary(self):
        # Prompt exactly fills one block; the next token must start a new block.
        bm = BlockManager(num_blocks=10, block_size=16)
        r = Request("a", prompt_len=16, max_tokens=8)
        bm.allocate(r)
        r.num_computed_tokens = 16          # block 0 is full (indices 0..15)
        free_before = bm.num_free_blocks()
        assert bm.can_append_slot(r)
        bm.append_slot(r)
        assert bm.num_free_blocks() == free_before - 1
        assert len(r.block_ids) == 2

    def test_append_slot_noop_within_block(self):
        # Prompt leaves room in the last block; the next token needs no new block.
        bm = BlockManager(num_blocks=10, block_size=16)
        r = Request("a", prompt_len=15, max_tokens=8)
        bm.allocate(r)
        r.num_computed_tokens = 15          # block 0 has one slot left
        free_before = bm.num_free_blocks()
        bm.append_slot(r)
        assert bm.num_free_blocks() == free_before
        assert len(r.block_ids) == 1

    def test_can_append_slot_false_when_pool_empty_at_boundary(self):
        bm = BlockManager(num_blocks=1, block_size=16)
        r = Request("a", prompt_len=16, max_tokens=8)
        bm.allocate(r)                      # consumes the only block
        r.num_computed_tokens = 16
        assert bm.num_free_blocks() == 0
        assert not bm.can_append_slot(r)    # boundary crossing, nothing free


class TestSchedulingBudget:

    def test_token_cap(self):
        b = SchedulingBudget(max_num_batched_tokens=100, max_num_seqs=10)
        assert b.can_add(100)
        assert not b.can_add(101)
        b.add(60)
        assert b.can_add(40)
        assert not b.can_add(41)

    def test_seq_cap(self):
        b = SchedulingBudget(max_num_batched_tokens=10_000, max_num_seqs=2)
        assert b.can_add(1)
        b.add(1)
        b.add(1)
        assert not b.can_add(1)


class TestRequestState:

    def test_needs_prefill_transitions(self):
        r = Request("a", prompt_len=20, max_tokens=8)
        assert r.needs_prefill
        r.num_computed_tokens = 20
        assert not r.needs_prefill

    def test_output_len_derivation(self):
        r = Request("a", prompt_len=20, max_tokens=8)
        assert r.output_len == 0            # nothing computed
        r.num_computed_tokens = 20
        assert r.output_len == 0            # prefill done, no tokens generated
        r.num_computed_tokens = 23
        assert r.output_len == 3


# ============================================================================
# Preemption contract (white-box, no loop-timing ambiguity)
# ============================================================================

class TestPreemption:

    def test_preempt_frees_resets_and_requeues(self):
        sch, bm = build(num_blocks=16, block_size=16)
        r = Request("victim", prompt_len=48, max_tokens=8)  # 3 blocks
        bm.allocate(r)
        r.num_computed_tokens = 50
        r.status = SeqStatus.RUNNING
        sch.running.append(r)
        free_before = bm.num_free_blocks()

        sch._preempt(r)

        assert r not in sch.running,                 "victim must leave the running queue"
        assert sch.waiting and sch.waiting[0] is r,  "victim must go to the FRONT of waiting (FCFS)"
        assert r.status == SeqStatus.WAITING
        assert r.num_computed_tokens == 0,           "recomputation must drop all progress"
        assert r.block_ids == [],                    "KV blocks must be freed"
        assert bm.num_free_blocks() == free_before + 3

    def test_tight_pool_forces_preemption_and_still_drains(self):
        # Enough blocks for the prompts, but not for everyone to decode far at
        # once -> the scheduler must preempt to make progress, and must still finish.
        block_size = 16
        # 4 requests, each prompt ~1 block, generating many tokens; pool sized so
        # they can't all grow simultaneously.
        sch, bm = build(num_blocks=6, block_size=block_size,
                        max_num_batched_tokens=4096, max_num_seqs=8)
        reqs = add_requests(sch, [(16, 40, 0.0) for _ in range(4)])

        _, _, outputs = run_to_completion(sch, bm, reqs, num_blocks=6, block_size=block_size)

        assert any(out.preempted for out in outputs), (
            "tight pool should have forced at least one preemption"
        )
        assert all(r.status == SeqStatus.FINISHED for r in reqs)
        assert bm.num_free_blocks() == 6, "pool must be fully restored after drain"


# ============================================================================
# Deterministic end-to-end scenarios (stop_prob = 0)
# ============================================================================

class TestDeterministicScenarios:

    def test_single_request_exact_ticks_and_output(self):
        sch, bm = build(num_blocks=64, block_size=16)
        [r] = add_requests(sch, [(30, 10, 0.0)])
        order, ticks, _ = run_to_completion(sch, bm, [r], num_blocks=64, block_size=16)

        # 1 prefill tick + max_tokens decode ticks, scheduled alone every step.
        assert ticks == 1 + 10
        assert r.output_len == 10
        assert r.status == SeqStatus.FINISHED
        assert order[-1] == (11, "r0")

    def test_all_requests_finish_and_pool_restored(self):
        sch, bm = build(num_blocks=64, block_size=16)
        reqs = add_requests(sch, [(20, 5, 0.0), (48, 12, 0.0), (8, 3, 0.0), (100, 7, 0.0)])
        run_to_completion(sch, bm, reqs, num_blocks=64, block_size=16)

        assert all(r.status == SeqStatus.FINISHED for r in reqs)
        for r in reqs:
            assert r.output_len == r.max_tokens
        assert bm.num_free_blocks() == 64

    def test_no_request_lost_or_duplicated(self):
        sch, bm = build(num_blocks=64, block_size=16)
        reqs = add_requests(sch, [(random.Random(i).randint(8, 120), 6, 0.0) for i in range(20)])
        order, _, _ = run_to_completion(sch, bm, reqs, num_blocks=64, block_size=16)

        finished_ids = [rid for _, rid in order]
        assert sorted(finished_ids) == sorted(r.request_id for r in reqs)
        assert len(finished_ids) == len(set(finished_ids)), "a request finished twice"

    @pytest.mark.parametrize("prompt_len", [15, 16, 17, 31, 32, 33, 48])
    def test_block_boundary_prompt_sizes(self, prompt_len):
        # Exercise off-by-one-block conditions in allocate/append_slot.
        sch, bm = build(num_blocks=64, block_size=16)
        [r] = add_requests(sch, [(prompt_len, 20, 0.0)])
        run_to_completion(sch, bm, [r], num_blocks=64, block_size=16)
        assert r.status == SeqStatus.FINISHED
        assert bm.num_free_blocks() == 64


# ============================================================================
# Determinism
# ============================================================================

class TestDeterminism:

    def test_same_seed_same_run(self):
        def one_run():
            random.seed(1234)
            sch, bm = build(num_blocks=12, block_size=16)
            reqs = add_requests(sch, [(24, 15, 0.1) for _ in range(6)])
            order, ticks, _ = run_to_completion(sch, bm, reqs, num_blocks=12, block_size=16)
            return order, ticks

        assert one_run() == one_run()


# ============================================================================
# Fuzz / property tests — the robustness centerpiece
# ============================================================================

class TestFuzz:

    @pytest.mark.parametrize("seed", range(25))
    def test_random_workloads_hold_all_invariants(self, seed):
        rng = random.Random(seed)
        random.seed(seed)  # also seed the module RNG the scheduler/executor use

        block_size = rng.choice([8, 16, 32])
        num_blocks = rng.randint(8, 40)
        # Cap prompt length so every prompt fits in the pool on its own; a prompt
        # needing more blocks than exist can never be scheduled under pure
        # recomputation (see TestKnownLimitations), which is out of scope here.
        max_prompt_tokens = num_blocks * block_size
        max_prompt = max(1, min(max_prompt_tokens, rng.randint(1, 200)))

        n = rng.randint(1, 20)
        specs = [
            (
                rng.randint(1, max_prompt),
                rng.randint(1, 30),
                rng.choice([0.0, 0.05, 0.2]),
            )
            for _ in range(n)
        ]

        sch, bm = build(
            num_blocks=num_blocks,
            block_size=block_size,
            max_num_batched_tokens=rng.choice([256, 1024, 4096]),
            max_num_seqs=rng.choice([2, 4, 16]),
        )
        reqs = add_requests(sch, specs)

        # run_to_completion checks partition + coverage each tick and budget at the
        # end, and fails on livelock. Reaching here means every invariant held.
        run_to_completion(sch, bm, reqs, num_blocks=num_blocks, block_size=block_size)

        assert all(r.status == SeqStatus.FINISHED for r in reqs)
        assert bm.num_free_blocks() == num_blocks


# ============================================================================
# Known limitations — documents behavior rather than asserting a bug
# ============================================================================

class TestKnownLimitations:

    def test_prompt_larger_than_pool_never_schedules(self):
        """Under pure recomputation, a prompt that needs more blocks than the whole
        pool can never be admitted. This isn't a bug — it's the tradeoff recomputation
        makes. If you later add swapping or chunked prefill, this test will flip and
        flag the behavior change, which is exactly what you want it to do."""
        block_size = 16
        num_blocks = 4                       # 64 tokens of capacity total
        sch, bm = build(num_blocks=num_blocks, block_size=block_size)
        [r] = add_requests(sch, [(200, 5, 0.0)])  # needs 13 blocks

        for _ in range(50):                  # bounded: this loop must NOT drain
            if not sch.has_unfinished():
                break
            sch.step()

        assert r.status != SeqStatus.FINISHED
        assert r in sch.waiting


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))