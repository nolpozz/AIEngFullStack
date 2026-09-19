"""
Mock single-GPU LLM scheduler — study scaffold.

Design boundary (the whole point of the exercise):
  Scheduler  = REAL  -> queues, step budget, block accounting, preemption
  Executor   = STUB  -> no model, no kernels; fabricates one token per seq

Fill in every method that raises NotImplementedError. The step() loop is
already wired, so the *order* of operations is fixed for you; the *decisions*
(who to batch, when to preempt, how blocks move) are what you implement.

Fill-in order that keeps the harness runnable soonest:
  1. Request properties + SchedulingBudget
  2. BlockManager (free list is the spine of everything)
  3. Scheduler._is_finished / _process_outputs   (loop closes, no preemption yet)
  4. Scheduler._schedule                          (prefill + decode batching)
  5. Scheduler._preempt                           (recomputation)
  6. MockExecutor.execute                         (trivial)
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto


# ----------------------------------------------------------------------------
# Requests / sequences
# ----------------------------------------------------------------------------

class SeqStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Request:
    request_id: str
    prompt_len: int          # number of prompt tokens
    max_tokens: int          # hard cap on generated tokens (EOS stand-in #1)
    stop_prob: float = 0.05  # per-decode-step random-stop probability (EOS stand-in #2)

    status: SeqStatus = SeqStatus.WAITING
    # Single source of truth for "how far along" a sequence is: the count of
    # tokens whose KV is materialized in the cache. Starts at 0; prefill jumps
    # it to prompt_len; each decode step adds 1. block_ids must always cover it.
    num_computed_tokens: int = 0
    block_ids: list[int] = field(default_factory=list)  # logical block idx -> physical block

    @property
    def needs_prefill(self) -> bool:
        """True while the prompt KV is not yet fully materialized.

        Naive (non-chunked) prefill is all-or-nothing, so this is just
        num_computed_tokens < prompt_len.
        """
        raise NotImplementedError

    @property
    def output_len(self) -> int:
        """Generated tokens so far (0 until prefill has completed)."""
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Step budget
# ----------------------------------------------------------------------------

@dataclass
class SchedulingBudget:
    """Caps a single step: total batched tokens and total sequences.

    Construct a fresh one at the top of every _schedule() call.
    """
    max_num_batched_tokens: int
    max_num_seqs: int
    num_batched_tokens: int = 0
    num_seqs: int = 0

    def can_add(self, num_tokens: int) -> bool:
        """Would adding a seq computing `num_tokens` this step stay within both caps?"""
        raise NotImplementedError

    def add(self, num_tokens: int) -> None:
        """Commit a seq to this step's budget."""
        raise NotImplementedError


# ----------------------------------------------------------------------------
# What a schedule decision produces
# ----------------------------------------------------------------------------

@dataclass
class ScheduledSeq:
    req: Request
    num_tokens: int   # tokens to compute this step: prefill -> remaining prompt; decode -> 1
    is_prefill: bool


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledSeq] = field(default_factory=list)
    preempted: list[str] = field(default_factory=list)  # request_ids, for logging/inspection


# ----------------------------------------------------------------------------
# KV cache block manager  (scheduler-side half of "paged attention")
# ----------------------------------------------------------------------------

class BlockManager:
    """Owns the physical block pool. Per-sequence block *tables* live on each
    Request as .block_ids; this class owns the free list and the alloc/free math.
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.free_blocks: deque[int] = deque(range(num_blocks))  # the free list

    def num_free_blocks(self) -> int:
        raise NotImplementedError

    def _blocks_for(self, num_tokens: int) -> int:
        """Physical blocks needed to hold `num_tokens` tokens = ceil(n / block_size)."""
        raise NotImplementedError

    # --- prefill path ---
    def can_allocate(self, req: Request) -> bool:
        """Prefill admission: enough free blocks for the entire prompt?"""
        raise NotImplementedError

    def allocate(self, req: Request) -> None:
        """Prefill: pop blocks off the free list into req.block_ids to cover prompt_len."""
        raise NotImplementedError

    # --- decode path ---
    def can_append_slot(self, req: Request) -> bool:
        """Decode: is there room for one more token?

        The next token needs a new block only when the current tokens exactly
        fill the allocated blocks (a boundary crossing). Otherwise it slots into
        the last block for free. Return True if no new block is needed, or if one
        is needed and at least one is free.
        """
        raise NotImplementedError

    def append_slot(self, req: Request) -> None:
        """Decode: if the next token crosses a block boundary, pop one free block
        onto req.block_ids. No-op when the current block still has room.
        Call this *before* execute, while num_computed_tokens is pre-increment.
        """
        raise NotImplementedError

    # --- teardown ---
    def free(self, req: Request) -> None:
        """Return all of req's blocks to the free list and clear its table.
        Used by both finish handling and recomputation preemption.
        """
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Stubbed executor  (stands in for model forward + attention kernels + sampling)
# ----------------------------------------------------------------------------

class MockExecutor:
    """No model, no paged-KV reads — just fabricates one token id per scheduled
    sequence so the scheduler has something to advance on. Swapping in a real
    executor later should not require the scheduler to change.
    """

    def execute(self, scheduled: list[ScheduledSeq]) -> dict[str, int]:
        """Return {request_id: next_token_id}. A random int is fine for the mock."""
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Scheduler
# ----------------------------------------------------------------------------

class Scheduler:
    def __init__(
        self,
        block_manager: BlockManager,
        executor: MockExecutor,
        max_num_batched_tokens: int,
        max_num_seqs: int,
    ):
        self.block_manager = block_manager
        self.executor = executor
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        # FCFS: append on the right, admit from the left. Oldest = leftmost.
        self.waiting: deque[Request] = deque()
        self.running: deque[Request] = deque()

    # --- public API ---
    def add_request(self, req: Request) -> None:
        """Enqueue a new request onto the waiting queue (tail)."""
        raise NotImplementedError

    def has_unfinished(self) -> bool:
        """True while anything is waiting or running (drives the outer loop)."""
        raise NotImplementedError

    def step(self) -> list[Request]:
        """One scheduler tick. Wiring is fixed; fill the three callees.

        Returns the requests that finished on this tick.
        """
        output = self._schedule()
        results = self.executor.execute(output.scheduled)
        finished = self._process_outputs(output.scheduled, results)
        return finished

    # --- decisions to implement ---
    def _schedule(self) -> SchedulerOutput:
        """Build this step's batch under a fresh SchedulingBudget(
        self.max_num_batched_tokens, self.max_num_seqs).

        FCFS ordering that mirrors basic vLLM behavior:

          1) DECODES FIRST. For each running req (oldest -> newest):
               - budget.can_add(1) ? if not, stop admitting decodes.
               - block_manager.can_append_slot(req) ?
                   yes -> append_slot, add a ScheduledSeq(req, 1, is_prefill=False),
                          budget.add(1).
                   no (out of blocks) -> PREEMPT the lowest-priority running req.
                          Under FCFS that's the *newest* still-running one
                          (tail of self.running). Call self._preempt on it,
                          record it in output.preempted, then retry the append.
                          If the victim would be req itself, preempt it and move on.

          2) PREFILLS NEXT. While waiting is non-empty (oldest -> newest):
               - peek the head; if not budget.can_add(prompt_len) -> stop.
               - if not block_manager.can_allocate(head) -> stop (blocks full).
               - pop it, block_manager.allocate, status = RUNNING, push to running,
                 add ScheduledSeq(head, prompt_len, is_prefill=True), budget.add(prompt_len).

        Return the SchedulerOutput. (Decides the batch only; token application
        happens in _process_outputs.)
        """
        raise NotImplementedError

    def _preempt(self, req: Request) -> None:
        """Recomputation preemption. Free the victim's KV, rewind it to a
        pre-prefill state, and requeue it so FCFS re-admits it first:
            - block_manager.free(req)
            - req.num_computed_tokens = 0
            - req.status = WAITING
            - remove from running; push to the FRONT (left) of waiting.
        """
        raise NotImplementedError

    def _process_outputs(
        self,
        scheduled: list[ScheduledSeq],
        results: dict[str, int],
    ) -> list[Request]:
        """Apply one step's results:
            - prefill seqs:  num_computed_tokens += num_tokens (the whole prompt)
            - decode seqs:   num_computed_tokens += 1 (the new token from results)
            - then _is_finished(req) ? -> block_manager.free, status = FINISHED,
              drop from running, collect into the returned list.
        Return the finished requests.
        """
        raise NotImplementedError

    def _is_finished(self, req: Request) -> bool:
        """Done = random stop fired this decode step OR output_len >= max_tokens.
        (Only meaningful once the request has left prefill.)
        """
        raise NotImplementedError


# Test harness
if __name__ == "__main__":
    random.seed(0)

    block_manager = BlockManager(num_blocks=64, block_size=16)
    scheduler = Scheduler(
        block_manager,
        MockExecutor(),
        max_num_batched_tokens=2048,
        max_num_seqs=8,
    )

    for i in range(12):
        scheduler.add_request(
            Request(
                request_id=f"r{i}",
                prompt_len=random.randint(8, 240),
                max_tokens=random.randint(16, 128),
            )
        )

    tick = 0
    while scheduler.has_unfinished():
        done = scheduler.step()
        tick += 1
        for r in done:
            print(
                f"[tick {tick:4d}] finished {r.request_id:>4}  "
                f"prompt={r.prompt_len:4d}  generated={r.output_len:4d}  "
                f"free_blocks={block_manager.num_free_blocks()}"
            )

    print(f"\nall requests drained in {tick} ticks")