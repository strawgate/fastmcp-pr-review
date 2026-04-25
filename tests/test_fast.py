"""Tests for the fast single-shot review mode."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from fastmcp_pr_review.fast import FastReview
from fastmcp_pr_review.models import (
    PRFile,
    PRReviewResult,
    ReviewInput,
    ReviewState,
)


def _make_inp(**overrides: object) -> ReviewInput:
    defaults = dict(
        files=[
            PRFile(
                filename="src/main.py",
                status="modified",
                additions=10,
                deletions=5,
                changes=15,
                patch="+new\n-old",
            )
        ],
        title="Test",
        description="",
        author="dev",
        head_ref="feat",
        base_ref="main",
        additions=10,
        deletions=5,
        changed_files=1,
    )
    defaults.update(overrides)  # ty: ignore[no-matching-overload]
    return ReviewInput(**defaults)  # ty: ignore[invalid-argument-type]


def _make_result() -> PRReviewResult:
    return PRReviewResult(
        verdict=ReviewState.APPROVED,
        summary="Looks good",
        risk_score=1,
        health_score=100,
    )


class TestFastReview:
    @pytest.mark.asyncio
    async def test_single_sample_call_no_tools(self) -> None:
        """Fast mode makes exactly one ctx.sample() call with no tools."""
        ctx = MagicMock()
        ctx.sample = AsyncMock(return_value=MagicMock(result=_make_result()))

        pipeline = FastReview()
        result = await pipeline.run(ctx, _make_inp())

        assert result.verdict == ReviewState.APPROVED
        ctx.sample.assert_awaited_once()
        call_kwargs = ctx.sample.call_args.kwargs
        assert call_kwargs["result_type"] is PRReviewResult
        assert "tools" not in call_kwargs

    @pytest.mark.asyncio
    async def test_includes_focus_areas(self) -> None:
        ctx = MagicMock()
        ctx.sample = AsyncMock(return_value=MagicMock(result=_make_result()))

        pipeline = FastReview()
        await pipeline.run(ctx, _make_inp(focus_areas="security"))

        messages = ctx.sample.call_args.kwargs["messages"]
        assert "security" in messages

    @pytest.mark.asyncio
    async def test_includes_diff_in_prompt(self) -> None:
        ctx = MagicMock()
        ctx.sample = AsyncMock(return_value=MagicMock(result=_make_result()))

        pipeline = FastReview()
        await pipeline.run(ctx, _make_inp())

        messages = ctx.sample.call_args.kwargs["messages"]
        assert "src/main.py" in messages

    @pytest.mark.asyncio
    async def test_handles_no_patches(self) -> None:
        """When files have no patches, prompt shows a fallback message."""
        ctx = MagicMock()
        ctx.sample = AsyncMock(return_value=MagicMock(result=_make_result()))

        inp = _make_inp(
            files=[
                PRFile(
                    filename="binary.png",
                    status="modified",
                    additions=0,
                    deletions=0,
                    changes=0,
                    patch=None,
                )
            ]
        )
        pipeline = FastReview()
        await pipeline.run(ctx, inp)

        messages = ctx.sample.call_args.kwargs["messages"]
        assert "(no patches available)" in messages

    def test_build_prompt_returns_string(self) -> None:
        """build_prompt() should return a string with key fields."""
        pipeline = FastReview()
        prompt = pipeline.build_prompt(_make_inp())
        assert "Test" in prompt
        assert "src/main.py" in prompt

    def test_subclass_overrides_system_prompt(self) -> None:
        """Subclassing should allow overriding SYSTEM_PROMPT."""

        class SecurityReview(FastReview):
            SYSTEM_PROMPT = "You are a security auditor."

        pipeline = SecurityReview()
        assert pipeline.SYSTEM_PROMPT == "You are a security auditor."
