"""Tests de l'isolation des erreurs par élément (``pipeline.batch``).

Vérifie le contrat de :func:`process_items_isolated` qui sous-tend les
correctifs de robustesse ROB-02/03/04 : une exception sur un élément
n'interrompt pas le traitement des autres, sauf annulation.
"""
from __future__ import annotations

import pytest

from pipeline.batch import process_items_isolated
from pipeline.cancellation import PipelineCancelled


class TestProcessItemsIsolated:
    def test_all_items_processed_in_order_when_no_error(self):
        seen = []
        succeeded, failures = process_items_isolated(
            ["a", "b", "c"], lambda i, item: seen.append((i, item))
        )
        assert seen == [(1, "a"), (2, "b"), (3, "c")]  # index 1-based
        assert succeeded == 3
        assert failures == []

    def test_failing_item_is_isolated_others_continue(self):
        seen = []

        def process(index, item):
            if item == "b":
                raise ValueError("boom")
            seen.append(item)

        succeeded, failures = process_items_isolated(["a", "b", "c"], process)

        assert seen == ["a", "c"]  # "b" a échoué mais "c" est quand même traité
        assert succeeded == 2
        assert [idx for idx, _ in failures] == [2]
        assert [item for _, item in failures] == ["b"]

    def test_on_failure_callback_receives_index_item_and_exception(self):
        captured = []

        def process(index, item):
            raise RuntimeError(f"fail-{item}")

        def on_failure(index, item, exc):
            captured.append((index, item, str(exc)))

        succeeded, failures = process_items_isolated(
            ["x", "y"], process, on_failure=on_failure
        )

        assert succeeded == 0
        assert captured == [(1, "x", "fail-x"), (2, "y", "fail-y")]

    def test_pipeline_cancelled_propagates(self):
        def process(index, item):
            raise PipelineCancelled()

        with pytest.raises(PipelineCancelled):
            process_items_isolated(["a"], process)

    def test_cancel_check_stops_before_next_item(self):
        seen = []
        # Annule après le premier élément traité.
        state = {"cancel": False}

        def process(index, item):
            seen.append(item)
            state["cancel"] = True

        succeeded, failures = process_items_isolated(
            ["a", "b", "c"], process, cancel=lambda: state["cancel"]
        )

        assert seen == ["a"]  # "b" et "c" non traités (annulation)
        assert succeeded == 1
        assert failures == []

    def test_cancel_check_true_from_start_processes_nothing(self):
        seen = []
        succeeded, failures = process_items_isolated(
            ["a", "b"], lambda i, item: seen.append(item), cancel=lambda: True
        )
        assert seen == []
        assert succeeded == 0
        assert failures == []

    def test_works_without_on_failure_callback(self):
        # Une erreur sans on_failure ne doit pas planter, juste être isolée.
        def process(index, item):
            if item == 1:
                raise KeyError("nope")

        succeeded, failures = process_items_isolated([1, 2], process)
        assert succeeded == 1
        assert [item for _, item in failures] == [1]

    def test_empty_iterable(self):
        succeeded, failures = process_items_isolated([], lambda i, item: None)
        assert succeeded == 0
        assert failures == []

    def test_keyboard_interrupt_is_not_swallowed(self):
        def process(index, item):
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            process_items_isolated(["a"], process)
