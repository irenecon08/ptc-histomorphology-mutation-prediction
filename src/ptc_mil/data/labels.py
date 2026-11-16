"""Prediction tasks and their label encodings, defined once for the whole project."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Task:
    """A prediction task.

    Attributes:
        name: Identifier used in configs and output file names.
        label_column: Column in the splits table that holds the raw label.
        label_map: Maps each raw label value to its integer class.
        class_names: Human-readable class names, in integer-class order.
    """

    name: str
    label_column: str
    label_map: dict[str | int, int]
    class_names: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def encode(self, labels: pd.Series) -> pd.Series:
        """Convert raw labels to integer classes.

        Raises:
            ValueError: If any label is not in the task's label map.
        """
        unknown = set(labels) - set(self.label_map)
        if unknown:
            raise ValueError(f"Task '{self.name}' got unknown labels: {sorted(map(str, unknown))}")
        return labels.map(self.label_map).astype(int)


MULTICLASS = Task(
    name="multiclass",
    label_column="mutation_label",
    label_map={"BRAF_V600E": 0, "RAS": 1, "Other": 2},
    class_names=("BRAF_V600E", "RAS", "Other"),
)

RET_BINARY = Task(
    name="ret_binary",
    label_column="RET",
    label_map={0: 0, 1: 1},
    class_names=("RET_negative", "RET_positive"),
)

TASKS = {task.name: task for task in (MULTICLASS, RET_BINARY)}


def get_task(name: str) -> Task:
    """Look up a task by name, e.g. get_task("multiclass")."""
    if name not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Choose from: {sorted(TASKS)}")
    return TASKS[name]
