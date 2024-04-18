"""
_summary_
"""

from typing import List
from typing import Dict

from pydantic import (
    BaseModel,
    RootModel,
    # FilePath,
    model_validator,
    field_validator,
    ValidationError,
    ConfigDict,
)


class Bout(BaseModel):
    """__summary__"""

    start: int
    stop: int
    behaviour: str
    actual: int
    user_defined: Dict[str, int]

    # # TODO: check that all extra keys are ints
    # @model_validator(mode="after")
    # @classmethod
    # def check_extra_keys(cls, values):
    #     for k, v in values:
    #         if not isinstance(v, int):
    #             raise ValueError(f"'{k}' must be an integer - it is {v} ({type(v)}).")
    #     return values


class Bouts(BaseModel):
    """__summary__"""

    start: int
    stop: int
    bouts: List[Bout]


if __name__ == "__main__":
    fp = (
        "/Users/timothylee/Desktop/Work/dev/ba_viewer/tests/resources/updated_abcd.json"
    )

    with open(fp, "r", encoding="utf-8") as f:
        model = Bouts.model_validate_json(f.read())
        print(model)
