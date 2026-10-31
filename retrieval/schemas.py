from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .llm import ModelError, parse_json_response


class StrictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class QueryRewriteOutput(StrictOutput):
    standalone_query: str = Field(min_length=1)
    rewritten_query: str = Field(min_length=1)


class DocumentGrade(StrictOutput):
    chunk_id: str = Field(min_length=1)
    relevant: bool
    reason: str


class DocumentGradingOutput(StrictOutput):
    grades: list[DocumentGrade] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_chunk_ids(self):
        ids = [grade.chunk_id for grade in self.grades]
        if len(ids) != len(set(ids)):
            raise ValueError("grades contains duplicate chunk IDs")
        return self


class EvaluationDimension(StrictOutput):
    score: int = Field(ge=0, le=100)
    reason: str


class AnswerEvaluationOutput(StrictOutput):
    faithfulness: EvaluationDimension
    relevance: EvaluationDimension
    correctness: EvaluationDimension
    unsupported_claims: list[str]
    feedback: str


def parse_structured_output(text: str, schema: type[StrictOutput]) -> StrictOutput:
    try:
        return schema.model_validate(parse_json_response(text))
    except ValidationError as exc:
        raise ModelError(f"Model response failed {schema.__name__} validation") from exc
