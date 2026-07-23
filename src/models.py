from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """
    A representation of a minimal source object.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """
    A representation of an unanswered question object.
    """

    question_id: str = Field(default_factory=lambda:
                             str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
    A representation of an answered question object.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    A representation of a RAG dataset object.
    """

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """
    A representation of a minimal search result object.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(BaseModel):
    """
    A representation of a minimal answer object.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """
    A representation of a student search result object.
    """

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsandAnswer(BaseModel):
    """
    A representation of a student search result and answer object.
    """

    search_results: List[MinimalSearchResults]
    k: int
