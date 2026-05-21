from pydantic import BaseModel, model_validator

# This is for defining the TOC strcture that will be used in the API request body when adding a document structure. It is not a database model, but a Pydantic model for request validation.
class StructureIn(BaseModel):
    section_title: str
    start_page: int
    end_page: int
    level: int = 1

    @model_validator(mode="after")
    def validate_page_range(self):
        if self.start_page < 1:
            raise ValueError("start_page must be >= 1")
        if self.end_page < self.start_page:
            raise ValueError("end_page must be >= start_page")
        return self