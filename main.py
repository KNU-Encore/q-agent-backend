import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

db = {
    'analysis_inputs': {},
    'reports': {},
}


class QueryMetadata(BaseModel):
    timestamp: str
    query: str
    qt: float
    lt: float
    rsent: int
    rexp: int

class ExplainAnalyze(BaseModel):
    plan_tree: str | dict | None

class DBMetadata(BaseModel):
    db_info: str | dict | None

class DBSchema(BaseModel):
    schema_info: str | dict | None

class AnalysisInput(BaseModel):
    query_metadata: QueryMetadata
    explain_analyze: ExplainAnalyze
    db_metadata: DBMetadata
    db_schema: DBSchema


@app.get('/')
def root():
    return {'message': 'Connected'}


@app.post('/analysis-sessions')
def create_analysis_session(inputs: AnalysisInput):
    session_id = str(uuid.uuid4())
    db['analysis_inputs'][session_id] = inputs.model_dump()

    return {
        'message': 'Data stored successfully',
        'session_id': session_id
    }
