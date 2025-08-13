import json
import uuid
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from pydantic import BaseModel, ValidationError
from redis import asyncio as aioredis

from services.ai_report_agent import run_ai_report_generation

redis_client = aioredis.from_url('redis://localhost:6379/0', decode_responses=True)

app = FastAPI()


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


@app.post('/analysis-sessions/upload')
async def create_analysis_session_from_file(file: UploadFile = File(...)):
    if file.content_type != 'application/json':
        raise HTTPException(status_code=400, detail='Please upload a JSON file')

    contents = await file.read()
    try:
        data = json.loads(contents)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail='Invalid JSON format')

    try:
        inputs = AnalysisInput(**data)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    session_id = str(uuid.uuid4())
    await redis_client.set(f'analysis_inputs:{session_id}', inputs.model_dump_json(), ex=3600)

    return {
        'message': 'File uploaded and data stored successfully',
        'session_id': session_id,
    }


@app.post('/analysis-sessions')
async def create_analysis_session(inputs: AnalysisInput):
    session_id = str(uuid.uuid4())
    await redis_client.set(f'analysis_inputs:{session_id}', inputs.model_dump_json(), ex=3600)

    return {
        'message': 'Data stored successfully',
        'session_id': session_id,
    }


@app.post('/reports/{session_id}', status_code=202)
async def generate_report(session_id: str, background_tasks: BackgroundTasks):
    input_key = f'analysis_inputs:{session_id}'
    if not await redis_client.exists(input_key):
        raise HTTPException(status_code=404, detail='Session id not found')

    report_key = f'report:{session_id}'
    report_json = await redis_client.get(report_key)
    if report_json:
        report_data = json.loads(report_json)
        status = report_data.get('status')
        if status in ['processing', 'complete']:
            return {
                'message': f'Report status: {status}',
                'session_id': session_id,
                'status': status,
            }

    initial_report_status = {
        'status': 'processing',
        'result': None,
    }
    await redis_client.set(report_key, json.dumps(initial_report_status), ex=3600)

    background_tasks.add_task(run_ai_report_generation, session_id)

    return {
        'message': 'AI report generation has started. Please check the results shortly',
        'session_id': session_id,
        'status': 'processing',
    }


@app.get('/reports/{session_id}')
async def get_report(session_id: str):
    report_key = f'report:{session_id}'
    report_json = await redis_client.get(report_key)
    if not report_json:
        raise HTTPException(status_code=404, detail='Please start the generation first')

    return json.loads(report_json)
