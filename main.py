import io
import json
import uuid
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from pydantic import BaseModel, ValidationError
from starlette.responses import StreamingResponse
from redis import asyncio as aioredis

from services.ai_report_agent import run_ai_report_generation
from services.pdf_generator import create_pdf_report

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

class ModelSelection(BaseModel):
    model: str


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

@app.post('/analysis-sessions/{session_id}/model')
async def select_model_for_session(session_id: str, selection: ModelSelection):
    input_key = f'analysis_inputs:{session_id}'

    input_data_json = await redis_client.get(input_key)
    if not input_data_json:
        raise HTTPException(status_code=404, detail='Session id not found')

    input_data = json.loads(input_data_json)
    input_data['model'] = selection.model

    await redis_client.set(input_key, json.dumps(input_data), ex=3600)

    return {
        'message': f'Model {selection.model} has been set for session {session_id}',
        'session_id': session_id,
        'selected_model': selection.model
    }


@app.post('/reports/{session_id}', status_code=202)
async def generate_report(session_id: str, background_tasks: BackgroundTasks):
    input_key = f'analysis_inputs:{session_id}'

    input_data_json = await redis_client.get(input_key)
    if not input_data_json:
        raise HTTPException(status_code=404, detail='Session id not found')

    input_data = json.loads(input_data_json)
    selected_model = input_data.get('model')
    if not selected_model:
        raise HTTPException(status_code=400, detail='Model has not been selected for this session. Please complete Step 2.')

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

    # background_tasks.add_task(run_ai_report_generation, session_id)
    if selected_model == 'gpt-4o-mini':
        pass
    elif selected_model == 'gemini-2.5-flash':
        pass
    elif selected_model == 'claude-sonnet-4-20250514':
        pass
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {selected_model}")

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

@app.get('/reports/{session_id}/download')
async def download_report(session_id: str):
    report_key = f'report:{session_id}'
    report_json = await redis_client.get(report_key)

    if not report_json:
        raise HTTPException(status_code=404, detail='Report not found. Please start the generation first')

    report_data = json.loads(report_json).get('result')
    pdf_bytes = create_pdf_report(report_data, session_id)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type='application/pdf',
        headers={
            'Content-Disposition': f'attachment; filename=slow_query_report_{session_id}.pdf'
        },
    )
