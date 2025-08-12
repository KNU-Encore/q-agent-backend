import uuid
from fastapi import FastAPI, BackgroundTasks, HTTPException
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
        'session_id': session_id,
    }


@app.post('/reports/{session_id}', status_code=202)
def generate_report(session_id: str, background_tasks: BackgroundTasks):
    if session_id not in db['analysis_inputs']:
        raise HTTPException(status_code=404, detail='Session id not found')

    if session_id in db['reports']:
        status = db['reports'][session_id]['status']
        if status == 'processing':
            return {
                'message': 'Report processing',
                'session_id': session_id,
                'status': status,
            }
        elif status == 'complete':
            return {
                'message': 'Report completed',
                'session_id': session_id,
                'status': status,
            }

    db['reports'][session_id] = {
        'status': 'processing',
        'result': None,
    }

    input_data = db['analysis_inputs'][session_id]

    # AI 리포트 생성 코드 호출
    # result = background_tasks.add_task(run_ai_report_geenration, session_id, input_data)

    return {
        'message': 'AI report generation has started. Please check the results shortly',
        'session_id': session_id,
        'status': db['reports'][session_id]['status'],
    }


@app.get('/reports/{session_id}')
def get_report(session_id: str):
    report = db['reports'].get(session_id)
    if not report:
        raise HTTPException(status_code=404, detail='Please start the generation first')

    return report
