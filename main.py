from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from urllib.parse import quote_plus

app = FastAPI()

db_urls = {}


class MySQLConnectionInfo(BaseModel):
    username: str
    password: str
    host: str
    port: int
    table: str


@app.get("/")
def root():
    return {"message": "connected"}


@app.post("/db-info/{user_id}")
async def connect_mysql(user_id: int, info: MySQLConnectionInfo):
    try:
        username = quote_plus(info.username)
        password = quote_plus(info.password)
        host = info.host
        port = info.port
        table = info.table

        mysql_url = f"mysql+pymysql://{username}:{password}@{host}:{port}/{table}"

        db_urls[user_id] = mysql_url

        return {"message": f"{user_id}의 DB 정보가 저장되었습니다: {mysql_url}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/db-info/{user_id}")
async def get_db_url(user_id: int):
    if user_id not in db_urls:
        raise HTTPException(status_code=404, detail="해당 사용자의 DB 정보가 없습니다.")
    return {"db_url": db_urls[user_id]}
