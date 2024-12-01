from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import status

from emotion_prediction.emotion_predict import predict

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:8080/emotions"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputText(BaseModel):
    text: str

@app.post("/emotions")
def emotion_prediction(inputText: InputText):
    try:
        emotion = predict(inputText)

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "statusCode": 201,
                "message": "사용자의 감정이 정상적으로 예측되었습니다.",
                "emotion": emotion
            }
        )
    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "statusCode": 500,
                "data": {
                    "message": ['서버 상의 이유로 감정 예측을 실패하였습니다.'],
                }
            }
        )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)