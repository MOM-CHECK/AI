from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import status

from text_processing.input_text_processing import text_processing
from emotion_prediction.emotion_predict import predict_emotion

app = FastAPI()

class InputText(BaseModel):
    text: str

# @app.post("/text-processing")
# async def text_processing(inputText: InputText):
#     try:
#         final_correct_text = text_processing(inputText.text)
#
#         return JSONResponse(
#             status_code=status.HTTP_201_CREATED,
#             content={
#                 "statusCode": 201,
#                 "message": "데이터 전처리 되었습니다.",
#                 "correct_text": final_correct_text
#             }
#         )
#     except Exception as e:
#         print(e)
#         return JSONResponse(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             content={
#                 "statusCode": 500,
#                 "data": {
#                     "message": ['텍스트 전처리를 실패했습니다.'],
#                 }
#             }
#         )

@app.post("/emotion")
def emotion_prediction(inputText: InputText):
    try:
        correct_text = text_processing(inputText.text)
        emotion_result = predict_emotion([correct_text])

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "statusCode": 201,
                "message": "사용자의 감정이 정상적으로 예측되었습니다.",
                "emotion": emotion_result
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