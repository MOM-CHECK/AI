from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import status

from text_processing.input_text_processing import *

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/text-processing")
async def text_processing(inputText: InputText):
    try:
        correct_spell = await orthography_examination(inputText.text)
        correct_spacing = word_spacing_correction(correct_spell)
        final_correct_text = morpheme_analyzer(correct_spacing)

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "statusCode": 201,
                "message": "데이터 전처리 되었습니다.",
                "correct_text": final_correct_text
            }
        )
    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "statusCode": 500,
                "data": {
                    "message": ['텍스트 전처리를 실패했습니다.'],
                }
            }
        )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)