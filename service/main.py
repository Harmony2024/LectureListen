import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import soundfile as sf
from espnet2.bin.asr_inference import Speech2Text
from starlette.responses import HTMLResponse
import asyncio


app = FastAPI()
load_dotenv('env/.env')
asr_config_path = "asr_config.yml"
asr_model_path = "asr.pth"
client = AsyncOpenAI(api_key=os.getenv('OPEN_API_KEY'))


def audio_to_text(audio_file_path):
    speech2text = Speech2Text(asr_config_path, asr_model_path)
    audio, rate = sf.read(audio_file_path)
    nbests = speech2text(audio)
    text = nbests[0].text
    return text


@app.get('/')
async def home():
    return HTMLResponse(content=open('index.html', 'r').read())


@app.post("/gpt-voice")
async def test_gpt(audio: UploadFile = File(...)):
    # 임시 파일 저장
    temp_file = f"temp_{audio.filename}"
    with open(temp_file, "wb+") as file_object:
        file_object.write(audio.file.read())

    # 오디오를 텍스트로 변환
    message = audio_to_text(temp_file)

    async def generator():
        yield f"{message}\n"
        await asyncio.sleep(0.01)
        stream = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system",
                       "content": f"너는 학생의 입장에서 컴퓨터공학 질문을 답변하는 assistant야."},
                      {"role": "user", "content": message}],
            stream=True,
        )
        async for chunk in stream:
            yield chunk.choices[0].delta.content or ""

    return StreamingResponse(generator(), media_type="text/event-stream")
