from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from brain import events

#from app_main import car
# from car import car
from brain import events

api_main = FastAPI()


def custom_openapi():
    if api_main.openapi_schema:
        return api_main.openapi_schema

    openapi_schema = get_openapi(
        title="Citi Dabai",
        version="1.0",
        description="Citi Dabai API",
        routes=api_main.routes,
    )
    api_main.openapi_schema = openapi_schema
    return api_main.openapi_schema


api_main.openapi = custom_openapi


@api_main.get("/api", tags=["Test"])
def read_main():
    return {"message": "Hello World"}


@api_main.get("/car/stop", tags=["Car"])
def stop_car():

    events.speaker_queue.put("thanks")
    # car.hello()
    # time.sleep(2)
    #car.brake()
    return {"message": "Car stopped"}


@api_main.get("/car/start", tags=["Car"])
def start_car():

    return {"message": "Car stated"}


@api_main.get("/car/thanks", tags=["Car"])
def say_thanks():
    events.speaker_queue.put("thanks")
    return {"message": "Car say thanks"}


