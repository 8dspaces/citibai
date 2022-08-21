import uvicorn
from fastapi.middleware.wsgi import WSGIMiddleware

from api.app import api_main
from web.app import web_main


api_main.mount("/", WSGIMiddleware(web_main))


def start_app(ip="127.0.0.1", port=9017):

	uvicorn.run(api_main, host=ip, port=port, log_level="info", debug=True)


if __name__ == '__main__':
	start_app()
	#start_web


