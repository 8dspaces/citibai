from flask import Response
from flask import Flask
from flask import render_template

#from web.stream import StreamServer
from web.streamLite import StreamServer

stream = StreamServer()
#stream.start()

web_main = Flask(__name__)


@web_main.route("/")
def index():
	# return the rendered template
	return render_template("index.html")


@web_main.route("/live_feed_people")
def live_feed_people():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(stream.generate_people(), mimetype ="multipart/x-mixed-replace; boundary=frame")

# @web_main.route("/live_feed_hand")
# def live_feed_hand():
# 	# return the response generated along with the specific media
# 	# type (mime type)
# 	return Response(stream.generate_hand(), mimetype ="multipart/x-mixed-replace; boundary=frame")
#

# @web_main.route("/live_feed_raw")
# def live_feed_raw():
# 	# return the response generated along with the specific media
# 	# type (mime type)
# 	return Response(stream.generate_raw(),mimetype = "multipart/x-mixed-replace; boundary=frame")