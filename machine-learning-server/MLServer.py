
"""
/*******************************************************************************
 * @author  Joseph Kamel
 * @email   josephekamel@gmail.com
 * @date    28/11/2018
 * @version 2.0
 *
 * SCA (Secure Cooperative Autonomous systems)
 * Copyright (c) 2013, 2018 Institut de Recherche Technologique SystemX
 * All rights reserved.
 *******************************************************************************/
"""

try:
    from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
except ImportError:
    from http.server import HTTPServer, BaseHTTPRequestHandler
from MLMain import MlMain

version = 'NOVER'


def make_handler(ml_type, save_data):
	class S(BaseHTTPRequestHandler):
		globalMlMain = MlMain()

		def setup(self):
			BaseHTTPRequestHandler.setup(self)
			self.request.settimeout(0.2)

		def _set_headers(self):
			self.send_response(200)
			self.send_header('Content-type', 'text/html')
			self.end_headers()

		def do_GET(self):
			self.wfile.write(ml_type.encode('utf-8'))

		def do_HEAD(self):
			self._set_headers()

		def do_PUT(self):
			self.globalMlMain.setCheckType(self.path)
			self.wfile.write("Successful PUT".encode('utf-8'))

		def do_POST(self):
			'''
			Handle POST requests.
			'''
			#print('The Request: %s' % (self.path))
			#requestStr = urllib2.unquote((self.path));
			#requestStr = unquote(self.path)
			pred = self.globalMlMain.mlMain(self.path, ml_type, save_data)

			# the response
			self.wfile.write(pred.encode('utf-8'))
	return S


def run(ml_type, save_data, server_class=HTTPServer, port=9997):
	server_address = ('', port)
	if (int(save_data) == 0):
		save_data = False
	elif (int(save_data) == 1):
		save_data = True
	else:
		raise ValueError("Save Data Parameter needs to be 0 or 1")
	httpd = server_class(server_address, make_handler(ml_type=ml_type, save_data=save_data))
	print('Starting MLServer...')
	print('Listening on port ' + str(port))
	httpd.serve_forever()



if __name__ == "__main__":
	from sys import argv
	if len(argv) == 4 or len(argv) == 3:
		run(ml_type=argv[2], save_data=argv[3], port=int(argv[1]))
	else:
		print('not enough argv')