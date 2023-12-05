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

version = "NOVER"


def make_handler(ml_type, save_data, positive_threshold, feat_start, feat_end, recurrence):
    class S(BaseHTTPRequestHandler):
        globalMlMain = MlMain()

        def setup(self):
            BaseHTTPRequestHandler.setup(self)
            self.request.settimeout(0.2)

        def _set_headers(self):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

        def do_GET(self):
            self.wfile.write(ml_type.encode("utf-8"))

        def do_HEAD(self):
            self._set_headers()

        def do_PUT(self):
            self.globalMlMain.setCheckType(self.path)
            self.wfile.write("Successful PUT".encode("utf-8"))

        def do_POST(self):
            """
            Handle POST requests.
            """
            # print('The Request: %s' % (self.path))
            # requestStr = urllib2.unquote((self.path));
            # requestStr = unquote(self.path)
            pred = self.globalMlMain.mlMain(
                self.path, ml_type, save_data, positive_threshold, feat_start, feat_end, recurrence
            )

            # the response
            self.wfile.write(pred.encode("utf-8"))

    return S


def run(
    ml_type,
    save_data,
    positive_threshold,
    feat_start,
    feat_end,
    recurrence=-1,
    server_class=HTTPServer,
    port=9997,
):
    server_address = ("", port)
    positive_threshold = float(positive_threshold)
    if int(save_data) == 0:
        save_data = False
    elif int(save_data) == 1:
        save_data = True
    else:
        raise ValueError("Save Data Parameter needs to be 0 or 1")
    if positive_threshold > 1:
        positive_threshold = positive_threshold / 100
    if recurrence == -1:
        httpd = server_class(
            server_address,
            make_handler(
                ml_type=ml_type,
                save_data=save_data,
                positive_threshold=positive_threshold,
                feat_start=feat_start,
                feat_end=feat_end,
            ),
        )
    else:
        httpd = server_class(
            server_address,
            make_handler(
                ml_type=ml_type,
                save_data=save_data,
                positive_threshold=positive_threshold,
                feat_start=feat_start,
                feat_end=feat_end,
                recurrence=recurrence
            ),
        )
    print("Starting MLServer...")
    print("Listening on port " + str(port))
    httpd.serve_forever()


if __name__ == "__main__":
    from sys import argv

    print(argv)
    if len(argv) == 8 or len(argv) == 7:
        if "LSTM" in argv[2]:
           run(
                ml_type=argv[2],
                save_data=argv[3],
                positive_threshold=argv[4],
                feat_start=int(argv[5]),
                feat_end=int(argv[6]),
                recurrence=int(argv[7]),
                port=int(argv[1]),
            ) 
        else:
            run(
                ml_type=argv[2],
                save_data=argv[3],
                positive_threshold=argv[4],
                feat_start=int(argv[5]),
                feat_end=int(argv[6]),
                port=int(argv[1]),
            )
    else:
        print("not enough argv")
