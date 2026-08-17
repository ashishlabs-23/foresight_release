"""
scripts.serve_frontend
~~~~~~~~~~~~~~~~~~~~~~
Serves the static ForeSight AI frontend on port 3000.
"""
import http.server
import os
import socketserver
import sys

PORT = 3000
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

def main():
    print(f"Starting ForeSight AI Frontend server on http://127.0.0.1:{PORT}")
    try:
        with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down frontend server.")

if __name__ == "__main__":
    sys.exit(main())
