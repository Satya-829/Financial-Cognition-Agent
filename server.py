"""
server.py — Python proxy server for FA Copilot
Serves static files and proxies AI API calls with the key from .env.
No external dependencies required — uses only Python stdlib.
"""

import http.server
import http.client
import json
import os
import ssl
import sys
import urllib.parse
from pathlib import Path

# ─── Configuration ──────────────────────────────────────────────
PORT = 3000
ROOT_DIR = Path(__file__).parent

# Provider configs (mirrors gemini.js PROVIDERS)
PROVIDERS = {
    'groq': {
        'host': 'api.groq.com',
        'path': '/openai/v1/chat/completions',
        'format': 'openai',
    },
    'openrouter': {
        'host': 'openrouter.ai',
        'path': '/api/v1/chat/completions',
        'format': 'openai',
    },
    'together': {
        'host': 'api.together.xyz',
        'path': '/v1/chat/completions',
        'format': 'openai',
    },
    'gemini': {
        'host': 'generativelanguage.googleapis.com',
        'path_template': '/v1beta/models/{model}:{action}',
        'format': 'gemini',
    },
}


def load_env():
    """Read .env file and return dict of key=value pairs."""
    env = {}
    env_path = ROOT_DIR / '.env'
    if not env_path.exists():
        print("ERROR: .env file not found! Create one with PROVIDER, MODEL, and API_KEY.")
        sys.exit(1)
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, val = line.split('=', 1)
                env[key.strip()] = val.strip()
    return env


ENV = load_env()
PROVIDER = ENV.get('PROVIDER', 'groq')
MODEL = ENV.get('MODEL', 'llama-3.3-70b-versatile')
API_KEY = ENV.get('API_KEY', '')

if not API_KEY or API_KEY == 'YOUR_GROQ_API_KEY_HERE':
    print("=" * 60)
    print("  WARNING: Set your API key in the .env file!")
    print("  Open .env and replace YOUR_GROQ_API_KEY_HERE")
    print("=" * 60)

print(f"Provider: {PROVIDER} | Model: {MODEL}")
print(f"API Key:  {'***' + API_KEY[-4:] if len(API_KEY) > 4 else '(not set)'}")


# ─── SSL context for HTTPS connections ──────────────────────────
ssl_ctx = ssl.create_default_context()


def proxy_openai_stream(body_dict, provider_cfg):
    """Forward a streaming chat completion to an OpenAI-compatible API."""
    body_dict['stream'] = True
    body_dict['model'] = MODEL

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}',
    }
    if PROVIDER == 'openrouter':
        headers['HTTP-Referer'] = 'http://localhost:3000'
        headers['X-Title'] = 'FA Copilot'

    payload = json.dumps(body_dict).encode('utf-8')

    conn = http.client.HTTPSConnection(provider_cfg['host'], context=ssl_ctx, timeout=120)
    conn.request('POST', provider_cfg['path'], body=payload, headers=headers)
    return conn.getresponse(), conn


def proxy_gemini_stream(body_dict):
    """Forward a streaming request to the Gemini API."""
    cfg = PROVIDERS['gemini']
    path = cfg['path_template'].format(model=MODEL, action='streamGenerateContent')
    path += f'?alt=sse&key={API_KEY}'

    payload = json.dumps(body_dict).encode('utf-8')
    headers = {'Content-Type': 'application/json'}

    conn = http.client.HTTPSConnection(cfg['host'], context=ssl_ctx, timeout=120)
    conn.request('POST', path, body=payload, headers=headers)
    return conn.getresponse(), conn


class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    """Combines static file serving with API proxy endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def do_GET(self):
        """Handle GET — serve static files or /api/config."""
        if self.path == '/api/config':
            self._send_json({
                'provider': PROVIDER,
                'model': MODEL,
                'connected': bool(API_KEY and API_KEY != 'YOUR_GROQ_API_KEY_HERE'),
            })
            return
        # Serve static files
        super().do_GET()

    def do_POST(self):
        """Handle POST — proxy /api/chat to AI provider."""
        if self.path == '/api/chat':
            self._handle_chat()
            return
        self.send_error(404)

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _handle_chat(self):
        """Read the client request, forward to AI provider, stream response back."""
        try:
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length)
            body = json.loads(raw) if raw else {}
        except Exception as e:
            self._send_json({'error': str(e)}, 400)
            return

        if not API_KEY or API_KEY == 'YOUR_GROQ_API_KEY_HERE':
            self._send_json({'error': 'API key not configured in .env file'}, 500)
            return

        provider_cfg = PROVIDERS.get(PROVIDER)
        if not provider_cfg:
            self._send_json({'error': f'Unknown provider: {PROVIDER}'}, 400)
            return

        try:
            if provider_cfg.get('format') == 'gemini':
                resp, conn = proxy_gemini_stream(body)
            else:
                resp, conn = proxy_openai_stream(body, provider_cfg)

            if resp.status != 200:
                error_body = resp.read().decode('utf-8', errors='replace')
                conn.close()
                self._send_json({'error': f'AI provider returned {resp.status}: {error_body}'}, 502)
                return

            # Stream the SSE response back to the client
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()

            conn.close()

        except Exception as e:
            try:
                self._send_json({'error': f'Proxy error: {str(e)}'}, 502)
            except Exception:
                pass  # Client may have disconnected

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        """Log all requests for debugging."""
        msg = format % args
        print(f"[{self.log_date_time_string()}] {msg}")


def main():
    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer(('', PORT), ProxyHandler)
    print(f"\n{'='*50}")
    print(f"  FA Copilot Server running on http://localhost:{PORT}")
    print(f"{'='*50}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down…')
        server.shutdown()


if __name__ == '__main__':
    main()
