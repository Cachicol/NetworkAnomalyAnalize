import http.server
import socketserver
import urllib.request
import json
import os

PORT = 3000

class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(__file__), **kwargs)
    
    def do_GET(self):
        # Se for requisição da API, redirecionar para backend
        if self.path == '/predict_latest' or self.path.startswith('/api/'):
            try:
                backend_url = f"http://localhost:8000{self.path}"
                print(f"🔁 Proxy: {self.path} -> {backend_url}")
                
                with urllib.request.urlopen(backend_url) as response:
                    # Copiar resposta do backend
                    self.send_response(response.status)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(response.read())
                    
            except Exception as e:
                print(f"❌ Erro no proxy: {e}")
                self.send_error(500, f"Erro no proxy: {str(e)}")
        else:
            # Servir arquivos estáticos (HTML, CSS, JS)
            # Para SPA, redirecionar tudo para index.html
            if self.path == '/':
                self.path = '/index.html'
            elif '.' not in self.path.split('/')[-1]:
                self.path = '/index.html'
                
            super().do_GET()

print(f"🚀 Frontend com Proxy rodando em http://localhost:{PORT}")
print("📁 Pressione Ctrl+C para parar")

with socketserver.TCPServer(("", PORT), ProxyHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Servidor parado")