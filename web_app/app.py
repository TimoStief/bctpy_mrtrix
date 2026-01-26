"""
Flask Web Application for BCT Analysis
Provides web interface for running brain connectivity analysis
"""

import os
import sys
import json
import webbrowser
import threading
import socket
import platform
import subprocess
from queue import Queue

from flask import Flask, request, jsonify, render_template, send_file
from waitress import serve

from bct_analyzer import BCTAnalyzer

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['JSON_SORT_KEYS'] = False

# Global state
analysis_queue = Queue()
current_analyzer = None
output_log = []


class OutputCapture:
    """Capture output from analyzer"""
    def __init__(self, max_lines=1000):
        self.lines = []
        self.max_lines = max_lines
    
    def write(self, message):
        if message.strip():
            self.lines.append(message)
            if len(self.lines) > self.max_lines:
                self.lines.pop(0)
    
    def get_all(self):
        return "\n".join(self.lines)
    
    def clear(self):
        self.lines = []


output_capture = OutputCapture()


# Routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Start analysis with selected folders"""
    global current_analyzer
    
    try:
        data = request.get_json()
        input_dir = data.get('input_dir', '')
        output_dir = data.get('output_dir', '')
        analysis_type = data.get('analysis_type', 'full')
        selected_metrics = data.get('selected_metrics', [])
        
        if not input_dir or not os.path.isdir(input_dir):
            return jsonify({'success': False, 'error': 'Invalid input directory'}), 400
        
        if not output_dir:
            output_dir = os.path.join(input_dir, 'bct_output')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create analyzer with output capture
        output_capture.clear()
        current_analyzer = BCTAnalyzer(output_callback=output_capture.write, selected_metrics=selected_metrics)
        
        output_capture.write(f"🚀 Starting analysis (type: {analysis_type})\n")
        if selected_metrics:
            output_capture.write(f"📊 Selected metrics: {', '.join(selected_metrics)}\n")
        
        if analysis_type == 'full' or analysis_type == 'analyze':
            df_results, summary = current_analyzer.analyze_matrices(input_dir, output_dir)
            return jsonify({
                'success': True,
                'output_dir': output_dir,
                'summary': summary,
                'results_count': len(df_results)
            })
        else:
            return jsonify({'success': False, 'error': 'Unknown analysis type'}), 400
    
    except Exception as e:
        output_capture.write(f"❌ Error: {str(e)}\n")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/logs')
def get_logs():
    """Get current analysis logs"""
    return jsonify({
        'logs': output_capture.get_all(),
        'lines': len(output_capture.lines)
    })


@app.route('/api/validate-path', methods=['POST'])
def validate_path():
    """Validate and get info about a folder path"""
    try:
        data = request.get_json()
        path = data.get('path', '').strip()
        
        if not path:
            return jsonify({
                'success': False,
                'error': 'Please enter a folder path'
            })
        
        # Expand home directory if needed
        path = os.path.expanduser(path)
        
        if not os.path.isdir(path):
            return jsonify({
                'success': False,
                'error': f'Folder not found: {path}'
            })
        
        return jsonify({
            'success': True,
            'path': path,
            'exists': True
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/pick-folder', methods=['POST'])
def pick_folder():
    """Open a native folder picker when supported (macOS best effort)"""
    try:
        data = request.get_json() or {}
        folder_type = data.get('type', 'input')

        system = platform.system().lower()
        if system == 'darwin':
            prompt = f"Select {folder_type.title()} Folder"
            script = f'''osascript -e 'set chosenFolder to POSIX path of (choose folder with prompt "{prompt}")' '''
            result = subprocess.run(script, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                path = result.stdout.strip()
                if path:
                    return jsonify({'success': True, 'path': path})
            # fallthrough if user cancels
            return jsonify({'success': False, 'message': 'No folder selected'})
        elif system == 'windows':
            return jsonify({
                'success': False,
                'message': 'Native picker not supported here. Please paste the folder path and click Validate.'
            })
        else:  # linux/others
            return jsonify({
                'success': False,
                'message': 'Native picker not available. Please paste the folder path and click Validate.'
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/check-sessions', methods=['POST'])
def check_sessions():
    """Check if directory has expected session structure"""
    try:
        path = request.get_json().get('path', '')
        
        if not os.path.isdir(path):
            return jsonify({'success': False, 'has_sessions': False})
        
        sessions = ['ses-1', 'ses-2', 'ses-3', 'ses-4']
        found = []
        
        for session in sessions:
            session_path = os.path.join(path, session)
            if os.path.isdir(session_path):
                npy_count = len([f for f in os.listdir(session_path) if f.endswith('.npy')])
                found.append({'name': session, 'files': npy_count})
        
        return jsonify({
            'success': True,
            'has_sessions': len(found) > 0,
            'sessions': found
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/download/<filename>')
def download_file(filename):
    """Download output file"""
    try:
        # Security: only allow alphanumeric and specific chars
        if not all(c.isalnum() or c in '._-' for c in filename):
            return 'Invalid filename', 400
        
        output_dir = request.args.get('output_dir', '')
        file_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(file_path):
            return 'File not found', 404
        
        return send_file(file_path, as_attachment=True)
    
    except Exception as e:
        return str(e), 500


@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Shutdown the server"""
    output_capture.write("\n⏹️ Shutdown requested...\n")
    os._exit(0)
    return jsonify({'success': True})


def find_free_port(start_port=5000, max_tries=10):
    """Find a free port on localhost"""
    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return start_port  # Fallback to start port


def open_browser(url):
    """Wait for server to start and open browser"""
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()


def main():
    """Main entry point"""
    # Configuration
    port = find_free_port()
    host = '127.0.0.1'
    url = f'http://{host}:{port}'
    
    print("\n" + "=" * 60)
    print("🧠 BCT Analysis Web Interface")
    print("=" * 60)
    print(f"🌐 Starting server at {url}")
    print("=" * 60 + "\n")
    
    # Open browser
    open_browser(url)
    
    # Run with Waitress
    try:
        serve(app, host=host, port=port, threads=4)
    except KeyboardInterrupt:
        print("\n⏹️ Stopping server...")
        sys.exit(0)


if __name__ == '__main__':
    main()
