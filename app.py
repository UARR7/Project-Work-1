from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json
import hashlib
import hmac
from datetime import datetime
import logging
from config import Config
from docs_generator import DocsGenerator
from github_webhook import GitHubWebhook
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Initialize components
docs_generator = DocsGenerator()
webhook_handler = GitHubWebhook()

@app.route('/')
def dashboard():
    """Main dashboard showing recent documentation updates"""
    try:
        recent_updates = get_recent_updates()
        return render_template('index.html', updates=recent_updates)
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return render_template('index.html', updates=[])

@app.route('/webhook', methods=['POST'])
def github_webhook():
    """Handle GitHub webhook events"""
    try:
        # Verify webhook signature
        signature = request.headers.get('X-Hub-Signature-256')
        if not webhook_handler.verify_signature(request.data, signature):
            return jsonify({'error': 'Invalid signature'}), 403
        
        payload = request.json
        event_type = request.headers.get('X-GitHub-Event')
        
        if event_type == 'push':
            # Process push event in background
            thread = threading.Thread(
                target=process_push_event,
                args=(payload,)
            )
            thread.daemon = True
            thread.start()
            
            return jsonify({'status': 'processing'}), 200
        
        return jsonify({'status': 'ignored'}), 200
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/docs/<path:filename>')
def serve_docs(filename):
    """Serve generated documentation files"""
    return send_from_directory('docs_output', filename)

@app.route('/api/regenerate', methods=['POST'])
def regenerate_docs():
    """Manually trigger documentation regeneration"""
    try:
        data = request.json
        repo_url = data.get('repo_url')
        branch = data.get('branch', 'main')
        
        if not repo_url:
            return jsonify({'error': 'Repository URL required'}), 400
        
        # Start regeneration in background
        thread = threading.Thread(
            target=docs_generator.generate_from_repo,
            args=(repo_url, branch)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'started'}), 202
        
    except Exception as e:
        logger.error(f"Manual regeneration error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    """Get current system status"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'docs_count': count_generated_docs()
    })

def process_push_event(payload):
    """Process GitHub push event"""
    try:
        repo_url = payload['repository']['clone_url']
        branch = payload['ref'].split('/')[-1]
        commits = payload['commits']
        
        logger.info(f"Processing push to {repo_url} branch {branch}")
        logger.info(f"Commits: {len(commits)}")
        
        # Generate documentation
        result = docs_generator.generate_from_repo(repo_url, branch)
        
        if result:
            logger.info("Documentation generated successfully")
            # Save update record
            save_update_record(repo_url, branch, commits)
        else:
            logger.error("Documentation generation failed")
            
    except Exception as e:
        logger.error(f"Push event processing error: {e}")

def get_recent_updates():
    """Get recent documentation updates"""
    try:
        updates_file = os.path.join(app.config['DOCS_OUTPUT_DIR'], 'updates.json')
        if os.path.exists(updates_file):
            with open(updates_file, 'r') as f:
                updates = json.load(f)
            return sorted(updates, key=lambda x: x['timestamp'], reverse=True)[:10]
        return []
    except:
        return []

def save_update_record(repo_url, branch, commits):
    """Save documentation update record"""
    try:
        updates_file = os.path.join(app.config['DOCS_OUTPUT_DIR'], 'updates.json')
        
        # Load existing updates
        updates = []
        if os.path.exists(updates_file):
            with open(updates_file, 'r') as f:
                updates = json.load(f)
        
        # Add new update
        update = {
            'repo_url': repo_url,
            'branch': branch,
            'timestamp': datetime.now().isoformat(),
            'commit_count': len(commits),
            'latest_commit': commits[-1]['message'] if commits else 'No commits'
        }
        
        updates.append(update)
        
        # Keep only last 100 updates
        updates = updates[-100:]
        
        # Save updates
        with open(updates_file, 'w') as f:
            json.dump(updates, f, indent=2)
            
    except Exception as e:
        logger.error(f"Failed to save update record: {e}")

def count_generated_docs():
    """Count generated documentation files"""
    try:
        docs_dir = app.config['DOCS_OUTPUT_DIR']
        if not os.path.exists(docs_dir):
            return 0
        
        count = 0
        for root, dirs, files in os.walk(docs_dir):
            count += len([f for f in files if f.endswith(('.html', '.md'))])
        return count
    except:
        return 0

if __name__ == '__main__':
    # Ensure output directory exists
    os.makedirs(app.config['DOCS_OUTPUT_DIR'], exist_ok=True)
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5001)),
        debug=app.config['DEBUG']
    )