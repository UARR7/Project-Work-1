# from flask import Flask, request, jsonify, render_template, send_from_directory
# import os
# import json
# import hashlib
# import hmac
# from datetime import datetime
# import logging

# from huggingface_hub import repo_info
# from config import Config
# from docs_generator import DocsGenerator
# from github_webhook import GitHubWebhook
# import threading

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)
# app.config.from_object(Config)

# # Initialize components
# docs_generator = DocsGenerator()
# webhook_handler = GitHubWebhook()

# @app.route('/')
# def dashboard():
#     """Main dashboard showing recent documentation updates"""
#     try:
#         recent_updates = get_recent_updates()
#         return render_template('index.html', updates=recent_updates)
#     except Exception as e:
#         logger.error(f"Dashboard error: {e}")
#         return render_template('index.html', updates=[])

# @app.route('/webhook', methods=['POST'])
# def github_webhook():
#     """Handle GitHub webhook events"""
#     try:
#         # Verify webhook signature
#         signature = request.headers.get('X-Hub-Signature-256')
#         if not webhook_handler.verify_signature(request.data, signature):
#             return jsonify({'error': 'Invalid signature'}), 403
        
#         payload = request.json
#         event_type = request.headers.get('X-GitHub-Event')
        
#         if event_type == 'push':
#             # Process push event in background
#             thread = threading.Thread(
#                 target=process_push_event,
#                 args=(payload,)
#             )
#             thread.daemon = True
#             thread.start()
            
#             return jsonify({'status': 'processing'}), 200
        
#         return jsonify({'status': 'ignored'}), 200
        
#     except Exception as e:
#         logger.error(f"Webhook error: {e}")
#         return jsonify({'error': 'Internal server error'}), 500

# @app.route('/docs/<path:filename>')
# def serve_docs(filename):
#     """Serve generated documentation files"""
#     return send_from_directory('docs_output', filename)

# @app.route('/api/regenerate', methods=['POST'])
# def regenerate_docs():
#     """Manually trigger documentation regeneration"""
#     try:
#         data = request.json
#         repo_url = data.get('repo_url')
#         branch = data.get('branch', 'main')
        
#         if not repo_url:
#             return jsonify({'error': 'Repository URL required'}), 400
        
#         # Start regeneration in background
#         thread = threading.Thread(
#             target=docs_generator.generate_from_repo,
#             args=(repo_url, branch)
#         )
#         thread.daemon = True
#         thread.start()
        
#         return jsonify({'status': 'started'}), 202
        
#     except Exception as e:
#         logger.error(f"Manual regeneration error: {e}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/status')
# def get_status():
#     """Get current system status"""
#     return jsonify({
#         'status': 'running',
#         'timestamp': datetime.now().isoformat(),
#         'docs_count': count_generated_docs()
#     })

# def process_push_event(payload):
#     """Process GitHub push event"""
#     try:
#         repo_url = payload['repository']['clone_url']
#         branch = payload['ref'].split('/')[-1]
#         commits = payload['commits']
        
#         logger.info(f"Processing push to {repo_url} branch {branch}")
#         logger.info(f"Commits: {len(commits)}")
        
#         # Generate documentation
#         result = docs_generator.generate_from_repo(repo_url, branch)
        
#         if result:
#             logger.info("Documentation generated successfully")
#             # Save update record
#             save_update_record(repo_url, branch, commits)
#         else:
#             logger.error("Documentation generation failed")
            
#     except Exception as e:
#         logger.error(f"Push event processing error: {e}")

# def get_recent_updates():
#     """Get recent documentation updates"""
#     try:
#         updates_file = os.path.join(app.config['DOCS_OUTPUT_DIR'], 'updates.json')
#         if os.path.exists(updates_file):
#             with open(updates_file, 'r') as f:
#                 updates = json.load(f)
#             return sorted(updates, key=lambda x: x['timestamp'], reverse=True)[:10]
#         return []
#     except:
#         return []

# def save_update_record(repo_url, branch, commits):
#     """Save documentation update record"""
#     try:
#         updates_file = os.path.join(app.config['DOCS_OUTPUT_DIR'], 'updates.json')
        
#         # Load existing updates
#         updates = []
#         if os.path.exists(updates_file):
#             with open(updates_file, 'r') as f:
#                 updates = json.load(f)
        
#         # Add new update
#         update = {
#             'repo_url': repo_url,
#             'branch': branch,
#             'timestamp': datetime.now().isoformat(),
#             'commit_count': len(commits),
#             'latest_commit': commits[-1]['message'] if commits else 'No commits'
#         }
        
#         updates.append(update)
        
#         # Keep only last 100 updates
#         updates = updates[-100:]
        
#         # Save updates
#         with open(updates_file, 'w') as f:
#             json.dump(updates, f, indent=2)
            
#     except Exception as e:
#         logger.error(f"Failed to save update record: {e}")

# def count_generated_docs():
#     """Count generated documentation files"""
#     try:
#         docs_dir = app.config['DOCS_OUTPUT_DIR']
#         if not os.path.exists(docs_dir):
#             return 0
        
#         count = 0
#         for root, dirs, files in os.walk(docs_dir):
#             count += len([f for f in files if f.endswith(('.html', '.md'))])
#         return count
#     except:
#         return 0
    

# from xai_validator import XAIDocumentationValidator
# import json

# # Load your existing repo_info (you'd need to extract this from your current system)
# validator = XAIDocumentationValidator()
# results = validator.validate_with_xai(repo_info)
# report = validator.generate_xai_report(results)

# # Save manually
# with open('docs_output/xai_validation_report.json', 'w') as f:
#     json.dump(report, f, indent=2)

# if __name__ == '__main__':
#     # Ensure output directory exists
#     os.makedirs(app.config['DOCS_OUTPUT_DIR'], exist_ok=True)
    
#     # Run the application
#     app.run(
#         host='0.0.0.0',
#         port=int(os.environ.get('PORT', 5001)),
#         debug=app.config['DEBUG']
#     )

from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json
import hashlib
import hmac
from datetime import datetime
import logging

from huggingface_hub import repo_info
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

# Initialize XAI validator (only if the module is available)
xai_validator = None
try:
    from xai_validator import XAIDocumentationValidator
    xai_validator = XAIDocumentationValidator()
    logger.info("XAI validator initialized successfully")
except ImportError:
    logger.warning("XAI validator not available. Install required dependencies: pip install scikit-learn shap lime transformers torch")
except Exception as e:
    logger.warning(f"Failed to initialize XAI validator: {e}")

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
        use_xai = data.get('use_xai_validation', True)
        
        if not repo_url:
            return jsonify({'error': 'Repository URL required'}), 400
        
        # Start regeneration in background
        thread = threading.Thread(
            target=generate_docs_with_xai,
            args=(repo_url, branch, use_xai)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'started'}), 202
        
    except Exception as e:
        logger.error(f"Manual regeneration error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/xai-validate', methods=['POST'])
def run_xai_validation():
    """Run XAI validation on existing documentation"""
    try:
        if not xai_validator:
            return jsonify({'error': 'XAI validator not available'}), 503
        
        data = request.json
        repo_name = data.get('repo_name')
        
        if not repo_name:
            return jsonify({'error': 'Repository name required'}), 400
        
        # Start XAI validation in background
        thread = threading.Thread(
            target=run_xai_validation_for_repo,
            args=(repo_name,)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'validation_started'}), 202
        
    except Exception as e:
        logger.error(f"XAI validation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    """Get current system status"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'docs_count': count_generated_docs(),
        'xai_available': xai_validator is not None
    })

@app.route('/api/xai-report/<repo_name>')
def get_xai_report(repo_name):
    """Get XAI validation report for a repository"""
    try:
        report_path = os.path.join(app.config['DOCS_OUTPUT_DIR'], repo_name, 'xai_validation_report.json')
        
        if not os.path.exists(report_path):
            return jsonify({'error': 'XAI report not found'}), 404
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        return jsonify(report)
        
    except Exception as e:
        logger.error(f"Failed to get XAI report: {e}")
        return jsonify({'error': 'Failed to load XAI report'}), 500

def process_push_event(payload):
    """Process GitHub push event"""
    try:
        repo_url = payload['repository']['clone_url']
        branch = payload['ref'].split('/')[-1]
        commits = payload['commits']
        
        logger.info(f"Processing push to {repo_url} branch {branch}")
        logger.info(f"Commits: {len(commits)}")
        
        # Generate documentation with XAI validation
        result = generate_docs_with_xai(repo_url, branch, use_xai=True)
        
        if result:
            logger.info("Documentation generated successfully")
            # Save update record
            save_update_record(repo_url, branch, commits)
        else:
            logger.error("Documentation generation failed")
            
    except Exception as e:
        logger.error(f"Push event processing error: {e}")

def generate_docs_with_xai(repo_url, branch='main', use_xai=True):
    """Generate documentation with optional XAI validation"""
    try:
        # Generate regular documentation
        result = docs_generator.generate_from_repo(repo_url, branch)
        
        if result and use_xai and xai_validator:
            # Run XAI validation
            logger.info("Starting XAI validation...")
            
            # Get repository info from the docs generator
            repo_info = docs_generator.repo_info
            
            if repo_info:
                xai_results = xai_validator.validate_with_xai(repo_info)
                xai_report = xai_validator.generate_xai_report(xai_results)
                
                # Save XAI reports
                repo_name = repo_info['name']
                repo_output_dir = os.path.join(app.config['DOCS_OUTPUT_DIR'], repo_name)
                
                # Ensure directory exists
                os.makedirs(repo_output_dir, exist_ok=True)
                
                # Save JSON report
                json_report_path = os.path.join(repo_output_dir, 'xai_validation_report.json')
                with open(json_report_path, 'w') as f:
                    json.dump(xai_report, f, indent=2)
                
                # Save HTML report (basic version)
                html_report = generate_xai_html_report(xai_report)
                html_report_path = os.path.join(repo_output_dir, 'xai_validation_report.html')
                with open(html_report_path, 'w') as f:
                    f.write(html_report)
                
                logger.info(f"XAI validation completed. Average quality score: {xai_report['summary']['average_quality_score']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Documentation generation with XAI failed: {e}")
        return False

def run_xai_validation_for_repo(repo_name):
    """Run XAI validation for an existing repository"""
    try:
        if not xai_validator:
            logger.error("XAI validator not available")
            return False
        
        # Load existing repository info
        repo_dir = os.path.join(app.config['DOCS_OUTPUT_DIR'], repo_name)
        repo_info_path = os.path.join(repo_dir, 'repo_info.json')
        
        if not os.path.exists(repo_info_path):
            logger.error(f"Repository info not found for {repo_name}")
            return False
        
        with open(repo_info_path, 'r') as f:
            repo_info = json.load(f)
        
        # Run XAI validation
        xai_results = xai_validator.validate_with_xai(repo_info)
        xai_report = xai_validator.generate_xai_report(xai_results)
        
        # Save reports
        json_report_path = os.path.join(repo_dir, 'xai_validation_report.json')
        with open(json_report_path, 'w') as f:
            json.dump(xai_report, f, indent=2)
        
        html_report = generate_xai_html_report(xai_report)
        html_report_path = os.path.join(repo_dir, 'xai_validation_report.html')
        with open(html_report_path, 'w') as f:
            f.write(html_report)
        
        logger.info(f"XAI validation for {repo_name} completed")
        return True
        
    except Exception as e:
        logger.error(f"XAI validation for {repo_name} failed: {e}")
        return False

def generate_xai_html_report(xai_report):
    """Generate a basic HTML report from XAI validation results"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>XAI Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .summary {{ background: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .score {{ font-size: 24px; font-weight: bold; color: #2e8b57; }}
            .recommendations {{ background: #fff8dc; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .low-quality {{ color: #dc143c; }}
            .medium-quality {{ color: #ff8c00; }}
            .high-quality {{ color: #2e8b57; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>XAI Documentation Validation Report</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Total Elements Analyzed: <strong>{xai_report['summary']['total_elements']}</strong></p>
            <p>Average Quality Score: <span class="score">{xai_report['summary']['average_quality_score']:.3f}</span></p>
            <p>Average Confidence: <strong>{xai_report['summary']['average_confidence']:.3f}</strong></p>
            <p>High Quality: <span class="high-quality">{xai_report['summary']['high_quality_count']}</span> | 
               Medium Quality: <span class="medium-quality">{xai_report['summary']['medium_quality_count']}</span> | 
               Low Quality: <span class="low-quality">{xai_report['summary']['low_quality_count']}</span></p>
        </div>
        
        <div class="recommendations">
            <h2>Global Recommendations</h2>
            <ul>
                {''.join(f'<li>{rec}</li>' for rec in xai_report['recommendations']['global_recommendations'])}
            </ul>
        </div>
        
        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>Element</th>
                <th>File</th>
                <th>Type</th>
                <th>Score</th>
                <th>Confidence</th>
                <th>Recommendations</th>
            </tr>
    """
    
    for result in xai_report['detailed_results']:
        quality_class = 'high-quality' if result['score'] >= 0.8 else 'medium-quality' if result['score'] >= 0.6 else 'low-quality'
        recommendations = '<br>'.join(result['recommendations'][:3])  # Show first 3 recommendations
        
        html += f"""
            <tr>
                <td>{result['element_name']}</td>
                <td>{result['file_path']}</td>
                <td>{result['validation_type']}</td>
                <td class="{quality_class}">{result['score']:.3f}</td>
                <td>{result['confidence']:.3f}</td>
                <td>{recommendations}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <div style="margin-top: 30px; padding: 15px; background: #f9f9f9; border-radius: 5px;">
            <h3>About XAI Validation</h3>
            <p>This report was generated using Explainable AI (XAI) techniques including SHAP and LIME to analyze 
               the quality of AI-generated documentation. Scores range from 0.0 (poor) to 1.0 (excellent).</p>
        </div>
    </body>
    </html>
    """
    
    return html

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