# # Auto Documentation Tool
# # Complete project structure with all necessary files

# """
# Project Structure:
# auto-docs-tool/
# ├── app.py                 # Main Flask application
# ├── config.py              # Configuration settings
# ├── requirements.txt       # Python dependencies
# ├── docs_generator.py      # Documentation generation logic
# ├── github_webhook.py      # GitHub webhook handler
# ├── utils.py              # Utility functions
# ├── templates/
# │   ├── index.html        # Dashboard template
# │   └── docs.html         # Documentation viewer
# ├── static/
# │   ├── style.css         # CSS styles
# │   └── script.js         # JavaScript
# ├── docs_output/          # Generated documentation
# ├── .env.example          # Environment variables template
# ├── Dockerfile           # Docker configuration
# ├── docker-compose.yml   # Docker Compose setup
# ├── setup.sh             # Setup script
# └── README.md            # Project documentation
# """

# # =============================================================================
# # FILE: app.py
# # =============================================================================



# # =============================================================================
# # FILE: config.py
# # =============================================================================


# # =============================================================================
# # FILE: requirements.txt
# # =============================================================================



# # =============================================================================
# # FILE: docs_generator.py
# # =============================================================================


# # =============================================================================
# # FILE: github_webhook.py
# # =============================================================================


# # =============================================================================
# # FILE: utils.py
# # =============================================================================



# # =============================================================================
# # FILE: templates/index.html
# # =============================================================================



# # =============================================================================
# # FILE: templates/docs.html
# # =============================================================================

# d
# # =============================================================================
# # FILE: static/style.css
# # =============================================================================

# style_css = """/* Auto Documentation Tool Styles */

# * {
#     margin: 0;
#     padding: 0;
#     box-sizing: border-box;
# }

# body {
#     font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#     line-height: 1.6;
#     color: #333;
#     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     min-height: 100vh;
# }

# .container {
#     max-width: 1200px;
#     margin: 0 auto;
#     padding: 20px;
# }

# header {
#     text-align: center;
#     margin-bottom: 40px;
#     color: white;
# }

# header h1 {
#     font-size: 2.5rem;
#     margin-bottom: 10px;
#     text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
# }

# header p {
#     font-size: 1.1rem;
#     opacity: 0.9;
# }

# .dashboard {
#     display: grid;
#     grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
#     gap: 20px;
#     margin-bottom: 30px;
# }

# .card {
#     background: white;
#     border-radius: 12px;
#     padding: 25px;
#     box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
#     backdrop-filter: blur(10px);
#     border: 1px solid rgba(255, 255, 255, 0.2);
# }

# .card h2 {
#     color: #333;
#     margin-bottom: 20px;
#     font-size: 1.4rem;
# }

# .status-indicator {
#     display: flex;
#     align-items: center;
#     gap: 10px;
#     font-weight: 600;
#     padding: 10px;
#     background: #f8f9fa;
#     border-radius: 6px;
#     margin-bottom: 20px;
# }

# .status-dot {
#     width: 12px;
#     height: 12px;
#     border-radius: 50%;
#     animation: pulse 2s infinite;
# }

# .status-dot.online {
#     background: #28a745;
# }

# .status-dot.offline {
#     background: #dc3545;
# }

# @keyframes pulse {
#     0% { opacity: 1; }
#     50% { opacity: 0.5; }
#     100% { opacity: 1; }
# }

# .stats {
#     display: flex;
#     justify-content: space-around;
# }

# .stat {
#     text-align: center;
# }

# .stat-number {
#     display: block;
#     font-size: 2rem;
#     font-weight: bold;
#     color: #007acc;
# }

# .stat-label {
#     font-size: 0.9rem;
#     color: #666;
# }

# .manual-form {
#     display: flex;
#     flex-direction: column;
#     gap: 15px;
# }

# .form-group {
#     display: flex;
#     flex-direction: column;
# }

# .form-group label {
#     margin-bottom: 5px;
#     font-weight: 600;
#     color: #555;
# }

# .form-group input {
#     padding: 10px;
#     border: 2px solid #e0e0e0;
#     border-radius: 6px;
#     font-size: 1rem;
#     transition: border-color 0.3s;
# }

# .form-group input:focus {
#     outline: none;
#     border-color: #007acc;
# }

# .btn-primary {
#     background: linear-gradient(135deg, #007acc, #0056b3);
#     color: white;
#     border: none;
#     padding: 12px 24px;
#     border-radius: 6px;
#     font-size: 1rem;
#     font-weight: 600;
#     cursor: pointer;
#     transition: transform 0.2s, box-shadow 0.2s;
# }

# .btn-primary:hover {
#     transform: translateY(-2px);
#     box-shadow: 0 4px 12px rgba(0, 122, 204, 0.3);
# }

# .btn-copy {
#     background: #6c757d;
#     color: white;
#     border: none;
#     padding: 5px 10px;
#     border-radius: 4px;
#     font-size: 0.8rem;
#     cursor: pointer;
#     margin-left: 10px;
# }

# .btn-back {
#     display: inline-block;
#     background: rgba(255, 255, 255, 0.2);
#     color: white;
#     text-decoration: none;
#     padding: 8px 16px;
#     border-radius: 6px;
#     font-weight: 600;
#     transition: background 0.3s;
# }

# .btn-back:hover {
#     background: rgba(255, 255, 255, 0.3);
# }

# .status-message {
#     margin-top: 15px;
#     padding: 10px;
#     border-radius: 4px;
#     display: none;
# }

# .status-message.success {
#     background: #d4edda;
#     border: 1px solid #c3e6cb;
#     color: #155724;
#     display: block;
# }

# .status-message.error {
#     background: #f8d7da;
#     border: 1px solid #f5c6cb;
#     color: #721c24;
#     display: block;
# }

# .status-message.info {
#     background: #cce7ff;
#     border: 1px solid #99d6ff;
#     color: #004085;
#     display: block;
# }

# .recent-updates {
#     grid-column: 1 / -1;
# }

# .updates-list {
#     max-height: 400px;
#     overflow-y: auto;
# }

# .update-item {
#     padding: 15px;
#     border-bottom: 1px solid #e0e0e0;
#     transition: background 0.2s;
# }

# .update-item:hover {
#     background: #f8f9fa;
# }

# .update-item:last-child {
#     border-bottom: none;
# }

# .update-header {
#     display: flex;
#     justify-content: space-between;
#     align-items: center;
#     margin-bottom: 8px;
# }

# .update-time {
#     font-size: 0.9rem;
#     color: #666;
# }

# .update-details {
#     display: flex;
#     gap: 10px;
#     margin-bottom: 8px;
# }

# .branch-tag {
#     background: #007acc;
#     color: white;
#     padding: 2px 8px;
#     border-radius: 12px;
#     font-size: 0.8rem;
#     font-weight: 600;
# }

# .commit-count {
#     background: #28a745;
#     color: white;
#     padding: 2px 8px;
#     border-radius: 12px;
#     font-size: 0.8rem;
#     font-weight: 600;
# }

# .update-message {
#     font-size: 0.9rem;
#     color: #555;
#     font-style: italic;
# }

# .no-updates {
#     text-align: center;
#     padding: 40px 20px;
#     color: #666;
# }

# .setup-instructions {
#     grid-column: 1 / -1;
# }

# .instruction-steps {
#     display: grid;
#     grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
#     gap: 20px;
# }

# .step {
#     padding: 20px;
#     background: #f8f9fa;
#     border-radius: 8px;
#     border-left: 4px solid #007acc;
# }

# .step h3 {
#     color: #007acc;
#     margin-bottom: 10px;
# }

# .webhook-url {
#     display: inline-block;
#     background: #f8f8f8;
#     padding: 8px 12px;
#     border-radius: 4px;
#     font-family: 'Courier New', monospace;
#     word-break: break-all;
#     border: 1px solid #e0e0e0;
# }

# /* Documentation Browser Styles */
# .docs-browser {
#     display: grid;
#     grid-template-columns: 300px 1fr;
#     gap: 20px;
#     height: 70vh;
# }

# .docs-list {
#     background: white;
#     border-radius: 8px;
#     padding: 20px;
#     overflow-y: auto;
# }

# .docs-viewer {
#     background: white;
#     border-radius: 8px;
#     padding: 0;
#     overflow: hidden;
# }

# #docs-frame {
#     width: 100%;
#     height: 100%;
#     border: none;
#     border-radius: 8px;
# }

# .loading {
#     text-align: center;
#     padding: 20px;
#     color: #666;
# }

# .no-docs {
#     text-align: center;
#     padding: 40px 20px;
#     color: #666;
# }

# /* Responsive Design */
# @media (max-width: 768px) {
#     .dashboard {
#         grid-template-columns: 1fr;
#     }
    
#     .instruction-steps {
#         grid-template-columns: 1fr;
#     }
    
#     .docs-browser {
#         grid-template-columns: 1fr;
#         height: auto;
#     }
    
#     .docs-list {
#         height: 300px;
#     }
    
#     .docs-viewer {
#         height: 500px;
#     }
    
#     header h1 {
#         font-size: 2rem;
#     }
    
#     .container {
#         padding: 10px;
#     }
# }

# /* Code syntax highlighting support */
# .codehilite {
#     background: #f8f8f8;
#     border-radius: 6px;
#     padding: 15px;
#     overflow-x: auto;
#     margin: 10px 0;
# }

# .codehilite pre {
#     margin: 0;
# }

# /* Loading animations */
# .spinner {
#     border: 2px solid #f3f3f3;
#     border-top: 2px solid #007acc;
#     border-radius: 50%;
#     width: 20px;
#     height: 20px;
#     animation: spin 1s linear infinite;
#     display: inline-block;
#     margin-right: 8px;
# }

# @keyframes spin {
#     0% { transform: rotate(0deg); }
#     100% { transform: rotate(360deg); }
# }"""

# # =============================================================================
# # FILE: static/script.js
# # =============================================================================

# script_js = """// Auto Documentation Tool JavaScript

# // Global variables
# let systemStatus = 'online';

# // Initialize the application
# document.addEventListener('DOMContentLoaded', function() {
#     initializeApp();
# });

# function initializeApp() {
#     loadSystemStatus();
#     setupEventListeners();
#     updateWebhookUrl();
    
#     // Refresh status every 30 seconds
#     setInterval(loadSystemStatus, 30000);
# }

# function setupEventListeners() {
#     // Manual form submission
#     const manualForm = document.getElementById('manual-form');
#     if (manualForm) {
#         manualForm.addEventListener('submit', handleManualGeneration);
#     }
# }

# function loadSystemStatus() {
#     fetch('/api/status')
#         .then(response => response.json())
#         .then(data => {
#             updateStatusIndicator(data.status);
#             updateDocsCount(data.docs_count);
#         })
#         .catch(error => {
#             console.error('Failed to load system status:', error);
#             updateStatusIndicator('offline');
#         });
# }

# function updateStatusIndicator(status) {
#     const statusElement = document.getElementById('status');
#     const statusDot = statusElement.querySelector('.status-dot');
    
#     if (status === 'running') {
#         statusDot.className = 'status-dot online';
#         statusElement.innerHTML = '<span class="status-dot online"></span>System Online';
#         systemStatus = 'online';
#     } else {
#         statusDot.className = 'status-dot offline';
#         statusElement.innerHTML = '<span class="status-dot offline"></span>System Offline';
#         systemStatus = 'offline';
#     }
# }

# function updateDocsCount(count) {
#     const docsCountElement = document.getElementById('docs-count');
#     if (docsCountElement) {
#         docsCountElement.textContent = count || '0';
#     }
# }

# function handleManualGeneration(event) {
#     event.preventDefault();
    
#     const form = event.target;
#     const formData = new FormData(form);
#     const data = {
#         repo_url: formData.get('repo_url'),
#         branch: formData.get('branch') || 'main'
#     };
    
#     // Validate input
#     if (!data.repo_url) {
#         showStatusMessage('Please enter a repository URL', 'error');
#         return;
#     }
    
#     if (!isValidRepoUrl(data.repo_url)) {
#         showStatusMessage('Please enter a valid repository URL', 'error');
#         return;
#     }
    
#     // Show loading state
#     const submitButton = form.querySelector('button[type="submit"]');
#     const originalText = submitButton.textContent;
#     submitButton.innerHTML = '<span class="spinner"></span>Generating...';
#     submitButton.disabled = true;
    
#     showStatusMessage('Starting documentation generation...', 'info');
    
#     // Send request
#     fetch('/api/regenerate', {
#         method: 'POST',
#         headers: {
#             'Content-Type': 'application/json',
#         },
#         body: JSON.stringify(data)
#     })
#     .then(response => response.json())
#     .then(result => {
#         if (result.status === 'started') {
#             showStatusMessage('Documentation generation started! This may take a few minutes.', 'success');
#             // Poll for completion (optional)
#             pollGenerationStatus();
#         } else {
#             showStatusMessage('Failed to start documentation generation', 'error');
#         }
#     })
#     .catch(error => {
#         console.error('Generation failed:', error);
#         showStatusMessage('Failed to generate documentation. Please try again.', 'error');
#     })
#     .finally(() => {
#         // Reset button
#         submitButton.textContent = originalText;
#         submitButton.disabled = false;
#     });
# }

# function pollGenerationStatus() {
#     // This could poll an endpoint to check generation status
#     // For now, we'll just refresh the page after a delay
#     setTimeout(() => {
#         location.reload();
#     }, 10000); // Refresh after 10 seconds
# }

# function isValidRepoUrl(url) {
#     // Basic URL validation for Git repositories
#     const gitUrlPatterns = [
#         /^https:\/\/github\.com\/[\w-]+\/[\w-]+\.git$/,
#         /^https:\/\/github\.com\/[\w-]+\/[\w-]+$/,
#         /^https:\/\/gitlab\.com\/[\w-]+\/[\w-]+\.git$/,
#         /^https:\/\/gitlab\.com\/[\w-]+\/[\w-]+$/,
#         /^https:\/\/bitbucket\.org\/[\w-]+\/[\w-]+\.git$/,
#         /^https:\/\/bitbucket\.org\/[\w-]+\/[\w-]+$/
#     ];
    
#     return gitUrlPatterns.some(pattern => pattern.test(url));
# }

# function showStatusMessage(message, type) {
#     const statusElement = document.getElementById('manual-status');
#     if (statusElement) {
#         statusElement.textContent = message;
#         statusElement.className = `status-message ${type}`;
        
#         // Hide after 5 seconds for non-error messages
#         if (type !== 'error') {
#             setTimeout(() => {
#                 statusElement.className = 'status-message';
#             }, 5000);
#         }
#     }
# }

# function updateWebhookUrl() {
#     const webhookUrlElement = document.getElementById('webhook-url');
#     if (webhookUrlElement) {
#         const baseUrl = window.location.origin;
#         const webhookUrl = `${baseUrl}/webhook`;
#         webhookUrlElement.textContent = webhookUrl;
#     }
# }

# function copyWebhookUrl() {
#     const webhookUrlElement = document.getElementById('webhook-url');
#     if (webhookUrlElement) {
#         const url = webhookUrlElement.textContent;
        
#         // Try to use the Clipboard API
#         if (navigator.clipboard && window.isSecureContext) {
#             navigator.clipboard.writeText(url).then(() => {
#                 showNotification('Webhook URL copied to clipboard!');
#             }).catch(err => {
#                 fallbackCopyTextToClipboard(url);
#             });
#         } else {
#             fallbackCopyTextToClipboard(url);
#         }
#     }
# }

# function fallbackCopyTextToClipboard(text) {
#     // Fallback method for copying text
#     const textArea = document.createElement("textarea");
#     textArea.value = text;
#     textArea.style.top = "0";
#     textArea.style.left = "0";
#     textArea.style.position = "fixed";
    
#     document.body.appendChild(textArea);
#     textArea.focus();
#     textArea.select();
    
#     try {
#         document.execCommand('copy');
#         showNotification('Webhook URL copied to clipboard!');
#     } catch (err) {
#         showNotification('Failed to copy URL. Please copy manually.');
#     }
    
#     document.body.removeChild(textArea);
# }

# function showNotification(message) {
#     // Create a temporary notification
#     const notification = document.createElement('div');
#     notification.textContent = message;
#     notification.style.cssText = `
#         position: fixed;
#         top: 20px;
#         right: 20px;
#         background: #28a745;
#         color: white;
#         padding: 15px 20px;
#         border-radius: 6px;
#         box-shadow: 0 4px 12px rgba(0,0,0,0.2);
#         z-index: 1000;
#         font-weight: 600;
#         animation: slideIn 0.3s ease-out;
#     `;
    
#     // Add animation keyframes if not already added
#     if (!document.querySelector('#notification-styles')) {
#         const style = document.createElement('style');
#         style.id = 'notification-styles';
#         style.textContent = `
#             @keyframes slideIn {
#                 from { transform: translateX(100%); opacity: 0; }
#                 to { transform: translateX(0); opacity: 1; }
#             }
#             @keyframes slideOut {
#                 from { transform: translateX(0); opacity: 1; }
#                 to { transform: translateX(100%); opacity: 0; }
#             }
#         `;
#         document.head.appendChild(style);
#     }
    
#     document.body.appendChild(notification);
    
#     // Remove notification after 3 seconds
#     setTimeout(() => {
#         notification.style.animation = 'slideOut 0.3s ease-in';
#         setTimeout(() => {
#             if (notification.parentNode) {
#                 notification.parentNode.removeChild(notification);
#             }
#         }, 300);
#     }, 3000);
# }

# // Utility functions
# function formatTimestamp(timestamp) {
#     const date = new Date(timestamp);
#     return date.toLocaleString();
# }

# function debounce(func, wait) {
#     let timeout;
#     return function executedFunction(...args) {
#         const later = () => {
#             clearTimeout(timeout);
#             func(...args);
#         };
#         clearTimeout(timeout);
#         timeout = setTimeout(later, wait);
#     };
# }

# // Export functions for global use
# window.copyWebhookUrl = copyWebhookUrl;"""

# # =============================================================================
# # FILE: .env.example
# # =============================================================================

# env_example = """# Flask Configuration
# FLASK_DEBUG=False
# SECRET_KEY=your-secret-key-here-change-in-production

# # GitHub Webhook Configuration
# GITHUB_WEBHOOK_SECRET=your-github-webhook-secret

# # Documentation Configuration
# DOCS_OUTPUT_DIR=docs_output
# TEMP_DIR=temp

# # Repository Configuration
# MAX_REPO_SIZE=100
# CLONE_TIMEOUT=300

# # Server Configuration
# PORT=5000"""

# # =============================================================================
# # FILE: Dockerfile
# # =============================================================================

# dockerfile = """FROM python:3.11-slim

# # Set working directory
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \\
#     git \\
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements first for better caching
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY . .

# # Create directories
# RUN mkdir -p docs_output temp

# # Set environment variables
# ENV PYTHONUNBUFFERED=1
# ENV FLASK_APP=app.py

# # Expose port
# EXPOSE 5000

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
#     CMD curl -f http://localhost:5000/api/status || exit 1

# # Run the application
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]"""

# # =============================================================================
# # FILE: docker-compose.yml
# # =============================================================================

# docker_compose = """version: '3.8'

# services:
#   auto-docs:
#     build: .
#     ports:
#       - "5000:5000"
#     environment:
#       - FLASK_DEBUG=False
#       - SECRET_KEY=change-this-secret-key-in-production
#       - GITHUB_WEBHOOK_SECRET=your-webhook-secret-here
#     volumes:
#       - ./docs_output:/app/docs_output
#       - ./temp:/app/temp
#     restart: unless-stopped
#     healthcheck:
#       test: ["CMD", "curl", "-f", "http://localhost:5000/api/status"]
#       interval: 30s
#       timeout: 10s
#       retries: 3