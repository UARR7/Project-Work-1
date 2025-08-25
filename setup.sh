# Make setup script executable and run
chmod +x setup.sh
./setup.sh

# Edit environment variables
nano .env

# Start the application
source venv/bin/activate
python app.py
