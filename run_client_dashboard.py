#!/usr/bin/env python3
"""
Client Dashboard Runner for Haggis Hopper
This script runs the executive dashboard on port 8505
"""

import subprocess
import sys
import os

def main():
    """Run the client dashboard"""
    try:
        # Run the client dashboard on port 8505
        cmd = [sys.executable, "-m", "streamlit", "run", "client_dashboard.py", "--server.port", "8505"]
        print("🚗 Starting Haggis Hopper Executive Dashboard...")
        print("📊 Dashboard will be available at: http://localhost:8505")
        print("🔄 Press Ctrl+C to stop the dashboard")
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

if __name__ == "__main__":
    main() 