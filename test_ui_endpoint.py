#!/usr/bin/env python3
"""
Test script to verify the new UI endpoint works correctly
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_ui_endpoint():
    """Test the UI dashboard endpoint"""
    print("🧪 Testing UI Dashboard Endpoint...")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/ui", timeout=10)
        
        if response.status_code == 200:
            print("✅ UI endpoint is accessible")
            print(f"📄 Response content type: {response.headers.get('content-type')}")
            print(f"📊 Response size: {len(response.content)} bytes")
            
            # Check if it's HTML
            if 'text/html' in response.headers.get('content-type', ''):
                print("✅ Returns HTML content")
                
                # Check for key elements in the HTML
                content = response.text
                checks = [
                    ("Dashboard title", "Smart Radiator AI Dashboard" in content),
                    ("Training section", "Latest Training Events" in content),
                    ("Predictions section", "Latest Predictions" in content),
                    ("Metrics section", "Room Performance Metrics" in content),
                    ("Training count (24h)", "Training Events (24h)" in content),
                ]
                
                print("\n🔍 HTML Content Checks:")
                for check_name, passed in checks:
                    status = "✅" if passed else "❌"
                    print(f"  {status} {check_name}")
                
                # Save HTML to file for inspection
                with open("/tmp/ui_dashboard.html", "w") as f:
                    f.write(content)
                print(f"\n💾 Full HTML saved to: /tmp/ui_dashboard.html")
            else:
                print("⚠️  Expected HTML but got different content type")
        else:
            print(f"❌ UI endpoint returned status code: {response.status_code}")
            print(f"Response: {response.text[:500]}")
    
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to AI service. Is it running?")
        print(f"   Try: docker-compose up -d")
    except Exception as e:
        print(f"❌ Error testing UI endpoint: {e}")
        import traceback
        traceback.print_exc()

def test_supporting_endpoints():
    """Test the database endpoints that support the UI"""
    print("\n🧪 Testing Supporting Database Endpoints...")
    print("=" * 60)
    
    endpoints = [
        ("/", "Service status"),
        ("/stats", "AI statistics"),
        ("/health", "Health check"),
    ]
    
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {description} ({endpoint})")
                data = response.json()
                if endpoint == "/":
                    print(f"   Database connected: {data.get('database_connected')}")
                    print(f"   Rooms: {len(data.get('rooms', {}))}")
                elif endpoint == "/stats":
                    summary = data.get('summary', {})
                    print(f"   Total training samples: {summary.get('total_training_samples', 0)}")
                    print(f"   Total predictions: {summary.get('total_predictions', 0)}")
            else:
                print(f"⚠️  {description} ({endpoint}) returned {response.status_code}")
        except Exception as e:
            print(f"❌ {description} ({endpoint}): {e}")

def generate_sample_data():
    """Generate some sample training and prediction data for testing"""
    print("\n🔧 Generating Sample Data for UI Testing...")
    print("=" * 60)
    
    rooms = ["Sovrum", "Kontor", "Vardagsrum", "Badrum"]
    
    try:
        # Generate some training data
        for room in rooms:
            training_data = {
                "room": room,
                "current_temp": 20.5,
                "target_temp": 21.0,
                "radiator_level": 3.5,
                "outdoor_temp": -2.0,
                "forecast_temp": -1.5,
                "timestamp": datetime.now().isoformat()
            }
            
            response = requests.post(f"{BASE_URL}/train", json=training_data, timeout=5)
            if response.status_code == 200:
                print(f"✅ Generated training data for {room}")
            else:
                print(f"⚠️  Failed to generate training data for {room}")
        
        # Generate some prediction data
        for room in rooms:
            prediction_data = {
                "room": room,
                "current_temp": 20.3,
                "target_temp": 21.0,
                "radiator_level": 3.0,
                "outdoor_temp": -2.0,
                "forecast_temp": -1.5,
                "timestamp": datetime.now().isoformat()
            }
            
            response = requests.post(f"{BASE_URL}/predict", json=prediction_data, timeout=5)
            if response.status_code == 200:
                print(f"✅ Generated prediction data for {room}")
            else:
                print(f"⚠️  Failed to generate prediction data for {room}")
    
    except Exception as e:
        print(f"❌ Error generating sample data: {e}")

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🏠 Smart Radiator AI - UI Endpoint Test Suite")
    print("=" * 60 + "\n")
    
    # Test basic endpoints first
    test_supporting_endpoints()
    
    # Generate some sample data if needed
    print("\n❓ Would you like to generate sample data? (Press Ctrl+C to skip)")
    try:
        import time
        time.sleep(2)
        generate_sample_data()
    except KeyboardInterrupt:
        print("\n⏭️  Skipping sample data generation")
    
    # Test the main UI endpoint
    test_ui_endpoint()
    
    print("\n" + "=" * 60)
    print("✅ Test suite completed!")
    print("=" * 60)
    print(f"\n🌐 Open the UI dashboard in your browser:")
    print(f"   http://localhost:8000/ui")
    print("\n")

if __name__ == "__main__":
    main()
