import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint returns healthy status."""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("✓ Health check passed")

def test_ask_education():
    """Test education domain question."""
    response = requests.post(
        f"{BASE_URL}/ask",
        json={"question": "How many AIIMS hospitals exist in India?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "domain" in data
    assert "response_time_ms" in data
    assert len(data["answer"]) > 0
    assert len(data["citations"]) > 0
    print(f"✓ Question answered in {data['response_time_ms']}ms")
    print(f"  Domain: {data['domain']}")
    print(f"  Citations: {len(data['citations'])}")
    print(f"  Answer preview: {data['answer'][:100]}...")

def test_empty_question():
    """Test that empty questions are rejected."""
    response = requests.post(
        f"{BASE_URL}/ask",
        json={"question": ""}
    )
    assert response.status_code == 422
    print("✓ Empty question correctly rejected")

def test_stats():
    """Test stats endpoint."""
    response = requests.get(f"{BASE_URL}/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    print(f"✓ Stats endpoint working — {data['total_requests']} total requests")

if __name__ == "__main__":
    print("Running API tests...\n")
    test_health()
    test_ask_education()
    test_empty_question()
    test_stats()
    print("\nAll tests passed ✓")