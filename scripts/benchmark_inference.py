import requests
import time
import statistics
import numpy as np
from uuid import uuid4

def benchmark_inference():
    print("Starting API Inference Benchmark...")
    # Initialize session
    res = requests.post("http://localhost:8001/api/v1/game/session")
    if res.status_code != 200:
        print("Backend not running or failed to create session.")
        return
        
    session_id = res.json()["session_id"]
    
    latencies = []
    
    for i in range(100):
        # We ping the recommend endpoint repeatedly for the same state
        start = time.time()
        try:
            res = requests.post("http://localhost:8001/api/v1/ai/recommend", json={"session_id": session_id}, timeout=5.0)
            end = time.time()
            if res.status_code != 200:
                print(f"Error on request {i}: {res.text}")
                break
            latencies.append((end - start) * 1000)
        except Exception as e:
            print(f"Request {i} failed: {e}")
            break
        
    if not latencies:
        return
        
    print(f"Total Requests: {len(latencies)}")
    print(f"Average Latency: {statistics.mean(latencies):.2f} ms")
    print(f"p50 Latency: {np.percentile(latencies, 50):.2f} ms")
    print(f"p95 Latency: {np.percentile(latencies, 95):.2f} ms")
    print(f"p99 Latency: {np.percentile(latencies, 99):.2f} ms")

if __name__ == "__main__":
    benchmark_inference()
