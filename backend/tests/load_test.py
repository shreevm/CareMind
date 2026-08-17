"""Load testing for CareMind API.

Tests performance under load using Locust.
Simulates realistic user behavior with mix of search, chat, and upload requests.

To run:
    locust -f backend/tests/load_test.py --host=http://127.0.0.1:8002 --web

Then visit http://localhost:8089 to start the test.
"""

import uuid
from locust import HttpUser, task, between, events
import json


class CareMindUser(HttpUser):
    """Simulates a CareMind user with realistic behavior patterns."""
    
    # Wait between 1-3 seconds between requests
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup before user starts making requests."""
        print("\n[LOAD TEST] User session started")
        
        # Seed demo data if not already loaded
        try:
            response = self.client.post(
                "/demo/seed",
                params={"workspace_id": "default"},
                timeout=10
            )
            if response.status_code == 200:
                print("[LOAD TEST] ✓ Demo data seeded")
        except Exception as e:
            print(f"[LOAD TEST] Warning: Could not seed demo data: {e}")
    
    @task(3)
    def search_documents(self):
        """
        Simulate document search requests (30% of traffic).
        
        Real-world scenario:
        - User searches for specific information in documents
        - Fast operation (should complete in <500ms)
        - No state required
        """
        queries = [
            "hemoglobin",
            "chest x-ray",
            "patient diagnosis",
            "lab results",
            "treatment plan",
        ]
        
        query = queries[hash(str(uuid.uuid4())) % len(queries)]
        
        with self.client.get(
            "/search",
            params={
                "query": query,
                "workspace_id": "default",
                "top_k": 5
            },
            catch_response=True,
            name="/search"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(7)
    def chat_query(self):
        """
        Simulate chat requests (70% of traffic).
        
        Real-world scenario:
        - User asks questions about documents
        - Moderate latency (should complete in <2s)
        - Maintains session state
        - Multiple turns in same session
        """
        messages = [
            "What is the patient diagnosis?",
            "What lab values were abnormal?",
            "What does the radiology report show?",
            "What treatment is recommended?",
            "Are there any follow-up recommendations?",
            "What is the patient's medication?",
            "Summarize the clinical findings",
        ]
        
        message = messages[hash(str(uuid.uuid4())) % len(messages)]
        
        # Use consistent session ID to maintain conversation context
        session_id = f"load-test-session-{id(self)}"
        
        payload = {
            "session_id": session_id,
            "workspace_id": "default",
            "message": message
        }
        
        with self.client.post(
            "/chat",
            json=payload,
            catch_response=True,
            name="/chat"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(2)
    def list_documents(self):
        """
        Simulate listing documents (20% of traffic).
        
        Real-world scenario:
        - User views available documents
        - Very fast operation (<100ms)
        - No state required
        """
        with self.client.get(
            "/documents",
            params={"workspace_id": "default"},
            catch_response=True,
            name="/documents"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """
        Simulate health check requests (10% of traffic).
        
        Real-world scenario:
        - Monitoring endpoints
        - Should be very fast (<50ms)
        - No resource usage
        """
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


class HighVolumeUser(HttpUser):
    """
    High-volume user making many rapid requests.
    
    Useful for stress testing to find breaking points.
    """
    wait_time = between(0.5, 1.5)
    
    @task(10)
    def rapid_search(self):
        """Make rapid search requests."""
        queries = ["hemoglobin", "chest", "labs", "diagnosis"]
        query = queries[hash(str(uuid.uuid4())) % len(queries)]
        
        self.client.get(
            "/search",
            params={
                "query": query,
                "workspace_id": "default",
                "top_k": 3
            },
            name="/search [high-volume]"
        )


# Event handlers for reporting

@events.quitting.add_listener
def _(environment, **kw):
    """Report summary when load test ends."""
    print("\n" + "="*80)
    print("LOAD TEST SUMMARY")
    print("="*80)
    
    # Print stats from Locust's stats
    stats = environment.stats
    
    print(f"\nTotal Requests: {stats.total.num_requests}")
    print(f"Failed Requests: {stats.total.num_failures}")
    print(f"Success Rate: {(1 - stats.total.fail_ratio) * 100:.1f}%")
    
    print(f"\nResponse Times (ms):")
    print(f"  Min: {stats.total.min_response_time:.1f}")
    print(f"  Max: {stats.total.max_response_time:.1f}")
    print(f"  Avg: {stats.total.avg_response_time:.1f}")
    print(f"  Median: {stats.total.get_response_time_percentile(0.5):.1f}")
    print(f"  95th percentile: {stats.total.get_response_time_percentile(0.95):.1f}")
    print(f"  99th percentile: {stats.total.get_response_time_percentile(0.99):.1f}")
    
    print(f"\nThroughput:")
    print(f"  Requests/sec: {stats.total.total_rps:.2f}")
    print(f"  Requests/min: {stats.total.total_rps * 60:.0f}")
    
    print("\n" + "="*80)
    print("Endpoint Performance:")
    print("="*80)
    
    for name, stat in stats.entries.items():
        if name != "Total":
            print(f"\n{name}:")
            print(f"  Requests: {stat.num_requests}")
            print(f"  Failures: {stat.num_failures} ({stat.fail_ratio*100:.1f}%)")
            print(f"  Avg Response: {stat.avg_response_time:.1f}ms")
            print(f"  95th: {stat.get_response_time_percentile(0.95):.1f}ms")


"""
Load Testing Guide:

1. **Basic Test (Light Load):**
   - Users: 10
   - Ramp-up: 2 min (adds 1 user every 12 seconds)
   - Duration: 5-10 minutes
   - Expected: Baseline performance

2. **Normal Load:**
   - Users: 50
   - Ramp-up: 5 min
   - Duration: 10-15 minutes
   - Expected: Should handle without degradation

3. **High Load (Stress Test):**
   - Users: 100+
   - Ramp-up: 10 min
   - Duration: 15-20 minutes
   - Expected: Find limits and breaking points

4. **Performance Metrics to Monitor:**
   - Response Time (avg/p95/p99)
   - Error Rate (should stay < 1%)
   - Throughput (requests/sec)
   - Resource Usage (CPU, Memory)

5. **Interpreting Results:**
   - Avg Response < 500ms → Good (search)
   - Avg Response < 2000ms → Good (chat)
   - Error Rate > 5% → Issue detected
   - P95 > 5s → Performance degradation

6. **Common Bottlenecks:**
   - Database (SQLite locks, Supabase limits)
   - Embedding API (rate limits)
   - Vector search (index size)
   - LLM API (concurrent requests)
   - Memory (large documents)

7. **Optimization Tips:**
   - Increase connection pool size
   - Enable caching
   - Reduce vector search top-k
   - Batch document uploads
   - Use async operations
"""
