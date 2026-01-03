import asyncio
import aiohttp
import time
import statistics
import psutil
import os

API_URL = "http://localhost:8000/v1/completions"

SAMPLE_PREFIXES = [
    "def calculate_total(items):\n    total = 0\n    for item in items:\n        ",
    "class UserService:\n    def __init__(self, db):\n        self.db = db\n    \n    def get_user(self, user_id):\n        ",
    "import os\nimport sys\n\ndef main():\n    args = sys.argv[1:]\n    if not args:\n        ",
    "async def fetch_data(url):\n    async with aiohttp.ClientSession() as session:\n        ",
    "def parse_config(path):\n    with open(path) as f:\n        data = json.load(f)\n        return ",
]

async def make_request(session, prefix, suffix=""):
    start = time.perf_counter()
    try:
        async with session.post(API_URL, json={
            "prompt": prefix,
            "suffix": suffix,
            "max_tokens": 30
        }) as resp:
            await resp.json()
            return time.perf_counter() - start, None
    except Exception as e:
        return time.perf_counter() - start, str(e)

async def benchmark(num_requests=50, concurrency=5):
    print(f"Benchmarking {API_URL}")
    print(f"Requests: {num_requests}, Concurrency: {concurrency}\n")
    
    proc = None
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in p.info['name'].lower():
                cmdline = ' '.join(p.info['cmdline'] or [])
                if 'serve_fim' in cmdline or 'uvicorn' in cmdline:
                    proc = p
                    break
        except:
            pass
    
    if proc:
        print(f"Found server process: PID {proc.pid}")
    else:
        print("Warning: Could not find server process for CPU monitoring")
    
    latencies = []
    errors = 0
    cpu_samples = []
    
    async with aiohttp.ClientSession() as session:
        # Warmup
        print("Warming up...")
        await make_request(session, SAMPLE_PREFIXES[0])
        
        print("Running benchmark...")
        start_time = time.perf_counter()
        
        sem = asyncio.Semaphore(concurrency)
        
        async def bounded_request(prefix):
            async with sem:
                if proc:
                    try:
                        cpu_samples.append(proc.cpu_percent())
                    except:
                        pass
                return await make_request(session, prefix)
        
        tasks = [
            bounded_request(SAMPLE_PREFIXES[i % len(SAMPLE_PREFIXES)])
            for i in range(num_requests)
        ]
        
        results = await asyncio.gather(*tasks)
        
        total_time = time.perf_counter() - start_time
        
        for latency, error in results:
            if error:
                errors += 1
            else:
                latencies.append(latency * 1000)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    
    if latencies:
        print(f"\nLatency (ms):")
        print(f"  Min:    {min(latencies):>8.1f}")
        print(f"  Max:    {max(latencies):>8.1f}")
        print(f"  Mean:   {statistics.mean(latencies):>8.1f}")
        print(f"  Median: {statistics.median(latencies):>8.1f}")
        print(f"  Stdev:  {statistics.stdev(latencies):>8.1f}" if len(latencies) > 1 else "")
        print(f"  p95:    {sorted(latencies)[int(len(latencies)*0.95)]:>8.1f}")
        print(f"  p99:    {sorted(latencies)[int(len(latencies)*0.99)]:>8.1f}")
    
    print(f"\nThroughput:")
    print(f"  Total time:     {total_time:.2f}s")
    print(f"  Requests/sec:   {num_requests/total_time:.2f}")
    print(f"  Successful:     {len(latencies)}/{num_requests}")
    print(f"  Errors:         {errors}")
    
    if cpu_samples:
        print(f"\nServer CPU (%):")
        print(f"  Mean:   {statistics.mean(cpu_samples):>8.1f}")
        print(f"  Max:    {max(cpu_samples):>8.1f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-requests", type=int, default=50)
    parser.add_argument("-c", "--concurrency", type=int, default=5)
    args = parser.parse_args()
    
    asyncio.run(benchmark(args.num_requests, args.concurrency))
