# FedZK Benchmarking System

The FedZK benchmarking system provides tools for evaluating the performance of the zero-knowledge proof generation and verification process in various scenarios. This document explains how to use the benchmarking system and interpret its results.

## Overview

The benchmark simulates a complete federated learning workflow:

1. A coordinator server is launched in a background process
2. Multiple client instances are created and run in parallel
3. Each client generates random gradients (simulating model training)
4. Clients generate zero-knowledge proofs for their gradients
5. Clients submit proofs to the coordinator
6. Performance metrics are collected and reported

## Running Benchmarks

### Via CLI

The simplest way to run a benchmark is through the CLI:

```bash
# Basic benchmark with 5 clients
python -m fedzk.cli benchmark run --clients 5

# With secure circuit (constrained proofs)
python -m fedzk.cli benchmark run --clients 5 --secure

# Save results to files
python -m fedzk.cli benchmark run --clients 10 --output report.json --csv report.csv

# Using MPC server for remote proof generation
python -m fedzk.cli benchmark run --clients 5 --mpc-server http://localhost:9000 --api-key your_api_key
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--clients`, `-c` | Number of clients to simulate | 5 |
| `--secure`, `-s` | Use secure circuit with constraints | False |
| `--mpc-server` | URL of MPC server for remote proof generation | None |
| `--api-key` | API key for MPC server authentication | None |
| `--output`, `-o` | Path to save JSON report | benchmark_report.json |
| `--csv` | Path to save CSV report | None |
| `--report-url` | URL to POST benchmark results | None |
| `--fallback-mode` | How to handle MPC server failures (strict/warn/silent) | warn |
| `--input-size` | Size of gradient tensors for benchmarking | 10 |
| `--coordinator-host` | Hostname for coordinator server | 127.0.0.1 |
| `--coordinator-port` | Port for coordinator server | 8000 |

### Via API

For more controlled benchmarking, you can use the API directly:

```python
from fedzk.benchmark.end_to_end import run_benchmark

# Basic benchmark
results = run_benchmark(num_clients=5, secure=False)

# With secure circuit and MPC server
results = run_benchmark(
    num_clients=10,
    secure=True,
    mpc_server="http://localhost:9000",
    output_json="secure_benchmark.json",
    output_csv="secure_report.csv"
)

# Post results to a dashboard/data warehouse
results = run_benchmark(
    num_clients=5,
    secure=True,
    report_url="https://metrics.example.com/api/fedzk"
)
```

## Benchmark Reports

### Console Output

During benchmark execution, the system displays a progress bar followed by two tables:

1. **Client Results Table**: Shows per-client metrics including training time, proof generation time, submission time, and status
2. **Summary Table**: Shows aggregate statistics such as success rate, average times, and overall duration

Example output:

```
Running FedZK end-to-end benchmark with 5 clients
Mode: Secure ZK circuit
Running client simulations... [############] 100%

┏━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Client ┃ Train(s) ┃ Proof(s) ┃ Submit(s) ┃ Total(s) ┃ Status    ┃
┡━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ #0     │ 0.0048   │ 0.1236   │ 0.0152   │ 0.1437   │ accepted   │
│ #1     │ 0.0042   │ 0.1154   │ 0.0124   │ 0.1320   │ accepted   │
│ #2     │ 0.0051   │ 0.1253   │ 0.0063   │ 0.1367   │ aggregated │
│ #3     │ 0.0045   │ 0.1198   │ 0.0098   │ 0.1341   │ accepted   │
│ #4     │ 0.0047   │ 0.1215   │ 0.0105   │ 0.1367   │ accepted   │
└────────┴──────────┴──────────┴──────────┴──────────┴────────────┘

┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Metric                 ┃ Value           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ Total Clients          │ 5               │
│ Successful Clients     │ 5 (100%)        │
│ Clients Triggering     │ 1               │
│ Aggregation            │                 │
│ Circuit Type           │ Secure          │
│ MPC Server Used        │ No              │
│ Avg. Training Time     │ 0.0047s         │
│ Avg. Proof Time        │ 0.1211s         │
│ Avg. Submission Time   │ 0.0108s         │
│ Avg. Total Client Time │ 0.1366s         │
│ Total Benchmark        │ 0.2485s         │
│ Duration               │                 │
└──────────────────────┴─────────────────┘

Report saved to benchmark_report.json
```

### JSON Report

The JSON report provides detailed metrics for analysis and comparison. It includes:

1. **Metadata**: Timestamp, configuration, and benchmark ID
2. **Configuration**: Settings used for the benchmark
3. **Summary**: Aggregate statistics
4. **Client Metrics**: Detailed per-client performance data

Example JSON structure:

```json
{
  "benchmarkId": "45c6df2a-98f1-4dbc-ba97-1234567890ab",
  "config": {
    "clients": 5,
    "secure": true,
    "batchSize": 128
  },
  "timestamp": "2025-04-24T12:34:56.789Z",
  "summary": {
    "total_duration": 0.2485,
    "successful_clients": 5,
    "aggregated_updates": 1,
    "avg_train_time": 0.0047,
    "avg_proof_time": 0.1211,
    "avg_submit_time": 0.0108,
    "avg_total_time": 0.1366
  },
  "client_metrics": [
    {
      "clientId": 0,
      "metrics": {
        "trainingTime": 0.0048,
        "proofTime": 0.1236,
        "verificationTime": 0.0152,
        "totalTime": 0.1437
      },
      "status": "accepted",
      "timestamp": "2025-04-24T12:34:56.123Z",
      "succeeded": true
    },
    {
      "clientId": 1,
      "metrics": {
        "trainingTime": 0.0042,
        "proofTime": 0.1154,
        "verificationTime": 0.0124,
        "totalTime": 0.1320
      },
      "status": "accepted",
      "timestamp": "2025-04-24T12:34:56.123Z",
      "succeeded": true
    },
    {
      "clientId": 2,
      "metrics": {
        "trainingTime": 0.0051,
        "proofTime": 0.1253,
        "verificationTime": 0.0063,
        "totalTime": 0.1367
      },
      "status": "aggregated",
      "timestamp": "2025-04-24T12:34:56.123Z",
      "succeeded": true
    },
    {
      "clientId": 3,
      "metrics": {
        "trainingTime": 0.0045,
        "proofTime": 0.1198,
        "verificationTime": 0.0098,
        "totalTime": 0.1341
      },
      "status": "accepted",
      "timestamp": "2025-04-24T12:34:56.123Z",
      "succeeded": true
    },
    {
      "clientId": 4,
      "metrics": {
        "trainingTime": 0.0047,
        "proofTime": 0.1215,
        "verificationTime": 0.0105,
        "totalTime": 0.1367
      },
      "status": "accepted",
      "timestamp": "2025-04-24T12:34:56.123Z",
      "succeeded": true
    }
  ]
}
```

### CSV Report

For simpler analysis or importing into spreadsheet applications, a CSV report can be generated with one row per client:

```csv
client_id,train_time,proof_time,submit_time,total_time,status,succeeded,model_version
0,0.0048,0.1236,0.0152,0.1437,accepted,True,1
1,0.0042,0.1154,0.0124,0.1320,accepted,True,1
2,0.0051,0.1253,0.0063,0.1367,aggregated,True,2
3,0.0045,0.1198,0.0098,0.1341,accepted,True,2
4,0.0047,0.1215,0.0105,0.1367,accepted,True,2
```

## Benchmark Analysis

### Key Metrics

The benchmarking system measures several key performance indicators:

1. **Training Time**: Time spent generating random gradients (simulates model training)
2. **Proof Time**: Time spent generating zero-knowledge proofs
3. **Submission Time**: Time spent sending updates to the coordinator
4. **Total Client Time**: Combined time for a client to complete all operations
5. **Total Benchmark Duration**: Wall-clock time for the entire benchmark

### Comparing Configurations

The benchmarking system allows comparing different configurations:

1. **Secure vs. Standard**: Compare performance with and without security constraints
2. **Local vs. MPC**: Compare local proof generation vs. remote MPC-based generation
3. **Scaling**: Analyze how performance changes with increasing client counts
4. **Input Size**: Evaluate the impact of gradient size on performance

### Integration with Monitoring Systems

For continuous performance monitoring, benchmark results can be automatically submitted to an external service by using the `--report-url` option. This allows:

1. Tracking performance across code changes
2. Maintaining historical performance data
3. Creating dashboards for visualization
4. Setting up alerts for performance regressions

## Extended Usage

### Custom Client Behavior

For more advanced benchmarks, you can create custom client classes:

```python
from fedzk.benchmark.end_to_end import Client

class CustomClient(Client):
    def _generate_gradients(self):
        # Custom gradient generation logic
        return {"weights": [1.0, 2.0, 3.0], "bias": [0.5]}
    
    def _generate_proof(self, gradients):
        # Custom proof generation logic
        return {"proof": "custom"}, {"public": [1, 2, 3]}

# Use custom client in benchmark
benchmark = FedZKBenchmark(client_class=CustomClient)
results = benchmark.run()
```

### Fallback Behavior

When using an MPC server, you can configure how the benchmark handles server failures:

- **strict**: Fail immediately if the MPC server is unavailable
- **warn**: Log a warning and fall back to local proof generation
- **silent**: Silently fall back to local proof generation without logging

```bash
# Strict mode (no fallback)
python -m fedzk.cli benchmark run --mpc-server http://localhost:9000 --fallback-mode strict

# With warning and fallback
python -m fedzk.cli benchmark run --mpc-server http://localhost:9000 --fallback-mode warn
``` 
 
 
 
 