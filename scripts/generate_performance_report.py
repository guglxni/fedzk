#!/usr/bin/env python3
"""
Performance Report Generator for FEDZK

This script generates HTML performance reports from pytest benchmark JSON results.
Used in the CI/CD pipeline to create performance analysis reports.
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import statistics


def load_benchmark_data(json_file: Path) -> Dict[str, Any]:
    """Load benchmark data from pytest JSON file."""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Benchmark file {json_file} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_file}: {e}")
        sys.exit(1)


def analyze_benchmarks(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze benchmark data and extract key metrics."""
    if 'benchmarks' not in data:
        return {"error": "No benchmarks found in data"}
    
    benchmarks = data['benchmarks']
    analysis = {
        "total_tests": len(benchmarks),
        "tests": [],
        "summary": {
            "fastest_test": None,
            "slowest_test": None,
            "average_time": 0,
            "total_time": 0
        }
    }
    
    times = []
    
    for benchmark in benchmarks:
        test_info = {
            "name": benchmark.get("name", "Unknown"),
            "group": benchmark.get("group", "default"),
            "mean_time": benchmark.get("stats", {}).get("mean", 0),
            "min_time": benchmark.get("stats", {}).get("min", 0),
            "max_time": benchmark.get("stats", {}).get("max", 0),
            "stddev": benchmark.get("stats", {}).get("stddev", 0),
            "rounds": benchmark.get("stats", {}).get("rounds", 0),
            "iterations": benchmark.get("stats", {}).get("iterations", 0)
        }
        
        analysis["tests"].append(test_info)
        times.append(test_info["mean_time"])
    
    if times:
        analysis["summary"]["average_time"] = statistics.mean(times)
        analysis["summary"]["total_time"] = sum(times)
        
        # Find fastest and slowest tests
        fastest_idx = times.index(min(times))
        slowest_idx = times.index(max(times))
        
        analysis["summary"]["fastest_test"] = analysis["tests"][fastest_idx]
        analysis["summary"]["slowest_test"] = analysis["tests"][slowest_idx]
    
    return analysis


def generate_html_report(analysis: Dict[str, Any], output_file: Path) -> None:
    """Generate HTML performance report."""
    
    if "error" in analysis:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FEDZK Performance Report - Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .error {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>FEDZK Performance Report</h1>
            <div class="error">Error: {analysis['error']}</div>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </body>
        </html>
        """
    else:
        # Generate test rows
        test_rows = ""
        for test in analysis["tests"]:
            test_rows += f"""
            <tr>
                <td>{test['name']}</td>
                <td>{test['group']}</td>
                <td>{test['mean_time']:.6f}s</td>
                <td>{test['min_time']:.6f}s</td>
                <td>{test['max_time']:.6f}s</td>
                <td>{test['stddev']:.6f}s</td>
                <td>{test['rounds']}</td>
                <td>{test['iterations']}</td>
            </tr>
            """
        
        summary = analysis["summary"]
        fastest = summary.get("fastest_test", {})
        slowest = summary.get("slowest_test", {})
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FEDZK Performance Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1, h2 {{
                    color: #333;
                    border-bottom: 2px solid #4CAF50;
                    padding-bottom: 10px;
                }}
                .summary {{
                    background-color: #f9f9f9;
                    padding: 20px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px 20px 10px 0;
                    padding: 10px;
                    background-color: #e8f5e8;
                    border-radius: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .highlight {{
                    font-weight: bold;
                    color: #4CAF50;
                }}
                .footer {{
                    margin-top: 30px;
                    text-align: center;
                    color: #666;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 FEDZK Performance Report</h1>
                
                <div class="summary">
                    <h2>📊 Summary</h2>
                    <div class="metric">
                        <strong>Total Tests:</strong> {analysis['total_tests']}
                    </div>
                    <div class="metric">
                        <strong>Average Time:</strong> {summary['average_time']:.6f}s
                    </div>
                    <div class="metric">
                        <strong>Total Time:</strong> {summary['total_time']:.6f}s
                    </div>
                    
                    {f'''
                    <div style="margin-top: 20px;">
                        <div class="metric">
                            <strong>⚡ Fastest Test:</strong> {fastest.get('name', 'N/A')} 
                            ({fastest.get('mean_time', 0):.6f}s)
                        </div>
                        <div class="metric">
                            <strong>🐌 Slowest Test:</strong> {slowest.get('name', 'N/A')} 
                            ({slowest.get('mean_time', 0):.6f}s)
                        </div>
                    </div>
                    ''' if fastest and slowest else ''}
                </div>
                
                <h2>📋 Detailed Results</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Test Name</th>
                            <th>Group</th>
                            <th>Mean Time</th>
                            <th>Min Time</th>
                            <th>Max Time</th>
                            <th>Std Dev</th>
                            <th>Rounds</th>
                            <th>Iterations</th>
                        </tr>
                    </thead>
                    <tbody>
                        {test_rows}
                    </tbody>
                </table>
                
                <div class="footer">
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>FEDZK Performance Analysis | Enterprise-Grade Federated Learning</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    try:
        with open(output_file, 'w') as f:
            f.write(html_content)
        print(f"Performance report generated: {output_file}")
    except IOError as e:
        print(f"Error writing report to {output_file}: {e}")
        sys.exit(1)


def main():
    """Main function to generate performance report."""
    parser = argparse.ArgumentParser(
        description="Generate HTML performance report from pytest benchmark JSON"
    )
    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to pytest benchmark JSON file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default="performance-report.html",
        help="Output HTML file path (default: performance-report.html)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Loading benchmark data from: {args.json_file}")
        print(f"Output report will be saved to: {args.output}")
    
    # Load and analyze benchmark data
    data = load_benchmark_data(args.json_file)
    analysis = analyze_benchmarks(data)
    
    if args.verbose:
        if "error" not in analysis:
            print(f"Analyzed {analysis['total_tests']} benchmark tests")
            print(f"Average execution time: {analysis['summary']['average_time']:.6f}s")
        else:
            print(f"Analysis error: {analysis['error']}")
    
    # Generate HTML report
    generate_html_report(analysis, args.output)
    
    if args.verbose:
        print("Performance report generation completed successfully!")


if __name__ == "__main__":
    main()
