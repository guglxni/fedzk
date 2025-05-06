# Copyright (c) 2025 Aaryan Guglani and FedZK Contributors
# SPDX-License-Identifier: MIT

"""
Benchmark script for ZK proof generation and verification.

This script benchmarks the performance of the ZK proof system across
different input sizes and configurations.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to allow running script directly
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from fedzk.benchmark import BenchmarkResults, benchmark, generate_random_gradients
from fedzk.prover.verifier import ZKVerifier
from fedzk.prover.zkgenerator import ZKProver


class ZKBenchmark:
    """Benchmark suite for ZK proof generation and verification."""

    def __init__(self,
                output_dir: str = "benchmark_results",
                zk_dir: str = "zk"):
        """
        Initialize ZK benchmarking suite.
        
        Args:
            output_dir: Directory to store benchmark results
            zk_dir: Directory containing ZK circuits and keys
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.zk_dir = Path(zk_dir)
        if not self.zk_dir.exists():
            raise ValueError(f"ZK directory {zk_dir} not found. Run setup_zk.sh first.")

        # Standard circuit files
        self.standard_circuit = self.zk_dir / "model_update.wasm"
        self.standard_proving_key = self.zk_dir / "proving_key.zkey"
        self.standard_verification_key = self.zk_dir / "verification_key.json"

        # Secure circuit files
        self.secure_circuit = self.zk_dir / "model_update_secure.wasm"
        self.secure_proving_key = self.zk_dir / "proving_key_secure.zkey"
        self.secure_verification_key = self.zk_dir / "verification_key_secure.json"

        # Check if files exist
        self._check_files_exist()

        # Create benchmark results container
        self.results = BenchmarkResults("zk_benchmark")

    def _check_files_exist(self):
        """Check if required ZK files exist, raising error if not."""
        missing_files = []

        for file_path in [
            self.standard_circuit, self.standard_proving_key, self.standard_verification_key,
            self.secure_circuit, self.secure_proving_key, self.secure_verification_key
        ]:
            if not file_path.exists():
                missing_files.append(str(file_path))

        if missing_files:
            raise FileNotFoundError(
                f"Missing ZK files: {', '.join(missing_files)}. "
                "Run setup_zk.sh first to generate these files."
            )

    @benchmark(operation_name="standard_proof_generation")
    def benchmark_standard_proof_generation(self, gradient_dict: dict, max_inputs: int = 4):
        """
        Benchmark standard proof generation.
        
        Args:
            gradient_dict: Dictionary of gradients to prove
            max_inputs: Maximum number of gradient values to use
            
        Returns:
            Generated proof and public inputs
        """
        prover = ZKProver(str(self.standard_circuit), str(self.standard_proving_key))
        return prover.generate_real_proof(gradient_dict, max_inputs=max_inputs)

    @benchmark(operation_name="standard_proof_verification")
    def benchmark_standard_proof_verification(self, proof: dict, public_inputs: list):
        """
        Benchmark standard proof verification.
        
        Args:
            proof: The proof to verify
            public_inputs: Public inputs for the proof
            
        Returns:
            Verification result (True/False)
        """
        verifier = ZKVerifier(str(self.standard_verification_key))
        return verifier.verify_real_proof(proof, public_inputs)

    @benchmark(operation_name="secure_proof_generation")
    def benchmark_secure_proof_generation(self, gradient_dict: dict,
                                        max_inputs: int = 4,
                                        max_norm: float = 100.0,
                                        min_active: int = 3):
        """
        Benchmark secure proof generation.
        
        Args:
            gradient_dict: Dictionary of gradients to prove
            max_inputs: Maximum number of gradient values to use
            max_norm: Maximum L2 norm allowed
            min_active: Minimum number of non-zero elements required
            
        Returns:
            Generated proof and public inputs
        """
        prover = ZKProver(str(self.secure_circuit), str(self.secure_proving_key))
        return prover.generate_real_proof_secure(
            gradient_dict,
            max_inputs=max_inputs,
            max_norm=max_norm,
            min_active=min_active
        )

    @benchmark(operation_name="secure_proof_verification")
    def benchmark_secure_proof_verification(self, proof: dict, public_inputs: list):
        """
        Benchmark secure proof verification.
        
        Args:
            proof: The proof to verify
            public_inputs: Public inputs for the proof
            
        Returns:
            Verification result (True/False)
        """
        verifier = ZKVerifier(str(self.secure_verification_key))
        return verifier.verify_real_proof_secure(proof, public_inputs)

    def run_all_benchmarks(self,
                          model_sizes: list = [(1, 4), (10, 10), (100, 10)],
                          num_iterations: int = 5,
                          secure_params: dict = None):
        """
        Run all benchmarks with different model sizes.
        
        Args:
            model_sizes: List of (num_params, param_size) tuples to test
            num_iterations: Number of iterations for each configuration
            secure_params: Parameters for secure circuit (max_norm, min_active)
        """
        if secure_params is None:
            secure_params = {"max_norm": 100.0, "min_active": 3}

        print(f"Running ZK benchmarks with {num_iterations} iterations per configuration")

        for num_params, param_size in model_sizes:
            print(f"\nBenchmarking with {num_params} parameters of size {param_size}")

            # Create test gradient dictionary
            shape_dict = {f"param_{i}": (param_size,) for i in range(num_params)}

            for i in range(num_iterations):
                # Generate random gradients
                gradients = generate_random_gradients(shape_dict, scale=0.5)

                # Standard proof benchmarks
                try:
                    proof, public_inputs = self.benchmark_standard_proof_generation(
                        self.results,
                        gradient_dict=gradients,
                        max_inputs=4
                    )

                    self.benchmark_standard_proof_verification(
                        self.results,
                        proof=proof,
                        public_inputs=public_inputs
                    )
                except Exception as e:
                    print(f"Error in standard proof benchmarking: {e}")

                # Secure proof benchmarks
                try:
                    proof, public_inputs = self.benchmark_secure_proof_generation(
                        self.results,
                        gradient_dict=gradients,
                        max_inputs=4,
                        max_norm=secure_params["max_norm"],
                        min_active=secure_params["min_active"]
                    )

                    self.benchmark_secure_proof_verification(
                        self.results,
                        proof=proof,
                        public_inputs=public_inputs["public_inputs"]
                    )
                except Exception as e:
                    print(f"Error in secure proof benchmarking: {e}")

        # Print summary
        self.results.print_summary()

        # Save results
        output_path = self.results.save_to_file(self.output_dir)
        print(f"Benchmark results saved to: {output_path}")


def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark ZK proof generation and verification")
    parser.add_argument("--output-dir", default="benchmark_results",
                      help="Directory to store benchmark results")
    parser.add_argument("--zk-dir", default="zk",
                      help="Directory containing ZK circuits and keys")
    parser.add_argument("--iterations", type=int, default=3,
                      help="Number of iterations for each configuration")

    args = parser.parse_args()

    try:
        benchmark = ZKBenchmark(output_dir=args.output_dir, zk_dir=args.zk_dir)
        benchmark.run_all_benchmarks(num_iterations=args.iterations)
    except Exception as e:
        print(f"Error running benchmarks: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
