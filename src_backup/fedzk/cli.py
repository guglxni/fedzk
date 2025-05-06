"""Command-line interface for FedZK."""

import typer
from rich import print

app = typer.Typer()
client_app = typer.Typer()
benchmark_app = typer.Typer()

app.add_typer(client_app, name="client")
app.add_typer(benchmark_app, name="benchmark")


@client_app.command("train")
def client_train():
    """Train a model locally."""
    print("[bold green]Training model locally...[/bold green]")
    # Mock implementation for now
    print("[bold green]Training complete![/bold green]")


@client_app.command("prove")
def client_prove():
    """Generate zero-knowledge proof for model updates."""
    print("[bold green]Generating zero-knowledge proof...[/bold green]")
    # Mock implementation for now
    print("[bold green]Proof generation complete![/bold green]")


@benchmark_app.command("run")
def benchmark_run(
    dataset: str = typer.Option(None, help="Dataset to use for benchmarking"),
    iterations: int = typer.Option(3, help="Number of iterations to run"),
):
    """Run benchmarks."""
    print(f"[bold blue]Running benchmarks with {iterations} iterations...[/bold blue]")
    if dataset:
        print(f"Using dataset: {dataset}")


def main():
    """Run the CLI application."""
    app()


if __name__ == "__main__":
    main()
