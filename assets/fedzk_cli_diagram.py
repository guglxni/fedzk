import graphviz


def create_cli_diagram():
    # Create directed graph
    dot = graphviz.Digraph(comment="FedZK CLI Command Structure")
    dot.attr(rankdir="LR")  # Left to right layout
    dot.attr("node", shape="box", style="filled", fillcolor="lightblue", fontname="Arial")
    dot.attr("edge", fontname="Arial")

    # Main command
    dot.node("fedzk", "fedzk", shape="ellipse", fillcolor="#ADD8E6")

    # First level commands
    commands = ["setup", "generate", "verify", "mpc", "benchmark"]
    for cmd in commands:
        dot.node(cmd, cmd)
        dot.edge("fedzk", cmd)

    # MPC subcommands
    dot.node("mpc_serve", "serve")
    dot.edge("mpc", "mpc_serve")

    # Benchmark subcommands
    dot.node("benchmark_run", "run")
    dot.edge("benchmark", "benchmark_run")

    # Add options for each command
    # Generate command options
    generate_options = [
        "--input, -i", "--output, -o", "--secure, -s", "--batch, -b",
        "--chunk-size, -c", "--max-norm, -m", "--min-active, -a",
        "--mpc-server", "--api-key", "--fallback-disabled",
        "--fallback-mode"
    ]
    for i, opt in enumerate(generate_options):
        dot.node(f"gen_opt_{i}", opt, shape="note", fillcolor="#E6E6FA")
        dot.edge("generate", f"gen_opt_{i}", style="dashed")

    # Verify command options
    verify_options = [
        "--input, -i", "--secure, -s", "--batch, -b",
        "--mpc-server", "--api-key"
    ]
    for i, opt in enumerate(verify_options):
        dot.node(f"verify_opt_{i}", opt, shape="note", fillcolor="#E6E6FA")
        dot.edge("verify", f"verify_opt_{i}", style="dashed")

    # MPC server options
    mpc_options = ["--host, -H", "--port, -p"]
    for i, opt in enumerate(mpc_options):
        dot.node(f"mpc_opt_{i}", opt, shape="note", fillcolor="#E6E6FA")
        dot.edge("mpc_serve", f"mpc_opt_{i}", style="dashed")

    # Benchmark options
    benchmark_options = [
        "--clients, -c", "--secure, -s", "--mpc-server",
        "--output, -o", "--csv", "--report-url",
        "--fallback-mode", "--input-size",
        "--coordinator-host", "--coordinator-port"
    ]
    for i, opt in enumerate(benchmark_options):
        dot.node(f"bench_opt_{i}", opt, shape="note", fillcolor="#E6E6FA")
        dot.edge("benchmark_run", f"bench_opt_{i}", style="dashed")

    # Set filename and format
    dot.attr(label="FedZK CLI Command Structure", fontsize="20")
    return dot

if __name__ == "__main__":
    # Create the diagram
    dot = create_cli_diagram()

    # Render the diagram to a file
    dot.render("fedzk_cli_diagram", format="png", cleanup=True)
    print("CLI diagram created: fedzk_cli_diagram.png")
