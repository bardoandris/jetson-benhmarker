import json as _json
import typing as _typing


def main():
    """
    This is the main function for the Jetson Benchmarker application.
    It initializes the application and starts the benchmarking processes.
    """
    # Initialize the application
    print("Initializing Jetson Benchmarker...")

    # Start the benchmarking processes
    print("Starting benchmarking processes...")
    import libraries.transformer.transformer_test as transformer_test
    