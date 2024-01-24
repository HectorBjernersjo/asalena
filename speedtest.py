import trace
import threading
import time
import functools
import sys

# Global variable to signal when to stop the script
stop_script = False

# Decorator to measure execution time of a function
def time_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if stop_script:
            sys.exit(0)  # Exit if stop signal is received
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.6f}s")
        return result
    return wrapper

def trace_script(script_path):
    tracer = trace.Trace(
        trace=0,
        count=1
    )
    tracer.run(f'exec(open("{script_path}").read())')
    return tracer.results()

def trace_and_time(script_path):
    global stop_script
    trace_thread = threading.Thread(target=trace_script, args=(script_path,))
    trace_thread.start()

    print("Press 'q' to stop the script and print statistics.")

    while not stop_script:
        if input() == 'q':
            stop_script = True
            print("\nStopping script and printing statistics...")

    trace_thread.join()  # Wait for the tracing thread to finish

    # Get and print trace results
    results = trace_script(script_path)
    results.write_results(show_missing=True, summary=True, coverdir=".")

# Example usage
trace_and_time('test.py')

