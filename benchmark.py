import subprocess
import matplotlib.pyplot as plt

# The N sizes for our square matrices
sizes = [2**i for i in range(15)]
times = []

# 1. Compile the CUDA code
print("Compiling matmul_1.cu...")
subprocess.run(["nvcc", "-o", "matmul_1", "matmul_1.cu"], check=True)

# 2. Run the benchmarks
print("Running benchmarks...")
for size in sizes:
    print(f"Testing {size}x{size}...")
    
    # Run the executable and capture standard output
    result = subprocess.run(["./matmul_1", str(size)], capture_output=True, text=True, check=True)
    
    # Parse the milliseconds printed by the C++ script
    time_ms = float(result.stdout.strip())
    times.append(time_ms)


# 2. Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sizes, times, marker='o', linestyle='-', color='b', label='Naive CUDA Kernel')

plt.title('Naive CUDA Matrix Multiplication Scaling')
plt.xlabel('Matrix Size (N)')
plt.ylabel('Execution Time (ms)')
plt.legend()
plt.grid(True)
plt.savefig('matmul_1_scaling.png')
print("Plot saved as matmul_1_scaling.png")