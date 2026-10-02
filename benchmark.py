import subprocess
import matplotlib.pyplot as plt

implementations = {
    "matmul_1": "Naive Kernel",
    "matmul_2": "Tiled Shared Memory Kernel"
}

# The N sizes: 2^4 to 2^13 (16 to 8192). 
# We skip 1, 2, 4, 8 because launch overhead dominates and ruins the GFLOPS metric
sizes = [2**i for i in range(4, 14)]

plt.figure(figsize=(10, 6))

for exe, label in implementations.items():
    source = f"{exe}.cu"
    gflops_list = []
    valid_sizes = []
    
    print(f"\nCompiling {source}...")
    try:
        subprocess.run(["nvcc", "-arch=native", "-o", exe, source], check=True)
    except subprocess.CalledProcessError:
        print(f"Failed to compile {source}, skipping.")
        continue

    print(f"Running benchmarks for {label}...")
    for size in sizes:
            
        print(f"  Testing {size}x{size}...", end="", flush=True)
        try:
            result = subprocess.run([f"./{exe}", str(size)], capture_output=True, text=True, check=True)
            time_ms = float(result.stdout.strip())
            
            # Calculate GFLOPS
            # Total operations = 2 * N^3 (N^3 multiplies + N^3 additions)
            # GFLOPS = Operations / (Time in seconds * 10^9)
            operations = 2 * (size ** 3)
            time_s = time_ms / 1000.0
            gflops = operations / (time_s * 1e9)
            
            gflops_list.append(gflops)
            valid_sizes.append(size)
            print(f" {time_ms:.2f} ms | {gflops:.2f} GFLOPS")
            
        except subprocess.CalledProcessError:
            print(f" Crash/Error!")
            break

    # Plot GFLOPS (Y) vs Matrix Size (X)
    plt.plot(valid_sizes, gflops_list, marker='o', linestyle='-', label=label)

# X-axis remains log2 so the sizes spread out evenly
plt.xscale('log', base=2)
# Y-axis is linear to easily compare throughput heights
plt.yscale('linear')

plt.title('CUDA Matrix Multiplication Throughput (Higher is Better)')
plt.xlabel('Matrix Size (N) [Log2 Scale]')
plt.ylabel('Performance (GFLOPS) [Linear Scale]')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.savefig('matmul_gflops.png')
print("\nPlot saved as matmul_gflops.png")