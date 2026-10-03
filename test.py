import os
import subprocess
import sys
import numpy as np

kernels = ["matmul_0", "matmul_1", "matmul_2", "matmul_3", "matmul_4"]
sizes = [64, 128, 256, 1024]  # kernels 3 and 4 need multiples of 64

ok = True
for exe in kernels:
    subprocess.run(["nvcc", "-arch=native", "-o", exe, f"{exe}.cu"], check=True)
    for n in sizes:
        subprocess.run([f"./{exe}", str(n), "out.bin"], check=True, capture_output=True)
        A, B, C = np.fromfile("out.bin", dtype=np.float32).reshape(3, n, n)
        ref = A.astype(np.float64) @ B.astype(np.float64)
        err = np.abs(C - ref).max() / np.abs(ref).max()
        passed = err < 1e-4
        ok &= passed
        print(f"{exe} {n:5d}: {'PASS' if passed else 'FAIL'} (max rel err {err:.1e})")
    os.remove(exe)

os.remove("out.bin")
sys.exit(0 if ok else 1)