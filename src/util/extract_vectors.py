"""
src/util/extract_vectors.py — Extract the first N vectors from an fbin file

fbin format (Microsoft DiskANN standard):
  [4B int32 npts][4B int32 dim][npts × dim × float32]

Purpose: quickly slice a subset out of a large dataset (e.g. sift100m or
spacev100m) for small-scale offline experiments or validation, without
copying the whole file.

Usage:
  python extract_vectors.py input.fbin output.fbin --n_vectors 10000
"""

import struct
import numpy as np
import argparse
from pathlib import Path

def extract_fbin(input_path, output_path, n_vectors):
    print(f"Reading from {input_path}...")
    with open(input_path, 'rb') as f:
        npts = struct.unpack('<i', f.read(4))[0]
        dim = struct.unpack('<i', f.read(4))[0]
        
        print(f"  Source: npts={npts}, dim={dim}")
        
        if n_vectors > npts:
            print(f"  Warning: Requested {n_vectors} vectors but file only has {npts}. Extracting all.")
            n_vectors = npts
            
        # Vectors are float32
        data_size = n_vectors * dim * 4
        f.seek(8) # Skip header
        data = f.read(data_size)
        
    print(f"Writing first {n_vectors} vectors to {output_path}...")
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<i', n_vectors))
        f.write(struct.pack('<i', dim))
        f.write(data)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract first N vectors from fbin")
    parser.add_argument("input", help="Input .fbin file")
    parser.add_argument("output", help="Output .fbin file")
    parser.add_argument("n", type=int, help="Number of vectors to extract")
    args = parser.parse_args()
    
    extract_fbin(args.input, args.output, args.n)
