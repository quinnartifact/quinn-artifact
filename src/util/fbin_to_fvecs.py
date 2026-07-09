"""
src/util/fbin_to_fvecs.py — Format conversion: fbin → fvecs

fbin  format (DiskANN): [4B npts][4B dim][npts × dim × float32]
fvecs format (FAISS)  : each vector prefixed with 4B dim, followed by dim × float32

Purpose: convert a DiskANN-format dataset into fvecs format that FAISS/SPTAG
can read directly.

Usage:
  python fbin_to_fvecs.py input.fbin output.fvecs
"""

import struct
import numpy as np
import argparse

def fbin_to_fvecs(input_path, output_path):
    print(f"Reading from {input_path}...")
    with open(input_path, 'rb') as f:
        npts = struct.unpack('<i', f.read(4))[0]
        dim = struct.unpack('<i', f.read(4))[0]
        data = f.read()
        vectors = np.frombuffer(data, dtype='<f4')
    
    vectors = vectors.reshape(npts, dim)
    print(f"  Shape: {vectors.shape}")
    
    print(f"Writing to {output_path}...")
    with open(output_path, 'wb') as f:
        for i in range(npts):
            f.write(struct.pack('<i', dim))
            f.write(vectors[i].tobytes())
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert fbin to fvecs")
    parser.add_argument("input", help="Input .fbin file")
    parser.add_argument("output", help="Output .fvecs file")
    args = parser.parse_args()
    
    fbin_to_fvecs(args.input, args.output)
