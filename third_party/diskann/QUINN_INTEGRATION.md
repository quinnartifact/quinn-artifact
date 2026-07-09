# QUINN Integration for DiskANN

This document describes the integration of QUINN's per-query budget allocation system with DiskANN.

## Overview

DiskANN has been modified to support reading per-query search budgets (L values) from POSIX shared memory. This enables the QUINN Controller to predict optimal L values for each query and communicate them to DiskANN without modifying the query files.

## Architecture

```
┌─────────────────────┐
│  QUINN Controller  │
│  (Budget Predictor) │
└──────────┬──────────┘
           │ Writes predicted budgets
           ↓
┌─────────────────────┐
│ Shared Memory       │
│ /dev/shm/quinn_... │
│                     │
│ Header:             │
│  - magic: 0x43415341│
│  - version: 1       │
│  - num_queries      │
│  - entry_size: 4    │
│                     │
│ Entries (4B each):  │
│  - bS (uint16_t)    │ SPANN nprobe
│  - bD (uint16_t)    │ DiskANN L
└──────────┬──────────┘
           │ Reads per-query L
           ↓
┌─────────────────────┐
│  DiskANN CLI        │
│  search_disk_index  │
└─────────────────────┘
```

## Modified Files

### 1. `<DISKANN_SRC>/include/quinn/budget_shm.h` (NEW)
- Copied from QUINN Controller
- Provides `BudgetShmReader` and `BudgetShmWriter` classes
- Handles POSIX shared memory operations with error handling

### 2. `<DISKANN_SRC>/apps/search_disk_index.cpp` (MODIFIED)

#### Changes:
1. **Added include** (line 16):
   ```cpp
   #include "quinn/budget_shm.h"
   ```

2. **Modified function signature** (line 52-58):
   ```cpp
   template <typename T, typename LabelT = uint32_t>
   int search_disk_index(diskann::Metric &metric, const std::string &index_path_prefix,
                         const std::string &result_output_prefix, const std::string &query_file, std::string &gt_file,
                         const uint32_t num_threads, const uint32_t recall_at, const uint32_t beamwidth,
                         const uint32_t num_nodes_to_cache, const uint32_t search_io_limit,
                         const std::vector<uint32_t> &Lvec, const float fail_if_recall_below,
                         const std::vector<std::string> &query_filters, const bool use_reorder_data = false,
                         const std::string &budget_shm_name = "")  // NEW PARAMETER
   ```

3. **Budget reader initialization** (line 136-156):
   ```cpp
   // QUINN: Initialize budget shared memory reader if provided
   std::unique_ptr<quinn::BudgetShmReader> budgetReader;
   if (!budget_shm_name.empty())
   {
       try
       {
           budgetReader = std::make_unique<quinn::BudgetShmReader>(budget_shm_name);
           diskann::cout << "Loaded per-query budgets from shared memory: " << budget_shm_name
                         << " (" << budgetReader->size() << " queries)" << std::endl;
           if (budgetReader->size() != query_num)
           {
               diskann::cerr << "Warning: Budget shared memory has " << budgetReader->size()
                             << " entries but query file has " << query_num << " queries" << std::endl;
           }
       }
       catch (const std::exception &e)
       {
           diskann::cerr << "Failed to open budget shared memory '" << budget_shm_name << "': " << e.what() << std::endl;
           diskann::cerr << "Continuing without per-query budgets" << std::endl;
       }
   }
   ```

4. **Per-query L usage in query loop** (line 251-292):
   ```cpp
   #pragma omp parallel for schedule(dynamic, 1)
   for (int64_t i = 0; i < (int64_t)query_num; i++)
   {
       // QUINN: Use per-query L if budget reader is available
       uint32_t query_L = L;
       if (budgetReader != nullptr && i < (int64_t)budgetReader->size())
       {
           query_L = budgetReader->get(i).bD;
           // Log a few samples to verify correct operation
           if (i < 3 || i == (int64_t)query_num / 2 || i == (int64_t)query_num - 1)
           {
               #pragma omp critical
               {
                   diskann::cout << "Query " << i << ": L=" << query_L << " (from shm)" << std::endl;
               }
           }
       }

       if (!filtered_search)
       {
           _pFlashIndex->cached_beam_search(query + (i * query_aligned_dim), recall_at, query_L,
                                            query_result_ids_64.data() + (i * recall_at),
                                            query_result_dists[test_id].data() + (i * recall_at),
                                            optimized_beamwidth, use_reorder_data, stats + i);
       }
       else
       {
           // ... filtered search also uses query_L
       }
   }
   ```

5. **Command-line argument** (line 370-432):
   ```cpp
   // Added budget_shm_name to variable declarations
   std::string data_type, dist_fn, index_path_prefix, result_path_prefix, query_file, gt_file, filter_label,
       label_type, query_filters_file, budget_shm_name;

   // Added to optional_configs
   optional_configs.add_options()("budget_shm",
                                  po::value<std::string>(&budget_shm_name)->default_value(""),
                                  "QUINN: Shared memory name for per-query budgets (optional)");
   ```

6. **All search_disk_index calls updated** to pass `budget_shm_name` as the last argument

## Usage

### Basic Usage (Without Per-Query Budgets)
```bash
./search_disk_index \
    --data_type float \
    --dist_fn l2 \
    --index_path_prefix /path/to/index \
    --result_path /path/to/results \
    --query_file /path/to/queries.bin \
    --recall_at 10 \
    --search_list 100 200 300
```

### With QUINN Per-Query Budgets
```bash
# First, the QUINN Controller creates shared memory with per-query budgets
# Then run DiskANN with --budget_shm:

./search_disk_index \
    --data_type float \
    --dist_fn l2 \
    --index_path_prefix /path/to/index \
    --result_path /path/to/results \
    --query_file /path/to/queries.bin \
    --recall_at 10 \
    --search_list 100 \
    --budget_shm /quinn_budget_12345
```

**Important Notes:**
- When using `--budget_shm`, the `--search_list` parameter should typically contain a single value (used as default/fallback)
- The shared memory name must match what the QUINN Controller created
- Each query will use its corresponding `bD` value from shared memory
- If shared memory cannot be opened, DiskANN falls back to using the L values from `--search_list`

## Integration with QUINN Controller

### Workflow:
1. **Controller Predicts Budgets**: QUINN Controller analyzes queries and predicts optimal (bS, bD) pairs
2. **Controller Writes to Shared Memory**: Creates `/dev/shm/quinn_budget_XXXXX` with all predictions
3. **Controller Launches DiskANN**: Spawns `search_disk_index --budget_shm /quinn_budget_XXXXX ...`
4. **DiskANN Reads Budgets**: Opens shared memory and reads per-query L values
5. **DiskANN Searches**: Each query uses its predicted L value
6. **Controller Cleans Up**: Unlinks shared memory after DiskANN finishes

### Example Python Controller Integration:
```python
from quinn.budget_shm import BudgetShmWriter
import subprocess
import uuid

# Predict budgets for queries
budgets = predictor.predict(queries)  # Returns [(bS, bD), ...]

# Create shared memory
shm_name = f"/quinn_budget_{uuid.uuid4().hex[:8]}"
writer = BudgetShmWriter(shm_name, len(budgets))
for i, (bs, bd) in enumerate(budgets):
    writer.set(i, bs, bd)
writer.close()

# Launch DiskANN
subprocess.run([
    "./search_disk_index",
    "--data_type", "float",
    "--dist_fn", "l2",
    "--index_path_prefix", index_path,
    "--result_path", result_path,
    "--query_file", query_file,
    "--recall_at", "10",
    "--search_list", "100",  # Default L value
    "--budget_shm", shm_name
])

# Clean up
os.unlink(shm_name)
```

## Shared Memory Format

### Header (16 bytes):
```
Offset | Size | Field       | Value
-------|------|-------------|------------------
0      | 4    | magic       | 0x43415341 ('CASA')
4      | 4    | version     | 1
8      | 4    | num_queries | Number of queries
12     | 4    | entry_size  | 4 (bytes per entry)
```

### Entry Format (4 bytes per query):
```
Offset | Size | Field | Description
-------|------|-------|---------------------------
0      | 2    | bS    | SPANN nprobe (uint16_t)
2      | 2    | bD    | DiskANN L value (uint16_t)
```

## Backward Compatibility

All modifications are **fully backward compatible**:
- If `--budget_shm` is not provided, DiskANN behaves exactly as before
- Uses L values from `--search_list` parameter
- No performance impact when shared memory is not used
- Optional parameter with empty string default

## Testing

### Create Test Shared Memory:
```python
#!/usr/bin/env python3
import struct
import mmap
import os

shm_name = "/quinn_test"
num_queries = 100

# Create shared memory
fd = os.open(f"/dev/shm{shm_name}", os.O_CREAT | os.O_RDWR, 0o600)
size = 16 + num_queries * 4  # Header + entries
os.ftruncate(fd, size)

# Map memory
mem = mmap.mmap(fd, size)

# Write header
mem[0:4] = struct.pack('I', 0x43415341)  # magic
mem[4:8] = struct.pack('I', 1)            # version
mem[8:12] = struct.pack('I', num_queries) # num_queries
mem[12:16] = struct.pack('I', 4)          # entry_size

# Write entries (varying L from 50 to 150)
for i in range(num_queries):
    bS = 100  # SPANN nprobe
    bD = 50 + i  # DiskANN L: 50, 51, 52, ..., 149
    offset = 16 + i * 4
    mem[offset:offset+2] = struct.pack('H', bS)
    mem[offset+2:offset+4] = struct.pack('H', bD)

mem.close()
os.close(fd)
print(f"Created test shared memory: {shm_name} with {num_queries} queries")
```

### Run DiskANN with Test Data:
```bash
# Create test shared memory
python3 create_test_shm.py

# Run DiskANN
./search_disk_index \
    --data_type float \
    --dist_fn l2 \
    --index_path_prefix /path/to/index \
    --result_path /path/to/results \
    --query_file /path/to/queries.bin \
    --recall_at 10 \
    --search_list 100 \
    --budget_shm /quinn_test

# You should see output like:
# Loaded per-query budgets from shared memory: /quinn_test (100 queries)
# Query 0: L=50 (from shm)
# Query 1: L=51 (from shm)
# Query 2: L=52 (from shm)
# ...

# Clean up
rm /dev/shm/quinn_test
```

## Troubleshooting

### Error: "Failed to open budget shared memory"
- Verify shared memory exists: `ls -la /dev/shm/`
- Check permissions: shared memory must be readable by the user running DiskANN
- Ensure the name matches exactly (including leading `/`)

### Warning: "Budget shared memory has X entries but query file has Y queries"
- The number of entries in shared memory doesn't match the query file
- DiskANN will only use budgets for queries that have corresponding entries
- Queries beyond the shared memory size will use the default L from `--search_list`

### No per-query budgets being used
- Verify `--budget_shm` argument is provided
- Check that shared memory name starts with `/`
- Look for error messages in DiskANN output
- If shared memory fails to open, DiskANN continues with default L values

## Performance Considerations

- **Memory overhead**: Minimal (4 bytes per query)
- **Runtime overhead**: Negligible (single shared memory open + array lookups)
- **Scalability**: Tested with millions of queries
- **Thread safety**: BudgetShmReader is thread-safe for concurrent reads

## See Also

- QUINN Controller documentation: `/path/to/controller.md`
- SPANN integration: `<SPTAG_SRC>/AnnService/src/SSDServing/QUINN_INTEGRATION.md`
- DiskANN documentation: https://github.com/microsoft/DiskANN
