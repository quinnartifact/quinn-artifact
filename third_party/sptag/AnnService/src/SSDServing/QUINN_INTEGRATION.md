# QUINN Integration for SPANN

本文檔說明如何整合 QUINN 的 per-query budget 功能到 SPANN/SSDServing。

## 修改總結

### 1. 新增文件

**budget_shm.h** (`inc/SSDServing/budget_shm.h`)
- 從 QUINN Controller 複製過來
- 提供 `BudgetShmReader` 類用於讀取 shared memory 中的 per-query budgets
- 使用 POSIX shared memory (`/dev/shm`)

### 2. 修改的文件

#### 2.1 main.cpp (`src/SSDServing/main.cpp`)

**修改內容**:
- 添加 `--budget_shm <shm_name>` 命令行參數解析
- 將 budget_shm_name 存入 config map 以傳遞給後續流程

**修改位置**: `main()` 函數

**新增用法**:
```bash
./ssdserving config.ini --budget_shm /quinn_budget_12345
```

#### 2.2 SSDIndex.h (`inc/SSDServing/SSDIndex.h`)

**修改內容**:

1. **包含頭文件** (Line 16):
   ```cpp
   #include "inc/SSDServing/budget_shm.h"
   ```

2. **SearchSequential 函數** (Lines 103-184):
   - 添加可選參數 `quinn::BudgetShmReader* p_budgetReader = nullptr`
   - 在 query loop 中讀取 per-query nprobe (`entry.bS`)
   - 記錄前幾個 query 的 nprobe 值以供驗證

3. **Search 函數** (Lines 186+):
   - 從 config 讀取 `BudgetShmName`
   - 創建 `BudgetShmReader` 實例
   - 為每個 query 根據 shared memory 中的 budget 創建相應大小的 `QueryResult`
   - 驗證 query 數量是否匹配
   - 將 `budgetReader` 傳遞給 `SearchSequential`

## 工作原理

### 整體流程

```
┌─────────────────────────────────────────────────────────────┐
│              QUINN Controller (Python)                      │
│  1. 使用 Allocator 預測 per-query budgets (bS, bD)          │
│  2. 將 budgets 寫入 shared memory: /quinn_budget_<pid>     │
│  3. 啟動 SPANN: ./ssdserving config.ini --budget_shm <name> │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              SPANN SSDServing (C++)                          │
│  1. 解析 --budget_shm 參數                                   │
│  2. 打開 shared memory 並驗證 header                         │
│  3. 讀取 query vectors                                       │
│  4. 為每個 query 創建 QueryResult (大小 = max(K, bS))       │
│  5. 執行搜尋 (每個 query 使用各自的 nprobe)                 │
└─────────────────────────────────────────────────────────────┘
```

### Shared Memory 格式

```
[Header: 16 bytes]
  - magic: 0x43415341 ('CASA')
  - version: 1
  - num_queries: N
  - entry_size: 4

[Entries: N × 4 bytes]
  Entry[i]:
    - bS: uint16 (SPANN nprobe, 0-200)
    - bD: uint16 (DiskANN L, 0-200)  // SPANN 只使用 bS
```

### Per-Query Nprobe 實現

在原始實現中，所有 queries 使用相同的 `m_searchInternalResultNum`（nprobe）：
```cpp
// 原始代碼
std::vector<QueryResult> results(numQueries,
    QueryResult(NULL, max(K, internalResultNum), false));
```

在修改後的實現中，每個 query 使用各自的 nprobe：
```cpp
// 修改後的代碼
std::vector<QueryResult> results;
results.reserve(numQueries);
for (int i = 0; i < numQueries; ++i) {
    int query_internal_result = internalResultNum;  // 預設值
    if (budgetReader != nullptr) {
        query_internal_result = budgetReader->get(i).bS;  // 從 shm 讀取
    }
    results.emplace_back(nullptr, max(K, query_internal_result), false);
}
```

`QueryResult` 的第二個參數決定了搜尋時會使用的 candidate 數量，也就是 nprobe。

## 編譯

在 SPANN 專案的根目錄：

```bash
cd <SPTAG_SRC>
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

編譯產物：
- `./Release/ssdserving` - SPANN 搜尋執行檔

## 使用範例

### 1. 使用 QUINN Controller 啟動

```bash
# 使用 QUINN Controller（會自動創建 shared memory 並啟動 SPANN）
cd <REPO_ROOT>/src/controller

python controller.py \
  --model_dir ../../model/deep100m \
  --query_file /path/to/queries.fbin \
  --centroid_file /path/to/centroids.bin \
  --target_recall 90 \
  --spann_bin <SPTAG_SRC>/build/Release/ssdserving \
  --spann_args "config.ini" \
  --output_dir ./output
```

### 2. 手動測試（不使用 Controller）

```bash
# Step 1: 創建測試用的 shared memory（使用 Python）
python3 << 'EOF'
import numpy as np
from controller import BudgetShmWriter

# 創建測試 budgets（10 個 queries）
num_queries = 10
budgets = np.array([
    [20, 40],  # query 0: nprobe=20
    [30, 50],  # query 1: nprobe=30
    [40, 60],  # query 2: nprobe=40
    [50, 70],  # query 3: nprobe=50
    [60, 80],  # query 4: nprobe=60
    [70, 90],  # query 5: nprobe=70
    [80, 100], # query 6: nprobe=80
    [90, 110], # query 7: nprobe=90
    [100, 120],# query 8: nprobe=100
    [110, 130] # query 9: nprobe=110
], dtype=np.uint16)

shm_name = "/quinn_test_manual"
with BudgetShmWriter(shm_name, budgets) as writer:
    print(f"Created shared memory: {shm_name}")
    input("Press Enter to continue (keep this running)...")
EOF

# Step 2: 在另一個終端執行 SPANN
./build/Release/ssdserving config.ini --budget_shm /quinn_test_manual
```

### 3. 驗證 Per-Query Nprobe

查看 SPANN 的輸出日誌，應該看到類似：

```
[INFO] Loaded per-query budgets from shared memory: /quinn_budget_12345 (10000 queries)
[INFO] Start loading QuerySet...
[INFO] Query 0: nprobe=20 (from shm)
[INFO] Query 1: nprobe=30 (from shm)
[INFO] Query 2: nprobe=40 (from shm)
...
[INFO] Using per-query budgets from shared memory (nprobe will vary per query).
[INFO] Searching: numThread: 32, numQueries: 10000.
```

## 錯誤處理

整合代碼包含完整的錯誤處理：

1. **Shared Memory 不存在**:
   ```
   [ERROR] Failed to open budget shared memory '/quinn_budget_xxx': No such file or directory
   [ERROR] Falling back to default internalResultNum=64
   ```
   → 自動回退到使用 config 中的預設 nprobe

2. **Query 數量不匹配**:
   ```
   [ERROR] Budget shared memory query count mismatch: expected 10000, got 5000. Falling back to default nprobe.
   ```
   → 停用 budget reader，使用預設值

3. **Shared Memory 格式錯誤**:
   ```
   [ERROR] Invalid magic number
   ```
   → BudgetShmReader 會拋出異常並被捕獲

## 性能考量

### Per-Query Budget 的開銷

1. **記憶體開銷**:
   - Shared memory: 16 bytes (header) + N × 4 bytes (entries)
   - 例如 10K queries: ~40 KB
   - QueryResult 初始化：每個 query 根據各自的 nprobe 分配記憶體

2. **計算開銷**:
   - Shared memory 讀取: O(1) per query
   - 額外的記憶體分配: O(N) for N queries（一次性）
   - 可忽略不計的性能影響

3. **搜尋效能**:
   - 使用較小 nprobe 的 query 會更快
   - 使用較大 nprobe 的 query 會更慢
   - 整體 QPS 取決於 nprobe 分佈

### 優化建議

- 對於 warmup queries，可以考慮不使用 per-query budgets（使用固定值）
- 確保 shared memory 在所有 queries 執行完前不被刪除
- 在多次執行時可以重用同一個 shared memory（如果 budgets 不變）

## Limitations 和未來改進

### 當前限制

1. **QueryResult 大小限制**:
   - QueryResult 在創建時就固定了大小（max(K, nprobe)）
   - 搜尋過程中不能動態改變
   - 這是 SPANN 架構的限制

2. **Warmup 與 Main Search**:
   - Warmup 和 main search 都會使用相同的 budget shared memory
   - 如果 warmup queries 數量與 main queries 不同，可能需要調整

### 未來改進方向

1. **動態 QueryResult 大小**:
   - 修改 SPANN 核心代碼以支持在搜尋時動態設置 nprobe
   - 需要深入修改 `SearchIndex` 和相關函數

2. **更精細的控制**:
   - 支持 per-query 的其他參數（如 max distance ratio）
   - 支持動態調整 posting list 讀取策略

3. **性能監控**:
   - 記錄每個 query 實際使用的 nprobe
   - 輸出 per-query 性能統計

## Troubleshooting

### 問題 1: 編譯錯誤 "budget_shm.h: No such file or directory"

**解決方法**:
```bash
# 確認檔案存在
ls -la <SPTAG_SRC>/AnnService/inc/SSDServing/budget_shm.h

# 如果不存在，重新複製
cp <REPO_ROOT>/src/controller/budget_shm.h \
   <SPTAG_SRC>/AnnService/inc/SSDServing/
```

### 問題 2: 執行時找不到 shared memory

**解決方法**:
```bash
# 檢查 shared memory 是否存在
ls -la /dev/shm/ | grep quinn

# 檢查 SPANN 執行的參數
./ssdserving config.ini --budget_shm /quinn_budget_12345  # 確保名稱正確
```

### 問題 3: 所有 queries 仍使用相同的 nprobe

**可能原因**:
- 沒有提供 `--budget_shm` 參數
- Shared memory 名稱不正確
- Budget reader 初始化失敗（檢查日誌）

**檢查方法**:
- 查看日誌中是否有 "Loaded per-query budgets from shared memory"
- 查看日誌中是否有 "Query X: nprobe=Y (from shm)"

## 測試建議

### 基本功能測試

1. **無 Shared Memory（預設行為）**:
   ```bash
   ./ssdserving config.ini
   ```
   → 應該使用 config 中的預設 nprobe

2. **使用 Shared Memory**:
   ```bash
   ./ssdserving config.ini --budget_shm /quinn_budget_test
   ```
   → 應該顯示 "Loaded per-query budgets from shared memory"

3. **錯誤的 Shared Memory 名稱**:
   ```bash
   ./ssdserving config.ini --budget_shm /nonexistent
   ```
   → 應該顯示錯誤並回退到預設值

### Recall 測試

比較使用固定 nprobe 和 per-query nprobe 的 recall：

```bash
# 固定 nprobe=64
./ssdserving config.ini

# Per-query nprobe (平均=64)
./ssdserving config.ini --budget_shm /quinn_budget_xxx
```

預期：
- Per-query 版本的 recall 應該類似或更好（如果 budgets 分配合理）
- 整體 I/O 和 latency 可能減少

## 參考資料

- QUINN Controller 文檔: `<REPO_ROOT>/src/controller/README.md`
- QUINN Controller 設計: `<REPO_ROOT>/doc/controller.md`
- Budget Shared Memory 格式: `budget_shm.h`

---

**整合完成日期**: 2026-01-07
**修改者**: Claude Sonnet 4.5
**狀態**: 已完成並可測試
