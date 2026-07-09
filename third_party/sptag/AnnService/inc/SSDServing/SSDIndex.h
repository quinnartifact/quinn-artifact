// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <chrono>
#include <iostream>
#include <limits>
#include <map>
#include <string>

#include "inc/Core/Common.h"
#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/SPANN/ExtraFullGraphSearcher.h"
#include "inc/Helper/VectorSetReader.h"
#include "inc/Helper/StringConvert.h"
#include "inc/SSDServing/Utils.h"
#include "io_trace.h"
#include "inc/SSDServing/budget_shm.h"
#include "inc/SSDServing/early_exit_shm.h"
#include "inc/quinn/thread_count_shm.h"
#include "inc/quinn/latency_shm.h"
#include <thread>

namespace SPTAG {
	namespace SSDServing {
		namespace SSDIndex {

            template <typename ValueType>
            ErrorCode OutputResult(const std::string& p_output, std::vector<QueryResult>& p_results, int p_resultNum)
            {
                if (!p_output.empty())
                {
                    auto ptr = f_createIO();
                    if (ptr == nullptr || !ptr->Initialize(p_output.c_str(), std::ios::binary | std::ios::out)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed create file: %s\n", p_output.c_str());
                        return ErrorCode::FailedCreateFile;
                    }
                    int32_t i32Val = static_cast<int32_t>(p_results.size());
                    if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                        LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                        return ErrorCode::DiskIOFail;
                    }
                    i32Val = p_resultNum;
                    if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                        LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                        return ErrorCode::DiskIOFail;
                    }

                    float fVal = 0;
                    for (size_t i = 0; i < p_results.size(); ++i)
                    {
                        for (int j = 0; j < p_resultNum; ++j)
                        {
                            i32Val = p_results[i].GetResult(j)->VID;
                            if (ptr->WriteBinary(sizeof(i32Val), reinterpret_cast<char*>(&i32Val)) != sizeof(i32Val)) {
                                LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                                return ErrorCode::DiskIOFail;
                            }

                            fVal = p_results[i].GetResult(j)->Dist;
                            if (ptr->WriteBinary(sizeof(fVal), reinterpret_cast<char*>(&fVal)) != sizeof(fVal)) {
                                LOG(Helper::LogLevel::LL_Error, "Fail to write result file!\n");
                                return ErrorCode::DiskIOFail;
                            }
                        }
                    }
                }
                return ErrorCode::Success;
            }

            template<typename T, typename V>
            void PrintPercentiles(const std::vector<V>& p_values, std::function<T(const V&)> p_get, const char* p_format)
            {
                double sum = 0;
                std::vector<T> collects;
                collects.reserve(p_values.size());
                for (const auto& v : p_values)
                {
                    T tmp = p_get(v);
                    sum += tmp;
                    collects.push_back(tmp);
                }

                std::sort(collects.begin(), collects.end());

                LOG(Helper::LogLevel::LL_Info, "Avg\t50tiles\t90tiles\t95tiles\t99tiles\t99.9tiles\tMax\n");

                std::string formatStr("%.3lf");
                for (int i = 1; i < 7; ++i)
                {
                    formatStr += '\t';
                    formatStr += p_format;
                }

                formatStr += '\n';

                LOG(Helper::LogLevel::LL_Info,
                    formatStr.c_str(),
                    sum / collects.size(),
                    collects[static_cast<size_t>(collects.size() * 0.50)],
                    collects[static_cast<size_t>(collects.size() * 0.90)],
                    collects[static_cast<size_t>(collects.size() * 0.95)],
                    collects[static_cast<size_t>(collects.size() * 0.99)],
                    collects[static_cast<size_t>(collects.size() * 0.999)],
                    collects[static_cast<size_t>(collects.size() - 1)]);
            }


            template <typename ValueType>
            void SearchSequential(SPANN::Index<ValueType>* p_index,
                int p_numThreads,
                std::vector<QueryResult>& p_results,
                std::vector<SPANN::SearchStats>& p_stats,
                int p_maxQueryCount, int p_internalResultNum,
                quinn::BudgetShmReader* p_budgetReader = nullptr,
                quinn::EarlyExitShmAccessor* p_earlyExitAccessor = nullptr,
                int p_earlyExitKRef = 0,
                quinn::ThreadCountShmReader* p_threadCountShm = nullptr,
                quinn::LatencyShmAccessor* p_latencyShm = nullptr)
            {
                int numQueries = min(static_cast<int>(p_results.size()), p_maxQueryCount);

                std::atomic_size_t queriesSent(0);
                std::atomic<int> active_spann_count{0};

                std::vector<std::thread> threads;

                LOG(Helper::LogLevel::LL_Info, "Searching: numThread: %d, numQueries: %d.\n", p_numThreads, numQueries);
                if (p_budgetReader != nullptr) {
                    LOG(Helper::LogLevel::LL_Info, "Using per-query budgets from shared memory (nprobe will vary per query).\n");
                }

                Utils::StopW sw;

                auto func = [&]()
                {
                    Utils::StopW threadws;
                    size_t index = 0;
                    while (true)
                    {
                        // QUINN: Dynamic throttling — wait if over allowed thread count
                        if (p_threadCountShm) {
                            int allowed = p_threadCountShm->get_thread_s();
                            while (active_spann_count.load(std::memory_order_acquire) >= allowed) {
                                std::this_thread::sleep_for(std::chrono::microseconds(50));
                                allowed = p_threadCountShm->get_thread_s();
                            }
                            active_spann_count.fetch_add(1, std::memory_order_release);
                        }

                        index = queriesSent.fetch_add(1);
                        if (index < numQueries)
                        {
                            if ((index & ((1 << 14) - 1)) == 0)
                            {
                                LOG(Helper::LogLevel::LL_Info, "Sent %.2lf%%...\n", index * 100.0 / numQueries);
                            }

                            // Get per-query nprobe from shared memory if available
                            int query_nprobe = p_internalResultNum;
                            if (p_budgetReader != nullptr) {
                                quinn::BudgetEntry entry = p_budgetReader->get(index);
                                query_nprobe = entry.bS;

                                // Log first few queries for verification
                                if (index < 5) {
                                    LOG(Helper::LogLevel::LL_Info, "Query %zu: using nprobe=%d (bS from shm)\n",
                                        index, query_nprobe);
                                }
                            }

                            double startTime = threadws.getElapsedMs();
                            int64_t t_start_ns = quinn::LatencyShmAccessor::now_ns();

                            // Note: QueryResult size was already set during initialization
                            // The actual nprobe used is controlled by the QueryResult size
                            // For full per-query support, we would need to modify QueryResult
                            // For now, we log the intended value and use the pre-set size
                            p_index->GetMemoryIndex()->SearchIndex(p_results[index]);
                            double endTime = threadws.getElapsedMs();
                            p_index->SearchDiskIndex(p_results[index], &(p_stats[index]));
                            double exEndTime = threadws.getElapsedMs();
                            int64_t t_end_ns = quinn::LatencyShmAccessor::now_ns();

                            if (p_latencyShm) {
                                p_latencyShm->record_spann_start(static_cast<uint32_t>(index), t_start_ns);
                                p_latencyShm->record_spann_end  (static_cast<uint32_t>(index), t_end_ns);
                            }

                            // QUINN: Update early exit shared memory with k-th distance + topk IDs
                            if (p_earlyExitAccessor != nullptr) {
                                int num_results = p_results[index].GetResultNum();
                                float kth_dist = std::numeric_limits<float>::max();
                                int k_idx = p_earlyExitKRef - 1;
                                if (k_idx >= 0 && k_idx < num_results &&
                                    p_results[index].GetResult(k_idx)->VID >= 0) {
                                    kth_dist = p_results[index].GetResult(k_idx)->Dist;
                                }

                                uint32_t topk_k = p_earlyExitAccessor->topk_k();
                                if (topk_k > 0) {
                                    // Collect VIDs into a temporary buffer
                                    std::vector<int64_t> vids(topk_k, -1);
                                    for (uint32_t j = 0; j < topk_k && j < (uint32_t)num_results; j++) {
                                        vids[j] = p_results[index].GetResult(j)->VID;
                                    }
                                    // Write IDs then dist (release fence on dist)
                                    p_earlyExitAccessor->update_topk(index, vids.data(),
                                                                     (uint32_t)vids.size(), kth_dist);
                                } else {
                                    p_earlyExitAccessor->update(index, kth_dist);
                                }
                            }

                            p_stats[index].m_exLatency = exEndTime - endTime;
                            p_stats[index].m_totalLatency = p_stats[index].m_totalSearchLatency = exEndTime - startTime;
                            if (p_threadCountShm) active_spann_count.fetch_sub(1, std::memory_order_release);
                        }
                        else
                        {
                            if (p_threadCountShm) active_spann_count.fetch_sub(1, std::memory_order_release);
                            io_trace::flush_thread();
                            return;
                        }
                    }
                };

                for (int i = 0; i < p_numThreads; i++) { threads.emplace_back(func); }
                for (auto& thread : threads) { thread.join(); }

                double sendingCost = sw.getElapsedSec();

                LOG(Helper::LogLevel::LL_Info,
                    "Finish sending in %.3lf seconds, actuallQPS is %.2lf, query count %u.\n",
                    sendingCost,
                    numQueries / sendingCost,
                    static_cast<uint32_t>(numQueries));

                for (int i = 0; i < numQueries; i++) { p_results[i].CleanQuantizedTarget(); }
            }

            template <typename ValueType>
            void Search(SPANN::Index<ValueType>* p_index)
            {
                SPANN::Options& p_opts = *(p_index->GetOptions());
                std::string outputFile = p_opts.m_searchResult;
                std::string truthFile = p_opts.m_truthPath;
                std::string warmupFile = p_opts.m_warmupPath;

                if (p_index->m_pQuantizer)
                {
                   p_index->m_pQuantizer->SetEnableADC(p_opts.m_enableADC);
                }

                if (!p_opts.m_logFile.empty())
                {
                    g_pLogger.reset(new Helper::FileLogger(Helper::LogLevel::LL_Info, p_opts.m_logFile.c_str()));
                }
                int numThreads = p_opts.m_iSSDNumberOfThreads;
                int internalResultNum = p_opts.m_searchInternalResultNum;
                int K = p_opts.m_resultNum;
                int truthK = (p_opts.m_truthResultNum <= 0) ? K : p_opts.m_truthResultNum;

                // Check for budget shared memory configuration from options
                std::string budget_shm_name = p_opts.m_budgetShmName;

                // Initialize budget reader if shared memory is provided
                std::unique_ptr<quinn::BudgetShmReader> budgetReader;
                if (!budget_shm_name.empty()) {
                    try {
                        budgetReader = std::make_unique<quinn::BudgetShmReader>(budget_shm_name);
                        LOG(Helper::LogLevel::LL_Info, "Loaded per-query budgets from shared memory: %s (%zu queries)\n",
                            budget_shm_name.c_str(), budgetReader->size());
                    } catch (const std::exception& e) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to open budget shared memory '%s': %s\n",
                            budget_shm_name.c_str(), e.what());
                        LOG(Helper::LogLevel::LL_Error, "Falling back to default internalResultNum=%d\n", internalResultNum);
                    }
                }



                if (!warmupFile.empty())
                {
                    LOG(Helper::LogLevel::LL_Info, "Start loading warmup query set...\n");
                    std::shared_ptr<Helper::ReaderOptions> queryOptions(new Helper::ReaderOptions(p_opts.m_valueType, p_opts.m_dim, p_opts.m_warmupType, p_opts.m_warmupDelimiter));
                    auto queryReader = Helper::VectorSetReader::CreateInstance(queryOptions);
                    if (ErrorCode::Success != queryReader->LoadFile(p_opts.m_warmupPath))
                    {
                        LOG(Helper::LogLevel::LL_Error, "Failed to read query file.\n");
                        exit(1);
                    }
                    auto warmupQuerySet = queryReader->GetVectorSet();
                    int warmupNumQueries = warmupQuerySet->Count();

                    // Create warmup results - use per-query nprobe if available
                    std::vector<QueryResult> warmupResults;
                    warmupResults.reserve(warmupNumQueries);
                    std::vector<SPANN::SearchStats> warmpUpStats(warmupNumQueries);
                    for (int i = 0; i < warmupNumQueries; ++i)
                    {
                        int query_internal_result = internalResultNum;
                        if (budgetReader != nullptr && i < budgetReader->size()) {
                            query_internal_result = budgetReader->get(i).bS;
                        }
                        int queryResultSize = max(K, query_internal_result);
                        warmupResults.emplace_back(nullptr, queryResultSize, false);
                        warmupResults[i].SetSearchBudget(query_internal_result);  // Set per-query search budget
                        (*((COMMON::QueryResultSet<ValueType>*)&warmupResults[i])).SetTarget(reinterpret_cast<ValueType*>(warmupQuerySet->GetVector(i)), p_index->m_pQuantizer);
                        warmupResults[i].Reset();
                    }

                    LOG(Helper::LogLevel::LL_Info, "Start warmup...\n");
                    SearchSequential(p_index, numThreads, warmupResults, warmpUpStats, p_opts.m_queryCountLimit, internalResultNum, budgetReader.get(), nullptr, 0); // No early exit for warmup
                    LOG(Helper::LogLevel::LL_Info, "\nFinish warmup...\n");
                }

                LOG(Helper::LogLevel::LL_Info, "Start loading QuerySet...\n");
                std::shared_ptr<Helper::ReaderOptions> queryOptions(new Helper::ReaderOptions(p_opts.m_valueType, p_opts.m_dim, p_opts.m_queryType, p_opts.m_queryDelimiter));
                auto queryReader = Helper::VectorSetReader::CreateInstance(queryOptions);
                if (ErrorCode::Success != queryReader->LoadFile(p_opts.m_queryPath))
                {
                    LOG(Helper::LogLevel::LL_Error, "Failed to read query file.\n");
                    exit(1);
                }
                auto querySet = queryReader->GetVectorSet();
                int numQueries = querySet->Count();

                // Initialize Early Exit SHM Accessor if provided
                std::unique_ptr<quinn::EarlyExitShmAccessor> earlyExitAccessor;
                if (!p_opts.m_earlyExitShm.empty()) {
                    try {
                        earlyExitAccessor = std::make_unique<quinn::EarlyExitShmAccessor>(p_opts.m_earlyExitShm, numQueries);
                        LOG(Helper::LogLevel::LL_Info, "Enabled Early Exit updates to SHM: %s (k_ref=%d)\n", 
                            p_opts.m_earlyExitShm.c_str(), p_opts.m_earlyExitKRef);
                    } catch (const std::exception& e) {
                        LOG(Helper::LogLevel::LL_Error, "Failed to open Early Exit SHM '%s': %s\n", 
                            p_opts.m_earlyExitShm.c_str(), e.what());
                    }
                }

                // QUINN: Thread count dynamic control
                std::unique_ptr<quinn::ThreadCountShmReader> threadCountShm;
                if (!p_opts.m_threadCountShmName.empty()) {
                    try {
                        threadCountShm = std::make_unique<quinn::ThreadCountShmReader>(p_opts.m_threadCountShmName);
                        LOG(Helper::LogLevel::LL_Info, "Dynamic thread control enabled via: %s\n", p_opts.m_threadCountShmName.c_str());
                    } catch (const std::exception& e) {
                        LOG(Helper::LogLevel::LL_Warning, "Failed to open ThreadCountShm: %s\n", e.what());
                    }
                }

                // QUINN: Per-query latency timestamps
                std::unique_ptr<quinn::LatencyShmAccessor> latencyShm;
                if (!p_opts.m_latencyShmName.empty()) {
                    try {
                        latencyShm = std::make_unique<quinn::LatencyShmAccessor>(
                            p_opts.m_latencyShmName, static_cast<uint32_t>(numQueries));
                        LOG(Helper::LogLevel::LL_Info, "Latency SHM opened: %s\n", p_opts.m_latencyShmName.c_str());
                    } catch (const std::exception& e) {
                        LOG(Helper::LogLevel::LL_Warning, "Failed to open LatencySHM '%s': %s\n",
                            p_opts.m_latencyShmName.c_str(), e.what());
                    }
                }

                // Verify budget reader has correct number of queries
                if (budgetReader != nullptr && budgetReader->size() != static_cast<size_t>(numQueries)) {
                    LOG(Helper::LogLevel::LL_Error,
                        "Budget shared memory query count mismatch: expected %d, got %zu. Falling back to default nprobe.\n",
                        numQueries, budgetReader->size());
                    budgetReader.reset();  // Disable budget reader
                }

                // Create query results - use per-query nprobe if available
                std::vector<QueryResult> results;
                results.reserve(numQueries);
                std::vector<SPANN::SearchStats> stats(numQueries);
                for (int i = 0; i < numQueries; ++i)
                {
                    int query_internal_result = internalResultNum;
                    if (budgetReader != nullptr) {
                        int budget_bS = budgetReader->get(i).bS;
                        if (budget_bS > internalResultNum) {
                             LOG(Helper::LogLevel::LL_Warning, "Query %d: budget %d from shm is larger than max configured internalResultNum %d. Capping to max.\n", i, budget_bS, internalResultNum);
                             query_internal_result = internalResultNum;
                        } else {
                             query_internal_result = budget_bS;
                        }
                    }
                    // QueryResult size: max(K, budget) to ensure enough space for final K results
                    int queryResultSize = max(K, query_internal_result);
                    results.emplace_back(nullptr, queryResultSize, false);
                    results[i].SetSearchBudget(query_internal_result);  // Set per-query search budget
                    (*((COMMON::QueryResultSet<ValueType>*)&results[i])).SetTarget(reinterpret_cast<ValueType*>(querySet->GetVector(i)), p_index->m_pQuantizer);
                    results[i].Reset();
                }

                // QUINN: READY/START protocol for aligned QPS measurement
                // Output READY to stdout and wait for START from stdin
                std::cout << "READY" << std::endl;  // Flush immediately
                std::string start_signal;
                std::getline(std::cin, start_signal);  // Wait for START

                // Start timing for search only
                LOG(Helper::LogLevel::LL_Info, "Start ANN Search...\n");
                auto search_start = std::chrono::high_resolution_clock::now();

                SearchSequential(p_index, numThreads, results, stats, p_opts.m_queryCountLimit, internalResultNum, budgetReader.get(), earlyExitAccessor.get(), p_opts.m_earlyExitKRef, threadCountShm.get(), latencyShm.get());

                auto search_end = std::chrono::high_resolution_clock::now();
                double search_time_ms = std::chrono::duration<double, std::milli>(search_end - search_start).count();

                LOG(Helper::LogLevel::LL_Info, "\nFinish ANN Search...\n");

                // Output search-only time to stdout for controller
                std::cout << "SEARCH_TIME_MS " << search_time_ms << std::endl;

                // QUINN: Flush I/O trace as soon as search finishes
                fprintf(stderr, "[SPANN] Flushing I/O trace...\n");
                io_trace::flush();
                std::cout << "DONE" << std::endl;
                fprintf(stderr, "[SPANN] I/O trace flushed and DONE signaled.\n");

                std::shared_ptr<VectorSet> vectorSet;

                if (!p_opts.m_vectorPath.empty() && fileexists(p_opts.m_vectorPath.c_str())) {
                    std::shared_ptr<Helper::ReaderOptions> vectorOptions(new Helper::ReaderOptions(p_opts.m_valueType, p_opts.m_dim, p_opts.m_vectorType, p_opts.m_vectorDelimiter));
                    auto vectorReader = Helper::VectorSetReader::CreateInstance(vectorOptions);
                    if (ErrorCode::Success == vectorReader->LoadFile(p_opts.m_vectorPath))
                    {
                        vectorSet = vectorReader->GetVectorSet();
                        if (p_opts.m_distCalcMethod == DistCalcMethod::Cosine) vectorSet->Normalize(numThreads);
                        LOG(Helper::LogLevel::LL_Info, "\nLoad VectorSet(%d,%d).\n", vectorSet->Count(), vectorSet->Dimension());
                    }
                }

                if (p_opts.m_rerank > 0 && vectorSet != nullptr) {
                    LOG(Helper::LogLevel::LL_Info, "\n Begin rerank...\n");
                    for (int i = 0; i < results.size(); i++)
                    {
                        for (int j = 0; j < K; j++)
                        {
                            if (results[i].GetResult(j)->VID < 0) continue;
                            results[i].GetResult(j)->Dist = COMMON::DistanceUtils::ComputeDistance((const ValueType*)querySet->GetVector(i),
                                (const ValueType*)vectorSet->GetVector(results[i].GetResult(j)->VID), querySet->Dimension(), p_opts.m_distCalcMethod);
                        }
                        BasicResult* re = results[i].GetResults();
                        std::sort(re, re + K, COMMON::Compare);
                    }
                    K = p_opts.m_rerank;
                }

                float recall = 0, MRR = 0;
                std::vector<std::set<SizeType>> truth;
                if (!truthFile.empty())
                {
                    LOG(Helper::LogLevel::LL_Info, "Start loading TruthFile...\n");

                    auto ptr = f_createIO();
                    if (ptr == nullptr || !ptr->Initialize(truthFile.c_str(), std::ios::in | std::ios::binary)) {
                        LOG(Helper::LogLevel::LL_Error, "Failed open truth file: %s\n", truthFile.c_str());
                        exit(1);
                    }
                    int originalK = truthK;
                    COMMON::TruthSet::LoadTruth(ptr, truth, numQueries, originalK, truthK, p_opts.m_truthType);
                    char tmp[4];
                    if (ptr->ReadBinary(4, tmp) == 4) {
                        LOG(Helper::LogLevel::LL_Error, "Truth number is larger than query number(%d)!\n", numQueries);
                    }

                    recall = COMMON::TruthSet::CalculateRecall<ValueType>((p_index->GetMemoryIndex()).get(), results, truth, K, truthK, querySet, vectorSet, numQueries, nullptr, false, &MRR);
                    LOG(Helper::LogLevel::LL_Info, "Recall%d@%d: %f MRR@%d: %f\n", truthK, K, recall, K, MRR);
                }

                LOG(Helper::LogLevel::LL_Info, "\nEx Elements Count:\n");
                PrintPercentiles<double, SPANN::SearchStats>(stats,
                    [](const SPANN::SearchStats& ss) -> double
                    {
                        return ss.m_totalListElementsCount;
                    },
                    "%.3lf");

                LOG(Helper::LogLevel::LL_Info, "\nHead Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(stats,
                    [](const SPANN::SearchStats& ss) -> double
                    {
                        return ss.m_totalSearchLatency - ss.m_exLatency;
                    },
                    "%.3lf");

                LOG(Helper::LogLevel::LL_Info, "\nEx Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(stats,
                    [](const SPANN::SearchStats& ss) -> double
                    {
                        return ss.m_exLatency;
                    },
                    "%.3lf");

                LOG(Helper::LogLevel::LL_Info, "\nTotal Latency Distribution:\n");
                PrintPercentiles<double, SPANN::SearchStats>(stats,
                    [](const SPANN::SearchStats& ss) -> double
                    {
                        return ss.m_totalSearchLatency;
                    },
                    "%.3lf");

                LOG(Helper::LogLevel::LL_Info, "\nTotal Disk Page Access Distribution:\n");
                PrintPercentiles<int, SPANN::SearchStats>(stats,
                    [](const SPANN::SearchStats& ss) -> int
                    {
                        return ss.m_diskAccessCount;
                    },
                    "%4d");

                LOG(Helper::LogLevel::LL_Info, "\nTotal Disk IO Distribution:\n");
                PrintPercentiles<int, SPANN::SearchStats>(stats,
                    [](const SPANN::SearchStats& ss) -> int
                    {
                        return ss.m_diskIOCount;
                    },
                    "%4d");

                LOG(Helper::LogLevel::LL_Info, "\n");

                if (!outputFile.empty())
                {
                    LOG(Helper::LogLevel::LL_Info, "Start output to %s\n", outputFile.c_str());
                    OutputResult<ValueType>(outputFile, results, K);
                }

                // io_trace::flush() and DONE moved earlier
                
                LOG(Helper::LogLevel::LL_Info,
                    "Recall@%d: %f MRR@%d: %f\n", K, recall, K, MRR);

                // Generate CSV file with per-query statistics
                std::string queryStatsFile = "stat.csv";
                LOG(Helper::LogLevel::LL_Info, "Generating per-query statistics CSV: %s\n", queryStatsFile.c_str());

                auto queryStatsCsvPtr = f_createIO();
                if (queryStatsCsvPtr != nullptr && queryStatsCsvPtr->Initialize(queryStatsFile.c_str(), std::ios::out)) {
                    // Write CSV header
                    std::string header = "qid,total_us,io_us,cpu_us,n_ios,n_4k\n";
                    queryStatsCsvPtr->WriteBinary(header.length(), header.c_str());

                    // Write data for each query
                    // Note: stats are in milliseconds, convert to microseconds for consistency with DiskANN
                    for (int i = 0; i < numQueries; ++i) {
                        double total_us = stats[i].m_totalSearchLatency * 1000.0;  // ms to us
                        double io_us = stats[i].m_exLatency * 1000.0;              // ms to us
                        double cpu_us = total_us - io_us;
                        std::string line = std::to_string(i) + "," +
                                          std::to_string(static_cast<int>(total_us)) + "," +
                                          std::to_string(static_cast<int>(io_us)) + "," +
                                          std::to_string(static_cast<int>(cpu_us)) + "," +
                                          std::to_string(stats[i].m_diskIOCount) + "," +
                                          std::to_string(stats[i].m_diskAccessCount) + "\n";
                        queryStatsCsvPtr->WriteBinary(line.length(), line.c_str());
                    }
                    LOG(Helper::LogLevel::LL_Info, "Per-query statistics CSV generated successfully.\n");
                } else {
                    LOG(Helper::LogLevel::LL_Error, "Failed to create per-query statistics CSV file: %s\n", queryStatsFile.c_str());
                }

                // Generate CSV file with vector statistics
                if (!truthFile.empty()) {
                    std::string csvOutputFile = p_opts.m_searchResult + "_vector_stats.csv";
                    LOG(Helper::LogLevel::LL_Info, "Generating vector statistics CSV: %s\n", csvOutputFile.c_str());

                    // Maps to count occurrences
                    std::map<SizeType, int> groundTruthCount;
                    std::map<SizeType, int> recallCount;

                    // Count ground truth occurrences
                    for (int i = 0; i < numQueries; i++) {
                        for (SizeType vectorId : truth[i]) {
                            groundTruthCount[vectorId]++;
                        }
                    }

                    // Count recall occurrences (intersection of ground truth and search results)
                    for (int i = 0; i < numQueries; i++) {
                        std::set<SizeType> resultSet;
                        // Collect search result vector IDs for this query
                        for (int j = 0; j < K; j++) {
                            if (results[i].GetResult(j)->VID >= 0) {
                                resultSet.insert(results[i].GetResult(j)->VID);
                            }
                        }

                        // Count intersection of truth and results
                        for (SizeType vectorId : truth[i]) {
                            if (resultSet.count(vectorId) > 0) {
                                recallCount[vectorId]++;
                            }
                        }
                    }

                    // Output to CSV file
                    auto csvPtr = f_createIO();
                    if (csvPtr != nullptr && csvPtr->Initialize(csvOutputFile.c_str(), std::ios::out)) {
                        // Write CSV header
                        std::string header = "vector_id,ground_truth_count,recall_count\n";
                        csvPtr->WriteBinary(header.length(), header.c_str());

                        // Write data for all vectors that appear in ground truth
                        for (const auto& pair : groundTruthCount) {
                            SizeType vectorId = pair.first;
                            int gtCount = pair.second;
                            int recallCnt = recallCount.count(vectorId) > 0 ? recallCount[vectorId] : 0;

                            std::string line = std::to_string(vectorId) + "," +
                                             std::to_string(gtCount) + "," +
                                             std::to_string(recallCnt) + "\n";
                            csvPtr->WriteBinary(line.length(), line.c_str());
                        }

                        LOG(Helper::LogLevel::LL_Info, "Vector statistics CSV generated successfully.\n");
                    } else {
                        LOG(Helper::LogLevel::LL_Error, "Failed to create CSV file: %s\n", csvOutputFile.c_str());
                    }
                }

                LOG(Helper::LogLevel::LL_Info, "\n");

                if (p_opts.m_recall_analysis) {
                    LOG(Helper::LogLevel::LL_Info, "Start recall analysis...\n");

                    std::shared_ptr<VectorIndex> headIndex = p_index->GetMemoryIndex();
                    SizeType sampleSize = numQueries < 100 ? numQueries : 100;
                    SizeType sampleK = headIndex->GetNumSamples() < 1000 ? headIndex->GetNumSamples() : 1000;
                    float sampleE = 1e-6f;

                    std::vector<SizeType> samples(sampleSize, 0);
                    std::vector<float> queryHeadRecalls(sampleSize, 0);
                    std::vector<float> truthRecalls(sampleSize, 0);
                    std::vector<int> shouldSelect(sampleSize, 0);
                    std::vector<int> shouldSelectLong(sampleSize, 0);
                    std::vector<int> nearQueryHeads(sampleSize, 0);
                    std::vector<int> annNotFound(sampleSize, 0);
                    std::vector<int> rngRule(sampleSize, 0);
                    std::vector<int> postingCut(sampleSize, 0);
                    for (int i = 0; i < sampleSize; i++) samples[i] = COMMON::Utils::rand(numQueries);

#pragma omp parallel for schedule(dynamic)
                    for (int i = 0; i < sampleSize; i++)
                    {
                        COMMON::QueryResultSet<ValueType> queryANNHeads((const ValueType*)(querySet->GetVector(samples[i])), max(K, internalResultNum));
                        headIndex->SearchIndex(queryANNHeads);
                        float queryANNHeadsLongestDist = queryANNHeads.GetResult(internalResultNum - 1)->Dist;

                        COMMON::QueryResultSet<ValueType> queryBFHeads((const ValueType*)(querySet->GetVector(samples[i])), max(sampleK, internalResultNum));
                        for (SizeType y = 0; y < headIndex->GetNumSamples(); y++)
                        {
                            float dist = headIndex->ComputeDistance(queryBFHeads.GetQuantizedTarget(), headIndex->GetSample(y));
                            queryBFHeads.AddPoint(y, dist);
                        }
                        queryBFHeads.SortResult();

                        {
                            std::vector<bool> visited(internalResultNum, false);
                            for (SizeType y = 0; y < internalResultNum; y++)
                            {
                                for (SizeType z = 0; z < internalResultNum; z++)
                                {
                                    if (visited[z]) continue;

                                    if (fabs(queryANNHeads.GetResult(z)->Dist - queryBFHeads.GetResult(y)->Dist) < sampleE)
                                    {
                                        queryHeadRecalls[i] += 1;
                                        visited[z] = true;
                                        break;
                                    }
                                }
                            }
                        }

                        std::map<int, std::set<int>> tmpFound; // headID->truths
                        p_index->DebugSearchDiskIndex(queryBFHeads, internalResultNum, sampleK, nullptr, &truth[samples[i]], &tmpFound);

                        for (SizeType z = 0; z < K; z++) {
                            truthRecalls[i] += truth[samples[i]].count(queryBFHeads.GetResult(z)->VID);
                        }

                        for (SizeType z = 0; z < K; z++) {
                            truth[samples[i]].erase(results[samples[i]].GetResult(z)->VID);
                        }

                        for (std::map<int, std::set<int>>::iterator it = tmpFound.begin(); it != tmpFound.end(); it++) {
                            float q2truthposting = headIndex->ComputeDistance(querySet->GetVector(samples[i]), headIndex->GetSample(it->first));
                            for (auto vid : it->second) {
                                if (!truth[samples[i]].count(vid)) continue;

                                if (q2truthposting < queryANNHeadsLongestDist) shouldSelect[i] += 1;
                                else {
                                    shouldSelectLong[i] += 1;

                                    std::set<int> nearQuerySelectedHeads;
                                    float v2vhead = headIndex->ComputeDistance(vectorSet->GetVector(vid), headIndex->GetSample(it->first));
                                    for (SizeType z = 0; z < internalResultNum; z++) {
                                        if (queryANNHeads.GetResult(z)->VID < 0) break;
                                        float v2qhead = headIndex->ComputeDistance(vectorSet->GetVector(vid), headIndex->GetSample(queryANNHeads.GetResult(z)->VID));
                                        if (v2qhead < v2vhead) {
                                            nearQuerySelectedHeads.insert(queryANNHeads.GetResult(z)->VID);
                                        }
                                    }
                                    if (nearQuerySelectedHeads.size() == 0) continue;

                                    nearQueryHeads[i] += 1;

                                    COMMON::QueryResultSet<ValueType> annTruthHead((const ValueType*)(vectorSet->GetVector(vid)), p_opts.m_debugBuildInternalResultNum);
                                    headIndex->SearchIndex(annTruthHead);

                                    bool found = false;
                                    for (SizeType z = 0; z < annTruthHead.GetResultNum(); z++) {
                                        if (nearQuerySelectedHeads.count(annTruthHead.GetResult(z)->VID)) {
                                            found = true;
                                            break;
                                        }
                                    }

                                    if (!found) {
                                        annNotFound[i] += 1;
                                        continue;
                                    }

                                    // RNG rule and posting cut
                                    std::set<int> replicas;
                                    for (SizeType z = 0; z < annTruthHead.GetResultNum() && replicas.size() < p_opts.m_replicaCount; z++) {
                                        BasicResult* item = annTruthHead.GetResult(z);
                                        if (item->VID < 0) break;

                                        bool good = true;
                                        for (auto r : replicas) {
                                            if (p_opts.m_rngFactor * headIndex->ComputeDistance(headIndex->GetSample(r), headIndex->GetSample(item->VID)) < item->Dist) {
                                                good = false;
                                                break;
                                            }
                                        }
                                        if (good) replicas.insert(item->VID);
                                    }

                                    found = false;
                                    for (auto r : nearQuerySelectedHeads) {
                                        if (replicas.count(r)) {
                                            found = true;
                                            break;
                                        }
                                    }

                                    if (found) postingCut[i] += 1;
                                    else rngRule[i] += 1;
                                }
                            }
                        }
                    }
                    float headacc = 0, truthacc = 0, shorter = 0, longer = 0, lost = 0, buildNearQueryHeads = 0, buildAnnNotFound = 0, buildRNGRule = 0, buildPostingCut = 0;
                    for (int i = 0; i < sampleSize; i++) {
                        headacc += queryHeadRecalls[i];
                        truthacc += truthRecalls[i];

                        lost += shouldSelect[i] + shouldSelectLong[i];
                        shorter += shouldSelect[i];
                        longer += shouldSelectLong[i];

                        buildNearQueryHeads += nearQueryHeads[i];
                        buildAnnNotFound += annNotFound[i];
                        buildRNGRule += rngRule[i];
                        buildPostingCut += postingCut[i];
                    }

                    LOG(Helper::LogLevel::LL_Info, "Query head recall @%d:%f.\n", internalResultNum, headacc / sampleSize / internalResultNum);
                    LOG(Helper::LogLevel::LL_Info, "BF top %d postings truth recall @%d:%f.\n", sampleK, truthK, truthacc / sampleSize / truthK);

                    LOG(Helper::LogLevel::LL_Info,
                        "Percent of truths in postings have shorter distance than query selected heads: %f percent\n",
                        shorter / lost * 100);
                    LOG(Helper::LogLevel::LL_Info,
                        "Percent of truths in postings have longer distance than query selected heads: %f percent\n",
                        longer / lost * 100);


                    LOG(Helper::LogLevel::LL_Info,
                        "\tPercent of truths no shorter distance in query selected heads: %f percent\n",
                        (longer - buildNearQueryHeads) / lost * 100);
                    LOG(Helper::LogLevel::LL_Info,
                        "\tPercent of truths exists shorter distance in query selected heads: %f percent\n",
                        buildNearQueryHeads / lost * 100);

                    LOG(Helper::LogLevel::LL_Info,
                        "\t\tRNG rule ANN search loss: %f percent\n", buildAnnNotFound / lost * 100);
                    LOG(Helper::LogLevel::LL_Info,
                        "\t\tPosting cut loss: %f percent\n", buildPostingCut / lost * 100);
                    LOG(Helper::LogLevel::LL_Info,
                        "\t\tRNG rule loss: %f percent\n", buildRNGRule / lost * 100);
                }
            }
		}
	}
}
