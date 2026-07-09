// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"
#include <boost/program_options.hpp>

#include "index.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition.h"
#include "pq_flash_index.h"
#include "timer.h"
#include "percentile_stats.h"
#include "program_options_utils.hpp"
#include "program_options_utils.hpp"
#include "quinn/budget_shm.h"
#include "quinn/early_exit_shm.h"
#include "quinn/latency_shm.h"
#include "quinn/thread_count_shm.h"
#include "io_trace.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

#define WARMUP false

namespace po = boost::program_options;

void print_stats(std::string category, std::vector<float> percentiles, std::vector<float> results)
{
    diskann::cout << std::setw(20) << category << ": " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(8) << percentiles[s] << "%";
    }
    diskann::cout << std::endl;
    diskann::cout << std::setw(22) << " " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(9) << results[s];
    }
    diskann::cout << std::endl;
}

template <typename T, typename LabelT = uint32_t>
int search_disk_index(diskann::Metric &metric, const std::string &index_path_prefix,
                      const std::string &result_output_prefix, const std::string &query_file, std::string &gt_file,
                      const uint32_t num_threads, const uint32_t recall_at, const uint32_t beamwidth,
                      const uint32_t num_nodes_to_cache, const uint32_t search_io_limit,
                      const std::vector<uint32_t> &Lvec, const float fail_if_recall_below,
                      const std::vector<std::string> &query_filters, const bool use_reorder_data = false,
                      const std::string &budget_shm_name = "",
                      const std::string &early_exit_shm_name = "",
                      float eps_stop = 0.05f,
                      uint32_t tau_k_spann = 100,
                      uint32_t tau_k_disk = 100,
                      uint32_t patience = 1,
                      const std::string &hop_trace_path = "",
                      const std::vector<uint32_t> &seed_indices_vec = {},
                      uint32_t seed_k = 0,
                      bool wait_for_spann = false,
                      bool deprioritize_spann = false,
                      const std::string &latency_shm_name = "",
                      const std::string &thread_count_shm_name = "")
{
    diskann::cout << "Search parameters: #threads: " << num_threads << ", ";
    if (beamwidth <= 0)
        diskann::cout << "beamwidth to be optimized for each L value" << std::flush;
    else
        diskann::cout << " beamwidth: " << beamwidth << std::flush;
    if (search_io_limit == std::numeric_limits<uint32_t>::max())
        diskann::cout << "." << std::endl;
    else
        diskann::cout << ", io_limit: " << search_io_limit << "." << std::endl;

    std::string warmup_query_file = index_path_prefix + "_sample_data.bin";

    // load query bin
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    bool filtered_search = false;
    if (!query_filters.empty())
    {
        filtered_search = true;
        if (query_filters.size() != 1 && query_filters.size() != query_num)
        {
            std::cout << "Error. Mismatch in number of queries and size of query "
                         "filters file"
                      << std::endl;
            return -1; // To return -1 or some other error handling?
        }
    }

    bool calc_recall_flag = false;
    if (gt_file != std::string("null") && gt_file != std::string("NULL") && file_exists(gt_file))
    {
        diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num)
        {
            diskann::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    }

    std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
    reader.reset(new WindowsAlignedFileReader());
#else
    reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
    reader.reset(new LinuxAlignedFileReader());
#endif

    std::unique_ptr<diskann::PQFlashIndex<T, LabelT>> _pFlashIndex(
        new diskann::PQFlashIndex<T, LabelT>(reader, metric));

    int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str());

    if (res != 0)
    {
        return res;
    }

    std::vector<uint32_t> node_list;
    diskann::cout << "Caching " << num_nodes_to_cache << " nodes around medoid(s)" << std::endl;
    _pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
    // if (num_nodes_to_cache > 0)
    //     _pFlashIndex->generate_cache_list_from_sample_queries(warmup_query_file, 15, 6, num_nodes_to_cache,
    //     num_threads, node_list);
    _pFlashIndex->load_cache_list(node_list);
    node_list.clear();
    node_list.shrink_to_fit();

    omp_set_num_threads(num_threads);

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

    // QUINN: Initialize Early Exit SHM
    std::unique_ptr<quinn::EarlyExitShmAccessor> earlyExitReader;
    if (!early_exit_shm_name.empty()) {
        try {
            earlyExitReader = std::make_unique<quinn::EarlyExitShmAccessor>(early_exit_shm_name, query_num);
            diskann::cout << "Enabled Early Exit with SHM: " << early_exit_shm_name << std::endl;
        } catch (const std::exception &e) {
            diskann::cerr << "Failed to open Early Exit SHM: " << e.what() << std::endl;
        }
    }

    // QUINN: Initialize Latency SHM
    std::unique_ptr<quinn::LatencyShmAccessor> latencyShm;
    if (!latency_shm_name.empty()) {
        try {
            latencyShm = std::make_unique<quinn::LatencyShmAccessor>(latency_shm_name, static_cast<uint32_t>(query_num));
            diskann::cout << "Enabled per-query latency recording: " << latency_shm_name << std::endl;
        } catch (const std::exception &e) {
            diskann::cerr << "Failed to open Latency SHM: " << e.what() << std::endl;
        }
    }

    // QUINN: Initialize ThreadCount SHM for dynamic thread throttling
    std::unique_ptr<quinn::ThreadCountShmReader> threadCountShm;
    if (!thread_count_shm_name.empty()) {
        try {
            threadCountShm = std::make_unique<quinn::ThreadCountShmReader>(thread_count_shm_name);
            diskann::cout << "Dynamic thread control enabled via: " << thread_count_shm_name << std::endl;
        } catch (const std::exception& e) {
            diskann::cerr << "Warning: failed to open ThreadCountShm: " << e.what() << std::endl;
        }
    }

    uint64_t warmup_L = 20;
    uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
    T *warmup = nullptr;

    if (WARMUP)
    {
        if (file_exists(warmup_query_file))
        {
            diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num, warmup_dim, warmup_aligned_dim);
        }
        else
        {
            warmup_num = (std::min)((uint32_t)150000, (uint32_t)15000 * num_threads);
            warmup_dim = query_dim;
            warmup_aligned_dim = query_aligned_dim;
            diskann::alloc_aligned(((void **)&warmup), warmup_num * warmup_aligned_dim * sizeof(T), 8 * sizeof(T));
            std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-128, 127);
            for (uint32_t i = 0; i < warmup_num; i++)
            {
                for (uint32_t d = 0; d < warmup_dim; d++)
                {
                    warmup[i * warmup_aligned_dim + d] = (T)dis(gen);
                }
            }
        }
        diskann::cout << "Warming up index... " << std::flush;
        std::vector<uint64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<float> warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)warmup_num; i++)
        {
            _pFlashIndex->cached_beam_search(warmup + (i * warmup_aligned_dim), 1, warmup_L,
                                             warmup_result_ids_64.data() + (i * 1),
                                             warmup_result_dists.data() + (i * 1), 4);
        }
        diskann::cout << "..done" << std::endl;
    }

    diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    diskann::cout.precision(2);

    std::string recall_string = "Recall@" + std::to_string(recall_at);
    diskann::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth" << std::setw(16) << "QPS" << std::setw(16)
                  << "Mean Latency" << std::setw(16) << "99.9 Latency" << std::setw(16) << "Mean IOs" << std::setw(16)
                  << "Mean IO (us)" << std::setw(16) << "CPU (s)";
    if (calc_recall_flag)
    {
        diskann::cout << std::setw(16) << recall_string << std::endl;
    }
    else
        diskann::cout << std::endl;
    diskann::cout << "=================================================================="
                     "================================================================="
                  << std::endl;

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());

    uint32_t optimized_beamwidth = 2;

    double best_recall = 0.0;
    double total_search_time_s = 0.0;  // QUINN: Accumulate search time across all L values

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        uint32_t L = Lvec[test_id];

        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        if (beamwidth <= 0)
        {
            diskann::cout << "Tuning beamwidth.." << std::endl;
            optimized_beamwidth =
                optimize_beamwidth(_pFlashIndex, warmup, warmup_num, warmup_aligned_dim, L, optimized_beamwidth);
        }
        else
            optimized_beamwidth = beamwidth;

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);

        auto stats = new diskann::QueryStats[query_num];

        std::vector<uint64_t> query_result_ids_64(recall_at * query_num);

        // QUINN: Allocate per-query hop traces if enabled
        std::vector<std::vector<diskann::HopTraceEntry>> hop_traces;
        if (!hop_trace_path.empty())
            hop_traces.resize(query_num);

        // QUINN: READY/START protocol for aligned QPS measurement
        // Output READY to stdout and wait for START from stdin
        std::cout << "READY" << std::endl;  // Flush immediately
        std::string start_signal;
        std::getline(std::cin, start_signal);  // Wait for START

        // QUINN: Active thread counter for dynamic throttling
        std::atomic<int> active_diskann_count{0};

        auto s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            // QUINN: Dynamic thread throttling
            if (threadCountShm) {
                int allowed = threadCountShm->get_thread_d();
                while (active_diskann_count.load(std::memory_order_acquire) >= allowed) {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                    allowed = threadCountShm->get_thread_d();
                }
                active_diskann_count.fetch_add(1, std::memory_order_release);
            }

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

            diskann::EarlyExitContext early_exit_ctx;
            if (earlyExitReader) {
                early_exit_ctx.enabled = true;
                early_exit_ctx.shm = earlyExitReader.get();
                early_exit_ctx.query_id = i;
                early_exit_ctx.eps_stop = eps_stop;
                early_exit_ctx.tau_k_spann = tau_k_spann;
                early_exit_ctx.tau_k_disk = tau_k_disk;
                early_exit_ctx.patience = patience;
                early_exit_ctx.seed_indices = seed_indices_vec;
                early_exit_ctx.seed_k = seed_k;
                early_exit_ctx.wait_for_spann = wait_for_spann;
                early_exit_ctx.deprioritize_spann = deprioritize_spann;
            }
            if (!hop_traces.empty())
                early_exit_ctx.hop_trace = &hop_traces[i];

            int64_t diskann_t_start = quinn::LatencyShmAccessor::now_ns();
            if (!filtered_search)
            {
                _pFlashIndex->cached_beam_search(query + (i * query_aligned_dim), recall_at, query_L,
                                                 query_result_ids_64.data() + (i * recall_at),
                                                 query_result_dists[test_id].data() + (i * recall_at),
                                                 optimized_beamwidth, use_reorder_data, stats + i,
                                                 early_exit_ctx);
            }
            else
            {
                LabelT label_for_search;
                if (query_filters.size() == 1)
                { // one label for all queries
                    label_for_search = _pFlashIndex->get_converted_label(query_filters[0]);
                }
                else
                { // one label for each query
                    label_for_search = _pFlashIndex->get_converted_label(query_filters[i]);
                }
                _pFlashIndex->cached_beam_search(
                    query + (i * query_aligned_dim), recall_at, query_L, query_result_ids_64.data() + (i * recall_at),
                    query_result_dists[test_id].data() + (i * recall_at), optimized_beamwidth, true, label_for_search,
                    use_reorder_data, stats + i, early_exit_ctx);
            }
            if (latencyShm) {
                int64_t diskann_t_end = quinn::LatencyShmAccessor::now_ns();
                latencyShm->record_diskann_start(static_cast<uint32_t>(i), diskann_t_start);
                latencyShm->record_diskann_end  (static_cast<uint32_t>(i), diskann_t_end);
            }
            if (threadCountShm) {
                active_diskann_count.fetch_sub(1, std::memory_order_release);
            }
        }

        // QUINN: Flush io_trace thread-local buffers from all OMP threads
        #pragma omp parallel
        {
            io_trace::flush_thread();
        }

        // QUINN: Write hop trace CSV if enabled
        if (!hop_traces.empty())
        {
            std::ofstream ht_csv(hop_trace_path);
            ht_csv << "query_id,hop,node_id,distance\n";
            for (size_t qi = 0; qi < query_num; ++qi)
            {
                for (auto &entry : hop_traces[qi])
                {
                    ht_csv << qi << ',' << entry.hop << ',' << entry.node_id << ',' << entry.distance << '\n';
                }
            }
            ht_csv.close();
            diskann::cout << "Hop trace written to: " << hop_trace_path << std::endl;
        }

        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        double search_time_ms = diff.count() * 1000.0;  // seconds to milliseconds
        std::cout << "SEARCH_TIME_MS " << search_time_ms << std::endl;
        std::cout << "DONE" << std::endl;
        total_search_time_s += diff.count();  // QUINN: Accumulate search time
        
        double qps = (1.0 * query_num) / (1.0 * diff.count());
        
        diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(), query_result_ids[test_id].data(),
        query_num, recall_at);
        
        std::string stats_csv = "stat_L" + std::to_string(L) + ".csv";
        std::ofstream csv(stats_csv);
        csv << "qid,total_us,io_us,cpu_us,n_ios,n_4k,n_cache_hits,n_hops,n_cmps\n";
        for (size_t i = 0; i < query_num; ++i) {
            csv << i << ',' << stats[i].total_us << ',' << stats[i].io_us << ',' << stats[i].cpu_us << ','
            << stats[i].n_ios << ',' << stats[i].n_4k << ',' << stats[i].n_cache_hits << ','
            << stats[i].n_hops << ',' << stats[i].n_cmps << '\n';
        }
        csv.close();
        
        // std::string ids_bin = result_output_prefix + "_" + std::to_string(L) + "_topk_ids_u32.bin";
        // diskann::save_bin<uint32_t>(ids_bin, query_result_ids[test_id].data(), query_num, recall_at);                
        
        auto mean_latency = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto latency_999 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.999, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto latency_50 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.50, [](const diskann::QueryStats &stats) { return stats.total_us; });
        auto latency_90 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.90, [](const diskann::QueryStats &stats) { return stats.total_us; });
        auto latency_99 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.99, [](const diskann::QueryStats &stats) { return stats.total_us; });
                
                auto mean_ios = diskann::get_mean_stats<uint32_t>(stats, query_num,
                    [](const diskann::QueryStats &stats) { return stats.n_ios; });

        auto mean_cpuus = diskann::get_mean_stats<float>(stats, query_num,
                                                         [](const diskann::QueryStats &stats) { return stats.cpu_us; });

        auto mean_io_us = diskann::get_mean_stats<float>(stats, query_num,
                                                         [](const diskann::QueryStats &stats) { return stats.io_us; });

        double recall = 0;
        if (calc_recall_flag)
        {
            recall = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                               query_result_ids[test_id].data(), recall_at, recall_at);
            best_recall = std::max(recall, best_recall);
        }

        diskann::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth << std::setw(16) << qps
                      << std::setw(16) << mean_latency << std::setw(16) << latency_999 << std::setw(16) << mean_ios
                      << std::setw(16) << mean_io_us << std::setw(16) << mean_cpuus;
                      if (calc_recall_flag)
                      {
                          diskann::cout << std::setw(16) << recall << std::endl;
                      }
                      else
                          diskann::cout << std::endl;

        // Print tail latency percentiles on a dedicated line (does not affect existing log parsers)
        diskann::cout << "LATENCY_PCTS L=" << L
                      << " p50=" << latency_50
                      << " p90=" << latency_90
                      << " p99=" << latency_99
                      << " p999=" << latency_999
                      << std::endl;

                    delete[] stats;
                }

    // QUINN: Output total search time (accumulated across all L values) to stdout for controller

    diskann::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    for (auto L : Lvec)
    {
        if (L < recall_at)
            continue;

        std::string cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
        diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

        cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
        diskann::save_bin<float>(cur_result_path, query_result_dists[test_id++].data(), query_num, recall_at);
    }

    diskann::aligned_free(query);
    if (warmup != nullptr)
        diskann::aligned_free(warmup);

    // QUINN: Flush I/O trace to disk
    io_trace::flush();

    return best_recall >= fail_if_recall_below ? 0 : -1;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path_prefix, result_path_prefix, query_file, gt_file, filter_label,
        label_type, query_filters_file, budget_shm_name, io_trace_path, early_exit_shm_name, hop_trace_path,
        seed_indices_str, latency_shm_name, thread_count_shm_name;
    uint32_t num_threads, K, W, num_nodes_to_cache, search_io_limit;
    uint32_t tau_k_spann, tau_k_disk, patience;
    std::vector<uint32_t> Lvec;
    bool use_reorder_data = false;
    uint32_t seed_k = 0;
    bool wait_for_spann = false;
    bool deprioritize_spann = false;
    float fail_if_recall_below = 0.0f;
    float eps_stop = 0.05f;

    po::options_description desc{
        program_options_utils::make_program_description("search_disk_index", "Searches on-disk DiskANN indexes")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("result_path", po::value<std::string>(&result_path_prefix)->required(),
                                       program_options_utils::RESULT_PATH_DESCRIPTION);
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        required_configs.add_options()("search_list,L",
                                       po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(),
                                       program_options_utils::SEARCH_LIST_DESCRIPTION);

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        optional_configs.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
                                       program_options_utils::BEAMWIDTH);
        optional_configs.add_options()("num_nodes_to_cache", po::value<uint32_t>(&num_nodes_to_cache)->default_value(0),
                                       program_options_utils::NUMBER_OF_NODES_TO_CACHE);
        optional_configs.add_options()(
            "search_io_limit",
            po::value<uint32_t>(&search_io_limit)->default_value(std::numeric_limits<uint32_t>::max()),
            "Max #IOs for search.  Default value: uint32::max()");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("use_reorder_data", po::bool_switch()->default_value(false),
                                       "Include full precision data in the index. Use only in "
                                       "conjuction with compressed data on SSD.  Default value: false");
        optional_configs.add_options()("filter_label",
                                       po::value<std::string>(&filter_label)->default_value(std::string("")),
                                       program_options_utils::FILTER_LABEL_DESCRIPTION);
        optional_configs.add_options()("query_filters_file",
                                       po::value<std::string>(&query_filters_file)->default_value(std::string("")),
                                       program_options_utils::FILTERS_FILE_DESCRIPTION);
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);
        optional_configs.add_options()("fail_if_recall_below",
                                       po::value<float>(&fail_if_recall_below)->default_value(0.0f),
                                       program_options_utils::FAIL_IF_RECALL_BELOW);
        optional_configs.add_options()("budget_shm",
                                       po::value<std::string>(&budget_shm_name)->default_value(""),
                                       "QUINN: Shared memory name for per-query budgets (optional)");
        optional_configs.add_options()("io_trace", po::value<std::string>(&io_trace_path)->default_value(""),
                                       "QUINN: Output path for I/O trace JSON (Perfetto-compatible). Empty = disabled.");
        optional_configs.add_options()("early_exit_shm", po::value<std::string>(&early_exit_shm_name)->default_value(""),
                                       "QUINN: Shared memory name for early exit (optional)");
        optional_configs.add_options()("eps_stop", po::value<float>(&eps_stop)->default_value(0.05f),
                                       "QUINN: Slack factor for incumbent-gated frontier bound early exit (e.g. 0.02~0.10)");
        optional_configs.add_options()("tau_k_spann", po::value<uint32_t>(&tau_k_spann)->default_value(100),
                                       "QUINN: Use SPANN's tau_k_spann-th nearest distance as tau_spann");
        optional_configs.add_options()("tau_k_disk", po::value<uint32_t>(&tau_k_disk)->default_value(100),
                                       "QUINN: Use DiskANN's tau_k_disk-th nearest distance as tau_disk");
        optional_configs.add_options()("patience", po::value<uint32_t>(&patience)->default_value(1),
                                       "QUINN: Consecutive rounds frontier bound must hold before early termination");
        optional_configs.add_options()("hop_trace", po::value<std::string>(&hop_trace_path)->default_value(""),
                                       "QUINN: Output path for hop trace CSV (query_id,hop,node_id,distance). Empty = disabled.");
        optional_configs.add_options()("seed_indices", po::value<std::string>(&seed_indices_str)->default_value(""),
                                       "QUINN: Comma-separated 0-based positions in SPANN topk to use as seeds "
                                       "(e.g. '0,25,50,75' picks 4 spread points). "
                                       "Empty = sequential top-N injection (default).");
        optional_configs.add_options()("seed_k", po::value<uint32_t>(&seed_k)->default_value(0),
                                       "QUINN: Number of SPANN top-K IDs to inject as DiskANN seeds (sequential mode). "
                                       "0 = no seed injection. Ignored when --seed_indices is set.");
        optional_configs.add_options()("wait_for_spann", po::bool_switch(&wait_for_spann)->default_value(false),
                                       "QUINN: Spin-wait (yield) until SPANN has written its topk IDs to SHM "
                                       "before starting beam search. Guarantees seeds are always injected.");
        optional_configs.add_options()("deprioritize_spann", po::bool_switch(&deprioritize_spann)->default_value(false),
                                       "QUINN: Move SPANN top-K nodes to back of beam frontier each iteration. "
                                       "Non-SPANN candidates expand first. Requires topk_k > 0.");
        optional_configs.add_options()("latency_shm", po::value<std::string>(&latency_shm_name)->default_value(""),
                                       "QUINN: POSIX SHM name for per-query latency timestamps.");
        optional_configs.add_options()("thread_count_shm",
            po::value<std::string>(&thread_count_shm_name)->default_value(""),
            "QUINN: POSIX SHM name for dynamic thread count control");

        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
        if (vm["use_reorder_data"].as<bool>())
            use_reorder_data = true;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    // QUINN: Parse --seed_indices into a vector
    std::vector<uint32_t> seed_indices_vec;
    if (!seed_indices_str.empty())
    {
        std::istringstream ss(seed_indices_str);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            if (!token.empty())
                seed_indices_vec.push_back(static_cast<uint32_t>(std::stoul(token)));
        }
    }

    // QUINN: Initialize I/O tracing if --io_trace is provided
    io_trace::init("diskann", io_trace_path);

    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    if ((data_type != std::string("float")) && (metric == diskann::Metric::INNER_PRODUCT))
    {
        std::cout << "Currently support only floating point data for Inner Product." << std::endl;
        return -1;
    }

    if (use_reorder_data && data_type != std::string("float"))
    {
        std::cout << "Error: Reorder data for reordering currently only "
                     "supported for float data type."
                  << std::endl;
        return -1;
    }

    if (filter_label != "" && query_filters_file != "")
    {
        std::cerr << "Only one of filter_label and query_filters_file should be provided" << std::endl;
        return -1;
    }

    std::vector<std::string> query_filters;
    if (filter_label != "")
    {
        query_filters.push_back(filter_label);
    }
    else if (query_filters_file != "")
    {
        query_filters = read_file_to_vector_of_strings(query_filters_file);
    }

    try
    {
        if (!query_filters.empty() && label_type == "ushort")
        {
            if (data_type == std::string("float"))
                return search_disk_index<float, uint16_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data,
                    budget_shm_name, early_exit_shm_name, eps_stop, tau_k_spann, tau_k_disk, patience, hop_trace_path, seed_indices_vec, seed_k, wait_for_spann, deprioritize_spann, latency_shm_name, thread_count_shm_name);
            else if (data_type == std::string("int8"))
                return search_disk_index<int8_t, uint16_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data,
                    budget_shm_name, early_exit_shm_name, eps_stop, tau_k_spann, tau_k_disk, patience, hop_trace_path, seed_indices_vec, seed_k, wait_for_spann, deprioritize_spann, latency_shm_name, thread_count_shm_name);
            else if (data_type == std::string("uint8"))
                return search_disk_index<uint8_t, uint16_t>(
                    metric, index_path_prefix, result_path_prefix, query_file, gt_file, num_threads, K, W,
                    num_nodes_to_cache, search_io_limit, Lvec, fail_if_recall_below, query_filters, use_reorder_data,
                    budget_shm_name, early_exit_shm_name, eps_stop, tau_k_spann, tau_k_disk, patience, hop_trace_path, seed_indices_vec, seed_k, wait_for_spann, deprioritize_spann, latency_shm_name, thread_count_shm_name);
            else
            {
                std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
                return -1;
            }
        }
        else
        {
            if (data_type == std::string("float"))
                return search_disk_index<float>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                                num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                                fail_if_recall_below, query_filters, use_reorder_data, budget_shm_name,
                                                early_exit_shm_name, eps_stop, tau_k_spann, tau_k_disk, patience, hop_trace_path, seed_indices_vec, seed_k, wait_for_spann, deprioritize_spann, latency_shm_name);
            else if (data_type == std::string("int8"))
                return search_disk_index<int8_t>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                                 num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                                 fail_if_recall_below, query_filters, use_reorder_data, budget_shm_name,
                                                 early_exit_shm_name, eps_stop, tau_k_spann, tau_k_disk, patience, hop_trace_path, seed_indices_vec, seed_k, wait_for_spann, deprioritize_spann, latency_shm_name);
            else if (data_type == std::string("uint8"))
                return search_disk_index<uint8_t>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                                  num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                                  fail_if_recall_below, query_filters, use_reorder_data, budget_shm_name,
                                                  early_exit_shm_name, eps_stop, tau_k_spann, tau_k_disk, patience, hop_trace_path, seed_indices_vec, seed_k, wait_for_spann, deprioritize_spann, latency_shm_name);
            else
            {
                std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
                return -1;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index search failed." << std::endl;
        return -1;
    }
}