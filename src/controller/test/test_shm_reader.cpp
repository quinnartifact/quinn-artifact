/**
 * test_shm_reader.cpp
 *
 * Test program: verifies that BudgetShmReader can correctly read shared memory
 *
 * Usage:
 *   1. First run the Python controller to create the shared memory
 *   2. Then run this program in another terminal:
 *      $ g++ -std=c++17 -o test_shm_reader test_shm_reader.cpp -lrt
 *      $ ./test_shm_reader /quinn_budget_12345
 *
 * Or run it automatically alongside the Python test script
 */

#include "budget_shm.h"
#include <iostream>
#include <cstdlib>

using namespace quinn;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <shm_name>" << std::endl;
        std::cerr << "Example: " << argv[0] << " /quinn_budget_12345" << std::endl;
        return 1;
    }

    std::string shm_name = argv[1];

    std::cout << "=============================================================\n";
    std::cout << "BudgetShmReader Test\n";
    std::cout << "=============================================================\n";
    std::cout << "Shared memory name: " << shm_name << "\n";
    std::cout << "-------------------------------------------------------------\n";

    try {
        // Open and verify the shared memory
        BudgetShmReader reader(shm_name);

        std::cout << "Total queries: " << reader.size() << "\n";
        std::cout << "-------------------------------------------------------------\n";

        // Show the first 10 budget entries
        size_t display_count = std::min(size_t(10), reader.size());
        std::cout << "First " << display_count << " budget entries:\n";
        std::cout << "-------------------------------------------------------------\n";
        std::cout << "QID\tbS\tbD\n";

        for (size_t qid = 0; qid < display_count; ++qid) {
            BudgetEntry entry = reader.get(qid);
            std::cout << qid << "\t" << entry.bS << "\t" << entry.bD << "\n";
        }

        std::cout << "-------------------------------------------------------------\n";

        // Show summary statistics
        uint16_t min_bS = 65535, max_bS = 0;
        uint16_t min_bD = 65535, max_bD = 0;

        for (size_t qid = 0; qid < reader.size(); ++qid) {
            BudgetEntry entry = reader.get(qid);
            min_bS = std::min(min_bS, entry.bS);
            max_bS = std::max(max_bS, entry.bS);
            min_bD = std::min(min_bD, entry.bD);
            max_bD = std::max(max_bD, entry.bD);
        }

        std::cout << "Budget statistics:\n";
        std::cout << "  b_S range: [" << min_bS << ", " << max_bS << "]\n";
        std::cout << "  b_D range: [" << min_bD << ", " << max_bD << "]\n";
        std::cout << "=============================================================\n";
        std::cout << "Test PASSED: BudgetShmReader successfully read " << reader.size() << " entries\n";
        std::cout << "=============================================================\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        std::cout << "=============================================================\n";
        std::cout << "Test FAILED\n";
        std::cout << "=============================================================\n";
        return 1;
    }
}
