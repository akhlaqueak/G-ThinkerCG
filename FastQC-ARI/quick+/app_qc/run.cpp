//########################################################################
//## Copyright 2019 Da Yan http://www.cs.uab.edu/yanda
//##
//## Licensed under the Apache License, Version 2.0 (the "License");
//## you may not use this file except in compliance with the License.
//## You may obtain a copy of the License at
//##
//## //http://www.apache.org/licenses/LICENSE-2.0
//##
//## Unless required by applicable law or agreed to in writing, software
//## distributed under the License is distributed on an "AS IS" BASIS,
//## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//## See the License for the specific language governing permissions and
//## limitations under the License.
//########################################################################

//########################################################################
//## Contributors
//##
//##
//########################################################################

#include "qc_app.h"
#include <math.h> // used for ceil method, as it won't compile without it on Windows.
#include "args.hxx"

// #include <chrono>
using namespace std::chrono;

int main(int argc, char **argv)
{
    args::ArgumentParser parser("Quick+ QC application\n");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> input_path_flag(parser, "dataset", "Path to dataset", {'f', "file"}, "");
    args::ValueFlag<int> num_compers_flag(parser, "threads", "Number of worker threads", {"cpu", "threads"}, num_compers);
    args::ValueFlag<double> deg_ratio_flag(parser, "ratio", "Minimum degree ratio", {'g', "g"}, gdmin_deg_ratio);
    args::ValueFlag<int> min_size_flag(parser, "size", "Minimum quasi-clique size", {'k', "k"}, gnmin_size);
    args::ValueFlag<double> time_threshold_flag(parser, "tau", "Time delay threshold", {'t', "time-threshold"}, TIME_THRESHOLD);
    args::ValueFlag<double> bigtask_threshold_flag(parser, "tau-split", "Big task split threshold", {"tau-split", "bigtask-threshold"}, BIGTASK_THRESHOLD);
    args::ValueFlag<int> tasks_per_fetch_flag(parser, "count", "Tasks per fetch", {"chunk", "tasks-per-fetch"}, tasks_per_fetch_g);
    args::ValueFlag<int> qbig_capacity_flag(parser, "size", "Big queue capacity", {"qbig", "qbig-capacity"}, Qbig_capacity);
    args::ValueFlag<int> qreg_capacity_flag(parser, "size", "Regular queue capacity", {"qreg", "qreg-capacity"}, Qreg_capacity);
    args::ValueFlag<int> bt_tasks_per_file_flag(parser, "count", "Big-task tasks per file", {"bt-file", "bt-tasks-per-file"}, BT_TASKS_PER_FILE);
    args::ValueFlag<int> mini_batch_num_flag(parser, "count", "Mini batch number", {"mini-batch", "mini-batch-num"}, MINI_BATCH_NUM);
    args::ValueFlag<int> rt_tasks_per_file_flag(parser, "count", "Regular-task tasks per file", {"rt-file", "rt-tasks-per-file"}, RT_TASKS_PER_FILE);
    args::ValueFlag<int> bt_threshold_refill_flag(parser, "count", "Big-task refill threshold", {"bt-refill", "bt-threshold-for-refill"}, BT_THRESHOLD_FOR_REFILL);
    args::ValueFlag<int> rt_threshold_refill_flag(parser, "count", "Regular-task refill threshold", {"rt-refill", "rt-threshold-for-refill"}, RT_THRESHOLD_FOR_REFILL);

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help)
    {
        std::cout << parser;
        return 0;
    }
    catch (args::ParseError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return -1;
    }
    catch (args::ValidationError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return -1;
    }

    std::string input_path_value = args::get(input_path_flag);
    if (input_path_value.empty())
    {
        std::cerr << "dataset path is required" << std::endl;
        std::cerr << parser;
        return -1;
    }
    std::string output_name = input_path_value;
    size_t slash_pos = output_name.find_last_of("/\\");
    if (slash_pos != std::string::npos)
    {
        output_name = output_name.substr(slash_pos + 1);
    }
    output_prefix = "output_" + output_name;

    num_compers = args::get(num_compers_flag); //number of compers
    gdmin_deg_ratio = args::get(deg_ratio_flag);
    gnmin_size = args::get(min_size_flag);
    gnmax_size = INT_MAX;
    gnmin_deg = ceil(gdmin_deg_ratio * (gnmin_size - 1));
    TIME_THRESHOLD = args::get(time_threshold_flag); // tau_time
    BIGTASK_THRESHOLD = args::get(bigtask_threshold_flag); // tau_split
    tasks_per_fetch_g = args::get(tasks_per_fetch_flag);
    Qbig_capacity = args::get(qbig_capacity_flag);
    Qreg_capacity = args::get(qreg_capacity_flag);
    BT_TASKS_PER_FILE = args::get(bt_tasks_per_file_flag);
    MINI_BATCH_NUM = args::get(mini_batch_num_flag);
    RT_TASKS_PER_FILE = args::get(rt_tasks_per_file_flag);
    BT_THRESHOLD_FOR_REFILL = args::get(bt_threshold_refill_flag);
    RT_THRESHOLD_FOR_REFILL = args::get(rt_threshold_refill_flag);

    cout << "input_path:" << input_path_value << endl;
    cout << "num_compers:" << num_compers << endl;
    cout << "gdmin_deg_ratio:" << gdmin_deg_ratio << endl;
    cout << "gnmin_size:" << gnmin_size << endl;
    cout << "TIME_THRESHOLD:" << TIME_THRESHOLD << endl;
    cout << "BIGTASK_THRESHOLD:" << BIGTASK_THRESHOLD << endl;
    cout << "tasks_per_fetch_g:" << tasks_per_fetch_g << endl;
    cout << "Qbig_capacity:" << Qbig_capacity << endl;
    cout << "Qreg_capacity:" << Qreg_capacity << endl;
    cout << "BT_TASKS_PER_FILE:" << BT_TASKS_PER_FILE << endl;
    cout << "MINI_BATCH_NUM:" << MINI_BATCH_NUM << endl;
    cout << "RT_TASKS_PER_FILE:" << RT_TASKS_PER_FILE << endl;
    cout << "BT_THRESHOLD_FOR_REFILL:" << BT_THRESHOLD_FOR_REFILL << endl;
    cout << "RT_THRESHOLD_FOR_REFILL:" << RT_THRESHOLD_FOR_REFILL << endl;
    

    QCWorker worker(num_compers);

    auto start = steady_clock::now();
    worker.load_data(const_cast<char *>(input_path_value.c_str()));
    auto end = steady_clock::now();
    float duration = (float)duration_cast<milliseconds>(end - start).count() / 1000;
    //cout << "load_data() execution time:" << duration << endl;
    worker.latest_elapsed_time = duration; // used to calculate step time inside the worker

    start = steady_clock::now();
    worker.run();
    end = steady_clock::now();
    duration = (float)duration_cast<milliseconds>(end - start).count() / 1000;
    //cout << "run() execution Time: " << duration << endl;
    // cout << "trie count: " << trie->print_result() << endl;
    // cout << "================================================" << endl;
    cout<<duration << "\n";


    log("Done");
}

//./run ...
