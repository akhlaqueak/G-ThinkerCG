
#ifndef COMMON_META_H
#define COMMON_META_H


const static double kDeviceMemoryUnit = 16;  // P100
const static size_t kDeviceMemoryLimits[8] = {(size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024), (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024), (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024),
                                              (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024), (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024), (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024),
                                              (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024), (size_t)(kDeviceMemoryUnit * 1024 * 1024 * 1024)};

enum CondOperator { LESS_THAN, LARGER_THAN, NON_EQUAL, OPERATOR_NONE };

enum StoreStrategy { EXPAND, PREFIX, COUNT };

enum ComputeStrategy { INTERSECTION, ENUMERATION };


#endif //COMMON_META_H