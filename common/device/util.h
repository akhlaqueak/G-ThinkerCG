#ifndef SYSTEM_UTIL_H
#define SYSTEM_UTIL_H

__device__ bool binarySearch(uintV *arr, uintE low, uintE high, ui val)
{
    uintE mid;
    if (high != 0)
        high--; // here in this search high must be inclusive, otherwise it can overflow

    while (low < high)
    {
        mid = low + (high - low) / 2;
        if (val == arr[mid])
            return true;
        else if (arr[mid] < val)
            low = mid + 1;
        else
            high = mid;
    }
    return arr[low] == val;
}
__device__ ull atomicDecrementNonNegative(ull *address, ui dec)
{
    ull old = *address;
    ull assumed;
    do
    {
        assumed = old;
        if (assumed == 0)
            return 0; // already zero, don't decrement
        old = atomicCAS(address, assumed, assumed - dec);
    } while (old != assumed);
    return old;
}
__device__ bool linearSearch(unsigned int *data, unsigned int st, unsigned int en, unsigned int v)
{
    bool pred;
    unsigned int laneid = LANEID;
    unsigned int res;
    for (unsigned int k; st < en; st += 32)
    {
        if (data[st] > v)
            return false; // this exploit the sorted nature of data, and can break early.
        k = st + laneid;
        pred = k < en && (v == data[k]);
        res = __ballot_sync(FULL, pred);
        if (res != 0)
            return true;
    }
    return false;
}

// returns index to write after scanning a warp
__device__ unsigned int scanIndex(bool pred)
{
    unsigned int bits = __ballot_sync(FULL, pred);
    unsigned int mask = FULL >> (31 - LANEID);
    unsigned int index = __popc(mask & bits) - pred; // to get exclusive sum subtract pred
    return index;
}

void deviceQuery()
{
    cudaDeviceProp prop;
    int nDevices = 0, i;
    cudaError_t ierr;

    ierr = cudaGetDeviceCount(&nDevices);
    if (ierr != cudaSuccess)
    {
        printf("Sync error: %s\n", cudaGetErrorString(ierr));
    }

    for (i = 0; i < nDevices; ++i)
    {
        ierr = cudaGetDeviceProperties(&prop, i);
        printf("Device number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Compute capability: %d.%d\n\n", prop.major, prop.minor);

        printf("  Clock Rate: %d kHz\n", prop.clockRate);
        printf("  Total SMs: %d \n", prop.multiProcessorCount);
        printf("  Shared Memory Per SM: %lu bytes\n", prop.sharedMemPerMultiprocessor);
        printf("  Registers Per SM: %d 32-bit\n", prop.regsPerMultiprocessor);
        printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  L2 Cache Size: %d bytes\n", prop.l2CacheSize);
        printf("  Total Global Memory: %lu bytes\n", prop.totalGlobalMem);
        printf("  Memory Clock Rate: %d kHz\n\n", prop.memoryClockRate);

        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max threads in X-dimension of block: %d\n", prop.maxThreadsDim[0]);
        printf("  Max threads in Y-dimension of block: %d\n", prop.maxThreadsDim[1]);
        printf("  Max threads in Z-dimension of block: %d\n\n", prop.maxThreadsDim[2]);

        printf("  Max blocks in X-dimension of grid: %d\n", prop.maxGridSize[0]);
        printf("  Max blocks in Y-dimension of grid: %d\n", prop.maxGridSize[1]);
        printf("  Max blocks in Z-dimension of grid: %d\n\n", prop.maxGridSize[2]);

        printf("  Shared Memory Per Block: %lu bytes\n", prop.sharedMemPerBlock);
        printf("  Registers Per Block: %d 32-bit\n", prop.regsPerBlock);
        printf("  Warp size: %d\n\n", prop.warpSize);
    }
}
void deviceSynch()
{
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cout << cudaGetErrorString(code) << std::endl;
        exit(-1);
    }
}

#endif