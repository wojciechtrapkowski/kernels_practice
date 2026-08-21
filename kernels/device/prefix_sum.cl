#define WORKGROUP_SIZE 256
#define SUB_GROUP_SIZE 32

__kernel void calculate_prefix_sum(__global const int* input, __global int* output, __global int* workgroupData, int N)
{
    __local int subGroupReductions[WORKGROUP_SIZE / SUB_GROUP_SIZE];

    // perform sub_group reduction
    int globalId = get_global_id(0);
    int localId  = get_local_id(0);
    int lane     = get_sub_group_local_id();

    int inputValue = ((globalId < N) ? input[globalId] : 0);

    int value = inputValue;
    for (int i = 1; i < get_sub_group_size(); i *= 2) {
        int shuffled = sub_group_shuffle_up(value, i);

        if (lane >= i) {
            value += shuffled;
        }
    }
    // could be
    // int value = sub_group_scan_inclusive_add(inputValue);

    // perform subgroup reduction inside local mem
    bool isLastLane = (lane == get_sub_group_size() - 1);
    if (isLastLane) {
        subGroupReductions[get_sub_group_id()] = value;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < get_sub_group_id(); i++) {
        value += subGroupReductions[i];
    }

    if (globalId < N) {
        output[globalId] = value - inputValue; // because we calculated inclusive sum
    }

    if (!workgroupData) {
        return;
    }

    bool isLastWorkgroupLane = (localId == get_local_size(0) - 1);
    if (isLastWorkgroupLane) {
        workgroupData[get_group_id(0)] = value;
    }
}

__kernel void finalize_prefix_sum(__global int* output, __global int* workgroupData, int N)
{
    int globalId = get_global_id(0);
    if (globalId < N) {
        output[globalId] += workgroupData[get_group_id(0)];
    }
}