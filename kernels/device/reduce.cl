__kernel void reduce(__global int* a, __global int* b, const unsigned int arrayCount)
{
    int localId = get_local_id(0);
    int localSize = get_local_size(0);
    int groupId = get_group_id(0);

    if (groupId != 0) {
        return;
    }


    if (arrayCount > 256) {
        return;
    } 

    // One array - one workgroup - 256 max size 
    local int localArray[256];
    for (int i=get_local_id(0); i<arrayCount;i+= get_local_size(0)) {
        localArray[i] = a[i];
    }

    int stride = arrayCount / 2;
    while (stride > 0) {
        if (localId < stride) {
            localArray[localId] += localArray[localId+stride];   
            printf("%d\n", localArray[localId]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        stride /= 2;
    }

    if (get_local_id(0) != 0) {
        return;
    }

    *b = localArray[0];
}