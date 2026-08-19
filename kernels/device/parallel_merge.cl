#define NUMBER_OF_ELEMENTS_PROCESSED_BY_ONE_LANE 8

int find_intersection_on_antydiagonal(__global float* A, int M, __global float* B, int N, int diag)
{
    // bIndex = diag - mid - 1 (so we move in the opposite direction)
    // bIndex < N
    // diag-mid-1 < N
    // diag - N < mid + 1
    // mid >= diag - N
    // so that's why we do max here, so we don't go out of bounds on B.
    int begin = max(0, diag - N);
    // guard against going out of bounds on A
    int end = min(diag, M);

    // binary search across the anty diagonal, to find where our merge path intersects with the anty diagonal, so where is the perfect point, where
    // the largest element from A is smaller than the next element from B
    // and the largest element from B is smaller than the next element from A
    while (begin < end) {
        int mid = (begin + end) >> 1;

        float aVal = A[mid];
        float bVal = B[diag - mid - 1];

        if (aVal < bVal) {
            // we can take this from A, because a is smaller, so we need to move right on the anty diagonal.
            begin = mid + 1;
        } else {
            // we can't take this from A, because b is smaller, so we need to move left on the anty diagonal.
            end = mid;
        }
    }

    return begin;
}

__kernel void parallel_merge(__global float* const A, int M, __global const float* B, int N, __global float* C)
{
    int globalId = get_global_id(0) * NUMBER_OF_ELEMENTS_PROCESSED_BY_ONE_LANE;

    int cBegin = globalId;
    int cEnd   = min(globalId + NUMBER_OF_ELEMENTS_PROCESSED_BY_ONE_LANE, M + N);

    int aBegin = find_intersection_on_antydiagonal(A, M, B, N, cBegin);
    int bBegin = cBegin - aBegin; // aBegin + bBegin = cBegin

    // merge results
    for (int i = cBegin; i < cEnd; i++) {
        bool useA = false;

        if (aBegin < M && bBegin < N) {
            useA = A[aBegin] <= B[bBegin];
        } else if (i < M) {
            useA = true;
        } else {
            useA = false;
        }

        if (useA) {
            C[i] = A[aBegin];
            aBegin++;
        } else {
            C[i] = B[bBegin];
            bBegin++;
        }
    }
}