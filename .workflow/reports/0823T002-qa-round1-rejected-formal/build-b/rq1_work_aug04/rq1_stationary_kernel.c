
#include <stdint.h>
#include <stddef.h>

int h0b_rq1_replicate(
    const int64_t *timestamps,
    const int16_t *segments,
    const int8_t *strata,
    const int8_t *outcomes,
    const int32_t *block_codes,
    const int32_t *pool,
    const int32_t *pool_offsets,
    const double *restart_uniform,
    const uint64_t *source_uniform,
    int64_t row_count,
    int32_t block_count,
    double restart_probability,
    int64_t *event_counts
) {
    for (int64_t i = 0; i < (int64_t)block_count * 2; ++i) {
        event_counts[i] = 0;
    }
    int32_t source = -1;
    for (int64_t i = 0; i < row_count; ++i) {
        int restart = (i == 0) || (restart_uniform[i] < restart_probability);
        if (i > 0 && (
            segments[i] != segments[i - 1] ||
            timestamps[i] != timestamps[i - 1] + 10000000LL
        )) {
            restart = 1;
        }
        if (!restart) {
            int32_t next = source + 1;
            if (
                next >= row_count ||
                segments[next] != segments[i] ||
                timestamps[next] != timestamps[source] + 10000000LL ||
                strata[next] != strata[i]
            ) {
                restart = 1;
            } else {
                source = next;
            }
        }
        if (restart) {
            int32_t group = (int32_t)segments[i] * 5 + (int32_t)strata[i];
            int32_t begin = pool_offsets[group];
            int32_t end = pool_offsets[group + 1];
            if (end <= begin) {
                return 2;
            }
            uint64_t width = (uint64_t)(end - begin);
            source = pool[begin + (int32_t)(source_uniform[i] % width)];
        }
        int32_t block = block_codes[i];
        if (block < 0 || block >= block_count) {
            return 3;
        }
        event_counts[(int64_t)block * 2] += outcomes[(int64_t)source * 2];
        event_counts[(int64_t)block * 2 + 1] += outcomes[(int64_t)source * 2 + 1];
    }
    return 0;
}
