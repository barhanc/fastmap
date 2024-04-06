#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "lap.h"

#define swap(type, x, y) \
    {                    \
        type SWAP = x;   \
        x = y;           \
        y = SWAP;        \
    }
#define cost(i, j) cost[(i) * nv + (j)]

int32_t
bap ()
{
    return 0;
}