#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int32_t v4si32_t __attribute__ ((vector_size (4 * sizeof (int32_t))));
typedef int32_t v8si32_t __attribute__ ((vector_size (8 * sizeof (int32_t))));

// v8si32_t vec = *(v8si32_t*)&buf[i]; // load
// *(v8si32_t*)(buf + i) = vec;        // store