#include <stdio.h>
#include <time.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int32_t v4si32_t __attribute__ ((vector_size (4 * sizeof (int32_t))));

int
main ()
{
    srand (1);
    size_t n = 100000000;
    int32_t *A = malloc (n * sizeof (int32_t));
    int32_t *B = malloc (n * sizeof (int32_t));
    int32_t *C = malloc (n * sizeof (int32_t));

    for (size_t i = 0; i < n; i++)
    {
        A[i] = rand () % n;
        B[i] = rand () % n;
    }

    clock_t t = clock ();
    for (size_t i = 0; i < n; i++)
        C[i] = A[i] * B[i];
    t = clock () - t;
    printf ("%.4f\n", (double)t / CLOCKS_PER_SEC);

    int32_t *D = malloc (n * sizeof (int32_t));
    v4si32_t a, b, c;

    t = clock ();
    for (size_t i = 0; i < n; i += 4)
    {
        memcpy (&a, &A[i], 4 * sizeof (int32_t));
        memcpy (&b, &B[i], 4 * sizeof (int32_t));
        c = a * b;
        memcpy (&D[i], &c, 4 * sizeof (int32_t));
    }
    t = clock () - t;
    printf ("%.4f\n", (double)t / CLOCKS_PER_SEC);

    for (size_t i = 0; i < n; i++)
        if (C[i] != D[i])
        {
            printf ("ERROR!\n");
            return -1;
        }

    printf ("OK\n");

    return 0;
}