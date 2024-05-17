#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "lap.h"
#include "queue.h"

#define d(i, j, k, l, m, n) ((pos_U[(i) + nv * (k)] - pos_U[(i) + nv * (l)]) * (pos_V[(j) + nv * (m)] - pos_V[(j) + nv * (n)]) < 0)

typedef struct Node
{
    bool *available;
    size_t *sigma;
    size_t n;
    int32_t bound;
} Node;

static int32_t
swap_bb (const int32_t *pos_U, const int32_t *pos_V, const size_t nv, const size_t nc)
{
    // Upper bound on swap distance
    int32_t B = 0x7FFFFFFF;

    size_t ITERS = 20;
    for (size_t i = 0; i < ITERS; i++)
    {
        int32_t bound = swap_aa (pos_U, pos_V, nv, nc);
        B = bound < B ? bound : B;
    }
    int calls = 0;

    // Cost matrix for LAP
    int32_t *cost = calloc (nv * nv, sizeof (int32_t));

    // Auxiliary variables required for J-V LAP algorithm
    int32_t *rowsol = calloc (nv, sizeof (int32_t));
    int32_t *colsol = calloc (nv, sizeof (int32_t));
    int32_t *x = calloc (nv, sizeof (int32_t));
    int32_t *y = calloc (nv, sizeof (int32_t));

    // FIFO queue (see: 'queue.h')
    Queue *q = queue_alloc ();

    Node *node = malloc (sizeof (Node));
    node->n = 0;
    node->bound = 0;
    node->sigma = calloc (nc, sizeof (size_t));
    node->available = calloc (nc, sizeof (bool));
    for (size_t i = 0; i < nc; i++)
        node->available[i] = true;

    enqueue (q, (void *)node);

    while (q->size > 0)
    {
        node = (Node *)dequeue (q);

        if (node->n == nc)
        {
            B = node->bound < B ? node->bound : B;
        }
        else
        {
            for (size_t el = 0; el < nc; el++)
            {
                if (!node->available[el])
                    continue;

                Node *new_node = malloc (sizeof (Node));

                new_node->sigma = calloc (nc, sizeof (size_t));
                new_node->available = calloc (nc, sizeof (bool));
                memcpy (new_node->sigma, node->sigma, nc * sizeof (size_t));
                memcpy (new_node->available, node->available, nc * sizeof (bool));

                new_node->n = node->n + 1;
                new_node->available[el] = false;
                new_node->sigma[node->n] = el;

                // TODO: Memory access pattern + vectorization. Expand macros and put data into registers.
                memset (cost, 0, nv * nv * sizeof (*cost));
                for (size_t i = 0; i < nv; i++)
                    for (size_t j = 0; j < nv; j++)
                        for (size_t k = 0; k < new_node->n; k++)
                            for (size_t l = k; l < new_node->n; l++)
                                cost[i * nv + j] += d (i, j, k, l, new_node->sigma[k], new_node->sigma[l]);

                calls++;
                new_node->bound = lap (nv, cost, rowsol, colsol, x, y);

                if (new_node->bound >= B)
                {
                    free (new_node->available);
                    free (new_node->sigma);
                    free (new_node);
                }
                else
                {
                    enqueue (q, (void *)new_node);
                }
            }
        }

        free (node->available);
        free (node->sigma);
        free (node);
    }

    free (q);
    free (cost);
    free (rowsol);
    free (colsol);
    free (x);
    free (y);

    printf ("%d\n", calls);

    return B;
}