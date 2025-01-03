/**
 * Source: https://github.com/gatagat/lap
 *
 * Copyright (c) 2012-2017, Tomas Kazmar All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this list of conditions
 * and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions
 * and the following disclaimer in the documentation and/or other materials provided with the
 * distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 * WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */
#ifndef LAPF_H
#define LAPF_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define LARGE 1000000

#define NEW(x, t, n)                               \
    if ((x = (t *)malloc (sizeof (t) * (n))) == 0) \
    {                                              \
        return -1;                                 \
    }
#define FREE(x)   \
    if (x != 0)   \
    {             \
        free (x); \
        x = 0;    \
    }
#define SWAP_INDICES(a, b)       \
    {                            \
        int32_t _temp_index = a; \
        a = b;                   \
        b = _temp_index;         \
    }

#define ASSERT(cond)
#define PRINTF(fmt, ...)
#define PRINT_COST_ARRAY(a, n)
#define PRINT_INDEX_ARRAY(a, n)

typedef double costf_t;

#ifdef __AVX2__

#include <x86intrin.h>

inline void
_lapf_find_umins_avx2 (const size_t n,
                          const int32_t free_i,
                          const costf_t *cost,
                          const costf_t *v,
                          costf_t *ptr_v1,
                          costf_t *ptr_v2,
                          int32_t *ptr_j1,
                          int32_t *ptr_j2) 
{
    costf_t *c = cost + free_i * n;

    __m256d idxs = _mm256_setr_pd(0.0, 1.0, 2.0, 3.0);
    __m256d incr = _mm256_set1_pd(4.0);

    __m256d hvec = _mm256_set1_pd(LARGE);
    __m256d hvec_backup = _mm256_set1_pd(LARGE);

    __m256d jvec = _mm256_set1_pd(-1.0);
    __m256d jvec_backup = _mm256_set1_pd(-1.0);

    __m256d h, cmp1, cmp2;
    //__mmask8 cmp1, cmp2;

    for (int32_t j = 0; j < n - 3; j += 4) 
    {
        h = _mm256_sub_pd (_mm256_loadu_pd ((costf_t *)(c + j)),
                           _mm256_loadu_pd ((costf_t *)(v + j)));


        cmp1 = _mm256_cmp_pd(hvec, h, _CMP_GT_OQ);
        cmp2 = _mm256_andnot_pd(cmp1, _mm256_cmp_pd(hvec_backup, h, _CMP_GT_OQ));

        hvec_backup = _mm256_blendv_pd(hvec_backup, hvec, cmp1);
        jvec_backup = _mm256_blendv_pd(jvec_backup, jvec, cmp1);

        hvec = _mm256_blendv_pd(hvec, h, cmp1);
        jvec = _mm256_blendv_pd(jvec, idxs, cmp1);

        hvec_backup = _mm256_blendv_pd(hvec_backup, h, cmp2);
        jvec_backup = _mm256_blendv_pd(jvec_backup, idxs, cmp2);

        idxs = _mm256_add_pd(idxs, incr);
    }

    costf_t h_dump[4], h_backup_dump[4];
    costf_t j_dump[4], j_backup_dump[4];
    _mm256_store_pd(h_dump, hvec);
    _mm256_store_pd(h_backup_dump, hvec_backup);
    _mm256_store_pd(j_dump, jvec);
    _mm256_store_pd(j_backup_dump, jvec_backup);

    costf_t j1 = -1.0;
    costf_t j2 = -1.0;
    costf_t v1 = LARGE;
    costf_t v2 = LARGE;


    for (int32_t j_ = 0; j_ < 4; j_++) 
    {
        costf_t h = h_dump[j_];
        if (h < v2)
        {
            if (h >= v1)
                v2 = h, j2 = j_dump[j_];
            else
                v2 = v1, v1 = h, j2 = j1, j1 = j_dump[j_];
        }

        costf_t h_b = h_backup_dump[j_];
        if (h_b < v2)
        {
            j2 = j_backup_dump[j_];
            v2 = h_b;
        }
    }

    for (int32_t j = n - n % 4; j < n; j += 1)
    {
        costf_t h = c[j] - v[j];
        if (h < v2)
        {
            if (h >= v1)
                v2 = h, j2 = j;
            else
                v2 = v1, v1 = h, j2 = j1, j1 = j;
        }
    }

    *ptr_v1 = v1, *ptr_v2 = v2, *ptr_j1 = j1, *ptr_j2 = j2;
}
#endif

inline void
_lapf_find_umins_regular (const size_t n,
                          const int32_t free_i,
                          const costf_t *cost,
                          const costf_t *v,
                          costf_t *ptr_v1,
                          costf_t *ptr_v2,
                          int32_t *ptr_j1,
                          int32_t *ptr_j2)
{
    int32_t j1 = 0;
    costf_t v1 = cost[free_i * n + 0] - v[0];
    int32_t j2 = -1;
    costf_t v2 = LARGE;
    for (size_t j = 1; j < n; j++)
    {
        PRINTF ("%d = %f %d = %f\n", j1, v1, j2, v2);
        const costf_t c = cost[free_i * n + j] - v[j];
        if (c < v2)
        {
            if (c >= v1)
            {
                v2 = c;
                j2 = j;
            }
            else
            {
                v2 = v1;
                v1 = c;
                j2 = j1;
                j1 = j;
            }
        }
    }

    *ptr_v1 = v1, *ptr_v2 = v2, *ptr_j1 = j1, *ptr_j2 = j2;
}

/** Column-reduction and reduction transfer for a dense cost matrix.
 */
static int32_t
_lapf_ccrrt_dense (const size_t n, costf_t *cost,
                   int32_t *free_rows, int32_t *x, int32_t *y, costf_t *v)
{
    int32_t n_free_rows;
    bool *unique;

    for (size_t i = 0; i < n; i++)
    {
        x[i] = -1;
        v[i] = LARGE;
        y[i] = 0;
    }
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            const costf_t c = cost[i * n + j];
            if (c < v[j])
            {
                v[j] = c;
                y[j] = i;
            }
            PRINTF ("i=%d, j=%d, c[i,j]=%f, v[j]=%f y[j]=%d\n", i, j, c, v[j], y[j]);
        }
    }
    PRINT_COST_ARRAY (v, n);
    PRINT_INDEX_ARRAY (y, n);
    NEW (unique, bool, n);
    memset (unique, true, n);
    {
        int32_t j = n;
        do
        {
            j--;
            const int32_t i = y[j];
            if (x[i] < 0)
            {
                x[i] = j;
            }
            else
            {
                unique[i] = false;
                y[j] = -1;
            }
        } while (j > 0);
    }
    n_free_rows = 0;
    for (size_t i = 0; i < n; i++)
    {
        if (x[i] < 0)
        {
            free_rows[n_free_rows++] = i;
        }
        else if (unique[i])
        {
            const int32_t j = x[i];
            costf_t min = LARGE;
            for (size_t j2 = 0; j2 < n; j2++)
            {
                if (j2 == (size_t)j)
                {
                    continue;
                }
                const costf_t c = cost[i * n + j2] - v[j2];
                if (c < min)
                {
                    min = c;
                }
            }
            PRINTF ("v[%d] = %f - %f\n", j, v[j], min);
            v[j] -= min;
        }
    }
    FREE (unique);
    return n_free_rows;
}

/** Augmenting row reduction for a dense cost matrix.
 */
static int32_t
_lapf_carr_dense (
    const size_t n, costf_t *cost,
    const size_t n_free_rows,
    int32_t *free_rows, int32_t *x, int32_t *y, costf_t *v)
{
    size_t current = 0;
    int32_t new_free_rows = 0;
    size_t rr_cnt = 0;
    PRINT_INDEX_ARRAY (x, n);
    PRINT_INDEX_ARRAY (y, n);
    PRINT_COST_ARRAY (v, n);
    PRINT_INDEX_ARRAY (free_rows, n_free_rows);
    while (current < n_free_rows)
    {
        int32_t i0;
        int32_t j1, j2;
        costf_t v1, v2, v1_new;
        bool v1_lowers;

        rr_cnt++;
        PRINTF ("current = %d rr_cnt = %d\n", current, rr_cnt);
        const int32_t free_i = free_rows[current++];
        // j1 = 0;
        // v1 = cost[free_i * n + 0] - v[0];
        // j2 = -1;
        // v2 = LARGE;
        // for (size_t j = 1; j < n; j++)
        // {
        //     PRINTF ("%d = %f %d = %f\n", j1, v1, j2, v2);
        //     const costf_t c = cost[free_i * n + j] - v[j];
        //     if (c < v2)
        //     {
        //         if (c >= v1)
        //         {
        //             v2 = c;
        //             j2 = j;
        //         }
        //         else
        //         {
        //             v2 = v1;
        //             v1 = c;
        //             j2 = j1;
        //             j1 = j;
        //         }
        //     }
        // }
    #ifdef __AVX2__
        if (n >= 4)
            _lapf_find_umins_avx2 (n, free_i, cost, v, &v1, &v2, &j1, &j2);
        else
            _lapf_find_umins_regular (n, free_i, cost, v, &v1, &v2, &j1, &j2);
    #else
            _lapf_find_umins_regular (n, free_i, cost, v, &v1, &v2, &j1, &j2);
    #endif

        i0 = y[j1];
        v1_new = v[j1] - (v2 - v1);
        v1_lowers = v1_new < v[j1];
        PRINTF ("%d %d 1=%d,%f 2=%d,%f v1'=%f(%d,%g) \n", free_i, i0, j1, v1, j2, v2, v1_new, v1_lowers, v[j1] - v1_new);
        if (rr_cnt < current * n)
        {
            if (v1_lowers)
            {
                v[j1] = v1_new;
            }
            else if (i0 >= 0 && j2 >= 0)
            {
                j1 = j2;
                i0 = y[j2];
            }
            if (i0 >= 0)
            {
                if (v1_lowers)
                {
                    free_rows[--current] = i0;
                }
                else
                {
                    free_rows[new_free_rows++] = i0;
                }
            }
        }
        else
        {
            PRINTF ("rr_cnt=%d >= %d (current=%d * n=%d)\n", rr_cnt, current * n, current, n);
            if (i0 >= 0)
            {
                free_rows[new_free_rows++] = i0;
            }
        }
        x[free_i] = j1;
        y[j1] = free_i;
    }
    return new_free_rows;
}

/** Find columns with minimum d[j] and put them on the SCAN list.
 */
static size_t
_lapf_find_dense (const size_t n, size_t lo, costf_t *d, int32_t *cols, int32_t *y)
{
    size_t hi = lo + 1;
    costf_t mind = d[cols[lo]];
    for (size_t k = hi; k < n; k++)
    {
        int32_t j = cols[k];
        if (d[j] <= mind)
        {
            if (d[j] < mind)
            {
                hi = lo;
                mind = d[j];
            }
            cols[k] = cols[hi];
            cols[hi++] = j;
        }
    }
    return hi;
}

// Scan all columns in TODO starting from arbitrary column in SCAN
// and try to decrease d of the TODO columns using the SCAN column.
static int32_t
_lapf_scan_dense (const size_t n, costf_t *cost,
                  size_t *plo, size_t *phi,
                  costf_t *d, int32_t *cols, int32_t *pred,
                  int32_t *y, costf_t *v)
{
    size_t lo = *plo;
    size_t hi = *phi;
    costf_t h, cred_ij;

    while (lo != hi)
    {
        int32_t j = cols[lo++];
        const int32_t i = y[j];
        const costf_t mind = d[j];
        h = cost[i * n + j] - v[j] - mind;
        PRINTF ("i=%d j=%d h=%f\n", i, j, h);
        // For all columns in TODO
        for (size_t k = hi; k < n; k++)
        {
            j = cols[k];
            cred_ij = cost[i * n + j] - v[j] - h;
            if (cred_ij < d[j])
            {
                d[j] = cred_ij;
                pred[j] = i;
                if (cred_ij == mind)
                {
                    if (y[j] < 0)
                    {
                        return j;
                    }
                    cols[k] = cols[hi];
                    cols[hi++] = j;
                }
            }
        }
    }
    *plo = lo;
    *phi = hi;
    return -1;
}

/** Single iteration of modified Dijkstra shortest path algorithm as explained in the JV paper.
 *
 * This is a dense matrix version.
 *
 * \return The closest free column index.
 */
static int32_t
_lapf_find_path_dense (
    const size_t n, costf_t *cost,
    const int32_t start_i,
    int32_t *y, costf_t *v,
    int32_t *pred)
{
    size_t lo = 0, hi = 0;
    int32_t final_j = -1;
    size_t n_ready = 0;
    int32_t *cols;
    costf_t *d;

    NEW (cols, int32_t, n);
    NEW (d, costf_t, n);

    for (size_t i = 0; i < n; i++)
    {
        cols[i] = i;
        pred[i] = start_i;
        d[i] = cost[start_i * n + i] - v[i];
    }
    PRINT_COST_ARRAY (d, n);
    while (final_j == -1)
    {
        // No columns left on the SCAN list.
        if (lo == hi)
        {
            PRINTF ("%d..%d -> find\n", lo, hi);
            n_ready = lo;
            hi = _lapf_find_dense (n, lo, d, cols, y);
            PRINTF ("check %d..%d\n", lo, hi);
            PRINT_INDEX_ARRAY (cols, n);
            for (size_t k = lo; k < hi; k++)
            {
                const int32_t j = cols[k];
                if (y[j] < 0)
                {
                    final_j = j;
                }
            }
        }
        if (final_j == -1)
        {
            PRINTF ("%d..%d -> scan\n", lo, hi);
            final_j = _lapf_scan_dense (
                n, cost, &lo, &hi, d, cols, pred, y, v);
            PRINT_COST_ARRAY (d, n);
            PRINT_INDEX_ARRAY (cols, n);
            PRINT_INDEX_ARRAY (pred, n);
        }
    }

    PRINTF ("found final_j=%d\n", final_j);
    PRINT_INDEX_ARRAY (cols, n);
    {
        const costf_t mind = d[cols[lo]];
        for (size_t k = 0; k < n_ready; k++)
        {
            const int32_t j = cols[k];
            v[j] += d[j] - mind;
        }
    }

    FREE (cols);
    FREE (d);

    return final_j;
}

/** Augment for a dense cost matrix.
 */
static int32_t
_lapf_ca_dense (
    const size_t n, costf_t *cost,
    const size_t n_free_rows,
    int32_t *free_rows, int32_t *x, int32_t *y, costf_t *v)
{
    int32_t *pred;

    NEW (pred, int32_t, n);

    for (int32_t *pfree_i = free_rows; pfree_i < free_rows + n_free_rows; pfree_i++)
    {
        int32_t i = -1, j;
        size_t k = 0;

        PRINTF ("looking at free_i=%d\n", *pfree_i);
        j = _lapf_find_path_dense (n, cost, *pfree_i, y, v, pred);
        ASSERT (j >= 0);
        ASSERT (j < n);
        while (i != *pfree_i)
        {
            PRINTF ("augment %d\n", j);
            PRINT_INDEX_ARRAY (pred, n);
            i = pred[j];
            PRINTF ("y[%d]=%d -> %d\n", j, y[j], i);
            y[j] = i;
            PRINT_INDEX_ARRAY (x, n);
            SWAP_INDICES (j, x[i]);
            k++;
            if (k >= n)
            {
                ASSERT (FALSE);
            }
        }
    }
    FREE (pred);
    return 0;
}

/** Solve dense sparse LAP.
 *
 * input:
 * n        - problem size
 * cost     - cost matrix

 * output:
 * x     - column assigned to row in solution
 * y     - row assigned to column in solution
 */
static costf_t
lapf (const size_t n, double *cost, int32_t *x, int32_t *y)
{
    int ret;
    int32_t *free_rows;
    costf_t *v;

    NEW (free_rows, int32_t, n);
    NEW (v, costf_t, n);
    ret = _lapf_ccrrt_dense (n, cost, free_rows, x, y, v);
    int i = 0;
    while (ret > 0 && i < 2)
    {
        ret = _lapf_carr_dense (n, cost, ret, free_rows, x, y, v);
        i++;
    }
    if (ret > 0)
    {
        ret = _lapf_ca_dense (n, cost, ret, free_rows, x, y, v);
    }

    FREE (v);
    FREE (free_rows);

    if (ret < 0)
        return ret;

    costf_t res = 0;
    for (size_t i = 0; i < n; i++)
        res += cost[i * n + x[i]];

    return res;
}
#endif
