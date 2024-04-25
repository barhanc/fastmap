#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <immintrin.h>

#include "cpuid.h"

typedef int32_t cost;
typedef int32_t idx_t;

#define INF_32 0x7FFFFFFF
#define AVX_MIN_DIM 64

#define d_ref(ptr) (*ptr)
#define cost(i, j) costmatrix[(i) * dim + (j)]

#ifdef __GNUC__
#define always_inline __attribute__((always_inline)) inline
#define ALIGN(n) __attribute__((aligned(n)))
#define restrict __restrict__
#elif _WIN32
#define always_inline __forceinline
#define restrict __restrict
#else
#define always_inline inline
#define restrict
#endif

SIMDFlags flags;

always_inline void
find_umins_regular(idx_t dim, idx_t i, const cost *restrict costmatrix, const cost *restrict v, cost* umin, cost* usubmin, idx_t *j1, idx_t *j2)
{
    const cost *l_cost = &cost(i, 0);
    cost umin_ = l_cost[0] - v[0];
    idx_t j1_ = 0;
    idx_t j2_ = -1;
    cost usubmin_ = INF_32;

    for (idx_t j = 1; j < dim; j++)
    {
        cost h = l_cost[j] - v[j];
        if (h < usubmin_)
        {
            if (h >= umin_) usubmin_ = h, j2_ = j;
            else usubmin_ = umin_, umin_ = h, j2_ = j1_, j1_ = j;
        }
    }

    *umin = umin_, *usubmin = usubmin_, *j1 = j1_, *j2 = j2_;
}

always_inline void
find_umins_avx2(idx_t dim, idx_t i, const cost *restrict costmatrix, const cost *restrict v, cost* umin, cost* usubmin, idx_t *j1, idx_t *j2)
{
    const cost *local_cost = costmatrix + i * dim;
    __m256i idxvec = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i hvec = _mm256_set1_epi32(INF_32);
    __m256i hvec_backup = _mm256_set1_epi32(INF_32);
    __m256i jvec = _mm256_set1_epi32(-1);
    __m256i jvec_backup = _mm256_set1_epi32(-1);
    const __m256i iterator = _mm256_set1_epi32(8);

    __m256i acvec, vvec, h, cmp1, cmp2;

    for (idx_t j = 0; j < dim - 7; j += 8)
    {
        acvec = _mm256_loadu_si256((__m256i*)(local_cost + j));
        vvec =  _mm256_loadu_si256((__m256i*)(v + j));
        h = _mm256_sub_epi32(acvec, vvec);

        cmp1 = _mm256_cmpgt_epi32(hvec, h);
        cmp2 = _mm256_andnot_si256(cmp1, _mm256_cmpgt_epi32(hvec_backup, h));

        hvec_backup = _mm256_blendv_epi8(hvec_backup, hvec, cmp1);
        jvec_backup = _mm256_blendv_epi8(jvec_backup, jvec, cmp1);

        hvec = _mm256_blendv_epi8(hvec, h, cmp1);
        jvec = _mm256_blendv_epi8(jvec, idxvec, cmp1);

        hvec_backup = _mm256_blendv_epi8(hvec_backup, h, cmp2);
        jvec_backup = _mm256_blendv_epi8(jvec_backup, idxvec, cmp2);

        idxvec = _mm256_add_epi32(idxvec, iterator);
    }

    cost h_dump[8], h_backup_dump[8]; 
    idx_t j_dump[8], j_backup_dump[8];
    _mm256_store_si256((__m256i *)&h_dump, hvec);
    _mm256_store_si256((__m256i *)&h_backup_dump, hvec_backup);
    _mm256_store_si256((__m256i *)&j_dump, jvec);
    _mm256_store_si256((__m256i *)&j_backup_dump, jvec_backup);

    idx_t j1_ = -1;
    idx_t j2_ = -1; 
    cost umin_ = INF_32;
    cost usubmin_ = INF_32;

    for (idx_t j_ = 0; j_ < 8; j_++)
    {
        cost h = h_dump[j_];
        if (h < usubmin_)
        {
            if (h >= umin_) usubmin_ = h, j2_ = j_dump[j_];
            else usubmin_ = umin_, umin_ = h, j2_ = j1_, j1_ = j_dump[j_];
        }

        cost h_b = h_backup_dump[j_];
        if (h_b < usubmin_)
        {
            j2_ = j_backup_dump[j_];
            usubmin_ = h_b;
        }
    }

    for (idx_t j=dim-dim%8; j<dim; j+=1)
    {
        cost h = local_cost[j] - v[j];
        if (h < usubmin_)
        {
            if (h >= umin_) usubmin_ = h, j2_ = j;
            else usubmin_ = umin_, umin_ = h, j2_ = j1_, j1_ = j;
        }
    }

    *umin = umin_, *usubmin = usubmin_, *j1 = j1_, *j2 = j2_;
}

always_inline void
find_umins(idx_t dim, idx_t i, const cost *restrict costmatrix, const cost *restrict v, cost* umin, cost* usubmin, idx_t *j1, idx_t *j2)
{
    if (dim > AVX_MIN_DIM && SIMDFlags_hasAVX2(&flags)) 
        return find_umins_avx2(dim, i, costmatrix, v, umin, usubmin, j1, j2);
    else
        return find_umins_regular(dim, i, costmatrix, v, umin, usubmin, j1, j2);
}


/**
 * @brief Jonker-Volgenant algorithm for computing Linear Problem Assignment (LAP)
 * ```
 *  minimize sum X_ij * C_ij
 * ```
 *  where C_ij is a cost matrix, where each row represents costs of tasks for given worker.
 *  X_ij is a solution matrix that satisfies: X_ij = {0, 1} && sum_{i} X_ij = 1 && sum_{j} X_ij = 1.
 *  More on linear assigment: https://en.wikipedia.org/wiki/Assignment_problem
 *  Implementation has a theoretical complexity of O(n^3). To speed up the computation, function uses
 *  AVX2 SIMD commands when available.
 *
 * @param dim problem size
 * @param costmatrix linear matrix of costs
 * @param rowsol index of smallest number in each row
 * @param colsol index of smallest number in each collumn
 * @param u values for optimal column reduction
 * @param v values for optimal row reduction
 * @return int32_t
 */
static cost
lap(int dim, cost *restrict costmatrix, idx_t *restrict rowsol, idx_t *restrict colsol, cost *restrict u, cost *restrict v)
{
    SIMDFlags_init(&flags);

    idx_t* free     = (idx_t*) malloc(dim * sizeof(idx_t));
    idx_t* collist  = (idx_t*) malloc(dim * sizeof(idx_t));
    idx_t* matches  = (idx_t*) malloc(dim * sizeof(idx_t));
    cost*  d        = (cost*)  malloc(dim * sizeof(idx_t));
    idx_t* pred     = (idx_t*) malloc(dim * sizeof(idx_t));

    memset(matches, 0, sizeof(idx_t) * dim);

    // COLUMN REDUCTION
    for(idx_t j=dim-1; j>=0; j--)
    {
        cost min = cost(0, j);
        idx_t imin = 0;
        for(idx_t i = 1; i < dim; i++)
        {
            const cost l_cost = cost(i, j);
            if (l_cost < min)
            {
                min = l_cost;
                imin = i;
            }
        }

        v[j] = min;

        if(++matches[imin] == 1) 
        {
            rowsol[imin] = j;
            colsol[j] = imin;
        }
        else
        {
            colsol[j] = -1;
        }
    }

    if (debug) printf("LAP: COLUMN REDUCTION finished\n");

    idx_t numfree = 0;
    for(idx_t i=0 ; i<dim; i++)
    {
        const cost *l_cost = &cost(i, 0);

        if (matches[i] == 0) 
        {
            free[numfree++] = i;
        }
        else if (matches[i] == 1)
        {
            idx_t j1 = rowsol[i];
            cost min = INF_32;

            for(idx_t j=0; j<dim; j++) 
                if (j != j1 && l_cost[j] - v[j] < min) 
                    min = l_cost[j] - v[j];

            v[j1] -= min;
        }
    }

    if (debug) printf("LAP: REDUCTION TRANSFER finished\n");

    // AUGMENTING ROW REDUCTION
    for (int loopcnt = 0; loopcnt < 2; loopcnt++)
    {
        idx_t k = 0;
        idx_t prevnumfree = numfree;
        numfree = 0;

        while (k < prevnumfree)
        {
            idx_t i = free[k++];
            cost umin, usubmin;
            idx_t j1, j2;

            find_umins(dim, i, costmatrix, v, &umin, &usubmin, &j1, &j2);

            idx_t i0 = colsol[j1];
            cost vj1_new = v[j1] - (usubmin - umin);
            bool vj1_lowers = vj1_new < v[j1];

            if (vj1_lowers) v[j1] = vj1_new;
            else
            {
                j1 = j2;
                 i0 = colsol[j2];
            }

            rowsol[i] = j1;
            colsol[j1] = i;

            if (i0 >= 0)
            {
                if (vj1_lowers) free[--k] = i0;
                else free[numfree++] = i0;
            }
        }

        if (debug) printf("lapjv: AUGMENTING ROW REDUCTION %d / %d\n", loopcnt + 1, 2);
    }


    for (idx_t f = 0; f < numfree; f++) 
    {
        idx_t endofpath;
        idx_t freerow = free[f];
        if (debug) printf("lapjv: AUGMENT SOLUTION row %d [%d / %d]\n", freerow, f + 1, numfree);

        #if _OPENMP >= 201307
        #pragma omp simd
        #endif
        for (idx_t j = 0; j < dim; j++) 
        {
            d[j] = cost(freerow, j) - v[j];
            pred[j] = freerow;
            collist[j] = j;
        }

        idx_t low = 0;
        idx_t up = 0;
        bool unassigned_found = false;

        idx_t last = 0;
        cost min = 0;

        do 
        {
            if (up == low) 
            {
                last = low - 1;
                min = d[collist[up++]];
                for(idx_t k = up; k < dim; k++) 
                {
                    idx_t j = collist[k];
                    cost h = d[j];
                    if (h <= min) 
                    {
                        if (h < min) up = low, min = h;
                        collist[k] = collist[up];
                        collist[up++] = j;
                    }
                }

                for (idx_t k = low; k < up; k++) 
                {
                    if (colsol[collist[k]] < 0) 
                    {
                        endofpath = collist[k];
                        unassigned_found = true;
                        break;
                    }
                }
            }

            if (!unassigned_found) 
            {
                idx_t j1 = collist[low];
                low++;
                idx_t i = colsol[j1];
                const cost *local_cost = &cost(i, 0);
                cost h = local_cost[j1] - v[j1] - min;

                for (idx_t k = up; k < dim; k++) 
                {
                    idx_t j = collist[k];
                    cost v2 = local_cost[j] - v[j] - h;
                    if (v2 < d[j]) 
                    {
                        pred[j] = i;

                        if (v2 == min) 
                        {
                            if (colsol[j] < 0) 
                            {
                                endofpath = j;
                                unassigned_found = true;
                                break;
                            } 
                            else 
                            {
                                collist[k] = collist[up];
                                collist[up++] = j;
                            }
                        }

                        d[j] = v2;
                    }
                }
            }

        } while (!unassigned_found);

        #if _OPENMP >= 201307
        #pragma omp simd
        #endif
        for (idx_t k = 0; k <= last; k++) 
        {
            idx_t j1 = collist[k];
            v[j1] = v[j1] + d[j1] - min;
        }

        {
            idx_t i;
            do 
            {
                i = pred[endofpath];
                colsol[endofpath] = i;
                idx_t j1 = endofpath;
                endofpath = rowsol[i];
                rowsol[i] = j1;

            } while (i != freerow);
        }
    }

    if (debug) printf("lapjv: AUGMENT SOLUTION finished\n");

    cost lapcost = 0;

    #if _OPENMP >= 201307
    #pragma omp simd reduction(+:lapcost)
    #endif

    for (idx_t i = 0; i < dim; i++) 
    {
        const cost *local_cost = &cost(i, 0);
        idx_t j = rowsol[i];
        u[i] = local_cost[j] - v[j];
        lapcost += local_cost[j];
    }

    if (debug) printf("lapjv: optimal cost calculated\n");

    free(free);
    free(collist);
    free(matches);
    free(d);
    free(pred);

    return lapcost;
}



