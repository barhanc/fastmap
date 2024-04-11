/**
 * Temp file for randomXYgreedy construction heuristics from arXiv:1707.07057 but it performs worse
 * than random initialization and multiple sampling.
 */
void
randomXYgreedy ()
{
    // size_t *arg_nv = calloc (nv, sizeof (size_t));
    // size_t *val_nv = calloc (nv, sizeof (size_t));
    // size_t *arg_nc = calloc (nc, sizeof (size_t));
    // size_t *val_nc = calloc (nc, sizeof (size_t));

    // for (size_t i = 0; i < nv; i++)
    //     arg_nv[i] = i, val_nv[i] = i;

    // for (size_t i = 0; i < nc; i++)
    //     arg_nc[i] = i, val_nc[i] = i;

    // size_t len_nv = nv, len_nc = nc;
    // while (len_nv > 0 || len_nc > 0)
    // {
    //     bool turn = rand () % 2;
    //     if (len_nv > 0 && (turn || len_nc == 0))
    //     {
    //         size_t r = rand () % len_nv;
    //         size_t i = arg_nv[r], j = val_nv[r];
    //         swap (size_t, arg_nv[r], arg_nv[len_nv - 1]);

    //         int64_t DELTA = 0;
    //         for (size_t k = len_nc; k < nc; k++)
    //             DELTA += d (i, j, arg_nc[k], val_nc[k]);

    //         for (size_t _r = 0; _r < len_nv; _r++)
    //         {
    //             size_t _j = val_nv[_r];
    //             int64_t _DELTA = 0;
    //             for (size_t k = len_nc; k < nc; k++)
    //                 _DELTA += d (i, _j, arg_nc[k], val_nc[k]);
    //             if (_DELTA < DELTA)
    //                 r = _r, DELTA = _DELTA;
    //         }

    //         swap (size_t, val_nv[r], val_nv[len_nv - 1]);
    //         len_nv--;
    //     }
    //     if (len_nc > 0 && (turn || len_nv == 0))
    //     {
    //         size_t r = rand () % len_nc;
    //         size_t i = arg_nc[r], j = val_nc[r];
    //         swap (size_t, arg_nc[r], arg_nc[len_nc - 1]);

    //         int64_t DELTA = 0;
    //         for (size_t k = len_nv; k < nv; k++)
    //             DELTA += d (arg_nv[k], val_nv[k], i, j);

    //         for (size_t _r = 0; _r < len_nc; _r++)
    //         {
    //             size_t _j = val_nc[_r];
    //             int64_t _DELTA = 0;
    //             for (size_t k = len_nc; k < nv; k++)
    //                 _DELTA += d (arg_nv[k], val_nv[k], i, _j);
    //             if (_DELTA < DELTA)
    //                 r = _r, DELTA = _DELTA;
    //         }

    //         swap (size_t, val_nc[r], val_nc[len_nc - 1]);
    //         len_nc--;
    //     }
    // }

    // for (size_t i = 0; i < nv; i++)
    //     sigma_nv[arg_nv[i]] = val_nv[i];
    // for (size_t i = 0; i < nc; i++)
    //     sigma_nc[arg_nc[i]] = val_nc[i];

    // free (arg_nv);
    // free (arg_nc);
    // free (val_nv);
    // free (val_nc);
}