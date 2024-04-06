
int32_t *_POS_U, *_POS_V;
#define d(i, j, k, l) abs (_POS_U[(i) * nc + (k)] - _POS_V[(j) * nc + (l)])
#include "bap_bf.h"

int32_t
spear (const int32_t *pos_U, const int32_t *pos_V, const size_t nv, const size_t nc)
{
    _POS_U = pos_U, _POS_V = pos_V;
    return bap (nv, nc);
}