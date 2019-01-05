#ifndef CLM_H_
#define CLM_H_

#include "lbfgs.h"

lbfgsfloatval_t evaluate_clm(void *instance, const lbfgsfloatval_t *x,
                             lbfgsfloatval_t *g, const int n,
                             const lbfgsfloatval_t step);

int progress_clm(void *instance, const lbfgsfloatval_t *x,
                 const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
                 const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
                 const lbfgsfloatval_t step, int n, int k, int ls);

#endif
