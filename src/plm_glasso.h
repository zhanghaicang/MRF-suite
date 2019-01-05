#ifndef PLM_GLASSO_H_
#define PLM_GLASSO_H_

#include "lbfgs.h"
#include "model.h"

int plm_glasso(model_t *m);

lbfgsfloatval_t evaluate_plm_glasso(void *instance, const lbfgsfloatval_t *x,
                                    lbfgsfloatval_t *g, const int n,
                                    const lbfgsfloatval_t step);

int progress_plm_glasso(void *instance, const lbfgsfloatval_t *x,
                        const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
                        const lbfgsfloatval_t xnorm,
                        const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step,
                        int n, int k, int ls);

#endif
