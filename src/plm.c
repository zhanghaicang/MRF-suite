#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#
#include "model.h"
#include "omp.h"
#include "plm.h"
#include "util.h"

lbfgsfloatval_t evaluate_plm(void *instance, const lbfgsfloatval_t *x,
                             lbfgsfloatval_t *g, const int n,
                             const lbfgsfloatval_t step) {
  model *_model = (model *)instance;
  unsigned char *msa = _model->msa;
  int nrow = _model->nrow;
  int ncol = _model->ncol;
  double *w = _model->w;
  double neff = _model->neff;

  int threads_num = _model->threads_num;
  int nsingle = ncol * ALPHA;

  // initialize
  double *objective_all = (double *)malloc(threads_num * sizeof(double));
  memset(objective_all, 0.0, sizeof(double) * threads_num);

  double **gradient_all = _model->gradient_all;
  for (int t = 0; t < threads_num; ++t) {
    memset(gradient_all[t], 0.0, sizeof(double) * n);
  }

  int per = (ncol) / threads_num;
#pragma omp parallel for
  for (int t = 0; t < threads_num; t++) {
    int pos_begin = per * t;
    int pos_end = per * (t + 1);
    if (t == threads_num - 1) {
      pos_end = ncol;
    }

    lbfgsfloatval_t *pre_prob =
        (lbfgsfloatval_t *)malloc(sizeof(lbfgsfloatval_t) * ALPHA);
    lbfgsfloatval_t *prob =
        (lbfgsfloatval_t *)malloc(sizeof(lbfgsfloatval_t) * ALPHA);

    for (int c = pos_begin; c < pos_end; c++) {
      for (int r = 0; r < nrow; r++) {
        unsigned char *seq = msa + r * ncol;
        char aa_c = seq[c];
        for (int aa = 0; aa < ALPHA; aa++) {
          pre_prob[aa] = x[INDEX1(c, aa)];
        }

        memset(pre_prob, 0, sizeof(lbfgsfloatval_t) * ALPHA);
        for (int i = 0; i < ncol; i++) {
          for (int aa = 0; aa < ALPHA; aa++) {
            if (i < c) {
              pre_prob[aa] += x[INDEX2(i, c, seq[i], aa)];
            } else if (i > c) {
              pre_prob[aa] += x[INDEX2(c, i, aa, seq[i])];
            }
          }
        }

        double sum = 0.0;
        for (int aa = 0; aa < ALPHA; aa++) {
          sum += exp(pre_prob[aa]);
        }

        double logz = log(sum);
        for (int aa = 0; aa < ALPHA; aa++) {
          prob[aa] = exp(pre_prob[aa]) / sum;
        }

        // objective function
        objective_all[t] += logz - pre_prob[aa_c];

        // cal gradients
        gradient_all[t][INDEX1(c, aa_c)] -= w[r];
        for (int aa = 0; aa < ALPHA; aa++) {
          gradient_all[t][INDEX1(c, aa)] += w[r] * prob[aa];
        }

        for (int i = 0; i < ncol; i++) {
          if (i < c) {
            gradient_all[t][INDEX2(i, c, seq[i], aa_c)] -= w[r];
          } else if (i > c) {
            gradient_all[t][INDEX2(c, i, aa_c, seq[i])] -= w[r];
          }

          for (int aa = 0; aa < ALPHA; aa++) {
            if (i < c) {
              gradient_all[t][INDEX2(i, c, seq[i], aa)] += w[r] * prob[aa];
            } else if (i > c) {
              gradient_all[t][INDEX2(c, i, aa, seq[i])] += w[r] * prob[aa];
            }
          }
        }
      }  // end r
    }    // end c
    free(pre_prob);
    free(prob);
  }

  // reduction data
  lbfgsfloatval_t fx = 0.0;
  memset(g, 0, sizeof(lbfgsfloatval_t) * n);
  for (int t = 0; t < threads_num; ++t) {
    fx += objective_all[t];
    for (int i = 0; i < n; ++i) {
      g[i] += gradient_all[t][i];
    }
  }

  // add regularization
  lbfgsfloatval_t lambda_single = _model->lambda_single * neff;
  lbfgsfloatval_t lambda_pair = _model->lambda_pair * neff;
  for (int i = 0; i < nsingle; i++) {
    fx += lambda_single * x[i] * x[i];
    g[i] += 2.0 * lambda_single * x[i];
  }
  for (int i = nsingle; i < n; i++) {
    fx += lambda_pair * x[i] * x[i];
    g[i] += 2.0 * lambda_pair * x[i];
  }
  free(objective_all);

  return fx;
}

int progress_plm(void *instance, const lbfgsfloatval_t *x,
                 const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
                 const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
                 const lbfgsfloatval_t step, int n, int k, int ls) {
  model *_model = (model *)instance;
  _model->iter = k;
  fprintf(_model->flog, "iter= %d  fx= %f xnorm = %f gnorm = %f step= %f ", k,
          fx, xnorm, gnorm, step);
  evaluate_model(_model);

  printf("iter= %d  fx= %f, xnorm = %f, gnorm = %f, step= %f ", k, fx, xnorm,
         gnorm, step);
  printf("orig_acc ");
  for (int i = 0; i < 8; i++) {
    printf("%.4f ", _model->mat_acc[i]);
  }
  printf("apc_acc ");
  for (int i = 0; i < 8; i++) {
    printf("%.4f ", _model->apc_acc[i]);
  }
  printf("\n");

  return 0;
}
