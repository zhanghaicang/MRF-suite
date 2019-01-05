#include "clm.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "type.h"
#include "util.h"

lbfgsfloatval_t evaluate_clm(void *instance, const lbfgsfloatval_t *x,
                             lbfgsfloatval_t *g, const int n,
                             const lbfgsfloatval_t step) {
  model_t *m = (model_t *)instance;
  unsigned char *msa = m->msa;
  int nrow = m->nrow;
  int ncol = m->ncol;
  double *w = m->w;
  double neff = m->neff;

  int threads_num = m->threads_num;
  int nsingle = ncol * ALPHA;

  // initialize
  double *objective_all = malloc(threads_num * sizeof(double));
  memset(objective_all, 0.0, sizeof(double) * threads_num);
  double **gradient_all = m->gradient_all;
  for (int t = 0; t < threads_num; ++t) {
    memset(gradient_all[t], 0.0, sizeof(double) * n);
  }

  // start openmp parallel
  int pair_num = ncol * (ncol - 1) / 2;
  int *left = (int *)malloc(sizeof(int) * pair_num);
  int *right = (int *)malloc(sizeof(int) * pair_num);
  int cnt = 0;
  for (int i = 0; i < ncol; i++) {
    for (int j = 0; j < ncol; j++) {
      left[cnt] = i;
      left[cnt] = j;
      cnt += 1;
    }
  }
  int per = pair_num / threads_num;
  omp_set_num_threads(threads_num);

#pragma omp parallel for
  for (int t = 0; t < threads_num; t++) {
    int pos_begin = per * t;
    int pos_end = per * (t + 1);
    if (t == threads_num - 1) {
      pos_end = pair_num;
    }
    lbfgsfloatval_t *pre_prob = lbfgs_malloc(ALPHA2);
    lbfgsfloatval_t *prob = lbfgs_malloc(ALPHA2);
    lbfgsfloatval_t *prob1 = lbfgs_malloc(ALPHA);
    lbfgsfloatval_t *prob2 = lbfgs_malloc(ALPHA);
    for (int c = pos_begin; c < pos_end; c++) {
      int pos1 = left[c];
      int pos2 = right[c];
      // for each instance
      for (int r = 0; r < nrow; r++) {
        // observed amino acids
        unsigned char *seq = msa + r * ncol;
        char o_aa1 = seq[pos1];
        char o_aa2 = seq[pos2];
        // calcuate the conditional likelihood
        memset(pre_prob, 0, sizeof(lbfgsfloatval_t) * ALPHA2);
        for (int aa1 = 0; aa1 < ALPHA; aa1++) {
          for (int aa2 = 0; aa2 < ALPHA; aa2++) {
            pre_prob[aa1 * ALPHA + aa2] = x[INDEX1(pos1, aa1)] +
                                          x[INDEX1(pos2, aa2)] +
                                          x[INDEX2(pos1, pos2, aa1, aa2)];
          }
        }
        memset(prob1, 0, sizeof(lbfgsfloatval_t) * ALPHA);
        memset(prob2, 0, sizeof(lbfgsfloatval_t) * ALPHA);
        for (int i = 0; i < ncol; i++) {
          if (i == pos1 || i == pos2) {
            continue;
          }
          for (int aa = 0; aa < ALPHA; aa++) {
            if (i < pos1) {
              prob1[aa] += x[INDEX2(i, pos1, seq[i], aa)];
            } else if (i > pos1) {
              prob1[aa] += x[INDEX2(pos1, i, aa, seq[i])];
            }
            if (i < pos2) {
              prob2[aa] += x[INDEX2(i, pos2, seq[i], aa)];
            } else if (i > pos1) {
              prob2[aa] += x[INDEX2(pos2, i, aa, seq[i])];
            }
          }
        }
        double sum = 0.0;
        for (int aa1 = 0; aa1 < ALPHA; aa1++) {
          for (int aa2 = 0; aa2 < ALPHA; aa2++) {
            pre_prob[aa1 * ALPHA + aa2] += prob1[aa1] + prob2[aa2];
            sum += exp(pre_prob[aa1 * ALPHA + aa2]);
          }
        }
        double logz = log(sum);
        memset(prob1, 0, sizeof(lbfgsfloatval_t) * ALPHA);
        memset(prob2, 0, sizeof(lbfgsfloatval_t) * ALPHA);
        for (int aa1 = 0; aa1 < ALPHA; aa1++) {
          for (int aa2 = 0; aa2 < ALPHA; aa2++) {
            prob[aa1 * ALPHA + aa2] = exp(pre_prob[aa1 * ALPHA + aa2] - logz);
            prob1[aa1] += prob[aa1 * ALPHA + aa2];
            prob2[aa2] += prob[aa1 * ALPHA + aa2];
          }
        }

        // end: calculate the conditional probatility

        // objective
        objective_all[t] += logz - pre_prob[o_aa1 * ALPHA + o_aa2];

        // debug
        // printf("%3d %3d %.4f %.4f %4.f %4.f %4.f\n", pos1, pos2,
        //        pre_prob[o_aa1 * ALPHA + o_aa2], prob[o_aa1 * ALPHA + o_aa2],
        //       prob1[o_aa1], prob2[o_aa2], prob1[o_aa1] * prob2[o_aa2]);

        // gradient for signle item
        gradient_all[t][INDEX1(pos1, o_aa1)] -= w[r];
        gradient_all[t][INDEX1(pos2, o_aa2)] -= w[r];
        // expection
        for (int aa = 0; aa < ALPHA; aa++) {
          gradient_all[t][INDEX1(pos1, aa)] += w[r] * prob1[aa];
          gradient_all[t][INDEX1(pos2, aa)] += w[r] * prob2[aa];
        }

        // gradient for pair couplings
        gradient_all[t][INDEX2(pos1, pos2, o_aa1, o_aa2)] -= w[r];
        for (int i = 0; i < ncol; i++) {
          if (i == pos1 || i == pos2) {
            continue;
          }
          if (i < pos1) {
            gradient_all[t][INDEX2(i, pos1, seq[i], o_aa1)] -= w[r];
          } else if (i > pos1) {
            gradient_all[t][INDEX2(pos1, i, o_aa1, seq[i])] -= w[r];
          }

          if (i < pos2) {
            gradient_all[t][INDEX2(i, pos2, seq[i], o_aa2)] -= w[r];
          } else if (i > pos2) {
            gradient_all[t][INDEX2(pos2, i, o_aa2, seq[i])] -= w[r];
          }
        }
        // expectation
        for (int aa1 = 0; aa1 < ALPHA; aa1++) {
          for (int aa2 = 0; aa2 < ALPHA; aa2++) {
            gradient_all[t][INDEX2(pos1, pos2, aa1, aa2)] +=
                w[r] * prob[aa1 * ALPHA + aa2];
          }
        }
        for (int i = 0; i < ncol; i++) {
          if (i == pos1 || i == pos2) {
            continue;
          }
          for (int aa = 0; aa < ALPHA; aa++) {
            if (i < pos1) {
              gradient_all[t][INDEX2(i, pos1, seq[i], aa)] += w[r] * prob1[aa];
            } else if (i > pos1) {
              gradient_all[t][INDEX2(pos1, i, aa, seq[i])] += w[r] * prob1[aa];
            }

            if (i < pos2) {
              gradient_all[t][INDEX2(i, pos2, seq[i], aa)] += w[r] * prob2[aa];
            } else if (i > pos2) {
              gradient_all[t][INDEX2(pos2, i, aa, seq[i])] += w[r] * prob2[aa];
            }
          }
        }

        // pair gradients
      }
      // end row
    }
    free(prob);
    free(prob1);
    free(prob2);
    free(pre_prob);
  }
  free(left);
  free(right);

  // reduction model
  lbfgsfloatval_t fx = 0.0;
  memset(g, 0, sizeof(lbfgsfloatval_t) * n);
  for (int t = 0; t < threads_num; ++t) {
    // printf("objective[%d]=%f\n", r, objective_all[r]);
    fx += objective_all[t];
    // printf("fx= %f\n", fx);
    for (int i = 0; i < n; ++i) {
      g[i] += gradient_all[t][i];
    }
  }

  // free memory
  free(objective_all);

  double norm = pair_num * 2.0 / ncol;
  // printf("fx %f neff %f\n", fx, m->mNeff);
  fx /= norm;
  for (int i = 0; i < n; i++) {
    g[i] /= norm;
  }

  // add regularization
  lbfgsfloatval_t lambda_single = m->neff * m->lambda_single;
  lbfgsfloatval_t lambda_pair = m->neff * m->lambda_pair;
  for (int i = 0; i < nsingle; i++) {
    fx += lambda_single * x[i] * x[i];
    g[i] += 2.0 * lambda_single * x[i];
  }
  for (int i = nsingle; i < n; i++) {
    fx += lambda_pair * x[i] * x[i];
    g[i] += 2.0 * lambda_pair * x[i];
  }

  return fx;
}

int progress_clm(void *instance, const lbfgsfloatval_t *x,
                 const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
                 const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
                 const lbfgsfloatval_t step, int n, int k, int ls) {
  model_t *model = (model_t *)instance;
  model->iter = k;
  fprintf(model->flog, "iter= %d  fx= %f xnorm = %f gnorm = %f step= %f ", k,
          fx, xnorm, gnorm, step);
  evaluate_model(model);

  printf("iter= %d  fx= %f, xnorm = %f, gnorm = %f, step= %f ", k, fx, xnorm,
         gnorm, step);
  printf("orig_acc ");
  for (int i = 0; i < 8; i++) {
    printf("%.4f ", model->mat_acc[i]);
  }
  printf("apc_acc ");
  for (int i = 0; i < 8; i++) {
    printf("%.4f ", model->apc_acc[i]);
  }
  printf("\n");

  return 0;
}
