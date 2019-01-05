#include "fm.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "type.h"
#include "util.h"

void init_fm(model_t* model) {
  srand(time(NULL));

  int nvar = model->nvar;
  int ncol = model->ncol;
  int rank = model->rank;
  double* x = model->x;
  double init_factor = model->init_factor;
  printf(">init end\n");
  memset(x, 0, sizeof(double) * nvar);
  for (int k = 0; k < ncol; k++) {
    for (int i = 0; i < ALPHA; i++) {
      for (int j = 0; j < rank; j++) {
        x[INDEX_FM(k, i, j)] = init_factor * ((double)rand() / RAND_MAX - 0.5);
      }
    }
  }
  int nvar_fm = rank * ncol * ALPHA;
  for (int c = 0; c < ncol; c++) {
    for (int a = 0; a < ALPHA; a++) {
      x[INDEX_S(c, a)] = init_factor * ((double)rand() / RAND_MAX - 0.5);
    }
  }

  // char path[1024];
  // sprintf(path, "%s.init_param", model->out_prefix);
  // outputmodel(model, path);
}

int solve_fm_SGD(model_t* model) {
  init_fm(model);

  int ncol = model->ncol;
  int nrow = model->nrow;
  int nvar = model->nvar;
  int rank = model->rank;
  unsigned char* msa = model->msa;
  double* x = model->x;
  double* g = model->g;
  double eta = model->eta;
  double lambda_pair = model->lambda_pair;
  int max_iter = model->iter;

  double* prob = (double*)malloc(sizeof(double) * ALPHA);

  double exp_sum = 0.0;
  int nvar_fm = ncol * ALPHA * rank;
  double* v_sum = (double*)malloc(sizeof(double) * rank);
  double* v_other = (double*)malloc(sizeof(double) * rank);
  double* v_t1 = (double*)malloc(sizeof(double) * rank);
  double* v_t2 = (double*)malloc(sizeof(double) * rank);

  double his_best = DBL_MAX;
  int bad_hit = 0;

  for (int iter = 0; iter < max_iter; iter++) {
    memset(g, 0, sizeof(double) * nvar);
    double obj = 0.0;

    for (int r = 0; r < nrow; r++) {
      unsigned char* m = msa + ncol * r;
      memset(v_sum, 0, sizeof(double) * rank);

      for (int c1 = 0; c1 < ncol; c1++) {
        double* v2 = x + (c1 * ALPHA + m[c1]) * rank;
        v_plus(v_sum, v2, v_sum, rank);
      }

      for (int c = 0; c < ncol; c++) {
        // expectation
        memset(prob, 0, sizeof(double) * ALPHA);
        double* v_c = x + (c * ALPHA + m[c]) * rank;
        double exp_sum = 0.0;
        v_minus(v_sum, v_c, v_other, rank);

        for (int a = 0; a < ALPHA; a++) {
          double s = x[INDEX_S(c, a)];
          double* v_a = x + (c * ALPHA + a) * rank;
          s += v_dot(v_a, v_other, rank);
          prob[a] = exp(s);
          exp_sum += prob[a];

          if (m[c] == a) {
            obj -= s;
          }
        }
        for (int a = 0; a < ALPHA; a++) {
          prob[a] /= exp_sum;
        }
        obj += log(exp_sum);
        if (isinf(obj)) {
          printf("%d %d %f\n", r, c, exp_sum);
          exit(-1);
        }
        // gradients
        for (int a = 0; a < ALPHA; a++) {
          double* v_g = g + (c * ALPHA + a) * rank;
          double q = prob[a];
          if (a == m[c]) {
            q -= 1.0;
          }
          v_multiply(q, v_other, v_t2, rank);
          v_plus(v_g, v_t2, v_g, rank);
          g[INDEX_S(c, a)] += q;
        }
        // others gradients
        memset(v_t1, 0, sizeof(double) * rank);
        for (int a = 0; a < ALPHA; a++) {
          double q = prob[a];
          if (m[c] == a) {
            q -= 1.0;
          }
          double* v_a = x + (c * ALPHA + a) * rank;
          v_multiply(q, v_a, v_t2, rank);
          v_plus(v_t1, v_t2, v_t1, rank);
        }
        for (int c1 = 0; c1 < ncol; c1++) {
          if (c1 == c) {
            continue;
          }
          double* g_o = g + (c1 * ALPHA + m[c1]) * rank;
          v_plus(g_o, v_t1, g_o, rank);
        }
      }  // end col
    }    // end row

    obj /= nrow;
    double norm_g = 0.0, norm_x = 0.0;
    double like_g_norm = 0.0, reg_g_norm = 0.0;
    double like_obj = obj, reg_obj = 0.0;

    for (int k = 0; k < nvar; k++) {
      g[k] /= nrow;

      like_g_norm += g[k] * g[k];

      g[k] += 2 * lambda_pair * x[k];
      obj += lambda_pair * x[k] * x[k];

      reg_obj += lambda_pair * x[k] * x[k];
      reg_g_norm = 4 * lambda_pair * lambda_pair * x[k] * x[k];

      norm_g += g[k] * g[k];
      norm_x += x[k] * x[k];

      x[k] -= eta * g[k];
    }
    norm_g = sqrt(norm_g);
    norm_x = sqrt(norm_x);
    like_g_norm = sqrt(like_g_norm);
    reg_g_norm = sqrt(reg_g_norm);
    /*
            if(obj < his_best){
                his_best = obj;
            }else{
                bad_hit++;
                if(bad_hit == 2){
                    eta /= 2.0;
                    bad_hit = 0;
                }
            }
    */
    printf(
        ">iter= %d obj= %f like_obj= %f reg_obj= %f ||x||= %f ||g||=%f "
        "||like_g||= %f ||reg_g|| = %f\n",
        iter, obj, like_obj, reg_obj, norm_x, norm_g, like_g_norm, reg_g_norm);

    if ((iter + 1) % 5 == 0) {
      evaluatemodel(model, iter);
    }
  }  // end iter

  char path[1024];
  sprintf(path, "%s.raw_param", model->out_prefix);
  outputmodel(model, path);

  free(prob);
  free(v_sum);
  free(v_other);
  free(v_t1);
  free(v_t2);

  return 0;
}

void outputmodel(model_t* model, char* path) {
  FILE* f = fopen(path, "w");
  int ncol = model->ncol;
  int rank = model->rank;
  double* x = model->x;
  int nvar_fm = ncol * ALPHA * rank;

  fprintf(f, "ncol= %d rank= %d\n", ncol, rank);
  for (int c = 0; c < ncol; c++) {
    fprintf(f, "col= %d\n", c);
    fprintf(f, "single= ");
    for (int a = 0; a < ALPHA; a++) {
      fprintf(f, " %8.4f", x[INDEX_S(c, a)]);
    }
    fprintf(f, "\n");

    for (int a = 0; a < ALPHA; a++) {
      fprintf(f, "a= %2d ", a);
      for (int k = 0; k < rank; k++) {
        fprintf(f, " %8.4f", x[INDEX_FM(c, a, k)]);
      }
      fprintf(f, "\n");
    }
  }
  fclose(f);
}

void evaluatemodel(model_t* model, int iter) {
  int ncol = model->ncol;
  int rank = model->rank;
  double* x = model->x;
  // cal_score_matrix(x, ncol, rank, model->mat_acc);
  // cal_apc(model->mat_acc, model->apc_acc, ncol);
  double acc1[8], acc2[8];
  cal_acc(model->mat_acc, model->dis, ncol, acc1);
  cal_acc(model->apc_acc, model->dis, ncol, acc2);
  printf("raw: ");
  for (int k = 0; k < 8; k++) {
    printf("%.3f ", acc1[k]);
  }
  printf("apc: ");
  for (int k = 0; k < 8; k++) {
    printf("%.3f ", acc2[k]);
  }
  printf("\n");

  // char path[1024];
  // sprintf(path, "%s_iter_%d.param", model->out_prefix, iter);
  // outputmodel(model, path);
}

void cal_plm_obj(model_t* model, int iter) {
  int ncol = model->ncol;
  int nrow = model->nrow;
  int nvar = model->nvar;
  int rank = model->rank;
  unsigned char* msa = model->msa;
  double* x = model->x;
  double* g = model->g;
  double eta = model->eta;
  double lambda_pair = model->lambda_pair;
  int max_iter = model->iter;

  double* prob = (double*)malloc(sizeof(double) * ALPHA);

  double exp_sum = 0.0;
  int nvar_fm = ncol * ALPHA * rank;
  double* v_sum = (double*)malloc(sizeof(double) * rank);
  double* v_other = (double*)malloc(sizeof(double) * rank);
  double* v_t1 = (double*)malloc(sizeof(double) * rank);
  double* v_t2 = (double*)malloc(sizeof(double) * rank);

  double his_best = DBL_MAX;
  int bad_hit = 0;

  memset(g, 0, sizeof(double) * nvar);
  double obj = 0.0;

  for (int r = 0; r < nrow; r++) {
    unsigned char* m = msa + ncol * r;
    memset(v_sum, 0, sizeof(double) * rank);

    for (int c1 = 0; c1 < ncol; c1++) {
      double* v2 = x + (c1 * ALPHA + m[c1]) * rank;
      v_plus(v_sum, v2, v_sum, rank);
    }

    for (int c = 0; c < ncol; c++) {
      // expectation
      memset(prob, 0, sizeof(double) * ALPHA);
      double* v_c = x + (c * ALPHA + m[c]) * rank;
      double exp_sum = 0.0;
      v_minus(v_sum, v_c, v_other, rank);

      for (int a = 0; a < ALPHA; a++) {
        double s = x[INDEX_S(c, a)];
        double* v_a = x + (c * ALPHA + a) * rank;
        s += v_dot(v_a, v_other, rank);
        prob[a] = exp(s);
        exp_sum += prob[a];

        if (m[c] == a) {
          obj -= s;
        }
      }
      for (int a = 0; a < ALPHA; a++) {
        prob[a] /= exp_sum;
      }
      obj += log(exp_sum);
      if (isinf(obj)) {
        printf("%d %d %f\n", r, c, exp_sum);
        exit(-1);
      }
      // gradients
      for (int a = 0; a < ALPHA; a++) {
        double* v_g = g + (c * ALPHA + a) * rank;
        double q = prob[a];
        if (a == m[c]) {
          q -= 1.0;
        }
        v_multiply(q, v_other, v_t2, rank);
        v_plus(v_g, v_t2, v_g, rank);
        g[INDEX_S(c, a)] += q;
      }
      // others gradients
      memset(v_t1, 0, sizeof(double) * rank);
      for (int a = 0; a < ALPHA; a++) {
        double q = prob[a];
        if (m[c] == a) {
          q -= 1.0;
        }
        double* v_a = x + (c * ALPHA + a) * rank;
        v_multiply(q, v_a, v_t2, rank);
        v_plus(v_t1, v_t2, v_t1, rank);
      }
      for (int c1 = 0; c1 < ncol; c1++) {
        if (c1 == c) {
          continue;
        }
        double* g_o = g + (c1 * ALPHA + m[c1]) * rank;
        v_plus(g_o, v_t1, g_o, rank);
      }
    }  // end col
  }    // end row

  obj /= nrow;
  double norm_g = 0.0, norm_x = 0.0;
  double like_g_norm = 0.0, reg_g_norm = 0.0;
  double like_obj = obj, reg_obj = 0.0;

  for (int k = 0; k < nvar; k++) {
    g[k] /= nrow;

    like_g_norm += g[k] * g[k];

    g[k] += 2 * lambda_pair * x[k];
    obj += lambda_pair * x[k] * x[k];

    reg_obj += lambda_pair * x[k] * x[k];
    reg_g_norm = 4 * lambda_pair * lambda_pair * x[k] * x[k];

    norm_g += g[k] * g[k];
    norm_x += x[k] * x[k];
  }
  norm_g = sqrt(norm_g);
  norm_x = sqrt(norm_x);
  like_g_norm = sqrt(like_g_norm);
  reg_g_norm = sqrt(reg_g_norm);

  printf(
      ">iter= %d obj= %f like_obj= %f reg_obj= %f ||x||= %f ||g||=%f "
      "||like_g||= %f ||reg_g|| = %f\n",
      iter, obj, like_obj, reg_obj, norm_x, norm_g, like_g_norm, reg_g_norm);

  free(prob);
  free(v_sum);
  free(v_other);
  free(v_t1);
  free(v_t2);
}
