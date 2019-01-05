#ifndef MODEL_H_
#define MODEL_H_

#include <stdlib.h>
#include <stdio.h>

typedef struct _model_t {
  unsigned char *msa;
  int ncol;
  int nrow;
  int nvar;

  int iter;

  double *x;

  double *w;
  double neff;

  double lambda_single;
  double lambda_pair;

  int threads_num;
  double **gradient_all;
  double *obj_all;

  FILE *flog;
  double *dis;
  double *mat_acc;
  double *apc_acc;

  char *out_prefix;

  // parameters for glasso
  double glasso_rho;
  double glasso_lambda;
  int glasso_iter;
  double tolerance_abs;
  double tolerance_ret;
  double *glasso_z;
  double *glasso_u;
} model_t;

int evaluate_model(model_t *model);

#endif
