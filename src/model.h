#ifndef MODEL_H_
#define MODEL_H_

#include <stdio.h>
#include <stdlib.h>

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

  // paramters for MRF-FM
  int rank;
  double eta;
  double *g;
  double init_factor;
} model_t;

int evaluate_model(model_t *model);

#endif
