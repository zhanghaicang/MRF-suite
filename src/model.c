#include "model.h"

#include "type.h"
#include "util.h"

int evaluate_model(model_t* model) {
  int nvar = model->nvar;
  int ncol = model->ncol;
  int nrow = model->nrow;
  double* x = model->x;
  double* mat = (double*)malloc(sizeof(double) * ncol * ncol);
  double* apc = (double*)malloc(sizeof(double) * ncol * ncol);
  double* pair_x = x + ALPHA * ncol;

  cal_score_matrix(ncol, pair_x, mat, 1);
  cal_apc(mat, apc, ncol);

  cal_acc(mat, model->dis, ncol, model->mat_acc);
  cal_acc(apc, model->dis, ncol, model->apc_acc);

  fprintf(model->flog, "orig_acc ");
  for (int i = 0; i < 8; i++) {
    fprintf(model->flog, "%.4f ", model->mat_acc[i]);
  }
  fprintf(model->flog, "apc_acc ");
  for (int i = 0; i < 8; i++) {
    fprintf(model->flog, "%.4f ", model->apc_acc[i]);
  }
  fprintf(model->flog, "\n");

  // char path[1024];
  // sprintf(path, "%s_iter_%d.param", model->out_prefix, model->iter);
  // out_raw_param(path, model->ncol, model->x);

  free(mat);
  free(apc);

  return 0;
}
