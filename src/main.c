#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

#include "type.h"
#include "clm.h"
#include "lbfgs.h"
#include "model.h"
#include "plm_glasso.h"
#include "sequence.h"
#include "util.h"

void usage();

int main(int argc, char** argv) {
  // initialize parameters
  int threads_num = 1;
  double lambda_pair = 0.1, lambda_single = 0.1;
  int max_iter = 50;
  char* aln_path = NULL;
  char* out_prefix = NULL;
  char* dis_path = NULL;

  int opt;
  int alg_type = 0;

  double glasso_rho = 0.5;
  double glasso_lambda = 0.1;
  int glasso_iter = 20;
  while ((opt = getopt(argc, argv, "c:g:r:a:t:l:s:i:")) >= 0) switch (opt) {
      case 'c':
        glasso_lambda = atof(optarg);
        break;
      case 'g':
        glasso_iter = atoi(optarg);
        break;
      case 'r':
        glasso_rho = atof(optarg);
        break;
      case 'a':
        alg_type = atoi(optarg);
        break;
      case 't':
        threads_num = atoi(optarg);
        break;
      case 'l':
        lambda_pair = atof(optarg);
        break;
      case 's':
        lambda_single = atof(optarg);
        break;
      case 'i':
        max_iter = atoi(optarg);
        break;
      default:
        break;
    }
  if (optind + 3 != argc) {
    usage();
    return -1;
  }
  aln_path = argv[optind];
  out_prefix = argv[optind + 1];
  dis_path = argv[optind + 2];

  // load MSA
  int ncol, nrow;
  unsigned char* msa = read_msa(aln_path, &ncol, &nrow);

  model_t* model = (model_t*)malloc(sizeof(model));
  model->msa = msa;
  model->ncol = ncol;
  model->nrow = nrow;
  model->lambda_single = lambda_single;
  model->lambda_pair = lambda_pair;
  model->threads_num = threads_num;

  model->iter = max_iter;
  model->glasso_lambda = glasso_lambda;
  model->glasso_rho = glasso_rho;
  model->glasso_iter = glasso_iter;
  model->tolerance_ret = 1e-3;

  // calcuate sequence weights
  model->w = (double*)malloc(sizeof(double) * nrow);
  double neff = cal_seq_weight(model->w, msa, ncol, nrow, 0.8);
  model->neff = neff;
  model->dis = (double*)malloc(sizeof(double) * ncol * ncol);
  load_matrix(dis_path, ncol, model->dis);
  model->mat_acc = (double*)malloc(sizeof(double) * ncol * ncol);
  model->apc_acc = (double*)malloc(sizeof(double) * ncol * ncol);

  // init lbfgs parameters
  int nsingle = ncol * ALPHA;
  int nvar = ncol * ALPHA + ncol * (ncol - 1) / 2 * ALPHA * ALPHA;
  model->nvar = nvar;
  model->x = (double*)malloc(nvar * sizeof(double));
  // lbfgsfloatval_t *x =  (lbfgsfloatval_t*)malloc(nvar *
  // sizeof(lbfgsfloatval_t));
  lbfgsfloatval_t* g = (lbfgsfloatval_t*)malloc(nvar * sizeof(lbfgsfloatval_t));
  memset(model->x, 0, nvar * sizeof(double));
  memset(g, 0, nvar * sizeof(lbfgsfloatval_t));

  int ret = -1;
  model->gradient_all = malloc(threads_num * sizeof(double*));
  for (int t = 0; t < threads_num; ++t) {
    model->gradient_all[t] = malloc(nvar * sizeof(double));
  }

  char log_path[1024];
  sprintf(log_path, "%s_%.3f_%.3f.log", out_prefix, lambda_pair, lambda_single);
  FILE* flog = fopen(log_path, "w");
  model->flog = flog;

  model->out_prefix = out_prefix;

  lbfgs_parameter_t param;
  lbfgs_parameter_init(&param);
  param.max_iterations = max_iter;
  param.epsilon = 1e-20;
  param.max_linesearch = 20;
  lbfgsfloatval_t fx;

  omp_set_num_threads(threads_num);
  // printf("optimiztion\n");
  if (alg_type == 1) {
    ret =
        lbfgs(nvar, model->x, &fx, evaluate_clm, progress_clm, model, &param);
  } else if (alg_type == 2) {
    ret = plm_glasso(model);
  }
  // printf(">lbfgs exit code = %d\n", ret);

  char param_path[1024];
  sprintf(param_path, "%s_%.3f_%.3f.own_plm", out_prefix, lambda_pair,
          lambda_single);
  out_raw_param(param_path, ncol, model->x);

  for (int i = 0; i < threads_num; ++i) {
    free(model->gradient_all[i]);
  }
  free(model->gradient_all);
  free(model->x);
  free(model->w);
  free(model);
  free(msa);
  free(g);

  return 0;
}

void usage() {
  printf("Usage: clm_singlet [options] aln-file out-prefix dis-file\n\n");
  printf("Options:\n");
  printf("-a\t: the type of algorithm (default 0)\n");
  printf("  \t: 1. clm\n");
  printf("  \t: 2. plm_glasso\n");
  printf("  \t: 3. MRF_FM\n");
  printf("-t\t: number of threads used (default 1)\n");
  printf("-l\t: the weight of regularization term (pair)  (default 0.1)\n");
  printf("-s\t: the weight of regularization term (single)(default 0.1)\n");
  printf("-i\t: the maximum number of iterations (default 50)\n");
  printf(
      "-g\t: the maximum number of admm iterations in glasso (default = 20)\n");
  printf("-r\t: the rho used in glasso (default = 0.5)\n");
  printf("-c\t: the lambda used in glasso (default = 0.1)\n");
  printf("-k\t: the dimension of latent factor (default = 64; only for FM)\n");
  return;
}
