#ifndef _MODEL_H
#define _MODEL_H

#include <stdlib.h>
#include <stdio.h>

typedef struct model{   
    unsigned char* msa;
    int ncol;
    int nrow;
    int nvar;

    int iter;

    double* x;

    double* w;
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
    
    //parameters for glasso
    double glasso_rho;
    double glasso_lambda;
    int glasso_iter;
    double tolerance_abs;
    double tolerance_ret;
    double* glasso_z;
    double* glasso_u;
}model;

void evaluate_model(model* _model);

#endif
