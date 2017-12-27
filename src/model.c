#include "model.h"
#include "util.h"

void evaluate_model(model* _model){
    
    int nvar = _model->nvar;
    int ncol = _model->ncol;
    int nrow = _model->nrow;
    double* x = _model->x;
    double *mat = (double*)malloc(sizeof(double)*ncol * ncol);
    double *apc = (double*)malloc(sizeof(double)*ncol * ncol);
    double* pair_x = x + ALPHA * ncol;
    
    cal_score_matrix(ncol, pair_x, mat, 1);
    cal_apc(mat, apc, ncol);
    
    cal_acc(mat, _model->dis, ncol, _model->mat_acc);
    cal_acc(apc, _model->dis, ncol, _model->apc_acc);
   
    fprintf(_model->flog, "orig_acc ");
    for(int i = 0; i < 8; i++){
        fprintf(_model->flog, "%.4f ", _model->mat_acc[i]);
    }
    fprintf(_model->flog, "apc_acc ");
    for(int i = 0; i < 8; i++){
        fprintf(_model->flog, "%.4f ", _model->apc_acc[i]);
    }
    fprintf(_model->flog, "\n");

    //char path[1024];
    //sprintf(path, "%s_iter_%d.param", _model->out_prefix, _model->iter);
    //out_raw_param(path, _model->ncol, _model->x);

    free(mat);
    free(apc);
}
