#include "plm_glasso.h"

#include <math.h>
#include <string.h>

int plm_glasso(model* m) {

    //set the parameters of 
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
	param.max_iterations = m->iter;
    param.epsilon = 1e-20;
    param.max_linesearch = 20;
    lbfgsfloatval_t fx;
   
    int nvar = m->nvar;
    int ncol = m->ncol;
    int nsingle = ncol * ALPHA;
    double *u = (double*) malloc(sizeof(double) * nvar);
    double *z = (double*) malloc(sizeof(double) * nvar);
    memset(u, 0, nvar*sizeof(double));
    memset(z, 0, nvar*sizeof(double));
    double *x = m->x;
    m->glasso_u = u;
    m->glasso_z = z;
    int t  = 0;
    while (t < m->glasso_iter) {
        double lambda_rho = m->glasso_lambda / m->glasso_rho;
        //printf("admm iter= %2d lambda_rho= %.4f\n", t + 1, lambda_rho);
        //step1: update x
        //x = min_x f(x) + \rho/2 ||x - z + u ||_2^2
        int lbfgs_ret = lbfgs(nvar, m->x, &fx, evaluate_plm_glasso, progress_plm_glasso, m, &param);

        //step2: update z
        //z_i  = S_{\lambda/\rho} (x_i + u_i)
        for (int c1 = 0; c1 < ncol; c1++) {
            for (int c2 = c1 + 1; c2 < ncol; c2++) {
                int offset = INDEX2(c1, c2, 0, 0);
                double* xx = x + offset;
                double* uu = u + offset;
                double* zz = z + offset;
                double norm2 = 0;
                for (int i = 0; i < ALPHA2; i++) {
                    norm2 += (xx[i] + uu[i]) * (xx[i] + uu[i]);
                }
                norm2 = sqrt(norm2);
                double ratio  = 1.0 - lambda_rho / norm2;
                if (ratio < 0.0) {
                    //inactive
                    memset(zz, 0, ALPHA2 * sizeof(double));
                } else {
                    for (int i = 0; i < ALPHA2; i++) {
                        zz[i] = ratio * (xx[i] + uu[i]);
                    }
                }
            }
        }

        //step3: update u
        double *uu = u + nsingle;
        double *xx = x + nsingle;
        double *zz = z + nsingle;
        double eps = 0;
        double x_norm, u_norm, z_norm = 0.0;
        for (int i = 0; i < nvar - nsingle; i++) {
            uu[i] += xx[i] - zz[i];
            eps += uu[i] * uu[i];

            x_norm += xx[i] * xx[i];
            z_norm += zz[i] * zz[i];
            u_norm += uu[i] * uu[i];
        }
        eps = sqrt(eps);
        x_norm = sqrt(x_norm);
        z_norm = sqrt(z_norm);
        u_norm = sqrt(u_norm);
        double eps_ret = eps / x_norm;
        printf("glasso iter = %3d rho= %.3f lambda_rho= %.3f eps= %.6f ret_eps= %.6f lbfgs_ret= %2d rho= %.4f x_norm= %.3f z_norm=%.3f u_norm= %.3f\n",
                t + 1, m->glasso_rho, lambda_rho, eps, eps_ret, lbfgs_ret, m->glasso_rho, x_norm, z_norm, u_norm);
        if (eps_ret < m->tolerance_ret) {
            break;
        }
        m->glasso_rho *= 1.2;
        t += 1;
    }
    free(u);
    free(z);
}

lbfgsfloatval_t evaluate_plm_glasso(
        void *instance,
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
        )
{
    model* _model = (model *)instance;
    unsigned char* msa = _model->msa;
    int nrow = _model->nrow;
    int ncol = _model->ncol;
    double* w = _model->w;
    double neff = _model->neff;

    int threads_num = _model->threads_num;
    int nsingle = ncol * ALPHA;

    //initialize
    double *objective_all = (double*) malloc(threads_num * sizeof(double));
    memset(objective_all, 0.0, sizeof(double) * threads_num);

    double **gradient_all = _model->gradient_all; 
    for (int t = 0; t < threads_num; ++ t) {
        memset(gradient_all[t], 0.0, sizeof(double) * n);
    }

    int per = (ncol) / threads_num; 
#pragma omp parallel for
    for(int t = 0; t < threads_num; t++){
        
        int pos_begin = per * t;
        int pos_end = per * (t+1);
        if (t == threads_num - 1){
            pos_end = ncol;
        }

        lbfgsfloatval_t* pre_prob = (lbfgsfloatval_t*)malloc(sizeof(lbfgsfloatval_t) * ALPHA);
        lbfgsfloatval_t* prob = (lbfgsfloatval_t*)malloc(sizeof(lbfgsfloatval_t) * ALPHA);

        for(int c = pos_begin; c < pos_end; c++){
            
            for(int r = 0; r < nrow; r++){
                
                unsigned char* seq = msa + r * ncol;
                char aa_c = seq[c];
                for(int aa = 0; aa < ALPHA; aa++){
                    pre_prob[aa] = x[INDEX1(c, aa)];
                }
                
                memset(pre_prob, 0, sizeof(lbfgsfloatval_t ) * ALPHA);
                for(int i = 0; i < ncol; i++){
                    for(int aa = 0; aa < ALPHA; aa++){
                        if (i < c){
                            pre_prob[aa] += x[INDEX2(i, c, seq[i], aa)];
                        }else if (i > c){
                            pre_prob[aa] += x[INDEX2(c, i, aa, seq[i])];
                        }
                    }
                }
                
                double sum = 0.0;
                for(int aa = 0; aa < ALPHA; aa++){
                    sum += exp(pre_prob[aa]);
                }
                
                double logz = log(sum);
                for(int aa = 0; aa < ALPHA; aa++){
                    prob[aa] = exp(pre_prob[aa]) / sum;
                }

                //objective function
                objective_all[t] += logz - pre_prob[aa_c];

                //cal gradients
                gradient_all[t][INDEX1(c, aa_c)] -= w[r];
                for(int aa = 0; aa < ALPHA; aa++){
                    gradient_all[t][INDEX1(c, aa)] += w[r] * prob[aa];
                }

                for(int i = 0; i < ncol; i++){
                    if (i < c){
                        gradient_all[t][INDEX2(i, c, seq[i], aa_c)] -= w[r];
                    }
                    else if(i > c){
                        gradient_all[t][INDEX2(c, i, aa_c, seq[i])] -= w[r];
                    }

                    for(int aa = 0;  aa < ALPHA; aa++){
                        if (i < c){
                            gradient_all[t][INDEX2(i, c, seq[i], aa)] += w[r] * prob[aa];
                        }
                        else if(i > c){
                            gradient_all[t][INDEX2(c, i, aa, seq[i])] += w[r] * prob[aa];
                        }
                    }
                }
            }//end r
        }//end c
        free(pre_prob);
        free(prob);
    }

    // reduction data
    lbfgsfloatval_t fx = 0.0;
    memset(g, 0, sizeof(lbfgsfloatval_t) * n);
    for(int t = 0; t < threads_num; ++ t){
        fx += objective_all[t];
        for(int i = 0; i < n; ++ i){
            g[i] += gradient_all[t][i];
        }
    }
    
    //add regularization
    lbfgsfloatval_t lambda_single = _model->lambda_single * neff;
    lbfgsfloatval_t lambda_pair = _model->lambda_pair * neff;
    for(int i = 0; i < nsingle; i++){
        fx += lambda_single * x[i] * x[i];
        g[i] += 2.0 * lambda_single * x[i];
    }

    //glasso
    double *u = _model->glasso_u;
    double *z = _model->glasso_z;
    double rho = _model->glasso_rho;
    for(int i = nsingle; i < n; i++){
        double temp = x[i] - z[i] + u[i];
        fx += rho / 2.0 * temp * temp;
        g[i] += rho * temp;
    }
    free(objective_all);
    
    return fx;
}

int progress_plm_glasso(
        void *instance,
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
        )
{
    model * _model = (model*) instance;
    _model->iter = k;
    fprintf(_model->flog, "iter= %d  fx= %f xnorm = %f gnorm = %f step= %f ", k, fx,  xnorm, gnorm, step);
    evaluate_model(_model);
    
    printf("iter= %d  fx= %f, xnorm = %f, gnorm = %f, step= %f ",k, fx,  xnorm, gnorm, step);
    printf("orig_acc ");
    for(int i = 0; i < 8; i++){
        printf("%.4f ", _model->mat_acc[i]);
    }
    printf("apc_acc ");
    for(int i = 0; i < 8; i++){
        printf("%.4f ", _model->apc_acc[i]);
    }
    printf("\n");

    return 0;
}
