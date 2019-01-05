#ifndef FM_H_
#define FM_H_

#include "model.h"

void init_fm(model_t* model);
int solve_fm_SGD(model_t* model);
void solve_fm_adam(model_t* model);

void cal_plm_obj(model_t* model, int iter);

void outputmodel(model_t* model, char* path);
void evaluatemodel(model_t* model, int iter);

double cal_score1(double* m);

#endif
