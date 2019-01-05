#ifndef UTIL_H_
#define UTIL_H_

#include <stdio.h>

typedef struct {
  double val;
  int left;
  int right;
  double dis;
} score_item_t;

void cal_score_matrix(int ncol, double* x, double* m, int type);
double cal_pair_score1(double* m);
double cal_pair_score2(double* m);

void cal_apc(double* m, double* apc, int ncol);
void cal_acc(double* mat, double* dis, int ncol, double* acc);
double cal_top_acc(score_item_t* scores, int num, double dis_cut, int seq_sep,
                   int rank);

void load_matrix(char* path, int ncol, double* mat);
void write_matrix(char* file_path, int ncol, double* m);
void out_raw_param(char* out_path, int ncol, double* x);

int compare(const void* a, const void* b);

#endif
