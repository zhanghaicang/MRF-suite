#ifndef _UTIL_H
#define _UTIL_H
#include <stdio.h>

#define ALPHA 21
#define ALPHA2 441
#define INDEX1(pos, aa) ALPHA * pos + aa
#define INDEX2(left_pos, right_pos, left_label, right_label) nsingle + ALPHA2 * (((left_pos * (2 *ncol - left_pos - 1)) >> 1) + right_pos - left_pos - 1) + ALPHA * left_label + right_label
#define INDEX6(pos1, pos2) (pos1*ncol - pos1*(pos1+1)/2 + pos2-pos1-1) * ALPHA2 

typedef struct score_item{
    double val;
    int left;
    int right;
    double dis;
} score_item;

void cal_score_matrix(int ncol, double* x, double* m, int type);
double cal_pair_score1(double *m);
double cal_pair_score2(double *m);

void cal_apc(double *m, double* apc, int ncol);
void cal_acc(double*mat, double* dis, int ncol, double*acc);
double cal_top_acc(score_item* scores, int num, double dis_cut, int seq_sep, int rank);

void load_matrix(char* path, int ncol, double*mat);
void write_matrix(char *file_path, int ncol, double* m);
void out_raw_param(char *out_path, int ncol, double* x);

int compare (const void * a, const void * b);
   
#endif
