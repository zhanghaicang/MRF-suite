#ifndef SEQUENCE_H_
#define SEQUENCE_H_

#include <stdio.h>

unsigned char* read_msa(const char* path, int* ncol, int* nrow);
unsigned char aatoi(unsigned char aa);
double cal_seq_weight(double* w, unsigned char* msa, int ncol, int nrow,
                      double seq_id);
double cal_seq_sim(unsigned char* seq1, unsigned char* seq2, int ncol);

#endif
