#include "sequence.h"

#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "type.h"

unsigned char* read_msa(const char* path, int* ncol, int* nrow) {
  FILE* f = fopen(path, "r");
  char buf[SEQ_BUFFER_SIZE];
  int nc;
  *nrow = 0;
  *ncol = 0;
  while (fgets(buf, SEQ_BUFFER_SIZE, f)) {
    (*nrow)++;
    nc = strlen(buf);
    *ncol = nc > *ncol ? nc : *ncol;
  }
  *ncol -= 1;
  unsigned char* out =
      (unsigned char*)malloc(sizeof(unsigned char) * (*ncol * *nrow));

  rewind(f);
  for (int i = 0; i < *nrow; i++) {
    fgets(buf, SEQ_BUFFER_SIZE, f);
    for (int j = 0; j < *ncol; j++) {
      out[i * *ncol + j] = aatoi(buf[j]);
    }
  }
  fclose(f);

  return out;
}

unsigned char aatoi(unsigned char aa) {
  char id;
  switch (aa) {
    case '-':
      id = 0;
      break;
    case 'A':
      id = 1;
      break;
    case 'C':
      id = 2;
      break;
    case 'D':
      id = 3;
      break;
    case 'E':
      id = 4;
      break;
    case 'F':
      id = 5;
      break;
    case 'G':
      id = 6;
      break;
    case 'H':
      id = 7;
      break;
    case 'I':
      id = 8;
      break;
    case 'K':
      id = 9;
      break;
    case 'L':
      id = 10;
      break;
    case 'M':
      id = 11;
      break;
    case 'N':
      id = 12;
      break;
    case 'P':
      id = 13;
      break;
    case 'Q':
      id = 14;
      break;
    case 'R':
      id = 15;
      break;
    case 'S':
      id = 16;
      break;
    case 'T':
      id = 17;
      break;
    case 'V':
      id = 18;
      break;
    case 'W':
      id = 19;
      break;
    case 'Y':
      id = 20;
      break;
    default:
      id = 0;
  }
  return id;
}

double cal_seq_weight(double* w, unsigned char* msa, int ncol, int nrow,
                      double seq_id) {
  for (int i = 0; i < nrow; i++) {
    // printf("%d\n", i);
    w[i] = 1.0;
  }
  int i, j;
  double sim;
#pragma omp parallel for default(shared) private(j, sim)
  for (i = 0; i < nrow; i++) {
    for (j = i + 1; j < nrow; j++) {
      sim = cal_seq_sim(msa + i * ncol, msa + j * ncol, ncol);
      if (sim >= seq_id) {
#pragma omp critical
        w[i] += 1.0;
        w[j] += 1.0;
      }
    }
  }

  double neff = 0.0;
  for (int i = 0; i < nrow; i++) {
    w[i] = 1.0 / w[i];
    neff += w[i];
  }
  printf(">msa ncol= %d nrow= %d neff= %.2f\n", ncol, nrow, neff);
  return neff;
}

double cal_seq_sim(unsigned char* seq1, unsigned char* seq2, int ncol) {
  double sum = 0.0;
  for (int i = 0; i < ncol; i++) {
    sum += seq1[i] == seq2[i];
  }
  return sum / ncol;
}
