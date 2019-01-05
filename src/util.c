#include "util.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "type.h"

void cal_score_matrix(int ncol, double* x, double* m, int type) {
  memset(m, 0, sizeof(double) * ncol * ncol);
  for (int c1 = 0; c1 < ncol; c1++) {
    for (int c2 = c1 + 1; c2 < ncol; c2++) {
      double score;
      if (type == 1) {
        score = cal_pair_score1(x + INDEX6(c1, c2));
      } else if (type == 2) {
        score = cal_pair_score2(x + INDEX6(c1, c2));
      }
      m[c1 * ncol + c2] = score;
      m[c2 * ncol + c1] = score;
    }
  }
}

double cal_pair_score1(double* m) {
  double mean = 0.0;
  double mean_col[ALPHA], mean_row[ALPHA];
  memset(mean_col, 0, sizeof(double) * ALPHA);
  memset(mean_row, 0, sizeof(double) * ALPHA);

  for (int i = 0; i < ALPHA; i++) {
    for (int j = 0; j < ALPHA; j++) {
      mean += m[i * ALPHA + j];
      mean_row[i] += m[i * ALPHA + j];
      mean_col[j] += m[i * ALPHA + j];
    }
  }
  for (int i = 0; i < ALPHA; i++) {
    mean_row[i] /= ALPHA;
    mean_col[i] /= ALPHA;
  }
  mean /= ALPHA * ALPHA;
  double res = 0.0;
  for (int i = 1; i < ALPHA; i++) {
    for (int j = 1; j < ALPHA; j++) {
      double temp = m[i * ALPHA + j] - mean_row[i] - mean_col[j] + mean;
      res += temp * temp;
    }
  }
  res = sqrt(res);
  return res;
}

double cal_pair_score2(double* m) {
  double mean = 0.0;
  double mean_col[ALPHA], mean_row[ALPHA];
  memset(mean_col, 0, sizeof(double) * ALPHA);
  memset(mean_row, 0, sizeof(double) * ALPHA);

  for (int i = 0; i < ALPHA; i++) {
    for (int j = 0; j < ALPHA; j++) {
      mean += m[i * ALPHA + j];
      mean_row[i] += m[i * ALPHA + j];
      mean_col[j] += m[i * ALPHA + j];
    }
  }
  for (int i = 0; i < ALPHA; i++) {
    mean_row[i] /= ALPHA;
    mean_col[i] /= ALPHA;
  }
  mean /= ALPHA * ALPHA;
  double res = 0.0;
  for (int i = 0; i < ALPHA; i++) {
    for (int j = 0; j < ALPHA; j++) {
      double temp = m[i * ALPHA + j] - mean;
      res += temp * temp;
    }
  }
  res = sqrt(res);
  return res;
}

void cal_apc(double* m, double* apc, int ncol) {
  double* mean = (double*)malloc(sizeof(double) * ncol);
  double sum = 0.0;

  for (int i = 0; i < ncol; i++) {
    for (int j = 0; j < ncol; j++) {
      if (i == j) {
        continue;
      }
      mean[i] += m[i * ncol + j];
      sum += m[i * ncol + j];
    }
    mean[i] /= ncol - 1.0;
  }

  sum /= ncol * (ncol - 1.0);

  for (int i = 0; i < ncol; i++) {
    for (int j = i + 1; j < ncol; j++) {
      apc[i * ncol + j] = apc[j * ncol + i] =
          m[i * ncol + j] - mean[i] * mean[j] / sum;
    }
  }
  free(mean);
}

void cal_acc(double* mat, double* dis, int ncol, double* acc) {
  int num = ncol * (ncol - 1) / 2;
  score_item_t* scores = (score_item_t*)malloc(sizeof(score_item_t) * num);
  int top[4] = {ceil(ncol / 10.0), ceil(ncol / 5.0), ceil(ncol / 2.0), ncol};
  int index = 0;
  for (int i = 0; i < ncol; i++) {
    for (int j = i + 1; j < ncol; j++) {
      scores[index].left = i;
      scores[index].right = j;
      scores[index].val = mat[i * ncol + j];
      scores[index].dis = dis[i * ncol + j];
      index++;
    }
  }
  qsort(scores, num, sizeof(score_item_t), compare);
  for (int i = 0; i < 4; i++) {
    acc[i] = cal_top_acc(scores, num, 8.0, 5, top[i]);
  }

  for (int i = 0; i < 4; i++) {
    acc[i + 4] = cal_top_acc(scores, num, 8.0, 23, top[i]);
  }
  free(scores);
}

double cal_top_acc(score_item_t* scores, int num, double dis_cut, int seq_sep,
                   int rank) {
  double acc = 0.0;
  int cnt = 0;
  int pos = 0;
  for (int k = 0; k < num; k++) {
    int left = scores[k].left;
    int right = scores[k].right;
    double dis = scores[k].dis;
    if (right - left < seq_sep || dis < 0.0) {
      continue;
    }
    cnt += 1;
    if (dis <= dis_cut) {
      pos++;
    }
    if (cnt == rank) {
      break;
    }
  }
  return (double)pos / (double)rank;
}

void load_matrix(char* path, int ncol, double* mat) {
  FILE* f = fopen(path, "r");
  double val;
  for (int i = 0; i < ncol; i++) {
    for (int j = 0; j < ncol; j++) {
      fscanf(f, "%lf", &val);
      mat[i * ncol + j] = val;
      mat[j * ncol + i] = val;
    }
  }
  fclose(f);
}

void write_matrix(char* file_path, int ncol, double* m) {
  FILE* fout = fopen(file_path, "w");
  for (int i = 0; i < ncol; i++) {
    for (int j = 0; j < ncol; j++) {
      fprintf(fout, "%8.5f ", m[i * ncol + j]);
    }
    fprintf(fout, "\n");
  }
  fclose(fout);
}

void out_raw_param(char* out_path, int ncol, double* x) {
  FILE* fout = fopen(out_path, "w");
  fprintf(fout, "%d\n", ncol);
  int nsingle = ncol * ALPHA;
  for (int i = 0; i < ncol; i++) {
    for (int aa = 0; aa < ALPHA; aa++) {
      fprintf(fout, "%8.5f ", x[INDEX1(i, aa)]);
    }
    fprintf(fout, "\n");
  }

  for (int i = 0; i < ncol; i++) {
    for (int j = i + 1; j < ncol; j++) {
      fprintf(fout, "%d %d\n", i, j);
      for (int aa1 = 0; aa1 < ALPHA; aa1++) {
        for (int aa2 = 0; aa2 < ALPHA; aa2++) {
          fprintf(fout, "%8.5f ", x[INDEX2(i, j, aa1, aa2)]);
        }
        fprintf(fout, "\n");
      }
    }
  }
  fclose(fout);
}

int compare(const void* a, const void* b) {
  score_item_t* a1 = (score_item_t*)a;
  score_item_t* b1 = (score_item_t*)b;

  if (a1->val < b1->val) return 1;
  if (a1->val == b1->val) return 0;
  if (a1->val > b1->val) return -1;
}
