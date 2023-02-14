#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <lapacke.h>

int main() {
  // Initialize features and targets
  int m, n;

  // bias is included in the X vector
  double *X = malloc(m * n * sizeof(double));
  double *y = malloc(m * sizeof(double));

  double *XT = malloc(m * n * sizeof(double));
}

void transpose(double *A, int m, int n) {
  // m is the n_rows, n is the n_cols
  double tmp;
  for (int i=0; i<m; i++) {
    for (int j=0; j < n; j++) {
      tmp = A[i * n + j];
      A[i * n + j] = A[j * n + i];
      A[j * n + i] = tmp;
      }
    }
  }

void copy_matrix(double *A, double *copy_to, int m, int n) {
  for (int i=0; i<m*n; i++) {
      copy_to[i] = A[i];
  }
}