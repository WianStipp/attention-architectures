#include <stdio.h>
#include <cblas.h>

int main() {
  double A[6] = {1, 2, 3, 4, 5, 6};
  double x[3] = {1, 2, 3};
  double y[2] = {0};

  // A*x
  cblas_dgemv(CblasRowMajor, CblasNoTrans, 2, 3, 1.0, A, 3, x, 1, 0.0, y, 1);
  printf("y = [%f, %f]\n", y[0], y[1]);
  return 0;
}
