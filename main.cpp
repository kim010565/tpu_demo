#include "test_bf16.h"

int main(int argc, char **argv) {
  test_exp_diff_bf16();

  test_special_bf16();

  test_ulp_bf16();

  return 0;
}

int sc_main(int argc, char **argv) {
  return 0;
}