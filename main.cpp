#include "ceml/operations.hpp"
#include "ceml/tensor.hpp"
#include "include/ceml/tensor.hpp"
#include <iostream>

int main() {
  ceml::Tensor<double, 5, 1, 1> x;
  ceml::Tensor<double, 5, 1, 1> y;

  x.allocate();
  y.allocate();
  for (uint i = 0; i < x.size1; i++) {
    x.assign(i, 0, 0, 1);
    y.assign(i, 0, 0, 1);
  }

  // auto z = ceml::ewiseMult<double>(ceml::add<double>(ceml::add<double>(x,y),
  // x), y);

  auto z = ceml::add<double>(ceml::add<double>(x, y), y);

  std::cout << x.size1 << std::endl;
  std::cout << y.size1 << std::endl;
  std::cout << z.size1 << std::endl;

  auto res = z.materialise();

  std::cout << z(0, 0, 0) << std::endl;

  return 0;
};