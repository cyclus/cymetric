#include <iostream>
#include <cyclus.h>

int main(int argc, char* argv[]) {
  std::cout << "Derp\n";
  cyclus::toolkit::ExponentialFunction* func = new cyclus::toolkit::ExponentialFunction(10.0, -1.0, 0.0);
  std::cout << func->Print() << "\n";
  delete func;
  return 0;
}

