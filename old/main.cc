#include <iostream>
#include <cyclus.h>

int main(int argc, char* argv[]) {
  using std::cout;
  cout << "Derp\n";
  cyclus::toolkit::ExponentialFunction* func = new cyclus::toolkit::ExponentialFunction(10.0, -1.0, 0.0);
  cout << func->Print() << "\n";
  delete func;
  return 0;
}

