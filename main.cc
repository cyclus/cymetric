#include <iostream>
#include <cyclus.h>

int main(int argc, char* argv[]) {
  std::cout << "Derp\n";
  std::string table = "Compositions";
  cyclus::QueryResult result = cyclus::QueryableBackend::Query(table, NULL);
  return 0;
}

