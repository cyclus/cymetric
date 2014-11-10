#include <iostream>
#include <cyclus.h>
#include <hdf5_back.h>

int main(int argc, char* argv[]) {
  using std::cout;
  cout << "Derp\n";
  std::string table = "Compositions";
  std::string fname = std::string(argv[1]);
  cout << "file name: " << fname << "\n";
  cyclus::FullBackend* fback = new cyclus::Hdf5Back(fname);
  cyclus::QueryResult result = fback->Query(table, NULL);
  delete fback;
  return 0;
}

