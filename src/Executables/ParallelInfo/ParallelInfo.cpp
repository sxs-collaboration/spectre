
// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <iostream>

#include "Informer/InfoFromBuild.hpp"

extern "C" void CkRegisterMainModule(void) {}

int main() {
  printf("%s", info_from_build().c_str());
  return 0;
}
