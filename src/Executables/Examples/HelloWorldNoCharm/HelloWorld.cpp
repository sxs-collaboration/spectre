// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"
#include "Parallel/Printf.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

int main() {
  DataVector a{1.0, 2.3, 8.9};
  Parallel::printf("%s\n", a);
}
