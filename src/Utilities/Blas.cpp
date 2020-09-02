// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/Blas.hpp"

extern "C" {
#ifdef DISABLE_OPENBLAS_MULTITHREADING
// Declaring this ourselves instead of including cblas.h because our `Blas`
// library does not provide include directories, so cblas.h might not be
// available.
void openblas_set_num_threads(int num_threads);
#endif  // DISABLE_OPENBLAS_MULTITHREADING
}  // extern "C"

void disable_openblas_multithreading() noexcept {
#ifdef DISABLE_OPENBLAS_MULTITHREADING
  openblas_set_num_threads(1);
#endif  // DISABLE_OPENBLAS_MULTITHREADING
}
