// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cerrno>
#include <cstring>

#include "Utilities/ErrorHandling/Strerror.hpp"

SPECTRE_TEST_CASE("Unit.ErrorHandling.Strerror", "[Unit][ErrorHandling]") {
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  CHECK(strerror_threadsafe(EINVAL) == strerror(EINVAL));
}
