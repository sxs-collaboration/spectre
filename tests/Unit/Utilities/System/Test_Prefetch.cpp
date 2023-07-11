// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <vector>

#include "Utilities/GetOutput.hpp"
#include "Utilities/System/Prefetch.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.System.Prefetch", "[Unit][Utilities]") {
  // The goal of this test is just to make sure the code compiles and doesn't
  // segfault.
  std::vector<double> vector(20);
  sys::prefetch<sys::PrefetchTo::L1Cache>(vector.data());
  sys::prefetch<sys::PrefetchTo::L2Cache>(vector.data());
  sys::prefetch<sys::PrefetchTo::L3Cache>(vector.data());

  sys::prefetch<sys::PrefetchTo::NonTemporal>(vector.data());

  sys::prefetch<sys::PrefetchTo::WriteL1Cache>(vector.data());
  sys::prefetch<sys::PrefetchTo::WriteL2Cache>(vector.data());

  CHECK(get_output(sys::PrefetchTo::L1Cache) == "L1Cache");
  CHECK(get_output(sys::PrefetchTo::L2Cache) == "L2Cache");
  CHECK(get_output(sys::PrefetchTo::L3Cache) == "L3Cache");
  CHECK(get_output(sys::PrefetchTo::NonTemporal) == "NonTemporal");
  CHECK(get_output(sys::PrefetchTo::WriteL1Cache) == "WriteL1Cache");
  CHECK(get_output(sys::PrefetchTo::WriteL2Cache) == "WriteL2Cache");
}
