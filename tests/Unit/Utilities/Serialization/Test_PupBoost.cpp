// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/rational.hpp>
#include <cstddef>

#include "Framework/TestHelpers.hpp"
#include "Utilities/Serialization/PupBoost.hpp"

SPECTRE_TEST_CASE("Unit.Serialization.PupBoost", "[Unit][Serialization]") {
  boost::rational<size_t> r1(3_st, 4_st);
  test_serialization(r1);
  boost::rational<int> r2(-5, 2);
  test_serialization(r2);
}
