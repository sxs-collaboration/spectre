// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>
// for __decay_and_strip<>::__type
// IWYU pragma: no_include <type_traits>
#include <utility>

#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Utilities/Literals.hpp"  // IWYU pragma: keep
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Time.SimpleBoundaryData", "[Unit][Time]") {
  const Slab slab(0., 1.);

  dg::SimpleBoundaryData<std::string, double> data;
  data = serialize_and_deserialize(data);
  data.local_insert(slab.start(), "string 1");
  data = serialize_and_deserialize(data);
  data.remote_insert(slab.start(), 1.234);
  CHECK(data.extract() == std::make_pair("string 1"s, 1.234));
  data = serialize_and_deserialize(data);
  data.remote_insert(slab.start(), 2.345);
  data = serialize_and_deserialize(data);
  data.local_insert(slab.start(), "string 2");
  CHECK(data.extract() == std::make_pair("string 2"s, 2.345));
}

// [[OutputRegex, Received local data at time Slab\[0,1\]:1/1 but
// already have remote data at time Slab\[0,1\]:0/1]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.SimpleBoundaryData.wrong_time.local",
                               "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const Slab slab(0., 1.);
  dg::SimpleBoundaryData<std::string, double> data;
  data.remote_insert(slab.start(), 0.);
  data.local_insert(slab.end(), "");
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Received remote data at time Slab\[0,1\]:0/1 but
// already have local data at time Slab\[0,1\]:1/1]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Time.SimpleBoundaryData.wrong_time.remote",
                               "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const Slab slab(0., 1.);
  dg::SimpleBoundaryData<std::string, double> data;
  data.local_insert(slab.end(), "");
  data.remote_insert(slab.start(), 0.);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Already received local data]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Time.SimpleBoundaryData.double_insert.local", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const Slab slab(0., 1.);
  dg::SimpleBoundaryData<std::string, double> data;
  data.local_insert(slab.end(), "");
  data.local_insert(slab.end(), "");
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Already received remote data]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Time.SimpleBoundaryData.double_insert.remote", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const Slab slab(0., 1.);
  dg::SimpleBoundaryData<std::string, double> data;
  data.remote_insert(slab.start(), 0.);
  data.remote_insert(slab.start(), 0.);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Tried to extract boundary data, but do not have any data]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Time.SimpleBoundaryData.bad_extract.none", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const Slab slab(0., 1.);
  dg::SimpleBoundaryData<std::string, double> data;
  data.extract();
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Tried to extract boundary data, but do not have remote data]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Time.SimpleBoundaryData.bad_extract.no_remote", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const Slab slab(0., 1.);
  dg::SimpleBoundaryData<std::string, double> data;
  data.local_insert(slab.end(), "");
  data.extract();
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Tried to extract boundary data, but do not have local data]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Time.SimpleBoundaryData.bad_extract.no_local", "[Unit][Time]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  const Slab slab(0., 1.);
  dg::SimpleBoundaryData<std::string, double> data;
  data.remote_insert(slab.start(), 0.);
  data.extract();
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
