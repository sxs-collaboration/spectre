// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>

#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::creators::time_dependence {

namespace {

template <size_t MeshDim>
void test() {
  const std::unique_ptr<TimeDependence<MeshDim>> time_dep =
      std::make_unique<None<MeshDim>>();
  CHECK(time_dep != nullptr);
  CHECK(time_dep->is_none());

  const std::unique_ptr<TimeDependence<MeshDim>> time_dep_clone =
      time_dep->get_clone();
  CHECK(time_dep_clone != nullptr);

  const std::unique_ptr<TimeDependence<MeshDim>> time_dep_factory =
      TestHelpers::test_creation<std::unique_ptr<TimeDependence<MeshDim>>>(
          "None\n");
  CHECK(time_dep_factory != nullptr);

  CHECK(None<MeshDim>{} == None<MeshDim>{});
  CHECK_FALSE(None<MeshDim>{} != None<MeshDim>{});

  CHECK_THROWS_WITH(
      (time_dep->block_maps_grid_to_inertial(5)),
      Catch::Matchers::Contains(
          "The 'block_maps_grid_to_inertial' function of the 'None'"));
  CHECK_THROWS_WITH(
      (time_dep->block_maps_grid_to_distorted(5)),
      Catch::Matchers::Contains(
          "The 'block_maps_grid_to_distorted' function of the 'None'"));
  CHECK_THROWS_WITH(
      (time_dep->block_maps_distorted_to_inertial(5)),
      Catch::Matchers::Contains(
          "The 'block_maps_distorted_to_inertial' function of the"));
  CHECK_THROWS_WITH((time_dep->functions_of_time()),
                    Catch::Matchers::Contains(
                        "The 'functions_of_time' function of the 'None'"));
}

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependence.None",
                  "[Domain][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace

}  // namespace domain::creators::time_dependence
