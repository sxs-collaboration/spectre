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
}

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependence.None",
                  "[Domain][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}

template <size_t MeshDim>
void test_error_block_maps_grid_to_inertial() {
  const std::unique_ptr<TimeDependence<MeshDim>> time_dep =
      std::make_unique<None<MeshDim>>();
  (void)time_dep->block_maps_grid_to_inertial(5);
}

template <size_t MeshDim>
void test_error_block_maps_grid_to_distorted() {
  const std::unique_ptr<TimeDependence<MeshDim>> time_dep =
      std::make_unique<None<MeshDim>>();
  (void)time_dep->block_maps_grid_to_distorted(5);
}

template <size_t MeshDim>
void test_error_block_maps_distorted_to_inertial() {
  const std::unique_ptr<TimeDependence<MeshDim>> time_dep =
      std::make_unique<None<MeshDim>>();
  (void)time_dep->block_maps_distorted_to_inertial(5);
}

template <size_t MeshDim>
void test_error_functions_of_time() {
  const std::unique_ptr<TimeDependence<MeshDim>> time_dep =
      std::make_unique<None<MeshDim>>();
  (void)time_dep->functions_of_time();
}

// [[OutputRegex, The 'block_maps_grid_to_inertial' function of the 'None']]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Domain.Creators.TimeDependence.None.ErrorBlockMaps0",
    "[Domain][Unit]") {
  ERROR_TEST();

  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> dist_size_t{1, 3};
  const size_t mesh_dim = dist_size_t(gen);

  if (mesh_dim == 1) {
    test_error_block_maps_grid_to_inertial<1>();
  } else if (mesh_dim == 2) {
    test_error_block_maps_grid_to_inertial<2>();
  } else if (mesh_dim == 3) {
    test_error_block_maps_grid_to_inertial<3>();
  }
  ERROR("Bad MeshDim in test: " << mesh_dim);
}

// [[OutputRegex, The 'block_maps_grid_to_distorted' function of the 'None']]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Domain.Creators.TimeDependence.None.ErrorBlockMaps1",
    "[Domain][Unit]") {
  ERROR_TEST();

  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> dist_size_t{1, 3};
  const size_t mesh_dim = dist_size_t(gen);

  if (mesh_dim == 1) {
    test_error_block_maps_grid_to_distorted<1>();
  } else if (mesh_dim == 2) {
    test_error_block_maps_grid_to_distorted<2>();
  } else if (mesh_dim == 3) {
    test_error_block_maps_grid_to_distorted<3>();
  }
  ERROR("Bad MeshDim in test: " << mesh_dim);
}

// [[OutputRegex, The 'block_maps_distorted_to_inertial' function of the ]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Domain.Creators.TimeDependence.None.ErrorBlockMaps2",
    "[Domain][Unit]") {
  ERROR_TEST();

  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> dist_size_t{1, 3};
  const size_t mesh_dim = dist_size_t(gen);

  if (mesh_dim == 1) {
    test_error_block_maps_distorted_to_inertial<1>();
  } else if (mesh_dim == 2) {
    test_error_block_maps_distorted_to_inertial<2>();
  } else if (mesh_dim == 3) {
    test_error_block_maps_distorted_to_inertial<3>();
  }
  ERROR("Bad MeshDim in test: " << mesh_dim);
}

// [[OutputRegex, The 'functions_of_time' function of the 'None']]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Domain.Creators.TimeDependence.None.ErrorFunctionsOfTime",
    "[Domain][Unit]") {
  ERROR_TEST();
  MAKE_GENERATOR(gen);

  UniformCustomDistribution<size_t> dist_size_t{1, 3};
  const size_t mesh_dim = dist_size_t(gen);

  if (mesh_dim == 1) {
    test_error_functions_of_time<1>();
  } else if (mesh_dim == 2) {
    test_error_functions_of_time<2>();
  } else if (mesh_dim == 3) {
    test_error_functions_of_time<3>();
  }
  ERROR("Bad MeshDim in test: " << mesh_dim);
}

}  // namespace

}  // namespace domain::creators::time_dependence
