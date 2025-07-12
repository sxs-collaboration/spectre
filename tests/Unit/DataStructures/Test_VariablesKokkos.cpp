// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "DataStructures/Tags/MirrorView.hpp"
#include "DataStructures/VariablesKokkos.hpp"
#include "Helpers/DataStructures/TestTags.hpp"

#ifdef SPECTRE_KOKKOS

SPECTRE_TEST_CASE("Unit.DataStructures.VariablesKokkos",
                  "[DataStructures][Unit]") {
  const size_t num_points = 5;
  using VectorTag = TestHelpers::Tags::Vector<DataVector>;
  using ScalarTag = TestHelpers::Tags::Scalar<DataVector>;
  using DeviceScalarTag = Tags::MirrorView<ScalarTag>;
  using HostMirrorScalarTag =
      Tags::MirrorView<DeviceScalarTag, Kokkos::HostSpace>;
  static_assert(
      std::is_same_v<DeviceScalarTag::type, Scalar<Kokkos::View<double*>>>);
  static_assert(std::is_same_v<HostMirrorScalarTag::type::type::memory_space,
                               Kokkos::HostSpace>);

  // Fill variables on host and copy to device
  Variables<tmpl::list<VectorTag, ScalarTag>> vars_host{num_points, 1.0};
  auto vars_device = copy_to_device(vars_host);
  CHECK(vars_device.number_of_grid_points() == num_points);
  CHECK(vars_device.size() == num_points * 4);

  // Compute on device
  Kokkos::parallel_for(
      "compute", num_points, KOKKOS_LAMBDA(const size_t i) {
        auto& [vec, scal] = vars_device;
        for (size_t d = 0; d < 3; ++d) {
          vec.get(d)[i] += static_cast<double>(d + i);
        }
        get(scal)[i] += static_cast<double>(i);
        // Test make_at_index and set_at_index
        auto vars_at_index = make_at_index(vars_device, i);
        auto& [vec_at_index, scal_at_index] = vars_at_index;
        get(scal_at_index) += 2.0;
        set_at_index(make_not_null(&vars_device), vars_at_index, i);
      });

  // Copy back to host and check
  copy_to_host(make_not_null(&vars_host), vars_device);
  for (size_t i = 0; i < num_points; ++i) {
    for (size_t d = 0; d < 3; ++d) {
      CHECK(get<VectorTag>(vars_host).get(d)[i] ==
            1. + static_cast<double>(d + i));
    }
    CHECK(get(get<ScalarTag>(vars_host))[i] == 3. + static_cast<double>(i));
  }

  // Zero tags
  const Variables<tmpl::list<>> empty_host{};
  Variables<tmpl::list<>> empty_result{};
  copy_to_host(make_not_null(&empty_result), copy_to_device(empty_host));
  CHECK(empty_result.size() == 0);
}

#endif  // SPECTRE_KOKKOS
