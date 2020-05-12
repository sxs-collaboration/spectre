// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"

namespace {
template <size_t Dim>
void test() {
  constexpr size_t num_points = 5;
  const tnsr::aa<DataVector, Dim> spacetime_metric{num_points};
  tnsr::aa<DataVector, Dim> normal_dot_flux_spacetime_metric{num_points};
  tnsr::aa<DataVector, Dim> normal_dot_flux_pi{num_points};
  tnsr::iaa<DataVector, Dim> normal_dot_flux_phi{num_points};

  GeneralizedHarmonic::ComputeNormalDotFluxes<Dim>::apply(
      make_not_null(&normal_dot_flux_spacetime_metric),
      make_not_null(&normal_dot_flux_pi), make_not_null(&normal_dot_flux_phi),
      spacetime_metric);

  const DataVector zero{num_points, 0.};
  for (size_t storage_index = 0;
       storage_index < normal_dot_flux_spacetime_metric.size();
       ++storage_index) {
    CHECK(normal_dot_flux_spacetime_metric[storage_index] == zero);
    CHECK(normal_dot_flux_pi[storage_index] == zero);
  }

  for (size_t storage_index = 0;
       storage_index < normal_dot_flux_phi.size();
       ++storage_index) {
    CHECK(normal_dot_flux_phi[storage_index] == zero);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.NormalDotFluxes",
                  "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
