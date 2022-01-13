// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Initialize.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace {
template <size_t Dim>
void test_initialize_constraint_damping_gammas() {
  auto box =
      db::create<db::AddSimpleTags<CurvedScalarWave::Tags::ConstraintGamma1,
                                   CurvedScalarWave::Tags::ConstraintGamma2,
                                   domain::Tags::Mesh<Dim>>>(
          Scalar<DataVector>{}, Scalar<DataVector>{},
          Mesh<Dim>{4, Spectral::Basis::Chebyshev,
                    Spectral::Quadrature::Gauss});
  db::mutate_apply<
      CurvedScalarWave::Initialization::InitializeConstraintDampingGammas<Dim>>(
      make_not_null(&box));
  CHECK(get<CurvedScalarWave::Tags::ConstraintGamma1>(box) ==
        Scalar<DataVector>{pow(4, Dim), 0.});
  CHECK(get<CurvedScalarWave::Tags::ConstraintGamma2>(box) ==
        Scalar<DataVector>{pow(4, Dim), 1.});
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.CurvedScalarWave.InitializeConstraintGammas",
    "[Unit][Evolution]") {
  test_initialize_constraint_damping_gammas<1>();
  test_initialize_constraint_damping_gammas<2>();
  test_initialize_constraint_damping_gammas<3>();
}
