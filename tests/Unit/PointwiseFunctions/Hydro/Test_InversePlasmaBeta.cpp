// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/InversePlasmaBeta.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename DataType>
void test_inverse_plasma_beta(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataType> (*)(const Scalar<DataType>&,
                                       const Scalar<DataType>&)>(
          &hydro::inverse_plasma_beta<DataType>),
      "InversePlasmaBeta", "inverse_plasma_beta", {{{0.01, 1.0}}},
      used_for_size);
}
}  // namespace

namespace hydro {
SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.InvserePlasmaBeta",
                  "[Unit][Hydro]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/Hydro"};

  test_inverse_plasma_beta(
      std::numeric_limits<double>::signaling_NaN());
  test_inverse_plasma_beta(DataVector(5));

  // Check compute item works correctly in DataBox
  TestHelpers::db::test_compute_tag<Tags::InversePlasmaBetaCompute<DataVector>>(
      "InversePlasmaBeta");
  const Scalar<DataVector> comoving_magnetic_field_magnitude{
      {{DataVector{5, 0.11}}}};
  const Scalar<DataVector> fluid_pressure{{{DataVector{5, 0.05}}}};

  const auto box =
      db::create<db::AddSimpleTags<
        Tags::ComovingMagneticFieldMagnitude<DataVector>,
        Tags::Pressure<DataVector>>,
                 db::AddComputeTags<
        Tags::InversePlasmaBetaCompute<DataVector>>>(
          comoving_magnetic_field_magnitude, fluid_pressure);
  CHECK(db::get<Tags::InversePlasmaBeta<DataVector>>(box) ==
        inverse_plasma_beta(comoving_magnetic_field_magnitude,
                            fluid_pressure));
}
}  // namespace hydro
