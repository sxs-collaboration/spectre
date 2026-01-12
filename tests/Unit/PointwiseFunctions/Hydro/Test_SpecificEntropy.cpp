// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/SpecificEntropy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace {

template <typename DataType, typename EquationOfStateType>
void test_compute_item_in_databox(
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& temperature,
    const Scalar<DataType>& electron_fraction,
    const EquationOfStateType& equation_of_state) {
  TestHelpers::db::test_compute_tag<hydro::Tags::SpecificEntropyCompute<
      DataType>>("SpecificEntropy");
  const auto box = db::create<
      db::AddSimpleTags<hydro::Tags::RestMassDensity<DataType>,
                        hydro::Tags::Temperature<DataType>,
                        hydro::Tags::ElectronFraction<DataType>,
                        hydro::Tags::GrmhdEquationOfState>,
      db::AddComputeTags<hydro::Tags::SpecificEntropyCompute<DataType>>>(
      rest_mass_density, temperature, electron_fraction,
      equation_of_state.get_clone());

  const auto expected_specific_entropy = hydro::specific_entropy(
      rest_mass_density, temperature, electron_fraction, equation_of_state);
  CHECK(db::get<hydro::Tags::SpecificEntropy<DataType>>(box) ==
        expected_specific_entropy);
}

template <typename DataType>
void test_specific_entropy(const DataType& used_for_size) {
  MAKE_GENERATOR(generator);

  const auto rest_mass_density = TestHelpers::hydro::random_density(
      make_not_null(&generator), used_for_size);
  const auto temperature = TestHelpers::hydro::random_specific_internal_energy(
      make_not_null(&generator), used_for_size);
  const Scalar<DataType> electron_fraction{};

  // check with representative equation of state of two independent variables
  const EquationsOfState::Equilibrium3D<EquationsOfState::IdealFluid<true>> eos(
      EquationsOfState::IdealFluid<true>{5.0 / 3.0});
  CHECK(Scalar<DataType>{get(eos.specific_entropy_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction))} ==
        hydro::specific_entropy(rest_mass_density, temperature,
                                electron_fraction, eos));
  test_compute_item_in_databox(rest_mass_density, temperature,
                               electron_fraction, eos);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.SpecificEntropy",
                  "[Unit][Evolution]") {
  test_specific_entropy(std::numeric_limits<double>::signaling_NaN());
  test_specific_entropy(DataVector(5));
}
