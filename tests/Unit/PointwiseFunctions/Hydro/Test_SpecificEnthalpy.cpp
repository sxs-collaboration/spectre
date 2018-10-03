// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <limits>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {

template <typename DataType>
void test_specific_enthalpy(const DataType& used_for_size) noexcept {
  pypp::check_with_random_values<1>(&hydro::specific_enthalpy<DataType>,
                                    "TestFunctions", "specific_enthalpy",
                                    {{{0.01, 1.0}}}, used_for_size);
}
}  // namespace

namespace hydro {
SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.SpecificEnthalpy",
                  "[Unit][Hydro]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/Hydro"};

  test_specific_enthalpy(std::numeric_limits<double>::signaling_NaN());
  test_specific_enthalpy(DataVector(5));

  // Check compute item works correctly in DataBox
  CHECK(Tags::SpecificEnthalpyCompute<DataVector>::name() ==
        "SpecificEnthalpy");
  Scalar<DataVector> rest_mass_density{{{DataVector{5, 0.2}}}};
  Scalar<DataVector> specific_internal_energy{{{DataVector{5, 0.23}}}};
  Scalar<DataVector> pressure{{{DataVector{5, 0.234}}}};

  const auto box =
      db::create<db::AddSimpleTags<Tags::RestMassDensity<DataVector>,
                                   Tags::SpecificInternalEnergy<DataVector>,
                                   Tags::Pressure<DataVector>>,
                 db::AddComputeTags<Tags::SpecificEnthalpyCompute<DataVector>>>(
          rest_mass_density, specific_internal_energy, pressure);
  CHECK(
      db::get<Tags::SpecificEnthalpy<DataVector>>(box) ==
      specific_enthalpy(rest_mass_density, specific_internal_energy, pressure));
}
}  // namespace hydro
