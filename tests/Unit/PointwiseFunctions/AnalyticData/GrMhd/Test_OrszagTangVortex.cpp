// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <limits>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/OrszagTangVortex.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_include <pup.h>

namespace {
using grmhd::AnalyticData::OrszagTangVortex;

template <typename DataType>
void test_variables(const DataType& used_for_size) noexcept {
  using tags = tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                          hydro::Tags::SpatialVelocity<DataType, 3>,
                          hydro::Tags::SpecificInternalEnergy<DataType>,
                          hydro::Tags::Pressure<DataType>,
                          hydro::Tags::LorentzFactor<DataType>,
                          hydro::Tags::SpecificEnthalpy<DataType>,
                          hydro::Tags::MagneticField<DataType, 3>,
                          hydro::Tags::DivergenceCleaningField<DataType>>;
  const auto names = make_array(
      "rest_mass_density"s, "spatial_velocity"s, "specific_internal_energy"s,
      "pressure"s, "lorentz_factor"s, "specific_enthalpy"s, "magnetic_field"s,
      "divergence_cleaning_field"s);

  tmpl::for_each<tags>([&used_for_size,
                        name = names.begin() ](auto tag) mutable noexcept {
    using Tag = tmpl::type_from<decltype(tag)>;
    pypp::check_with_random_values<1>(
        +[](const tnsr::I<DataType, 3>& x) noexcept {
          const auto result =
              get<Tag>(OrszagTangVortex{}.variables(x, tmpl::list<Tag>{}));
          CHECK(result == get<Tag>(OrszagTangVortex{}.variables(x, tags{})));
          return result;
        },
        "OrszagTangVortex", *name++, {{{-10., 10.}}}, used_for_size);
  });

  // Check that metric variables are being forwarded
  using lapse_tag = gr::Tags::Lapse<DataType>;
  using LapseType = typename lapse_tag::type;
  CHECK(get<lapse_tag>(OrszagTangVortex{}.variables(
            make_with_value<tnsr::I<DataType, 3>>(used_for_size, 3.),
            tmpl::list<lapse_tag>{})) ==
        make_with_value<LapseType>(used_for_size, 1.));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.GrMhd.OrszagTangVortex",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/GrMhd"};

  test_variables(std::numeric_limits<double>::signaling_NaN());
  test_variables(DataVector(5));

  test_serialization(OrszagTangVortex{});
  test_creation<OrszagTangVortex>("");
}
