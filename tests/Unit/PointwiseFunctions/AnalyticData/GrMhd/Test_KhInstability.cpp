// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>

#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/KhInstability.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace {
struct KhInstabilityProxy : public ::grmhd::AnalyticData::KhInstability {
  using grmhd::AnalyticData::KhInstability::KhInstability;

  template <typename DataType>
  using variables_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataType>,
                 hydro::Tags::ElectronFraction<DataType>,
                 hydro::Tags::SpatialVelocity<DataType, 3>,
                 hydro::Tags::SpecificInternalEnergy<DataType>,
                 hydro::Tags::Pressure<DataType>,
                 hydro::Tags::LorentzFactor<DataType>,
                 hydro::Tags::SpecificEnthalpy<DataType>,
                 hydro::Tags::MagneticField<DataType, 3>,
                 hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<variables_tags<DataType>>
  primitive_variables(const tnsr::I<DataType, 3, Frame::Inertial>& x) const {
    return this->variables(x, variables_tags<DataType>{});
  }
};

template <typename DataType>
void test(const DataType& used_for_size) {
  const double adiabatic_index = 1.43;
  const double strip_bimedian_height = 0.5;
  const double strip_thickness = 0.4;
  const double strip_density = 2.1;
  const double strip_velocity = 0.3;
  const double background_density = 2.0;
  const double background_velocity = -0.2;
  const double pressure = 1.1;
  const double perturbation_amplitude = 0.1;
  const double perturbation_width = 0.01;
  const std::array<double, 3> magnetic_field{{1.0e-3, 0.0, 0.0}};
  const auto members = std::make_tuple(
      adiabatic_index, strip_bimedian_height, strip_thickness, strip_density,
      strip_velocity, background_density, background_velocity, pressure,
      perturbation_amplitude, perturbation_width, magnetic_field);

  register_classes_with_charm<grmhd::AnalyticData::KhInstability>();
  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData,
          grmhd::AnalyticData::KhInstability>(
          "KhInstability:\n"
          "  AdiabaticIndex: 1.43\n"
          "  StripBimedianHeight: 0.5\n"
          "  StripThickness: 0.4\n"
          "  StripDensity: 2.1\n"
          "  StripVelocity: 0.3\n"
          "  BackgroundDensity: 2.0\n"
          "  BackgroundVelocity: -0.2\n"
          "  Pressure: 1.1\n"
          "  PerturbAmplitude: 0.1\n"
          "  PerturbWidth: 0.01\n"
          "  MagneticField: [1.0e-3, 0.0, 0.0]\n")
          ->get_clone();
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& kh_instability =
      dynamic_cast<const grmhd::AnalyticData::KhInstability&>(
          *deserialized_option_solution);

  CHECK(kh_instability == grmhd::AnalyticData::KhInstability(
                              adiabatic_index, strip_bimedian_height,
                              strip_thickness, strip_density, strip_velocity,
                              background_density, background_velocity, pressure,
                              perturbation_amplitude, perturbation_width,
                              magnetic_field));

  KhInstabilityProxy kh_inst_to_move(
      adiabatic_index, strip_bimedian_height, strip_thickness, strip_density,
      strip_velocity, background_density, background_velocity, pressure,
      perturbation_amplitude, perturbation_width, magnetic_field);
  KhInstabilityProxy kh_inst(
      adiabatic_index, strip_bimedian_height, strip_thickness, strip_density,
      strip_velocity, background_density, background_velocity, pressure,
      perturbation_amplitude, perturbation_width, magnetic_field);
  test_move_semantics(std::move(kh_inst_to_move), kh_inst);  //  NOLINT

  // run post-serialized state through checks with random numbers
  pypp::check_with_random_values<1>(
      &KhInstabilityProxy::template primitive_variables<DataType>,
      serialize_and_deserialize(kh_inst), "KhInstability",
      {"rest_mass_density", "electron_fraction", "velocity",
       "specific_internal_energy", "pressure", "lorentz_factor",
       "specific_enthalpy", "magnetic_field", "divergence_cleaning_field"},
      {{{0.0, 1.0}}}, members, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.GrMhd.KhInstability",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/GrMhd"};

  test(std::numeric_limits<double>::signaling_NaN());
  test(DataVector(5));
}
