// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>

#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/SlabJet.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace {
struct SlabJetProxy : public ::grmhd::AnalyticData::SlabJet {
  using grmhd::AnalyticData::SlabJet::SlabJet;

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
  const double adiabatic_index = 1.33333333333333;
  const double ambient_density = 10.;
  const double ambient_pressure = 0.01;
  const double ambient_electron_fraction = 0.;
  const double jet_density = 0.1;
  const double jet_pressure = 0.01;
  const double jet_electron_fraction = 0.;
  const std::array<double, 3> jet_velocity{{0.9987523388778445, 0., 0.}};
  const double inlet_radius = 1.;
  const std::array<double, 3> magnetic_field{{1., 0., 0.}};
  const auto members = std::make_tuple(
      adiabatic_index, ambient_density, ambient_pressure,
      ambient_electron_fraction, jet_density, jet_pressure,
      jet_electron_fraction, jet_velocity, inlet_radius, magnetic_field);

  register_classes_with_charm<grmhd::AnalyticData::SlabJet>();
  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData,
          grmhd::AnalyticData::SlabJet>(
          "SlabJet:\n"
          "  AdiabaticIndex: 1.33333333333333\n"
          "  AmbientDensity: 10.\n"
          "  AmbientPressure: 0.01\n"
          "  AmbientElectronFraction: 0.\n"
          "  JetDensity: 0.1\n"
          "  JetPressure: 0.01\n"
          "  JetElectronFraction: 0.\n"
          "  JetVelocity: [0.9987523388778445, 0., 0.]\n"
          "  InletRadius: 1.\n"
          "  MagneticField: [1., 0., 0.]\n")
          ->get_clone();
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& slab_jet = dynamic_cast<const grmhd::AnalyticData::SlabJet&>(
      *deserialized_option_solution);

  CHECK(slab_jet == grmhd::AnalyticData::SlabJet(
                        adiabatic_index, ambient_density, ambient_pressure,
                        ambient_electron_fraction, jet_density, jet_pressure,
                        jet_electron_fraction, jet_velocity, inlet_radius,
                        magnetic_field));

  {
    INFO("Semantics");
    auto to_move = slab_jet;
    test_move_semantics(std::move(to_move), slab_jet);
  }

  const SlabJetProxy proxy{adiabatic_index,       ambient_density,
                           ambient_pressure,      ambient_electron_fraction,
                           jet_density,           jet_pressure,
                           jet_electron_fraction, jet_velocity,
                           inlet_radius,          magnetic_field};
  pypp::check_with_random_values<1>(
      &SlabJetProxy::template primitive_variables<DataType>, proxy, "SlabJet",
      {"rest_mass_density", "electron_fraction", "spatial_velocity",
       "specific_internal_energy", "pressure", "lorentz_factor",
       "specific_enthalpy", "magnetic_field", "divergence_cleaning_field"},
      {{{0.0, 1.0}}}, members, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.GrMhd.SlabJet",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticData/GrMhd"};

  test(std::numeric_limits<double>::signaling_NaN());
  test(DataVector(5));
}
