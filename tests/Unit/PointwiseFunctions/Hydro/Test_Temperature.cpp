// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "IO/Connectivity.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/EosTable.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Header.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/OpenGroup.hpp"
#include "IO/H5/SourceArchive.hpp"
#include "IO/H5/Version.hpp"
#include "IO/H5/Wrappers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Factory.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Tabulated3d.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/Temperature.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace hydro {
namespace {
template <typename EosType, size_t Dim>
class DummySolution
    : public TemperatureInitialization<DummySolution<EosType, Dim>> {
 public:
  DummySolution(EosType eos) : eos_(std::move(eos)) {}

  template <typename DataType>
  tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>> variables(
      const tnsr::I<DataType, Dim>& coords,
      tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const {
    if constexpr (std::is_same_v<DataType, double>) {
      return {Scalar<DataType>(1.28e-3)};
    } else {
      return {Scalar<DataType>(get<0>(coords).size(), 1.28e-3)};
    }
  }

  template <typename DataType>
  tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>> variables(
      const tnsr::I<DataType, Dim>& coords,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/)
      const {
    if constexpr (std::is_same_v<DataType, double>) {
      return {Scalar<DataType>(1.)};
    } else {
      return {Scalar<DataType>(get<0>(coords).size(), 1.)};
    }
  }

  template <typename DataType>
  tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>> variables(
      const tnsr::I<DataType, Dim>& coords,
      tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/) const {
    if constexpr (std::is_same_v<DataType, double>) {
      return {Scalar<DataType>(0.1)};
    } else {
      return {Scalar<DataType>(get<0>(coords).size(), 0.1)};
    }
  }

  template <typename DataType>
  tuples::TaggedTuple<hydro::Tags::Temperature<DataType>> variables(
      const tnsr::I<DataType, Dim>& coords,
      tmpl::list<hydro::Tags::Temperature<DataType>> tvar) const {
    return TemperatureInitialization<DummySolution<EosType, Dim>>::variables(
        coords, tvar);
  }

  EosType const equation_of_state() const { return eos_; };

 private:
  EosType eos_{};
};
}  // namespace

template <size_t Dim>
void test() {
  const tnsr::I<DataVector, Dim> coords_dv{10u, 3.0};
  const tnsr::I<double, Dim> coords_double{13.0};

  {
    EquationsOfState::PolytropicFluid<true> polytrope{};
    DummySolution<EquationsOfState::PolytropicFluid<true>, Dim> solution{
        polytrope};

    CHECK(get<hydro::Tags::Temperature<DataVector>>(solution.variables(
              coords_dv, tmpl::list<hydro::Tags::Temperature<DataVector>>{})) ==
          polytrope.temperature_from_density(Scalar<DataVector>(10u, 1.28e-3)));

    CHECK(get<hydro::Tags::Temperature<double>>(solution.variables(
              coords_double, tmpl::list<hydro::Tags::Temperature<double>>{})) ==
          polytrope.temperature_from_density(Scalar<double>(1.28e-3)));
  }

  {
    const auto ideal_fluid = EquationsOfState::IdealFluid<true>{1.5};
    DummySolution<EquationsOfState::IdealFluid<true>, Dim> solution{
        ideal_fluid};

    CHECK(get<hydro::Tags::Temperature<DataVector>>(solution.variables(
              coords_dv, tmpl::list<hydro::Tags::Temperature<DataVector>>{})) ==
          ideal_fluid.temperature_from_density_and_energy(
              Scalar<DataVector>(10u, 1.28e-3), Scalar<DataVector>(10u, 1.)));

    CHECK(get<hydro::Tags::Temperature<double>>(solution.variables(
              coords_double, tmpl::list<hydro::Tags::Temperature<double>>{})) ==
          ideal_fluid.temperature_from_density_and_energy(
              Scalar<double>(1.28e-3), Scalar<double>(1.)));
  }

  {
    std::string h5_file_name{
        unit_test_src_path() +
        "PointwiseFunctions/Hydro/EquationsOfState/dd2_unit_test.h5"};

    h5::H5File<h5::AccessType::ReadOnly> eos_file{h5_file_name};
    const auto& compose_eos = eos_file.get<h5::EosTable>("/dd2");

    EquationsOfState::Tabulated3D<true> eos;
    eos.initialize(compose_eos);

    DummySolution<EquationsOfState::Tabulated3D<true>, Dim> solution{eos};

    CHECK(get<hydro::Tags::Temperature<DataVector>>(solution.variables(
              coords_dv, tmpl::list<hydro::Tags::Temperature<DataVector>>{})) ==
          eos.temperature_from_density_and_energy(
              Scalar<DataVector>(10u, 1.28e-3), Scalar<DataVector>(10u, 1.),
              Scalar<DataVector>(10u, 0.1)));

    CHECK(get<hydro::Tags::Temperature<double>>(solution.variables(
              coords_double, tmpl::list<hydro::Tags::Temperature<double>>{})) ==
          eos.temperature_from_density_and_energy(Scalar<double>(1.28e-3),
                                                  Scalar<double>(1.),
                                                  Scalar<double>(0.1)));
  }
}

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.Temperature",
                  "[Unit][Hydro]") {
  test<1>();
  test<2>();
  test<3>();
}

}  // namespace hydro
