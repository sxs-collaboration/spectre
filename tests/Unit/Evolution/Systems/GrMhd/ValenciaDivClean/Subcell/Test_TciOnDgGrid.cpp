// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOnDgGrid.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/Gsl.hpp"

namespace {
enum class TestThis {
  AllGood,
  SmallTildeD,
  InAtmosphere,
  TildeB2TooBig,
  PrimRecoveryFailed,
  PerssonTildeD,
  PerssonTildeTau
};

void test(const TestThis test_this) {
  const EquationsOfState::PolytropicFluid<true> eos{100.0, 2.0};
  const Mesh<3> mesh{6, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  using ConsVars =
      typename grmhd::ValenciaDivClean::System::variables_tag::type;
  using PrimVars = Variables<hydro::grmhd_tags<DataVector>>;

  const double persson_exponent = 4.0;
  PrimVars prim_vars{mesh.number_of_grid_points(), 0.0};
  get(get<hydro::Tags::RestMassDensity<DataVector>>(prim_vars)) = 1.0;
  if (test_this == TestThis::InAtmosphere) {
    get(get<hydro::Tags::RestMassDensity<DataVector>>(prim_vars)) = 1.0e-12;
  }
  get<hydro::Tags::SpecificInternalEnergy<DataVector>>(prim_vars) =
      eos.specific_internal_energy_from_density(
          get<hydro::Tags::RestMassDensity<DataVector>>(prim_vars));
  get(get<hydro::Tags::LorentzFactor<DataVector>>(prim_vars)) = 1.0;
  get<hydro::Tags::Pressure<DataVector>>(prim_vars) = eos.pressure_from_density(
      get<hydro::Tags::RestMassDensity<DataVector>>(prim_vars));
  get<hydro::Tags::SpecificEnthalpy<DataVector>>(prim_vars) =
      hydro::relativistic_specific_enthalpy(
          get<hydro::Tags::RestMassDensity<DataVector>>(prim_vars),
          get<hydro::Tags::SpecificInternalEnergy<DataVector>>(prim_vars),
          get<hydro::Tags::Pressure<DataVector>>(prim_vars));
  // set magnetic field to tiny but non-zero value
  for (size_t i = 0; i < 3; ++i) {
    get<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>(prim_vars)
        .get(i) = 1.0e-50;
  }

  // Just use flat space since none of the TCI checks really depend on the
  // spacetime variables.
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{
      mesh.number_of_grid_points(), 0.0};
  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric{
      mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric.get(i, i) = inv_spatial_metric.get(i, i) = 1.0;
  }
  const Scalar<DataVector> sqrt_det_spatial_metric{mesh.number_of_grid_points(),
                                                   1.0};

  const grmhd::ValenciaDivClean::subcell::TciOptions tci_options{
      1.0e-20, 1.0e-40, 1.1e-12, 1.0e-12};

  auto box = db::create<db::AddSimpleTags<
      ::Tags::Variables<typename ConsVars::tags_list>,
      ::Tags::Variables<typename PrimVars::tags_list>, ::domain::Tags::Mesh<3>,
      hydro::Tags::EquationOfState<
          std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>>,
      gr::Tags::SqrtDetSpatialMetric<>, gr::Tags::SpatialMetric<3>,
      gr::Tags::InverseSpatialMetric<3>,
      grmhd::ValenciaDivClean::subcell::Tags::TciOptions>>(
      ConsVars{mesh.number_of_grid_points()}, prim_vars, mesh,
      std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>{
          std::make_unique<EquationsOfState::PolytropicFluid<true>>(eos)},
      sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric, tci_options);

  db::mutate_apply<grmhd::ValenciaDivClean::ConservativeFromPrimitive>(
      make_not_null(&box));

  // set B and Phi to NaN since they should be set by recovery
  db::mutate<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>,
             hydro::Tags::DivergenceCleaningField<DataVector>>(
      make_not_null(&box), [](const auto mag_field_ptr, const auto phi_ptr) {
        for (size_t i = 0; i < 3; ++i) {
          mag_field_ptr->get(i) = std::numeric_limits<double>::signaling_NaN();
        }
        get(*phi_ptr) = std::numeric_limits<double>::signaling_NaN();
      });

  const size_t point_to_change = mesh.number_of_grid_points() / 2;
  if (test_this == TestThis::SmallTildeD) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeD>(
        make_not_null(&box), [point_to_change](const auto tilde_d_ptr) {
          get(*tilde_d_ptr)[point_to_change] = 1.0e-30;
        });
  } else if (test_this == TestThis::InAtmosphere) {
    // Make sure the PerssonTCI would trigger in the atmosphere to verify that
    // the reason we didn't mark the cell as troubled is because we're in
    // atmosphere.
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeTau>(
        make_not_null(&box), [point_to_change](const auto tilde_tau_ptr) {
          get(*tilde_tau_ptr)[point_to_change] = 1.0e5;
        });
  } else if (test_this == TestThis::TildeB2TooBig) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeB<Frame::Inertial>>(
        make_not_null(&box), [point_to_change](const auto tilde_b_ptr) {
          get<0>(*tilde_b_ptr)[point_to_change] = 1.0e4;
          get<1>(*tilde_b_ptr)[point_to_change] = 1.0e4;
          get<2>(*tilde_b_ptr)[point_to_change] = 1.0e4;
        });
  } else if (test_this == TestThis::PrimRecoveryFailed) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeB<Frame::Inertial>,
               grmhd::ValenciaDivClean::Tags::TildeTau>(
        make_not_null(&box),
        [point_to_change, &tci_options](const auto tilde_b_ptr,
                                        const auto tilde_tau_ptr) {
          // The values here are chosen specifically so that the Newman-Hamlin
          // recovery scheme fails to obtain a solution.
          get(*tilde_tau_ptr)[point_to_change] = 4.32044e-24;

          get<0>(*tilde_b_ptr)[point_to_change] =
              sqrt(2.0 * (1.0 - tci_options.safety_factor_for_magnetic_field) *
                   get(*tilde_tau_ptr)[point_to_change]);
          get<1>(*tilde_b_ptr)[point_to_change] = 0.0;
          get<2>(*tilde_b_ptr)[point_to_change] = 0.0;
        });
  } else if (test_this == TestThis::PerssonTildeTau) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeTau>(
        make_not_null(&box), [point_to_change](const auto tilde_tau_ptr) {
          get(*tilde_tau_ptr)[point_to_change] *= 2.0;
        });
  } else if (test_this == TestThis::PerssonTildeD) {
    db::mutate<grmhd::ValenciaDivClean::Tags::TildeD>(
        make_not_null(&box), [point_to_change](const auto tilde_d_ptr) {
          get(*tilde_d_ptr)[point_to_change] *= 2.0;
        });
  }

  const bool result =
      db::mutate_apply<grmhd::ValenciaDivClean::subcell::TciOnDgGrid<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin>>(
          make_not_null(&box), persson_exponent);

  if (test_this == TestThis::AllGood or test_this == TestThis::InAtmosphere) {
    CHECK_FALSE(result);
    CHECK(db::get<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>(
              box) ==
          get<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>(
              prim_vars));
    CHECK(db::get<hydro::Tags::DivergenceCleaningField<DataVector>>(box) ==
          get<hydro::Tags::DivergenceCleaningField<DataVector>>(prim_vars));
  } else {
    CHECK(result);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ValenciaDivClean.Subcell.TciOnDgGrid",
                  "[Unit][Evolution]") {
  for (const TestThis& test_this :
       {TestThis::AllGood, TestThis::SmallTildeD, TestThis::InAtmosphere,
        TestThis::TildeB2TooBig, TestThis::PrimRecoveryFailed,
        TestThis::PerssonTildeD, TestThis::PerssonTildeTau}) {
    test(test_this);
  }
}
