// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/PrimsAfterRollback.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

namespace grmhd::ValenciaDivClean {
namespace {
void test(const gsl::not_null<std::mt19937*> gen,
          const gsl::not_null<std::uniform_real_distribution<>*> dist,
          const bool did_rollback) {
  const Mesh<3> dg_mesh{4, SpatialDiscretization::Basis::Legendre,
                        SpatialDiscretization::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  using cons_tag = typename grmhd::ValenciaDivClean::System::variables_tag;
  using prim_tag =
      typename grmhd::ValenciaDivClean::System::primitive_variables_tag;
  using ConsVars = typename cons_tag::type;
  using PrimVars = typename prim_tag::type;

  const size_t subcell_num_pts = subcell_mesh.number_of_grid_points();
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{subcell_num_pts, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric.get(i, i) = 1.0 + 0.01 * i;
  }
  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric{subcell_num_pts,
                                                              0.0};
  for (size_t i = 0; i < 3; ++i) {
    inv_spatial_metric.get(i, i) = 1.0 / spatial_metric.get(i, i);
  }
  const Scalar<DataVector> sqrt_det_spatial_metric{
      sqrt(get(determinant(spatial_metric)))};

  std::unique_ptr<EquationsOfState::EquationOfState<true, 1>> eos =
      std::make_unique<EquationsOfState::PolytropicFluid<true>>(1.4, 5.0 / 3.0);

  // Compute the conservatives on the FD grid by first computing the primitives
  // on the FD grid, then compute the conservatives from the primitives.
  auto subcell_prims = make_with_random_values<PrimVars>(
      gen, dist, subcell_mesh.number_of_grid_points());
  PrimVars dg_prims{};
  ConsVars subcell_cons{};
  if (did_rollback) {
    subcell_cons.initialize(subcell_mesh.number_of_grid_points());
    get<hydro::Tags::Pressure<DataVector>>(subcell_prims) =
        eos->pressure_from_density(
            get<hydro::Tags::RestMassDensity<DataVector>>(subcell_prims));
    get<hydro::Tags::SpecificInternalEnergy<DataVector>>(subcell_prims) =
        eos->specific_internal_energy_from_density(
            get<hydro::Tags::RestMassDensity<DataVector>>(subcell_prims));
    get<hydro::Tags::SpecificEnthalpy<DataVector>>(subcell_prims) =
        hydro::relativistic_specific_enthalpy(
            get<hydro::Tags::RestMassDensity<DataVector>>(subcell_prims),
            get<hydro::Tags::SpecificInternalEnergy<DataVector>>(subcell_prims),
            get<hydro::Tags::Pressure<DataVector>>(subcell_prims));
    {
      const auto& spatial_velocity =
          get<hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>>(
              subcell_prims);
      get(get<hydro::Tags::LorentzFactor<DataVector>>(subcell_prims)) =
          1.0 / sqrt(1.0 - get(dot_product(spatial_velocity, spatial_velocity,
                                           spatial_metric)));
    }
    ConservativeFromPrimitive::apply(
        make_not_null(&get<Tags::TildeD>(subcell_cons)),
        make_not_null(&get<Tags::TildeYe>(subcell_cons)),
        make_not_null(&get<Tags::TildeTau>(subcell_cons)),
        make_not_null(&get<Tags::TildeS<Frame::Inertial>>(subcell_cons)),
        make_not_null(&get<Tags::TildeB<Frame::Inertial>>(subcell_cons)),
        make_not_null(&get<Tags::TildePhi>(subcell_cons)),
        get<hydro::Tags::RestMassDensity<DataVector>>(subcell_prims),
        get<hydro::Tags::ElectronFraction<DataVector>>(subcell_prims),
        get<hydro::Tags::SpecificInternalEnergy<DataVector>>(subcell_prims),
        get<hydro::Tags::Pressure<DataVector>>(subcell_prims),
        get<hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>>(
            subcell_prims),
        get<hydro::Tags::LorentzFactor<DataVector>>(subcell_prims),
        get<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>(
            subcell_prims),
        sqrt_det_spatial_metric, spatial_metric,
        get<hydro::Tags::DivergenceCleaningField<DataVector>>(subcell_prims));
    dg_prims = evolution::dg::subcell::fd::reconstruct(
        subcell_prims, dg_mesh, subcell_mesh.extents(),
        evolution::dg::subcell::fd::ReconstructionMethod::AllDimsAtOnce);
  }

  const double cutoff_d_for_inversion = 0.0;
  const double density_when_skipping_inversion = 0.0;
  const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions
    primitive_from_conservative_options(cutoff_d_for_inversion,
                                        density_when_skipping_inversion);

  // The DG prims are used as an initial guess so we need to provide them
  auto box = db::create<db::AddSimpleTags<
      evolution::dg::subcell::Tags::DidRollback, cons_tag, prim_tag,
      gr::Tags::SpatialMetric<DataVector, 3>,
      gr::Tags::InverseSpatialMetric<DataVector, 3>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>, ::domain::Tags::Mesh<3>,
      evolution::dg::subcell::Tags::Mesh<3>,
      hydro::Tags::EquationOfState<
          std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>>,
      grmhd::ValenciaDivClean::Tags::PrimitiveFromConservativeOptions>>(
      did_rollback, subcell_cons, dg_prims, spatial_metric, inv_spatial_metric,
      sqrt_det_spatial_metric, dg_mesh, subcell_mesh, std::move(eos),
      primitive_from_conservative_options);

  using recovery_schemes = tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl,
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin,
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>;
  db::mutate_apply<subcell::PrimsAfterRollback<recovery_schemes>>(
      make_not_null(&box));

  if (did_rollback) {
    REQUIRE(db::get<prim_tag>(box).number_of_grid_points() == subcell_num_pts);
    PrimVars expected_subcell_prims = subcell_prims;
    get(get<hydro::Tags::Pressure<DataVector>>(expected_subcell_prims)) =
        evolution::dg::subcell::fd::project(
            get(get<hydro::Tags::Pressure<DataVector>>(dg_prims)), dg_mesh,
            subcell_mesh.extents());
    PrimitiveFromConservative<recovery_schemes>::apply(
        make_not_null(&get<hydro::Tags::RestMassDensity<DataVector>>(
            expected_subcell_prims)),
        make_not_null(&get<hydro::Tags::ElectronFraction<DataVector>>(
            expected_subcell_prims)),
        make_not_null(&get<hydro::Tags::SpecificInternalEnergy<DataVector>>(
            expected_subcell_prims)),
        make_not_null(&get<hydro::Tags::SpatialVelocity<DataVector, 3>>(
            expected_subcell_prims)),
        make_not_null(&get<hydro::Tags::MagneticField<DataVector, 3>>(
            expected_subcell_prims)),
        make_not_null(&get<hydro::Tags::DivergenceCleaningField<DataVector>>(
            expected_subcell_prims)),
        make_not_null(&get<hydro::Tags::LorentzFactor<DataVector>>(
            expected_subcell_prims)),
        make_not_null(
            &get<hydro::Tags::Pressure<DataVector>>(expected_subcell_prims)),
        make_not_null(&get<hydro::Tags::SpecificEnthalpy<DataVector>>(
            expected_subcell_prims)),
        make_not_null(
            &get<hydro::Tags::Temperature<DataVector>>(expected_subcell_prims)),
        get<grmhd::ValenciaDivClean::Tags::TildeD>(box),
        get<grmhd::ValenciaDivClean::Tags::TildeYe>(box),
        get<grmhd::ValenciaDivClean::Tags::TildeTau>(box),
        get<grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>>(box),
        get<grmhd::ValenciaDivClean::Tags::TildeB<Frame::Inertial>>(box),
        get<grmhd::ValenciaDivClean::Tags::TildePhi>(box), spatial_metric,
        inv_spatial_metric, sqrt_det_spatial_metric,
        db::get<hydro::Tags::EquationOfStateBase>(box),
        primitive_from_conservative_options);
    CHECK_VARIABLES_APPROX(db::get<prim_tag>(box), expected_subcell_prims);
  } else {
    CHECK(db::get<prim_tag>(box).number_of_grid_points() == 0);
  }
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.Subcell.PrimsAfterRollback",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  // Use a small range of random values since we need recovery to succeed and we
  // also reconstruct to the DG grid from the FD grid and need to maintain a
  // somewhat reasonable state on both grids.
  std::uniform_real_distribution<> dist(0.5, 0.50005);
  for (const bool did_rollback : {true, false}) {
    test(make_not_null(&gen), make_not_null(&dist), did_rollback);
  }
}
}  // namespace
}  // namespace grmhd::ValenciaDivClean
