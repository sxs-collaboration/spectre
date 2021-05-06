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
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/ResizeAndComputePrimitives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
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
          const evolution::dg::subcell::ActiveGrid active_grid) {
  CAPTURE(active_grid);
  const Mesh<3> dg_mesh{5, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  using cons_tag = typename grmhd::ValenciaDivClean::System::variables_tag;
  using prim_tag =
      typename grmhd::ValenciaDivClean::System::primitive_variables_tag;
  using ConsVars = typename cons_tag::type;
  using PrimVars = typename prim_tag::type;

  const size_t active_num_pts =
      active_grid == evolution::dg::subcell::ActiveGrid::Dg
          ? dg_mesh.number_of_grid_points()
          : subcell_mesh.number_of_grid_points();
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{active_num_pts, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric.get(i, i) = 1.0 + 0.01 * i;
  }
  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric{active_num_pts,
                                                              0.0};
  for (size_t i = 0; i < 3; ++i) {
    inv_spatial_metric.get(i, i) = 1.0 / spatial_metric.get(i, i);
  }
  const Scalar<DataVector> sqrt_det_spatial_metric{
      sqrt(get(determinant(spatial_metric)))};

  std::unique_ptr<EquationsOfState::EquationOfState<true, 1>> eos =
      std::make_unique<EquationsOfState::PolytropicFluid<true>>(1.4, 5.0 / 3.0);
  auto prim_vars = make_with_random_values<PrimVars>(
      gen, dist, subcell_mesh.number_of_grid_points());
  get<hydro::Tags::Pressure<DataVector>>(prim_vars) =
      eos->pressure_from_density(
          get<hydro::Tags::RestMassDensity<DataVector>>(prim_vars));
  get<hydro::Tags::SpecificInternalEnergy<DataVector>>(prim_vars) =
      eos->specific_internal_energy_from_density(
          get<hydro::Tags::RestMassDensity<DataVector>>(prim_vars));
  get<hydro::Tags::SpecificEnthalpy<DataVector>>(prim_vars) =
      hydro::relativistic_specific_enthalpy(
          get<hydro::Tags::RestMassDensity<DataVector>>(prim_vars),
          get<hydro::Tags::SpecificInternalEnergy<DataVector>>(prim_vars),
          get<hydro::Tags::Pressure<DataVector>>(prim_vars));
  {
    const auto& spatial_velocity =
        get<hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>>(
            prim_vars);
    tnsr::ii<DataVector, 3, Frame::Inertial> subcell_spatial_metric{
        subcell_mesh.number_of_grid_points(), 0.0};
    for (size_t i = 0; i < 3; ++i) {
      subcell_spatial_metric.get(i, i) = 1.0 + 0.01 * i;
    }
    get(get<hydro::Tags::LorentzFactor<DataVector>>(prim_vars)) =
        1.0 / sqrt(1.0 - get(dot_product(spatial_velocity, spatial_velocity,
                                         subcell_spatial_metric)));
  }
  ConsVars cons_vars{};
  const auto compute_cons = [&cons_vars, &spatial_metric,
                             &sqrt_det_spatial_metric](const auto& prims) {
    cons_vars.initialize(prims.number_of_grid_points());
    ConservativeFromPrimitive::apply(
        make_not_null(&get<Tags::TildeD>(cons_vars)),
        make_not_null(&get<Tags::TildeTau>(cons_vars)),
        make_not_null(&get<Tags::TildeS<Frame::Inertial>>(cons_vars)),
        make_not_null(&get<Tags::TildeB<Frame::Inertial>>(cons_vars)),
        make_not_null(&get<Tags::TildePhi>(cons_vars)),
        get<hydro::Tags::RestMassDensity<DataVector>>(prims),
        get<hydro::Tags::SpecificInternalEnergy<DataVector>>(prims),
        get<hydro::Tags::SpecificEnthalpy<DataVector>>(prims),
        get<hydro::Tags::Pressure<DataVector>>(prims),
        get<hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>>(
            prims),
        get<hydro::Tags::LorentzFactor<DataVector>>(prims),
        get<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>(prims),
        sqrt_det_spatial_metric, spatial_metric,
        get<hydro::Tags::DivergenceCleaningField<DataVector>>(prims));
  };
  if (active_grid == evolution::dg::subcell::ActiveGrid::Dg) {
    const auto dg_prims = evolution::dg::subcell::fd::reconstruct(
        prim_vars, dg_mesh, subcell_mesh.extents());
    compute_cons(dg_prims);
  } else {
    compute_cons(prim_vars);
  }

  auto box = db::create<db::AddSimpleTags<
      evolution::dg::subcell::Tags::ActiveGrid, cons_tag, prim_tag,
      gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>, ::domain::Tags::Mesh<3>,
      evolution::dg::subcell::Tags::Mesh<3>,
      hydro::Tags::EquationOfState<
          std::unique_ptr<EquationsOfState::EquationOfState<true, 1>>>>>(
      active_grid, cons_vars, prim_vars, spatial_metric, inv_spatial_metric,
      sqrt_det_spatial_metric, dg_mesh, subcell_mesh, std::move(eos));

  using recovery_schemes = tmpl::list<
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl,
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin,
      grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>;
  db::mutate_apply<grmhd::ValenciaDivClean::subcell::ResizeAndComputePrims<
      recovery_schemes>>(make_not_null(&box));

  REQUIRE(db::get<prim_tag>(box).number_of_grid_points() == active_num_pts);
  if (active_grid == evolution::dg::subcell::ActiveGrid::Dg) {
    prim_vars.initialize(cons_vars.number_of_grid_points());
    grmhd::ValenciaDivClean::PrimitiveFromConservative<recovery_schemes>::apply(
        make_not_null(
            &get<hydro::Tags::RestMassDensity<DataVector>>(prim_vars)),
        make_not_null(
            &get<hydro::Tags::SpecificInternalEnergy<DataVector>>(prim_vars)),
        make_not_null(
            &get<hydro::Tags::SpatialVelocity<DataVector, 3>>(prim_vars)),
        make_not_null(
            &get<hydro::Tags::MagneticField<DataVector, 3>>(prim_vars)),
        make_not_null(
            &get<hydro::Tags::DivergenceCleaningField<DataVector>>(prim_vars)),
        make_not_null(&get<hydro::Tags::LorentzFactor<DataVector>>(prim_vars)),
        make_not_null(&get<hydro::Tags::Pressure<DataVector>>(prim_vars)),
        make_not_null(
            &get<hydro::Tags::SpecificEnthalpy<DataVector>>(prim_vars)),
        get<grmhd::ValenciaDivClean::Tags::TildeD>(cons_vars),
        get<grmhd::ValenciaDivClean::Tags::TildeTau>(cons_vars),
        get<grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>>(cons_vars),
        get<grmhd::ValenciaDivClean::Tags::TildeB<Frame::Inertial>>(cons_vars),
        get<grmhd::ValenciaDivClean::Tags::TildePhi>(cons_vars), spatial_metric,
        inv_spatial_metric, sqrt_det_spatial_metric,
        db::get<hydro::Tags::EquationOfStateBase>(box));
  }
  CHECK_VARIABLES_APPROX(db::get<prim_tag>(box), prim_vars);
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.Subcell.ResizeAndComputePrims",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  // Use a small range of random values since we need recovery to succeed and we
  // also reconstruct to the DG grid from the FD grid and need to maintain a
  // somewhat reasonable state on both grids.
  std::uniform_real_distribution<> dist(0.5, 0.505);
  for (const auto active_grid : {evolution::dg::subcell::ActiveGrid::Dg,
                                 evolution::dg::subcell::ActiveGrid::Subcell}) {
    test(make_not_null(&gen), make_not_null(&dist), active_grid);
  }
}
}  // namespace
}  // namespace grmhd::ValenciaDivClean
