// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/PrimitiveGhostData.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using copied_tags =
    tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
               hydro::Tags::ElectronFraction<DataVector>,
               hydro::Tags::Pressure<DataVector>,
               hydro::Tags::MagneticField<DataVector, 3>,
               hydro::Tags::DivergenceCleaningField<DataVector>>;
using gh_tags = grmhd::GhValenciaDivClean::Tags::spacetime_reconstruction_tags;
using tags_for_reconstruction = grmhd::GhValenciaDivClean::Tags::
    primitive_grmhd_and_spacetime_reconstruction_tags;

void test_primitive_ghost_data_on_subcells(
    const gsl::not_null<std::mt19937*> gen,
    const gsl::not_null<std::uniform_real_distribution<>*> dist) {
  const Mesh<3> dg_mesh{5, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const auto prims =
      make_with_random_values<Variables<hydro::grmhd_tags<DataVector>>>(
          gen, dist, subcell_mesh.number_of_grid_points());
  const auto gh_vars = make_with_random_values<Variables<gh_tags>>(
      gen, dist, subcell_mesh.number_of_grid_points());

  auto box = db::create<
      db::AddSimpleTags<::Tags::Variables<hydro::grmhd_tags<DataVector>>,
                        ::Tags::Variables<gh_tags>>>(prims, gh_vars);
  DataVector recons_prims_rdmp = db::mutate_apply<
      grmhd::GhValenciaDivClean::subcell::PrimitiveGhostVariables>(
      make_not_null(&box), 2_st);
  const Variables<tags_for_reconstruction> recons_prims{
      recons_prims_rdmp.data(), recons_prims_rdmp.size() - 2};
  tmpl::for_each<copied_tags>([&prims, &recons_prims](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    CHECK_ITERABLE_APPROX(get<tag>(recons_prims), get<tag>(prims));
  });
  tmpl::for_each<gh_tags>([&gh_vars, &recons_prims](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    CHECK_ITERABLE_APPROX(get<tag>(recons_prims), get<tag>(gh_vars));
  });
  auto lorentz_factor_times_v_I =
      get<hydro::Tags::SpatialVelocity<DataVector, 3>>(prims);
  for (auto& component : lorentz_factor_times_v_I) {
    component *= get(get<hydro::Tags::LorentzFactor<DataVector>>(prims));
  }
  CHECK_ITERABLE_APPROX(
      (get<hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>>(
          recons_prims)),
      lorentz_factor_times_v_I);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GhValenciaDivClean.Subcell.PrimitiveGhostData",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.0, 1.0);
  test_primitive_ghost_data_on_subcells(make_not_null(&gen),
                                        make_not_null(&dist));
}
