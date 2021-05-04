// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/NewtonianEuler/Subcell/PrimitiveGhostData.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace {
template <size_t Dim>
void test_subcells(
    const gsl::not_null<std::mt19937*> gen,
    const gsl::not_null<std::uniform_real_distribution<>*> dist) {
  using MassDensity = NewtonianEuler::Tags::MassDensity<DataVector>;
  using Velocity = NewtonianEuler::Tags::Velocity<DataVector, Dim>;
  using SpecificInternalEnergy =
      NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>;
  using Pressure = NewtonianEuler::Tags::Pressure<DataVector>;
  using prim_tags =
      tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>;
  using PrimVars = Variables<prim_tags>;
  using prims_to_reconstruct_tags = tmpl::list<MassDensity, Velocity, Pressure>;
  using ReconsPrimVars = Variables<prims_to_reconstruct_tags>;

  const Mesh<Dim> subcell_mesh{9, Spectral::Basis::FiniteDifference,
                               Spectral::Quadrature::CellCentered};
  // NOLINTNEXTLINE(modernize-use-auto)
  const PrimVars prims = make_with_random_values<PrimVars>(
      gen, dist, subcell_mesh.number_of_grid_points());

  auto box = db::create<db::AddSimpleTags<::Tags::Variables<prim_tags>>>(prims);
  const ReconsPrimVars prims_to_reconstruct = db::mutate_apply<
      NewtonianEuler::subcell::PrimitiveGhostDataOnSubcells<Dim>>(
      make_not_null(&box));
  tmpl::for_each<prims_to_reconstruct_tags>(
      [&expected_prims = prims, &prims_to_reconstruct](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        CHECK_ITERABLE_APPROX(get<tag>(prims_to_reconstruct),
                              get<tag>(expected_prims));
      });
}

template <size_t Dim>
void test_dg(const gsl::not_null<std::mt19937*> gen,
             const gsl::not_null<std::uniform_real_distribution<>*> dist) {
  using MassDensity = NewtonianEuler::Tags::MassDensity<DataVector>;
  using Velocity = NewtonianEuler::Tags::Velocity<DataVector, Dim>;
  using SpecificInternalEnergy =
      NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>;
  using Pressure = NewtonianEuler::Tags::Pressure<DataVector>;
  using prim_tags =
      tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>;
  using PrimVars = Variables<prim_tags>;
  using prims_to_reconstruct_tags = tmpl::list<MassDensity, Velocity, Pressure>;
  using ReconsPrimVars = Variables<prims_to_reconstruct_tags>;

  const Mesh<Dim> dg_mesh{5, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  // NOLINTNEXTLINE(modernize-use-auto)
  const PrimVars prims = make_with_random_values<PrimVars>(
      gen, dist, dg_mesh.number_of_grid_points());

  auto box = db::create<
      db::AddSimpleTags<::Tags::Variables<prim_tags>, domain::Tags::Mesh<Dim>,
                        evolution::dg::subcell::Tags::Mesh<Dim>>>(
      prims, dg_mesh, subcell_mesh);
  const ReconsPrimVars prims_to_reconstruct =
      db::mutate_apply<NewtonianEuler::subcell::PrimitiveGhostDataToSlice<Dim>>(
          make_not_null(&box));

  const PrimVars expected_prims = evolution::dg::subcell::fd::project(
      prims, dg_mesh, subcell_mesh.extents());
  tmpl::for_each<prims_to_reconstruct_tags>(
      [&prims_to_reconstruct, &expected_prims](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        CHECK_ITERABLE_APPROX(get<tag>(prims_to_reconstruct),
                              get<tag>(expected_prims));
      });
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.Subcell.PrimitiveGhostData",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.0, 1.0);
  test_subcells<1>(make_not_null(&gen), make_not_null(&dist));
  test_subcells<2>(make_not_null(&gen), make_not_null(&dist));
  test_subcells<3>(make_not_null(&gen), make_not_null(&dist));

  test_dg<1>(make_not_null(&gen), make_not_null(&dist));
  test_dg<2>(make_not_null(&gen), make_not_null(&dist));
  test_dg<3>(make_not_null(&gen), make_not_null(&dist));
}
