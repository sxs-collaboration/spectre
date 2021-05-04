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
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.Subcell.PrimitiveGhostData",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.0, 1.0);
  test_subcells<1>(make_not_null(&gen), make_not_null(&dist));
  test_subcells<2>(make_not_null(&gen), make_not_null(&dist));
  test_subcells<3>(make_not_null(&gen), make_not_null(&dist));
}
