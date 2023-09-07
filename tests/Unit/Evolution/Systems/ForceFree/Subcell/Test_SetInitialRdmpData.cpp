// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/Systems/ForceFree/Subcell/SetInitialRdmpData.hpp"
#include "Evolution/Systems/ForceFree/System.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"

namespace ForceFree::subcell {
namespace {

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.Subcell.SetInitialRdmpData",
                  "[Unit][Evolution]") {
  const Mesh<3> dg_mesh{5, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  using EvolvedVars = typename System::variables_tag::type;
  EvolvedVars dg_vars{dg_mesh.number_of_grid_points(), 1.0};

  // While the code is supposed to be used on the subcells, that doesn't
  // actually matter.

  const auto subcell_vars = evolution::dg::subcell::fd::project(
      dg_vars, dg_mesh, subcell_mesh.extents());

  evolution::dg::subcell::RdmpTciData rdmp_data{};
  ForceFree::subcell::SetInitialRdmpData::apply(
      make_not_null(&rdmp_data), get<ForceFree::Tags::TildeE>(dg_vars),
      get<ForceFree::Tags::TildeB>(dg_vars),
      get<ForceFree::Tags::TildeQ>(dg_vars),
      evolution::dg::subcell::ActiveGrid::Dg, dg_mesh, subcell_mesh);

  const auto& dg_tilde_e_magnitude =
      magnitude(get<ForceFree::Tags::TildeE>(dg_vars));
  const auto dg_tilde_b_magnitude =
      magnitude(get<ForceFree::Tags::TildeB>(dg_vars));
  const auto& dg_tilde_q = get<ForceFree::Tags::TildeQ>(dg_vars);

  const auto projected_subcell_tilde_e_magnitude =
      Scalar<DataVector>(evolution::dg::subcell::fd::project(
          get(dg_tilde_e_magnitude), dg_mesh, subcell_mesh.extents()));
  const auto projected_subcell_tilde_b_magnitude =
      Scalar<DataVector>(evolution::dg::subcell::fd::project(
          get(dg_tilde_b_magnitude), dg_mesh, subcell_mesh.extents()));
  const auto& subcell_tilde_q = get<ForceFree::Tags::TildeQ>(subcell_vars);

  evolution::dg::subcell::RdmpTciData expected_dg_rdmp_data{};
  using std::max;
  using std::min;
  expected_dg_rdmp_data.max_variables_values =
      DataVector{max(max(get(dg_tilde_e_magnitude)),
                     max(get(projected_subcell_tilde_e_magnitude))),
                 max(max(get(dg_tilde_b_magnitude)),
                     max(get(projected_subcell_tilde_b_magnitude))),
                 max(max(get(dg_tilde_q)), max(get(subcell_tilde_q)))};
  expected_dg_rdmp_data.min_variables_values =
      DataVector{min(min(get(dg_tilde_e_magnitude)),
                     min(get(projected_subcell_tilde_e_magnitude))),
                 min(min(get(dg_tilde_b_magnitude)),
                     min(get(projected_subcell_tilde_b_magnitude))),
                 min(min(get(dg_tilde_q)), min(get(subcell_tilde_q)))};

  CHECK(rdmp_data == expected_dg_rdmp_data);

  ForceFree::subcell::SetInitialRdmpData::apply(
      make_not_null(&rdmp_data), get<ForceFree::Tags::TildeE>(subcell_vars),
      get<ForceFree::Tags::TildeB>(subcell_vars),
      get<ForceFree::Tags::TildeQ>(subcell_vars),
      evolution::dg::subcell::ActiveGrid::Subcell, dg_mesh, subcell_mesh);

  const auto subcell_tilde_e_magnitude =
      magnitude(get<ForceFree::Tags::TildeE>(subcell_vars));
  const auto subcell_tilde_b_magnitude =
      magnitude(get<ForceFree::Tags::TildeB>(subcell_vars));

  evolution::dg::subcell::RdmpTciData expected_subcell_rdmp_data{};
  expected_subcell_rdmp_data.max_variables_values = DataVector{
      max(get(subcell_tilde_e_magnitude)), max(get(subcell_tilde_b_magnitude)),
      max(get(subcell_tilde_q))};
  expected_subcell_rdmp_data.min_variables_values = DataVector{
      min(get(subcell_tilde_e_magnitude)), min(get(subcell_tilde_b_magnitude)),
      min(get(subcell_tilde_q))};

  CHECK(rdmp_data == expected_subcell_rdmp_data);
}

}  // namespace
}  // namespace ForceFree::subcell
