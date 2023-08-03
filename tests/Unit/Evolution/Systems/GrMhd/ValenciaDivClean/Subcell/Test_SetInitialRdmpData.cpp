// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/DgSubcell/ReconstructionMethod.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/SetInitialRdmpData.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ValenciaDivClean.Subcell.SetInitialRdmpData",
    "[Unit][Evolution]") {
  using ConsVars =
      typename grmhd::ValenciaDivClean::System::variables_tag::type;

  const Mesh<3> dg_mesh{5, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  ConsVars dg_vars{dg_mesh.number_of_grid_points(), 1.0};

  // While the code is supposed to be used on the subcells, that doesn't
  // actually matter.
  const auto subcell_vars = evolution::dg::subcell::fd::project(
      dg_vars, dg_mesh, subcell_mesh.extents());
  evolution::dg::subcell::RdmpTciData rdmp_data{};
  grmhd::ValenciaDivClean::subcell::SetInitialRdmpData::apply(
      make_not_null(&rdmp_data),
      get<grmhd::ValenciaDivClean::Tags::TildeD>(dg_vars),
      get<grmhd::ValenciaDivClean::Tags::TildeYe>(dg_vars),
      get<grmhd::ValenciaDivClean::Tags::TildeTau>(dg_vars),
      get<grmhd::ValenciaDivClean::Tags::TildeB<>>(dg_vars),
      evolution::dg::subcell::ActiveGrid::Dg, dg_mesh, subcell_mesh);
  const auto& dg_tilde_d = get<grmhd::ValenciaDivClean::Tags::TildeD>(dg_vars);
  const auto& dg_tilde_ye =
      get<grmhd::ValenciaDivClean::Tags::TildeYe>(dg_vars);
  const auto& dg_tilde_tau =
      get<grmhd::ValenciaDivClean::Tags::TildeTau>(dg_vars);
  const auto dg_tilde_b_magnitude =
      magnitude(get<grmhd::ValenciaDivClean::Tags::TildeB<>>(dg_vars));
  const auto& subcell_tilde_d =
      get<grmhd::ValenciaDivClean::Tags::TildeD>(subcell_vars);
  const auto& subcell_tilde_ye =
      get<grmhd::ValenciaDivClean::Tags::TildeYe>(subcell_vars);
  const auto& subcell_tilde_tau =
      get<grmhd::ValenciaDivClean::Tags::TildeTau>(subcell_vars);
  const auto projected_subcell_tilde_b_magnitude =
      Scalar<DataVector>(evolution::dg::subcell::fd::project(
          get(dg_tilde_b_magnitude), dg_mesh, subcell_mesh.extents()));

  evolution::dg::subcell::RdmpTciData expected_dg_rdmp_data{};
  using std::max;
  using std::min;
  expected_dg_rdmp_data.max_variables_values =
      DataVector{max(max(get(dg_tilde_d)), max(get(subcell_tilde_d))),
                 max(max(get(dg_tilde_ye)), max(get(subcell_tilde_ye))),
                 max(max(get(dg_tilde_tau)), max(get(subcell_tilde_tau))),
                 max(max(get(dg_tilde_b_magnitude)),
                     max(get(projected_subcell_tilde_b_magnitude)))};
  expected_dg_rdmp_data.min_variables_values =
      DataVector{min(min(get(dg_tilde_d)), min(get(subcell_tilde_d))),
                 min(min(get(dg_tilde_ye)), min(get(subcell_tilde_ye))),
                 min(min(get(dg_tilde_tau)), min(get(subcell_tilde_tau))),
                 min(min(get(dg_tilde_b_magnitude)),
                     min(get(projected_subcell_tilde_b_magnitude)))};
  CHECK(rdmp_data == expected_dg_rdmp_data);

  grmhd::ValenciaDivClean::subcell::SetInitialRdmpData::apply(
      make_not_null(&rdmp_data),
      get<grmhd::ValenciaDivClean::Tags::TildeD>(subcell_vars),
      get<grmhd::ValenciaDivClean::Tags::TildeYe>(subcell_vars),
      get<grmhd::ValenciaDivClean::Tags::TildeTau>(subcell_vars),
      get<grmhd::ValenciaDivClean::Tags::TildeB<>>(subcell_vars),
      evolution::dg::subcell::ActiveGrid::Subcell, dg_mesh, subcell_mesh);

  const auto subcell_tilde_b_magnitude =
      magnitude(get<grmhd::ValenciaDivClean::Tags::TildeB<>>(subcell_vars));
  evolution::dg::subcell::RdmpTciData expected_subcell_rdmp_data{};
  expected_subcell_rdmp_data.max_variables_values = DataVector{
      max(get(subcell_tilde_d)), max(get(subcell_tilde_ye)),
      max(get(subcell_tilde_tau)), max(get(subcell_tilde_b_magnitude))};
  expected_subcell_rdmp_data.min_variables_values = DataVector{
      min(get(subcell_tilde_d)), min(get(subcell_tilde_ye)),
      min(get(subcell_tilde_tau)), min(get(subcell_tilde_b_magnitude))};
  CHECK(rdmp_data == expected_subcell_rdmp_data);
}
