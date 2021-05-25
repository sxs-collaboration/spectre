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
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/SwapGrTags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ValenciaDivClean.Subcell.SwapGrTags",
                  "[Unit][Evolution]") {
  const Mesh<3> dg_mesh{5, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<3> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);

  const grmhd::ValenciaDivClean::System::spacetime_variables_tag::type
      active_gr_vars(dg_mesh.number_of_grid_points());
  const evolution::dg::subcell::Tags::Inactive<
      typename grmhd::ValenciaDivClean::System::spacetime_variables_tag>::type
      inactive_gr_vars(subcell_mesh.number_of_grid_points());

  auto box = db::create<db::AddSimpleTags<
      grmhd::ValenciaDivClean::System::spacetime_variables_tag,
      evolution::dg::subcell::Tags::Inactive<
          grmhd::ValenciaDivClean::System::spacetime_variables_tag>,
      domain::Tags::Mesh<3>, evolution::dg::subcell::Tags::Mesh<3>,
      evolution::dg::subcell::Tags::ActiveGrid>>(
      active_gr_vars, inactive_gr_vars, dg_mesh, subcell_mesh,
      evolution::dg::subcell::ActiveGrid::Dg);

  for (const auto active_grid : {evolution::dg::subcell::ActiveGrid::Dg,
                                 evolution::dg::subcell::ActiveGrid::Subcell,
                                 evolution::dg::subcell::ActiveGrid::Dg}) {
    db::mutate<evolution::dg::subcell::Tags::ActiveGrid>(
        make_not_null(&box), [&active_grid](const auto active_grid_ptr) {
          *active_grid_ptr = active_grid;
        });

    db::mutate_apply<grmhd::ValenciaDivClean::subcell::SwapGrTags>(
        make_not_null(&box));

    if (active_grid == evolution::dg::subcell::ActiveGrid::Dg) {
      CHECK(
          db::get<grmhd::ValenciaDivClean::System::spacetime_variables_tag>(box)
              .number_of_grid_points() == dg_mesh.number_of_grid_points());
      CHECK(db::get<evolution::dg::subcell::Tags::Inactive<
                grmhd::ValenciaDivClean::System::spacetime_variables_tag>>(box)
                .number_of_grid_points() ==
            subcell_mesh.number_of_grid_points());
    } else {
      CHECK(
          db::get<grmhd::ValenciaDivClean::System::spacetime_variables_tag>(box)
              .number_of_grid_points() == subcell_mesh.number_of_grid_points());
      CHECK(db::get<evolution::dg::subcell::Tags::Inactive<
                grmhd::ValenciaDivClean::System::spacetime_variables_tag>>(box)
                .number_of_grid_points() == dg_mesh.number_of_grid_points());
    }
  }
}
