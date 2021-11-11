// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/Systems/Burgers/Subcell/InitialDataTci.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename Tag>
using Inactive = evolution::dg::subcell::Tags::Inactive<Tag>;
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Burgers.Subcell.InitialDataTci",
                  "[Unit][Evolution]") {
  using Vars = Variables<tmpl::list<Burgers::Tags::U>>;
  using InactiveVars = Variables<tmpl::list<Inactive<Burgers::Tags::U>>>;

  const Mesh<1> dg_mesh{5, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const Mesh<1> subcell_mesh = evolution::dg::subcell::fd::mesh(dg_mesh);
  const size_t number_of_dg_grid_points{dg_mesh.number_of_grid_points()};
  const size_t number_of_subcell_grid_points{
      subcell_mesh.number_of_grid_points()};

  Vars dg_vars{number_of_dg_grid_points, 1.0};

  // TCI parameters
  const double persson_exponent{4.0};
  const double rdmp_delta0{1.0e-4};
  const double rdmp_epsilon{1.0e-3};

  {
    INFO("TCI is happy");
    // set dg_vars == subcell_vars
    const InactiveVars subcell_vars{number_of_subcell_grid_points, 1.0};
    CHECK_FALSE(Burgers::subcell::DgInitialDataTci::apply(
        dg_vars, subcell_vars, rdmp_delta0, rdmp_epsilon, persson_exponent,
        dg_mesh));
  }

  {
    INFO("Two mesh RDMP fails");
    // set subcell_vars to be smooth but quite different from dg_vars
    const InactiveVars subcell_vars{number_of_subcell_grid_points, 2.0};
    CHECK(Burgers::subcell::DgInitialDataTci::apply(dg_vars, subcell_vars,
                                                    rdmp_delta0, rdmp_epsilon,
                                                    persson_exponent, dg_mesh));
  }

  {
    INFO("Persson TCI fails");
    InactiveVars subcell_vars{number_of_subcell_grid_points, 1.0};
    // set dg_vars to have a sharp peak
    get(get<Burgers::Tags::U>(dg_vars))[number_of_dg_grid_points / 2] += 1.0;
    // set rdmp_delta0 to be very large to ensure that it's the Persson TCI
    // which triggers alarm here
    CHECK(Burgers::subcell::DgInitialDataTci::apply(dg_vars, subcell_vars,
                                                    rdmp_delta0, rdmp_epsilon,
                                                    persson_exponent, dg_mesh));
  }
}
