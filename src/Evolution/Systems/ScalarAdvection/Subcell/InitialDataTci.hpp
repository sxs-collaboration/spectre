// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/DataForRdmpTci.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;
/// \endcond

namespace ScalarAdvection::subcell {
/*!
 * \brief The troubled-cell indicator run on DG initial data to see if we need
 * to switch to subcell.
 *
 * Uses the two-mesh relaxed discrete maximum principle as well as the Persson
 * TCI applied to the scalar field \f$U\f$.
 */
template <size_t Dim>
struct DgInitialDataTci {
  using argument_tags = tmpl::list<domain::Tags::Mesh<Dim>,
                                   evolution::dg::subcell::Tags::Mesh<Dim>>;

  static std::tuple<bool, evolution::dg::subcell::RdmpTciData> apply(
      const Variables<tmpl::list<ScalarAdvection::Tags::U>>& dg_vars,
      double rdmp_delta0, double rdmp_epsilon, double persson_exponent,
      const Mesh<Dim>& dg_mesh, const Mesh<Dim>& subcell_mesh);
};

/// \brief Sets the initial RDMP data.
///
/// Used on the subcells after the TCI marked the DG solution as inadmissible.
struct SetInitialRdmpData {
  using argument_tags = tmpl::list<ScalarAdvection::Tags::U,
                                   evolution::dg::subcell::Tags::ActiveGrid>;
  using return_tags = tmpl::list<evolution::dg::subcell::Tags::DataForRdmpTci>;

  static void apply(
      gsl::not_null<evolution::dg::subcell::RdmpTciData*> rdmp_tci_data,
      const Scalar<DataVector>& subcell_u,
      evolution::dg::subcell::ActiveGrid active_grid);
};
}  // namespace ScalarAdvection::subcell
