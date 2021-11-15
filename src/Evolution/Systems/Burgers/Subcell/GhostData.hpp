// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Mesh;
template <typename T>
class Variables;
namespace Tags {
template <typename TagsList>
struct Variables;
}  // namespace Tags
/// \endcond

namespace Burgers::subcell {
/*!
 * \brief Returns \f$U\f$ on the subcells so it can be used for reconstruction.
 *
 * This mutator is passed to
 * `evolution::dg::subcell::Actions::SendDataForReconstruction`.
 *
 * \note Only called on elements using FD.
 */
class GhostDataOnSubcells {
 public:
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<::Tags::Variables<tmpl::list<Burgers::Tags::U>>>;

  static Variables<tmpl::list<Burgers::Tags::U>> apply(
      const Variables<tmpl::list<Burgers::Tags::U>>& vars);
};

/*!
 * \brief Projects \f$U\f$ from DG grid to subcell grid to send out ghost cell
 * data to neighbors for reconstruction.
 *
 * This mutator is passed what `Metavars::SubcellOptions::GhostDataToSlice` must
 * be set to.
 */
class GhostDataToSlice {
 public:
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<::Tags::Variables<tmpl::list<Burgers::Tags::U>>,
                 domain::Tags::Mesh<1>, evolution::dg::subcell::Tags::Mesh<1>>;

  static Variables<tmpl::list<Burgers::Tags::U>> apply(
      const Variables<tmpl::list<Burgers::Tags::U>>& vars,
      const Mesh<1>& dg_mesh, const Mesh<1>& subcell_mesh);
};
}  // namespace Burgers::subcell
