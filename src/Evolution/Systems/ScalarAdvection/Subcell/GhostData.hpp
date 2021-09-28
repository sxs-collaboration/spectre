// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
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

namespace ScalarAdvection::subcell {
/*!
 * \brief Returns \f$U\f$ on the subcells so it can be used for reconstruction.
 *
 * This mutator is passed to
 * `evolution::dg::subcell::Actions::SendDataForReconstruction`.
 *
 * \note Called only by FD-solving elements.
 */
class GhostDataOnSubcells {
 public:
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<::Tags::Variables<tmpl::list<ScalarAdvection::Tags::U>>>;

  static Variables<tmpl::list<ScalarAdvection::Tags::U>> apply(
      const Variables<tmpl::list<ScalarAdvection::Tags::U>>& vars) noexcept;
};

/*!
 * \brief Projects \f$U\f$ from DG grid to subcell grid to send out ghost cell
 * data to neighbors for reconstruction.
 *
 * This mutator is passed what `Metavars::SubcellOptions::GhostDataToSlice` must
 * be set to.
 */
template <size_t Dim>
class GhostDataToSlice {
 public:
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<::Tags::Variables<tmpl::list<ScalarAdvection::Tags::U>>,
                 domain::Tags::Mesh<Dim>,
                 evolution::dg::subcell::Tags::Mesh<Dim>>;

  static Variables<tmpl::list<ScalarAdvection::Tags::U>> apply(
      const Variables<tmpl::list<ScalarAdvection::Tags::U>>& vars,
      const Mesh<Dim>& dg_mesh, const Mesh<Dim>& subcell_mesh) noexcept;
};
}  // namespace ScalarAdvection::subcell
