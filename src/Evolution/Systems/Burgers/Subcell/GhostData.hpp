// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Mesh;
template <typename T>
class Variables;
/// \endcond

namespace Burgers::subcell {
/*!
 * \brief Returns \f$U\f$ on the subcells so it can be used for reconstruction.
 *
 * The computation copies the data to a new Variables. In the future we will
 * likely want to elide this copy but that requires support from the actions.
 *
 * This mutator is passed to
 * `evolution::dg::subcell::Actions::SendDataForReconstruction`.
 */
class GhostDataOnSubcells {
 public:
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<::Tags::Variables<tmpl::list<Burgers::Tags::U>>>;

  static Variables<tmpl::list<Burgers::Tags::U>> apply(
      const Variables<tmpl::list<Burgers::Tags::U>>& vars) noexcept;
};

/*!
 * \brief Projects \f$U\f$ to the subcells to be sent to neighbors for subcell
 * reconstruction.
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
      const Mesh<1>& dg_mesh, const Mesh<1>& subcell_mesh) noexcept;
};
}  // namespace Burgers::subcell
