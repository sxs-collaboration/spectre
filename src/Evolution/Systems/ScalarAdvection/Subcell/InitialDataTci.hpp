// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
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
 private:
  template <typename Tag>
  using Inactive = evolution::dg::subcell::Tags::Inactive<Tag>;

 public:
  using argument_tags = tmpl::list<domain::Tags::Mesh<Dim>>;

  static bool apply(
      const Variables<tmpl::list<ScalarAdvection::Tags::U>>& dg_vars,
      const Variables<tmpl::list<Inactive<ScalarAdvection::Tags::U>>>&
          subcell_vars,
      const Mesh<Dim>& dg_mesh, double persson_exponent, double rdmp_delta0,
      double rdmp_epsilon) noexcept;
};
}  // namespace ScalarAdvection::subcell
