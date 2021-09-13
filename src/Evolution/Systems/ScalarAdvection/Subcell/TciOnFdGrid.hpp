// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
/// \endcond

namespace ScalarAdvection::subcell {
/*!
 * \brief Troubled-cell indicator applied to the finite difference subcell
 * solution to check if the corresponding DG solution is admissible.
 *
 * Applies the Persson TCI to \f$U\f$ on the DG grid.
 *
 * \note TCI is run after reconstructing the solution to the DG grid during the
 * subcell(FD) time stepping procedure, therefore `Inactive<Tag>` is the
 * updated DG solution.
 */
template <size_t Dim>
struct TciOnFdGrid {
 private:
  template <typename Tag>
  using Inactive = ::evolution::dg::subcell::Tags::Inactive<Tag>;

 public:
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<Inactive<ScalarAdvection::Tags::U>, ::domain::Tags::Mesh<Dim>>;

  static bool apply(const Scalar<DataVector>& dg_u, const Mesh<Dim>& dg_mesh,
                    const double persson_exponent) noexcept;
};
}  // namespace ScalarAdvection::subcell
