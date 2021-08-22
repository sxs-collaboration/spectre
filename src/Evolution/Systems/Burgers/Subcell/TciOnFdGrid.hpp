// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
/// \endcond

namespace Burgers::subcell {
/*!
 * \brief Troubled-cell indicator applied to the finite difference subcell
 * solution to check if the corresponding DG solution is admissible.
 *
 * Runs the Persson TCI on \f$U\f$ on the DG grid. The Persson TCI only works
 * with spectral-type methods and is a direct check of whether or not the DG
 * solution is a good representation of the underlying data.
 *
 * Please note that the TCI is run after the subcell solution has been
 * reconstructed to the DG grid, and so `Inactive<Tag>` is the updated DG
 * solution.
 */
class TciOnFdGrid {
 private:
  template <typename Tag>
  using Inactive = evolution::dg::subcell::Tags::Inactive<Tag>;

 public:
  using return_tags = tmpl::list<>;
  using argument_tags =
      tmpl::list<Inactive<Burgers::Tags::U>, domain::Tags::Mesh<1>>;

  static bool apply(const Scalar<DataVector>& dg_u, const Mesh<1>& dg_mesh,
                    double persson_exponent) noexcept;
};
}  // namespace Burgers::subcell
