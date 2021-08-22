// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
/// \endcond

namespace Burgers::subcell {
/*!
 * \brief The troubled-cell indicator run on the DG grid to check if the
 * solution is admissible.
 *
 * Applies the Persson TCI to \f$U\f$.
 */
struct TciOnDgGrid {
 public:
  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<Burgers::Tags::U, domain::Tags::Mesh<1>>;

  static bool apply(const Scalar<DataVector>& u, const Mesh<1>& dg_mesh,
                    double persson_exponent) noexcept;
};
}  // namespace Burgers::subcell
