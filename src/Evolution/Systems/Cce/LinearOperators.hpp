// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce {
/*!
 * \brief Computes the partial derivative along a particular direction
 * determined by the `dimension_to_differentiate`.
 * The input `u` is differentiated with the spectral matrix and the solution is
 * placed in `d_u`.
 *
 * \note This is placed in Cce Utilities for its currently narrow use-case. If
 * more general uses desire a single partial derivative of complex values, this
 * should be moved to `NumericalAlgorithms`. This utility currently assumes the
 * spatial dimensionality is 3, which would also need to be generalized, likely
 * by creating a wrapping struct with partial template specializations.
 */
void logical_partial_directional_derivative_of_complex(
    gsl::not_null<ComplexDataVector*> d_u, const ComplexDataVector& u,
    const Mesh<3>& mesh, size_t dimension_to_differentiate);

namespace Tags {
/// \cond
struct LMax;
/// \endcond

/*!
 * \brief Compute tag for a manually-handled partial y derivative in the volume.
 *
 * \details Most of the partial y derivatives are handled by PreSwshDerivatives.
 * However, that infrastructure is specific to the CCE evolution system itself,
 * so it won't work for other uses (e.g. observers that are invoked
 * periodically). This compute tag instead manually computes a partial
 * y derivative.
 */
template <typename Tag>
struct DyCompute
    : Dy<Tag>, db::ComputeTag {
  using base = Dy<Tag>;
  using return_type = typename base::type;

  using argument_tags = tmpl::list<Tag, Tags::LMax>;

  static constexpr auto function(
      const gsl::not_null<
          Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>*>
          dy_val,
      const Scalar<SpinWeighted<ComplexDataVector, Tag::type::type::spin>>& val,
      const size_t l_max) {
    if (get(*dy_val).size() != get(val).size()) {
      get(*dy_val).destructive_resize(get(val).size());
    };
    logical_partial_directional_derivative_of_complex(
        make_not_null(&get(*dy_val).data()), get(val).data(),
        Mesh<3>{
            {{Spectral::Swsh::number_of_swsh_theta_collocation_points(l_max),
              Spectral::Swsh::number_of_swsh_phi_collocation_points(l_max),
              get(val).size() /
                  Spectral::Swsh::number_of_swsh_collocation_points(l_max)}},
            Spectral::Basis::Legendre,
            Spectral::Quadrature::GaussLobatto},
        // 2 for differentiating in y; coordinate ordering is:
        // {\phi, \theta, y}.
        2);
  }
};

}  // namespace Tags

}  // namespace Cce
