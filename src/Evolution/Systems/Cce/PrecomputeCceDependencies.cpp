// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/PrecomputeCceDependencies.hpp"

#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/VectorAlgebra.hpp"

namespace Cce {

namespace detail {
template <typename DerivKind>
void angular_derivative_of_r_divided_by_r_impl(
    const gsl::not_null<
        SpinWeighted<ComplexDataVector,
                     Spectral::Swsh::Tags::derivative_spin_weight<DerivKind>>*>
        d_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 0>& boundary_r, const size_t l_max,
    const size_t number_of_radial_points) {
  const size_t number_of_angular_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  // first set the first angular view
  SpinWeighted<ComplexDataVector,
               Spectral::Swsh::Tags::derivative_spin_weight<DerivKind>>
      d_r_divided_by_r_boundary;
  d_r_divided_by_r_boundary.data() = ComplexDataVector{
      d_r_divided_by_r->data().data(), number_of_angular_points};
  Spectral::Swsh::angular_derivatives<tmpl::list<DerivKind>>(
      l_max, 1, make_not_null(&d_r_divided_by_r_boundary), boundary_r);
  d_r_divided_by_r_boundary.data() /= boundary_r.data();
  // all of the angular shells after the innermost one
  ComplexDataVector d_r_divided_by_r_tail_shells{
      d_r_divided_by_r->data().data() + number_of_angular_points,
      (number_of_radial_points - 1) * number_of_angular_points};
  fill_with_n_copies(make_not_null(&d_r_divided_by_r_tail_shells),
                     d_r_divided_by_r_boundary.data(),
                     number_of_radial_points - 1);
}
}  // namespace detail

namespace detail {
// explicit  template instantiations
template void
angular_derivative_of_r_divided_by_r_impl<Spectral::Swsh::Tags::Eth>(
    const gsl::not_null<SpinWeighted<
        ComplexDataVector, Spectral::Swsh::Tags::derivative_spin_weight<
                               Spectral::Swsh::Tags::Eth>>*>
        d_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 0>& boundary_r, const size_t l_max,
    const size_t number_of_radial_points);

template void
angular_derivative_of_r_divided_by_r_impl<Spectral::Swsh::Tags::EthEth>(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 2>*> d_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 0>& boundary_r, const size_t l_max,
    const size_t number_of_radial_points);

template void
angular_derivative_of_r_divided_by_r_impl<Spectral::Swsh::Tags::EthEthbar>(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> d_r_divided_by_r,
    const SpinWeighted<ComplexDataVector, 0>& boundary_r, const size_t l_max,
    const size_t number_of_radial_points);
}  // namespace detail
}  // namespace Cce
