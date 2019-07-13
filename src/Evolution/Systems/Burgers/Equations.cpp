// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/Equations.hpp"

#include <array>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"  // IWYU pragma: keep

// IWYU pragma: no_forward_declare Tensor

namespace Burgers {
double ComputeLargestCharacteristicSpeed::apply(
    const Scalar<DataVector>& u) noexcept {
  return max(abs(get(u)));
}
}  // namespace Burgers

template Variables<tmpl::list<
    Tags::div<Tags::Flux<Burgers::Tags::U, tmpl::size_t<1>, Frame::Inertial>>>>
divergence(
    const Variables<tmpl::list<
        Tags::Flux<Burgers::Tags::U, tmpl::size_t<1>, Frame::Inertial>>>& F,
    const Mesh<1>& mesh,
    const InverseJacobian<DataVector, 1, Frame::Logical,
                          Frame::Inertial>& inverse_jacobian) noexcept;
