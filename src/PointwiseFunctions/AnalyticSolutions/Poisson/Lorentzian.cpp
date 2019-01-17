// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Poisson/Lorentzian.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
// IWYU pragma: no_forward_declare Tensor

namespace Poisson {
namespace Solutions {

template <>
tuples::TaggedTuple<Field> Lorentzian<3>::variables(
    const tnsr::I<DataVector, 3>& x, tmpl::list<Field> /*meta*/) noexcept {
  return {Scalar<DataVector>(1. / sqrt(1 + get(dot_product(x, x))))};
}

template <>
tuples::TaggedTuple<::Tags::Source<Field>> Lorentzian<3>::variables(
    const tnsr::I<DataVector, 3>& x,
    tmpl::list<::Tags::Source<Field>> /*meta*/) noexcept {
  return {Scalar<DataVector>(3. / pow<5>(sqrt(1. + get(dot_product(x, x)))))};
}

template <>
tuples::TaggedTuple<::Tags::Source<AuxiliaryField<3>>> Lorentzian<3>::variables(
    const tnsr::I<DataVector, 3>& x,
    tmpl::list<::Tags::Source<AuxiliaryField<3>>> /*meta*/) noexcept {
  return {make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(x, 0.)};
}

template <size_t Dim>
void Lorentzian<Dim>::pup(PUP::er& /*p*/) noexcept {}

template <size_t Dim>
bool operator==(const Lorentzian<Dim>& /*lhs*/,
                const Lorentzian<Dim>& /*rhs*/) noexcept {
  return true;
}

template <size_t Dim>
bool operator!=(const Lorentzian<Dim>& lhs,
                const Lorentzian<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit template instantiations
template class Lorentzian<3>;
template bool operator==(const Lorentzian<3>& lhs,
                         const Lorentzian<3>& rhs) noexcept;
template bool operator!=(const Lorentzian<3>& lhs,
                         const Lorentzian<3>& rhs) noexcept;

}  // namespace Solutions
}  // namespace Poisson
