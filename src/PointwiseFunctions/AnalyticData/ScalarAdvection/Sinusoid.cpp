// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/ScalarAdvection/Sinusoid.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Math.hpp"

namespace ScalarAdvection::AnalyticData {
template <typename T>
Scalar<T> Sinusoid::u(const tnsr::I<T, 1>& x) const noexcept {
  return Scalar<T>{sin(M_PI * get<0>(x))};
}

tuples::TaggedTuple<Tags::U> Sinusoid::variables(
    const tnsr::I<DataVector, 1>& x,
    tmpl::list<Tags::U> /*meta*/) const noexcept {
  return {u(x)};
}

void Sinusoid::pup(PUP::er& /*p*/) noexcept {}

bool operator==(const Sinusoid& /*lhs*/, const Sinusoid& /*rhs*/) noexcept {
  return true;
}

bool operator!=(const Sinusoid& lhs, const Sinusoid& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace ScalarAdvection::AnalyticData

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                               \
  template Scalar<DTYPE(data)> ScalarAdvection::AnalyticData::Sinusoid::u( \
      const tnsr::I<DTYPE(data), 1>& x) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
