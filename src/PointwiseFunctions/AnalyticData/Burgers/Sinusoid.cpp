// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Burgers/Sinusoid.hpp"

#include <array>
#include <cmath>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Math.hpp"  // IWYU pragma: keep

// IWYU pragma: no_forward_declare Tensor

namespace Burgers::AnalyticData {
template <typename T>
Scalar<T> Sinusoid::u(const tnsr::I<T, 1>& x) const {
  return Scalar<T>{sin(get<0>(x))};
}

tuples::TaggedTuple<Tags::U> Sinusoid::variables(
    const tnsr::I<DataVector, 1>& x, tmpl::list<Tags::U> /*meta*/) const {
  return {u(x)};
}

void Sinusoid::pup(PUP::er& /*p*/) {}

bool operator==(const Sinusoid& /*lhs*/, const Sinusoid& /*rhs*/) {
  return true;
}

bool operator!=(const Sinusoid& lhs, const Sinusoid& rhs) {
  return not(lhs == rhs);
}
}  // namespace Burgers::AnalyticData

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                       \
  template Scalar<DTYPE(data)> Burgers::AnalyticData::Sinusoid::u( \
      const tnsr::I<DTYPE(data), 1>& x) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
