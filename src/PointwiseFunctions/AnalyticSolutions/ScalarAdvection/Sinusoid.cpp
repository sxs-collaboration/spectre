// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/ScalarAdvection/Sinusoid.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace ScalarAdvection::Solutions {

template <typename DataType>
tuples::TaggedTuple<ScalarAdvection::Tags::U> Sinusoid::variables(
    const tnsr::I<DataType, 1>& x, double t,
    tmpl::list<ScalarAdvection::Tags::U> /*meta*/) const {
  auto u = make_with_value<Scalar<DataType>>(get<0>(x), 0.0);
  get(u) = sin(M_PI * (get<0>(x) - t));
  return u;
}

void Sinusoid::pup(PUP::er& /*p*/) {}

bool operator==(const Sinusoid& /*lhs*/, const Sinusoid& /*rhs*/) {
  return true;
}

bool operator!=(const Sinusoid& lhs, const Sinusoid& rhs) {
  return not(lhs == rhs);
}

}  // namespace ScalarAdvection::Solutions

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                             \
  template tuples::TaggedTuple<ScalarAdvection::Tags::U> \
  ScalarAdvection::Solutions::Sinusoid::variables(       \
      const tnsr::I<DTYPE(data), 1>& x, double t,        \
      tmpl::list<ScalarAdvection::Tags::U> /*meta*/) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector))

#undef DTYPE
#undef INSTANTIATE
