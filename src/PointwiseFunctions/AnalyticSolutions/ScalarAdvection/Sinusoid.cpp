// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/ScalarAdvection/Sinusoid.hpp"

#include <cmath>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace ScalarAdvection::Solutions {

template <typename T>
Scalar<T> Sinusoid::u(const tnsr::I<T, 1>& x, double t) const noexcept {
  return Scalar<T>(sin(M_PI * (get<0>(x) - t)));
}

template <typename T>
Scalar<T> Sinusoid::du_dt(const tnsr::I<T, 1>& x, double t) const noexcept {
  return Scalar<T>(-M_PI * cos(M_PI * (get<0>(x) - t)));
}

tuples::TaggedTuple<Tags::U> Sinusoid::variables(
    const tnsr::I<DataVector, 1>& x, double t,
    tmpl::list<Tags::U> /*meta*/) const noexcept {
  return {u(x, t)};
}

tuples::TaggedTuple<::Tags::dt<Tags::U>> Sinusoid::variables(
    const tnsr::I<DataVector, 1>& x, double t,
    tmpl::list<::Tags::dt<Tags::U>> /*meta*/) const noexcept {
  return {du_dt(x, t)};
}

void Sinusoid::pup(PUP::er& /*p*/) noexcept {}

}  // namespace ScalarAdvection::Solutions

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                \
  template Scalar<DTYPE(data)> ScalarAdvection::Solutions::Sinusoid::u(     \
      const tnsr::I<DTYPE(data), 1>& x, double t) const noexcept;           \
  template Scalar<DTYPE(data)> ScalarAdvection::Solutions::Sinusoid::du_dt( \
      const tnsr::I<DTYPE(data), 1>& x, double t) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
