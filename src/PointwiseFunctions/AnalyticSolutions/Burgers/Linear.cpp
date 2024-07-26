// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/Burgers/Linear.hpp"

#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace Burgers::Solutions {

Linear::Linear(const double shock_time) : shock_time_(shock_time) {}

std::unique_ptr<evolution::initial_data::InitialData> Linear::get_clone()
    const {
  return std::make_unique<Linear>(*this);
}

Linear::Linear(CkMigrateMessage* msg) : InitialData(msg) {}

template <typename T>
Scalar<T> Linear::u(const tnsr::I<T, 1>& x, double t) const {
  Scalar<T> result(get<0>(x));
  get(result) /= (t - shock_time_);
  return result;
}

template <typename T>
Scalar<T> Linear::du_dt(const tnsr::I<T, 1>& x, double t) const {
  Scalar<T> result(get<0>(x));
  get(result) /= -square(t - shock_time_);
  return result;
}

tuples::TaggedTuple<Tags::U> Linear::variables(
    const tnsr::I<DataVector, 1>& x, double t,
    tmpl::list<Tags::U> /*meta*/) const {
  return {u(x, t)};
}

tuples::TaggedTuple<::Tags::dt<Tags::U>> Linear::variables(
    const tnsr::I<DataVector, 1>& x, double t,
    tmpl::list<::Tags::dt<Tags::U>> /*meta*/) const {
  return {du_dt(x, t)};
}

void Linear::pup(PUP::er& p) {
  InitialData::pup(p);
  p | shock_time_;
}

PUP::able::PUP_ID Linear::my_PUP_ID = 0;
}  // namespace Burgers::Solutions

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                      \
  template Scalar<DTYPE(data)> Burgers::Solutions::Linear::u(     \
      const tnsr::I<DTYPE(data), 1>& x, double t) const;          \
  template Scalar<DTYPE(data)> Burgers::Solutions::Linear::du_dt( \
      const tnsr::I<DTYPE(data), 1>& x, double t) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
