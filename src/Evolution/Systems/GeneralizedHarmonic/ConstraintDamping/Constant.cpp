// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Constant.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime

namespace gh::ConstraintDamping {
template <size_t VolumeDim, typename Fr>
Constant<VolumeDim, Fr>::Constant(CkMigrateMessage* msg)
    : DampingFunction<VolumeDim, Fr>(msg) {}

template <size_t VolumeDim, typename Fr>
Constant<VolumeDim, Fr>::Constant(const double value) : value_(value) {}

template <size_t VolumeDim, typename Fr>
template <typename T>
void Constant<VolumeDim, Fr>::apply_call_operator(
    const gsl::not_null<Scalar<T>*> value_at_x) const {
  get(*value_at_x) = value_;
}

template <size_t VolumeDim, typename Fr>
void Constant<VolumeDim, Fr>::operator()(
    const gsl::not_null<Scalar<double>*> value_at_x,
    const tnsr::I<double, VolumeDim, Fr>& /*x*/, const double /*time*/,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
    /*functions_of_time*/) const {
  apply_call_operator(value_at_x);
}
template <size_t VolumeDim, typename Fr>
void Constant<VolumeDim, Fr>::operator()(
    const gsl::not_null<Scalar<DataVector>*> value_at_x,
    const tnsr::I<DataVector, VolumeDim, Fr>& x, const double /*time*/,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
    /*functions_of_time*/) const {
  destructive_resize_components(value_at_x, get<0>(x).size());
  apply_call_operator(value_at_x);
}

template <size_t VolumeDim, typename Fr>
void Constant<VolumeDim, Fr>::pup(PUP::er& p) {
  DampingFunction<VolumeDim, Fr>::pup(p);
  p | value_;
}

template <size_t VolumeDim, typename Fr>
auto Constant<VolumeDim, Fr>::get_clone() const
    -> std::unique_ptr<DampingFunction<VolumeDim, Fr>> {
  return std::make_unique<Constant<VolumeDim, Fr>>(*this);
}
}  // namespace gh::ConstraintDamping

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data)                                                  \
  template gh::ConstraintDamping::Constant<DIM(data), FRAME(data)>::Constant( \
      CkMigrateMessage* msg);                                                 \
  template gh::ConstraintDamping::Constant<DIM(data), FRAME(data)>::Constant( \
      const double value);                                                    \
  template void gh::ConstraintDamping::Constant<DIM(data), FRAME(data)>::pup( \
      PUP::er& p);                                                            \
  template auto                                                               \
  gh::ConstraintDamping::Constant<DIM(data), FRAME(data)>::get_clone()        \
      const->std::unique_ptr<DampingFunction<DIM(data), FRAME(data)>>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial))
#undef DIM
#undef FRAME
#undef INSTANTIATE

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                           \
  template void                                                        \
  gh::ConstraintDamping::Constant<DIM(data), FRAME(data)>::operator()( \
      const gsl::not_null<Scalar<DTYPE(data)>*> value_at_x,            \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& /*x*/,       \
      const double /*time*/,                                           \
      const std::unordered_map<                                        \
          std::string,                                                 \
          std::unique_ptr<domain::FunctionsOfTime::                    \
                              FunctionOfTime>>& /*functions_of_time*/) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))
#undef FRAME
#undef DIM
#undef DTYPE
#undef INSTANTIATE
