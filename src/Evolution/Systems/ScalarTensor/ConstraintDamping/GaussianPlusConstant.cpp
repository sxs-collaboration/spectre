// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarTensor/ConstraintDamping/GaussianPlusConstant.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime

namespace ScalarTensor::ConstraintDamping {
template <size_t VolumeDim, typename Fr>
GaussianPlusConstant<VolumeDim, Fr>::GaussianPlusConstant(CkMigrateMessage* msg)
    : DampingFunction<VolumeDim, Fr>(msg) {}

template <size_t VolumeDim, typename Fr>
GaussianPlusConstant<VolumeDim, Fr>::GaussianPlusConstant(
    const double constant, const double amplitude, const double width,
    const std::array<double, VolumeDim>& center)
    : constant_(constant),
      amplitude_(amplitude),
      inverse_width_(1.0 / width),
      center_(center) {}

template <size_t VolumeDim, typename Fr>
template <typename T>
void GaussianPlusConstant<VolumeDim, Fr>::apply_call_operator(
    const gsl::not_null<Scalar<T>*> value_at_x,
    const tnsr::I<T, VolumeDim, Fr>& x) const {
  tnsr::I<T, VolumeDim, Fr> centered_coords{x};
  for (size_t i = 0; i < VolumeDim; ++i) {
    centered_coords.get(i) -= gsl::at(center_, i);
  }
  dot_product(value_at_x, centered_coords, centered_coords);
  get(*value_at_x) =
      constant_ + amplitude_ * exp(-get(*value_at_x) * square(inverse_width_));
}

template <size_t VolumeDim, typename Fr>
void GaussianPlusConstant<VolumeDim, Fr>::operator()(
    const gsl::not_null<Scalar<double>*> value_at_x,
    const tnsr::I<double, VolumeDim, Fr>& x, const double /*time*/,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
    /*functions_of_time*/) const {
  apply_call_operator(value_at_x, x);
}
template <size_t VolumeDim, typename Fr>
void GaussianPlusConstant<VolumeDim, Fr>::operator()(
    const gsl::not_null<Scalar<DataVector>*> value_at_x,
    const tnsr::I<DataVector, VolumeDim, Fr>& x, const double /*time*/,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
    /*functions_of_time*/) const {
  set_number_of_grid_points(value_at_x, x);
  apply_call_operator(value_at_x, x);
}

template <size_t VolumeDim, typename Fr>
void GaussianPlusConstant<VolumeDim, Fr>::pup(PUP::er& p) {
  DampingFunction<VolumeDim, Fr>::pup(p);
  p | constant_;
  p | amplitude_;
  p | inverse_width_;
  p | center_;
}

template <size_t VolumeDim, typename Fr>
auto GaussianPlusConstant<VolumeDim, Fr>::get_clone() const
    -> std::unique_ptr<DampingFunction<VolumeDim, Fr>> {
  return std::make_unique<GaussianPlusConstant<VolumeDim, Fr>>(*this);
}
}  // namespace ScalarTensor::ConstraintDamping

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data)                                                 \
  template ScalarTensor::ConstraintDamping::GaussianPlusConstant<            \
      DIM(data), FRAME(data)>::GaussianPlusConstant(CkMigrateMessage* msg);  \
  template ScalarTensor::ConstraintDamping::                                 \
      GaussianPlusConstant<DIM(data), FRAME(data)>::GaussianPlusConstant(    \
          const double constant, const double amplitude, const double width, \
          const std::array<double, DIM(data)>& center);                      \
  template void ScalarTensor::ConstraintDamping::GaussianPlusConstant<       \
      DIM(data), FRAME(data)>::pup(PUP::er& p);                              \
  template auto ScalarTensor::ConstraintDamping::GaussianPlusConstant<       \
      DIM(data), FRAME(data)>::get_clone()                                   \
      const->std::unique_ptr<DampingFunction<DIM(data), FRAME(data)>>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial))
#undef DIM
#undef FRAME
#undef INSTANTIATE

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                               \
  template void ScalarTensor::ConstraintDamping::                          \
      GaussianPlusConstant<DIM(data), FRAME(data)>::operator()(            \
          const gsl::not_null<Scalar<DTYPE(data)>*> value_at_x,            \
          const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x,           \
          const double /*time*/,                                           \
          const std::unordered_map<                                        \
              std::string,                                                 \
              std::unique_ptr<domain::FunctionsOfTime::                    \
                                  FunctionOfTime>>& /*functions_of_time*/) \
          const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid, Frame::Inertial),
                        (double, DataVector))
#undef FRAME
#undef DIM
#undef DTYPE
#undef INSTANTIATE
