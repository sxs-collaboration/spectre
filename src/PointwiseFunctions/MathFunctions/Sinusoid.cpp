// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/MathFunctions/Sinusoid.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace MathFunctions {
template <typename Fr>
Sinusoid<1, Fr>::Sinusoid(const double amplitude, const double wavenumber,
                          const double phase)
    : amplitude_(amplitude), wavenumber_(wavenumber), phase_(phase) {}

template <typename Fr>
std::unique_ptr<MathFunction<1, Fr>> Sinusoid<1, Fr>::get_clone() const {
  return std::make_unique<Sinusoid<1, Fr>>(*this);
}

template <typename Fr>
double Sinusoid<1, Fr>::operator()(const double& x) const {
  return apply_call_operator(x);
}

template <typename Fr>
DataVector Sinusoid<1, Fr>::operator()(const DataVector& x) const {
  return apply_call_operator(x);
}

template <typename Fr>
double Sinusoid<1, Fr>::first_deriv(const double& x) const {
  return apply_first_deriv(x);
}

template <typename Fr>
DataVector Sinusoid<1, Fr>::first_deriv(const DataVector& x) const {
  return apply_first_deriv(x);
}

template <typename Fr>
double Sinusoid<1, Fr>::second_deriv(const double& x) const {
  return apply_second_deriv(x);
}

template <typename Fr>
DataVector Sinusoid<1, Fr>::second_deriv(const DataVector& x) const {
  return apply_second_deriv(x);
}

template <typename Fr>
double Sinusoid<1, Fr>::third_deriv(const double& x) const {
  return apply_third_deriv(x);
}

template <typename Fr>
DataVector Sinusoid<1, Fr>::third_deriv(const DataVector& x) const {
  return apply_third_deriv(x);
}

template <typename Fr>
template <typename T>
T Sinusoid<1, Fr>::apply_call_operator(const T& x) const {
  return amplitude_ * sin(wavenumber_ * x + phase_);
}

template <typename Fr>
template <typename T>
T Sinusoid<1, Fr>::apply_first_deriv(const T& x) const {
  return wavenumber_ * amplitude_ * cos(wavenumber_ * x + phase_);
}

template <typename Fr>
template <typename T>
T Sinusoid<1, Fr>::apply_second_deriv(const T& x) const {
  return -amplitude_ * square(wavenumber_) * sin(wavenumber_ * x + phase_);
}

template <typename Fr>
template <typename T>
T Sinusoid<1, Fr>::apply_third_deriv(const T& x) const {
  return -amplitude_ * cube(wavenumber_) * cos(wavenumber_ * x + phase_);
}

template <typename Fr>
bool Sinusoid<1, Fr>::operator==(const MathFunction<1, Fr>& other) const {
  const auto* derived_other = dynamic_cast<const Sinusoid<1, Fr>*>(&other);
  if (derived_other != nullptr) {
    return (this->amplitude_ == derived_other->amplitude_) and
           (this->wavenumber_ == derived_other->wavenumber_) and
           (this->phase_ == derived_other->phase_);
  }
  return false;
}

template <typename Fr>
bool Sinusoid<1, Fr>::operator!=(const MathFunction<1, Fr>& other) const {
  return not(*this == other);
}

template <typename Fr>
void Sinusoid<1, Fr>::pup(PUP::er& p) {
  MathFunction<1, Fr>::pup(p);
  p | amplitude_;
  p | wavenumber_;
  p | phase_;
}
}  // namespace MathFunctions

template class MathFunctions::Sinusoid<1, Frame::Grid>;
template class MathFunctions::Sinusoid<1, Frame::Inertial>;

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                        \
  template DTYPE(data)                                              \
      MathFunctions::Sinusoid<1, FRAME(data)>::apply_call_operator( \
          const DTYPE(data) & x) const;                             \
  template DTYPE(data)                                              \
      MathFunctions::Sinusoid<1, FRAME(data)>::apply_first_deriv(   \
          const DTYPE(data) & x) const;                             \
  template DTYPE(data)                                              \
      MathFunctions::Sinusoid<1, FRAME(data)>::apply_second_deriv(  \
          const DTYPE(data) & x) const;                             \
  template DTYPE(data)                                              \
      MathFunctions::Sinusoid<1, FRAME(data)>::apply_third_deriv(   \
          const DTYPE(data) & x) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (Frame::Grid, Frame::Inertial),
                        (double, DataVector))
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
