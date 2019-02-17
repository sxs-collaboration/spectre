// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/CubicScale.hpp"

#include <array>
#include <boost/none.hpp>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <utility>

#include "ControlSystem/FunctionOfTime.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/RootFinding/NewtonRaphson.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain {
namespace CoordMapsTimeDependent {

CubicScale::CubicScale(const double outer_boundary) noexcept
    : outer_boundary_(outer_boundary) {
  if (outer_boundary_ <= 0.0) {
    ERROR("For invertability, we require outer_boundary to be positive.\n");
  }
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 1> CubicScale::operator()(
    const std::array<T, 1>& source_coords, const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
    noexcept {
  const auto a_of_t = map_list.at(f_of_t_a_).func(time)[0][0];
  const auto b_of_t = map_list.at(f_of_t_b_).func(time)[0][0];
  return {{source_coords[0] *
           (a_of_t +
            (b_of_t - a_of_t) * square(source_coords[0] / outer_boundary_))}};
}

template <typename T>
boost::optional<std::array<tt::remove_cvref_wrap_t<T>, 1>> CubicScale::inverse(
    const std::array<T, 1>& target_coords, const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
    noexcept {
  // the original coordinates are found by solving for the roots
  // of (b-a)/X^2*\xi^3 + a*\xi - x = 0, where a and b are the FunctionsOfTime,
  // X is the outer_boundary, and x represents the mapped coordinates
  const auto a_of_t = map_list.at(f_of_t_a_).func(time)[0][0];
  const auto b_of_t = map_list.at(f_of_t_b_).func(time)[0][0];

  // these checks ensure that the function is monotonically increasing
  // and that there is one real root in the domain of \xi, [0,X]
  if (a_of_t <= 0.0) {
    ERROR("We require expansion_a > 0 for invertibility, however expansion_a = "
          << a_of_t << ".");
  }
  if (b_of_t < 2.0 / 3.0 * a_of_t or b_of_t <= 0.0) {
    ERROR("The map is invertible only if 0 < expansion_b < expansion_a*2/3, "
          << " but expansion_b = " << b_of_t << " and expansion_a = " << a_of_t
          << ".");
  }

  // Make the coordinates dimensionless
  const tt::remove_cvref_wrap_t<T> x_bar = target_coords[0] / outer_boundary_;

  // Check if x_bar is outside of the range of the map
  if (x_bar < 0.0 or x_bar > b_of_t) {
    return boost::none;
  }

  // with the assumptions above:
  // x_bar lies within the range [0,b]
  // and \xi_bar = \xi/X is restricted to the domain [0,1]
  // For an initial guess, we provide a linearly approximated solution for
  // \xi_bar, which is just x_bar/b.
  const tt::remove_cvref_wrap_t<T> initial_guess = x_bar / b_of_t;
  const double cubic_coef_a = (b_of_t - a_of_t);

  const auto cubic_and_deriv =
      [&cubic_coef_a, &a_of_t, &x_bar ](double x) noexcept {
    return std::make_pair(x * (cubic_coef_a * square(x) + a_of_t) - x_bar,
                          3.0 * cubic_coef_a * square(x) + a_of_t);
  };

  // The original implementation of this inverse function used a cubic
  // equation solver. However, given that the problem of finding the inverse
  // in this case is well constrained -- using a Newton-Raphson root find with
  // a linearly approximated guess is almost twice as fast.
  // Using google benchmark,
  // the CubicEquation solver: ~ 480 ns
  // boost implemented Newton-Raphson: ~ 280 ns
  // minimal Newton-Raphson from Numerical Recipes: ~ 255 ns
  // Despite the minimal Newton-Raphson being more efficient than the boost
  // version, here we utilize the boost implementation, as it includes
  // additional checks for zero derivative, checks on bounds, and can implement
  // bisection if necessary.
  return {
      {{outer_boundary_ * RootFinder::newton_raphson(
                              cubic_and_deriv, initial_guess, 0.0, 1.0, 14)}}};
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 1> CubicScale::frame_velocity(
    const std::array<T, 1>& source_coords, const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
    noexcept {
  const auto dt_a_of_t = map_list.at(f_of_t_a_).func_and_deriv(time)[1][0];
  const auto dt_b_of_t = map_list.at(f_of_t_b_).func_and_deriv(time)[1][0];
  const auto frame_vel =
      source_coords[0] *
      (dt_a_of_t +
       (dt_b_of_t - dt_a_of_t) * square(source_coords[0] / outer_boundary_));

  return {{frame_vel}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> CubicScale::jacobian(
    const std::array<T, 1>& source_coords, const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
    noexcept {
  const auto a_of_t = map_list.at(f_of_t_a_).func(time)[0][0];
  const auto b_of_t = map_list.at(f_of_t_b_).func(time)[0][0];
  auto jac{
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0)};

  get<0, 0>(jac) =
      a_of_t +
      3.0 * (b_of_t - a_of_t) * square(source_coords[0] / outer_boundary_);

  return jac;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame>
CubicScale::inv_jacobian(
    const std::array<T, 1>& source_coords, const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
    noexcept {
  const auto a_of_t = map_list.at(f_of_t_a_).func(time)[0][0];
  const auto b_of_t = map_list.at(f_of_t_b_).func(time)[0][0];
  auto inv_jac{
      make_with_value<tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0)};

  get<0, 0>(inv_jac) = 1.0 / (a_of_t +
                              3.0 * (b_of_t - a_of_t) *
                                  square(source_coords[0] / outer_boundary_));

  return inv_jac;
}

template <typename T>
tnsr::Iaa<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> CubicScale::hessian(
    const std::array<T, 1>& source_coords, const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
    noexcept {
  const auto a_of_t_and_derivs = map_list.at(f_of_t_a_).func_and_2_derivs(time);
  const auto b_of_t_and_derivs = map_list.at(f_of_t_b_).func_and_2_derivs(time);

  auto result{
      make_with_value<tnsr::Iaa<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0)};
  // time-time
  get<0, 0, 0>(result) = a_of_t_and_derivs[2][0] * source_coords[0] +
                         (b_of_t_and_derivs[2][0] - a_of_t_and_derivs[2][0]) *
                             cube(source_coords[0]) / square(outer_boundary_);
  // time-space
  get<0, 0, 1>(result) =
      a_of_t_and_derivs[1][0] +
      3.0 * (b_of_t_and_derivs[1][0] - a_of_t_and_derivs[1][0]) *
          square(source_coords[0] / outer_boundary_);
  // space-space
  get<0, 1, 1>(result) = 6.0 *
                         (b_of_t_and_derivs[0][0] - a_of_t_and_derivs[0][0]) *
                         source_coords[0] / square(outer_boundary_);

  return result;
}

void CubicScale::pup(PUP::er& p) noexcept {
  p | f_of_t_a_;
  p | f_of_t_b_;
  p | outer_boundary_;
}

bool operator==(const CoordMapsTimeDependent::CubicScale& lhs,
                const CoordMapsTimeDependent::CubicScale& rhs) noexcept {
  return lhs.f_of_t_a_ == rhs.f_of_t_a_ and lhs.f_of_t_b_ == rhs.f_of_t_b_ and
         lhs.outer_boundary_ == rhs.outer_boundary_;
}

// Explicit instantiations
/// \cond
template boost::optional<std::array<tt::remove_cvref_wrap_t<double>, 1>>
CubicScale::inverse(
    const std::array<double, 1>& target_coords, const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
    noexcept;
template boost::optional<std::array<
    tt::remove_cvref_wrap_t<std::reference_wrapper<const double>>, 1>>
CubicScale::inverse(
    const std::array<std::reference_wrapper<const double>, 1>& target_coords,
    const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
    noexcept;

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 1> CubicScale::    \
  operator()(const std::array<DTYPE(data), 1>& source_coords,                  \
             const double time,                                                \
             const std::unordered_map<std::string, FunctionOfTime&>& map_list) \
      const noexcept;                                                          \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 1>                 \
  CubicScale::frame_velocity(                                                  \
      const std::array<DTYPE(data), 1>& source_coords, const double time,      \
      const std::unordered_map<std::string, FunctionOfTime&>& map_list)        \
      const noexcept;                                                          \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 1, Frame::NoFrame>   \
  CubicScale::jacobian(                                                        \
      const std::array<DTYPE(data), 1>& source_coords, const double time,      \
      const std::unordered_map<std::string, FunctionOfTime&>& map_list)        \
      const noexcept;                                                          \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 1, Frame::NoFrame>   \
  CubicScale::inv_jacobian(                                                    \
      const std::array<DTYPE(data), 1>& source_coords, const double time,      \
      const std::unordered_map<std::string, FunctionOfTime&>& map_list)        \
      const noexcept;                                                          \
  template tnsr::Iaa<tt::remove_cvref_wrap_t<DTYPE(data)>, 1, Frame::NoFrame>  \
  CubicScale::hessian(                                                         \
      const std::array<DTYPE(data), 1>& source_coords, const double time,      \
      const std::unordered_map<std::string, FunctionOfTime&>& map_list)        \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))
#undef DTYPE
#undef INSTANTIATE
/// \endcond
}  // namespace CoordMapsTimeDependent
}  // namespace domain
