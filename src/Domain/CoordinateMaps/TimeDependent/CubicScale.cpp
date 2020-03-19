// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"

#include <array>
#include <boost/none.hpp>
#include <boost/optional.hpp>
#include <cstddef>
#include <memory>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/RootFinding/NewtonRaphson.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

namespace domain {
namespace CoordMapsTimeDependent {

template <size_t Dim>
CubicScale<Dim>::CubicScale(const double outer_boundary,
                            std::string function_of_time_name_a,
                            std::string function_of_time_name_b) noexcept
    : f_of_t_a_(std::move(function_of_time_name_a)),
      f_of_t_b_(std::move(function_of_time_name_b)),
      functions_of_time_equal_(f_of_t_a_ == f_of_t_b_) {
  if (outer_boundary <= 0.0) {
    ERROR("For invertability, we require outer_boundary to be positive, but is "
          << outer_boundary);
  }
  one_over_outer_boundary_ = 1.0 / outer_boundary;
}

template <size_t Dim>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, Dim> CubicScale<Dim>::operator()(
    const std::array<T, Dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  ASSERT(functions_of_time.find(f_of_t_a_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_a_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));
  ASSERT(functions_of_time.find(f_of_t_b_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_b_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));

  const double a_of_t = functions_of_time.at(f_of_t_a_)->func(time)[0][0];

  if (functions_of_time_equal_) {
    // optimization for linear radial scaling
    std::array<tt::remove_cvref_wrap_t<T>, Dim> result{};
    for (size_t i = 0; i < Dim; ++i) {
      gsl::at(result, i) = a_of_t * gsl::at(source_coords, i);
    }
    return result;
  }

  const double b_of_t = functions_of_time.at(f_of_t_b_)->func(time)[0][0];

  tt::remove_cvref_wrap_t<T> rho_squared =
      square(dereference_wrapper(source_coords[0]));
  for (size_t i = 1; i < Dim; ++i) {
    rho_squared += square(dereference_wrapper(gsl::at(source_coords, i)));
  }
  // Reuse rho^2 allocation
  rho_squared = a_of_t + (b_of_t - a_of_t) * square(one_over_outer_boundary_) *
                             rho_squared;

  std::array<tt::remove_cvref_wrap_t<T>, Dim> result{};
  for (size_t i = 0; i < Dim - 1; ++i) {
    gsl::at(result, i) = gsl::at(source_coords, i) * rho_squared;
  }
  rho_squared = source_coords[Dim - 1] * rho_squared;
  result[Dim - 1] = std::move(rho_squared);
  return result;
}

template <size_t Dim>
template <typename T>
boost::optional<std::array<tt::remove_cvref_wrap_t<T>, Dim>>
CubicScale<Dim>::inverse(
    const std::array<T, Dim>& target_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  ASSERT(functions_of_time.find(f_of_t_a_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_a_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));
  ASSERT(functions_of_time.find(f_of_t_b_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_b_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));

  if (functions_of_time_equal_) {
    // optimization for linear radial scaling
    const double one_over_a_of_t =
        1.0 / functions_of_time.at(f_of_t_a_)->func(time)[0][0];

    // Construct boost::optional to have a default value of an empty array.
    // Doing just result{} would construct a boost::optional that doesn't hold a
    // value and so *result would throw an exception.
    boost::optional<std::array<tt::remove_cvref_wrap_t<T>, Dim>> result{
        std::array<tt::remove_cvref_wrap_t<T>, Dim>{}};
    for (size_t i = 0; i < Dim; ++i) {
      gsl::at(*result, i) = one_over_a_of_t * gsl::at(target_coords, i);
    }
    return result;
  }

  // the source coordinates \xi^{\hat{i}} are found by solving for the roots of
  // (b-a)/R^2*\rho^3 + a*\rho - r = 0,
  // where a and b are the FunctionsOfTime, R is the outer_boundary, and r is
  // the mapped/target coordinates radius.
  const double a_of_t = functions_of_time.at(f_of_t_a_)->func(time)[0][0];
  const double b_of_t = functions_of_time.at(f_of_t_b_)->func(time)[0][0];

  // these checks ensure that the function is monotonically increasing
  // and that there is one real root in the domain of \rho, [0,R]
  if (a_of_t <= 0.0) {
    ERROR("We require expansion_a > 0 for invertibility, however expansion_a = "
          << a_of_t << ".");
  }
  if (b_of_t < 2.0 / 3.0 * a_of_t or b_of_t <= 0.0) {
    ERROR("The map is invertible only if 0 < expansion_b < expansion_a*2/3, "
          << " but expansion_b = " << b_of_t << " and expansion_a = " << a_of_t
          << ".");
  }

  // To invert the map we work in the dimensionless radius,
  // r / R_{outer_boundary}.
  tt::remove_cvref_wrap_t<T> target_dimensionless_radius =
      magnitude(target_coords) * one_over_outer_boundary_;

  if (UNLIKELY(target_dimensionless_radius == 0.0)) {
    return {make_array<Dim>(0.0)};
  }

  // Check if x_bar is outside of the range of the map.
  // We need a slight buffer because computing (r/R) is not equal to (r * (1/R))
  // at roundoff and thus to make sure we include the boundary we need to
  // support epsilon above b(t).
  if (UNLIKELY(target_dimensionless_radius >
               b_of_t * (1.0 + 2.0 * std::numeric_limits<double>::epsilon()))) {
    return boost::none;
  }

  // For an initial guess, we provide a linearly approximated solution for
  // q, which is just r / (R b).
  const tt::remove_cvref_wrap_t<T> initial_guess =
      target_dimensionless_radius / b_of_t;
  const double cubic_coef_a = (b_of_t - a_of_t);

  // Solve the modified equation:
  // q * ( (b-a) q^2 + a) - r / R = 0,
  // where q = rho / R, and rho is the source radius
  const auto cubic_and_deriv =
      [&cubic_coef_a, &a_of_t, &target_dimensionless_radius](
          const double source_dimensionless_radius) noexcept {
        return std::make_pair(
            source_dimensionless_radius *
                    (cubic_coef_a * square(source_dimensionless_radius) +
                     a_of_t) -
                target_dimensionless_radius,

            3.0 * cubic_coef_a * square(source_dimensionless_radius) + a_of_t);
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
  const double scale_factor =
      RootFinder::newton_raphson(cubic_and_deriv, initial_guess, 0.0, 1.0, 14) /
      target_dimensionless_radius;

  std::array<tt::remove_cvref_wrap_t<T>, Dim> result{};
  for (size_t i = 0; i < Dim; ++i) {
    gsl::at(result, i) = scale_factor * gsl::at(target_coords, i);
  }
  return {std::move(result)};
}

template <size_t Dim>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, Dim> CubicScale<Dim>::frame_velocity(
    const std::array<T, Dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  ASSERT(functions_of_time.find(f_of_t_a_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_a_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));
  ASSERT(functions_of_time.find(f_of_t_b_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_b_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));

  const double dt_a_of_t =
      functions_of_time.at(f_of_t_a_)->func_and_deriv(time)[1][0];

  if (functions_of_time_equal_) {
    // optimization for linear radial scaling
    std::array<tt::remove_cvref_wrap_t<T>, Dim> result{};
    for (size_t i = 0; i < Dim; ++i) {
      gsl::at(result, i) = dt_a_of_t * gsl::at(source_coords, i);
    }
    return result;
  }

  const double dt_b_of_t =
      functions_of_time.at(f_of_t_b_)->func_and_deriv(time)[1][0];

  tt::remove_cvref_wrap_t<T> rho_squared =
      square(dereference_wrapper(source_coords[0]));
  for (size_t i = 1; i < Dim; ++i) {
    rho_squared += square(dereference_wrapper(gsl::at(source_coords, i)));
  }
  // Reuse rho^2 allocation
  rho_squared = dt_a_of_t + (dt_b_of_t - dt_a_of_t) *
                                square(one_over_outer_boundary_) * rho_squared;

  std::array<tt::remove_cvref_wrap_t<T>, Dim> result{};
  for (size_t i = 0; i < Dim - 1; ++i) {
    gsl::at(result, i) = gsl::at(source_coords, i) * rho_squared;
  }
  rho_squared = source_coords[Dim - 1] * rho_squared;
  result[Dim - 1] = std::move(rho_squared);
  return result;
}

template <size_t Dim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>
CubicScale<Dim>::jacobian(
    const std::array<T, Dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  ASSERT(functions_of_time.find(f_of_t_a_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_a_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));
  ASSERT(functions_of_time.find(f_of_t_b_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_b_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));

  const double a_of_t = functions_of_time.at(f_of_t_a_)->func(time)[0][0];

  if (functions_of_time_equal_) {
    // optimization for linear radial scaling
    auto jac{make_with_value<
        tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>>(
        dereference_wrapper(source_coords[0]), 0.0)};
    for (size_t i = 0; i < Dim; ++i) {
      jac.get(i, i) = a_of_t;
    }
    return jac;
  }

  const double b_of_t = functions_of_time.at(f_of_t_b_)->func(time)[0][0];

  tt::remove_cvref_wrap_t<T> rho_squared =
      square(dereference_wrapper(source_coords[0]));
  for (size_t i = 1; i < Dim; ++i) {
    rho_squared += square(dereference_wrapper(gsl::at(source_coords, i)));
  }
  const double rho_squared_coeff =
      (b_of_t - a_of_t) * square(one_over_outer_boundary_);
  // Reuse rho^2 allocation
  rho_squared = a_of_t + rho_squared_coeff * rho_squared;

  const double coeff =
      2.0 * (b_of_t - a_of_t) * square(one_over_outer_boundary_);

  tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> jac{};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      if (i == j) {
        jac.get(i, j) = rho_squared + gsl::at(source_coords, i) *
                                          gsl::at(source_coords, j) * coeff;
      } else {
        jac.get(i, j) =
            gsl::at(source_coords, i) * gsl::at(source_coords, j) * coeff;
      }
    }
  }

  return jac;
}

template <size_t Dim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>
CubicScale<Dim>::inv_jacobian(
    const std::array<T, Dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const noexcept {
  ASSERT(functions_of_time.find(f_of_t_a_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_a_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));
  ASSERT(functions_of_time.find(f_of_t_b_) != functions_of_time.end(),
         "Could not find function of time: '"
             << f_of_t_b_ << "' in functions of time. Known functions are "
             << keys_of(functions_of_time));

  const double a_of_t = functions_of_time.at(f_of_t_a_)->func(time)[0][0];

  if (functions_of_time_equal_) {
    // optimization for linear radial scaling
    auto inv_jac{make_with_value<
        tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>>(
        dereference_wrapper(source_coords[0]), 0.0)};
    const double one_over_a = 1.0 / a_of_t;
    for (size_t i = 0; i < Dim; ++i) {
      inv_jac.get(i, i) = one_over_a;
    }
    return inv_jac;
  }

  const double b_of_t = functions_of_time.at(f_of_t_b_)->func(time)[0][0];

  tt::remove_cvref_wrap_t<T> rho_squared =
      square(dereference_wrapper(source_coords[0]));
  for (size_t i = 1; i < Dim; ++i) {
    rho_squared += square(dereference_wrapper(gsl::at(source_coords, i)));
  }
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> inv_jac{};
  get<0, 0>(inv_jac) =
      1.0 / (a_of_t + (b_of_t - a_of_t) * square(one_over_outer_boundary_) *
                          rho_squared);
  for (size_t i = 1; i < Dim; ++i) {
    inv_jac.get(i, i) = get<0, 0>(inv_jac);
  }

  // Factor out `double` computations to ensure minimal DataVector operations
  const double denom_constant_a = a_of_t / square(one_over_outer_boundary_);
  const double denom_constant_b = 3.0 * (b_of_t - a_of_t);
  const double numerator_constant = -2.0 * (b_of_t - a_of_t);
  if (Dim == 1) {
    get<0, 0>(inv_jac) *=
        (1.0 + numerator_constant /
                   (denom_constant_a + denom_constant_b * rho_squared) *
                   square(source_coords[0]));
  } else {
    // Reuse rho^2 allocation
    rho_squared = numerator_constant /
                  (denom_constant_a + denom_constant_b * rho_squared) *
                  get<0, 0>(inv_jac);

    for (size_t i = 0; i < Dim; ++i) {
      for (size_t j = 0; j < Dim; ++j) {
        if (i == j) {
          inv_jac.get(i, j) += rho_squared * gsl::at(source_coords, i) *
                               gsl::at(source_coords, j);
        } else {
          inv_jac.get(i, j) = rho_squared * gsl::at(source_coords, i) *
                              gsl::at(source_coords, j);
        }
      }
    }
  }

  return inv_jac;
}

template <size_t Dim>
void CubicScale<Dim>::pup(PUP::er& p) noexcept {
  p | f_of_t_a_;
  p | f_of_t_b_;
  p | one_over_outer_boundary_;
  p | functions_of_time_equal_;
}

template <size_t Dim>
bool operator==(const CubicScale<Dim>& lhs,
                const CubicScale<Dim>& rhs) noexcept {
  return lhs.f_of_t_a_ == rhs.f_of_t_a_ and lhs.f_of_t_b_ == rhs.f_of_t_b_ and
         lhs.one_over_outer_boundary_ == rhs.one_over_outer_boundary_ and
         lhs.functions_of_time_equal_ == rhs.functions_of_time_equal_;
}

// Explicit instantiations
/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template class CubicScale<DIM(data)>;                                      \
  template boost::optional<                                                  \
      std::array<tt::remove_cvref_wrap_t<double>, DIM(data)>>                \
  CubicScale<DIM(data)>::inverse(                                            \
      const std::array<double, DIM(data)>& target_coords, const double time, \
      const std::unordered_map<                                              \
          std::string,                                                       \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&         \
          functions_of_time) const noexcept;                                 \
  template boost::optional<std::array<                                       \
      tt::remove_cvref_wrap_t<std::reference_wrapper<const double>>,         \
      DIM(data)>>                                                            \
  CubicScale<DIM(data)>::inverse(                                            \
      const std::array<std::reference_wrapper<const double>, DIM(data)>&     \
          target_coords,                                                     \
      const double time,                                                     \
      const std::unordered_map<                                              \
          std::string,                                                       \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&         \
          functions_of_time) const noexcept;                                 \
  template bool operator==(const CubicScale<DIM(data)>&,                     \
                           const CubicScale<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                           \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data)> \
  CubicScale<DIM(data)>::operator()(                                   \
      const std::array<DTYPE(data), DIM(data)>& source_coords,         \
      const double time,                                               \
      const std::unordered_map<                                        \
          std::string,                                                 \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&   \
          functions_of_time) const noexcept;                           \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data)> \
  CubicScale<DIM(data)>::frame_velocity(                               \
      const std::array<DTYPE(data), DIM(data)>& source_coords,         \
      const double time,                                               \
      const std::unordered_map<                                        \
          std::string,                                                 \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&   \
          functions_of_time) const noexcept;                           \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),   \
                    Frame::NoFrame>                                    \
  CubicScale<DIM(data)>::jacobian(                                     \
      const std::array<DTYPE(data), DIM(data)>& source_coords,         \
      const double time,                                               \
      const std::unordered_map<                                        \
          std::string,                                                 \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&   \
          functions_of_time) const noexcept;                           \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),   \
                    Frame::NoFrame>                                    \
  CubicScale<DIM(data)>::inv_jacobian(                                 \
      const std::array<DTYPE(data), DIM(data)>& source_coords,         \
      const double time,                                               \
      const std::unordered_map<                                        \
          std::string,                                                 \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&   \
          functions_of_time) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3),
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))
#undef DIM
#undef DTYPE
#undef INSTANTIATE
/// \endcond
}  // namespace CoordMapsTimeDependent
}  // namespace domain
