// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/TimeDependent/Skew.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Identity.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/StlStreamDeclarations.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

namespace domain::CoordinateMaps::TimeDependent {

Skew::Skew(std::string function_of_time_name,
           const std::array<double, 3>& center, const double outer_radius)
    : f_of_t_name_(std::move(function_of_time_name)),
      center_(center),
      outer_radius_(outer_radius),
      f_of_t_names_({f_of_t_name_}) {
  if (outer_radius_ <= 0.0) {
    ERROR("Skew map outer radius must be positive, but is " << outer_radius_);
  }
  one_over_outer_radius_squared_ = 1.0 / square(outer_radius_);
  if (magnitude(center_) >= outer_radius) {
    ERROR("Center of Skew map "
          << center_ << " with radius " << magnitude(center_)
          << " must be within the outer radius " << outer_radius_);
  }
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Skew::operator()(
    const std::array<T, 3>& source_coords, const double time,
    const domain::FunctionsOfTimeMap& functions_of_time) const {
  return map_and_velocity_helper(source_coords, time, functions_of_time, false);
}

std::optional<std::array<double, 3>> Skew::inverse(
    const std::array<double, 3>& target_coords, const double time,
    const domain::FunctionsOfTimeMap& functions_of_time) const {
  const auto& function_of_time = functions_of_time.at(f_of_t_name_);
  const DataVector func = function_of_time->func_and_deriv(time)[0];
  ASSERT(func.size() == 2, "Expected a function of time with size 2, not "
                               << func.size() << " in the Skew map.");

  // Short circuit if there's no function of time
  if (equal_within_roundoff(func[0], 0.0) and
      equal_within_roundoff(func[1], 0.0)) {
    return target_coords;
  }

  // Short circuit if we're at the outer radius
  const double target_radius = magnitude(target_coords);
  if (equal_within_roundoff(target_radius, outer_radius_)) {
    return target_coords;
  } else if (target_radius >= outer_radius_) {
    return std::nullopt;
  }

  // Another short circuit. Target y & z are the same as source y & z
  const double tan_sum =
      -1.0 * (tan(func[0]) * (target_coords[1] - center_[1]) +
              tan(func[1]) * (target_coords[2] - center_[2]));
  if (equal_within_roundoff(tan_sum, 0.0)) {
    return target_coords;
  }

  std::array<double, 3> temporary_source_coord = target_coords;

  const auto root_func = [&](const double source_coord_x) -> double {
    temporary_source_coord[0] = source_coord_x;
    // Sometimes the temporary point is outside the outer radius, but we still
    // need to continue with the root find, so allow evaluation of the width
    // outside the outer radius without an error
    const double width = get_width(temporary_source_coord, true);
    return source_coord_x + width * tan_sum - target_coords[0];
  };

  // \bar{x} -> x - w * (tan(f_y)*(y-y_C) + tan(f_z)*(z-z_C)) and 0<=w<=1, then
  // x <= \bar{x} <= x - (tan(f_y)*(y-y_C) + tan(f_z)*(z-z_C)), ==>
  // 0 <= \bar{x} - x <= -(tan(f_y)*(y-y_C) + tan(f_z)*(z-z_C)), ==>
  // -\bar{x} <= - x <= -\bar{x} - (tan(f_y)*(y-y_C) + tan(f_z)*(z-z_C)), ==>
  // \bar{x} >= x >= \bar{x} + (tan(f_y)*(y-y_C) + tan(f_z)*(z-z_C))
  // This last line is our bracket for the root

  // We give a small epsilon to give the root find a little buffer. It is highly
  // unlikely there will be another root within these bounds
  const double eps = std::numeric_limits<double>::epsilon() * 100.0;

  const double x_0 = target_coords[0] - tan_sum - eps;
  const double x_1 = target_coords[0] + eps;
  const double lower_x = std::min(x_0, x_1);
  const double upper_x = std::max(x_0, x_1);
  const double lower_func = root_func(lower_x);
  const double upper_func = root_func(upper_x);

  // Since we adjusted the bounds by eps above, we have a slightly larger eps
  // here to account for the arithmetic of the root func
  if (equal_within_roundoff(lower_func, 0.0, 5 * eps)) {
    temporary_source_coord[0] = lower_x;
    return temporary_source_coord;
  }

  if (equal_within_roundoff(upper_func, 0.0, 5 * eps)) {
    temporary_source_coord[0] = upper_x;
    return temporary_source_coord;
  }

  if (lower_func * upper_func > 0.0) {
    ERROR("No root in Skew map inverse!. Target coord = "
          << target_coords << ", lower_x = " << lower_x
          << ", lower_func = " << lower_func << ", upper_x = " << upper_x
          << ", upper_func = " << upper_func << ", tan_sum = " << tan_sum);
  }

  temporary_source_coord[0] = RootFinder::toms748<true>(
      root_func, lower_x, upper_x, lower_func, upper_func, eps, eps);

  return temporary_source_coord;
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Skew::frame_velocity(
    const std::array<T, 3>& source_coords, const double time,
    const domain::FunctionsOfTimeMap& functions_of_time) const {
  return map_and_velocity_helper(source_coords, time, functions_of_time, true);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Skew::jacobian(
    const std::array<T, 3>& source_coords, const double time,
    const domain::FunctionsOfTimeMap& functions_of_time) const {
  using ResultT = tt::remove_cvref_wrap_t<T>;

  check_for_singular_map(source_coords, time, functions_of_time);

  const ResultT width = get_width(source_coords);
  const std::array<ResultT, 3> width_deriv = get_width_deriv(source_coords);

  const auto& function_of_time = functions_of_time.at(f_of_t_name_);
  const DataVector func = function_of_time->func_and_deriv(time)[0];
  ASSERT(func.size() == 2, "Expected a function of time with size 2, not "
                               << func.size() << " in the Skew map.");

  auto result = identity<3>(dereference_wrapper(source_coords[0]));

  const DataVector tan_func = -1.0 * tan(func);
  // Temporarily use component to avoid allocation
  auto& tan_sum = get<0, 2>(result);
  tan_sum = tan_func[0] * (source_coords[1] - center_[1]) +
            tan_func[1] * (source_coords[2] - center_[2]);

  get<0, 0>(result) += width_deriv[0] * tan_sum;
  get<0, 1>(result) = width_deriv[1] * tan_sum + width * tan_func[0];
  get<0, 2>(result) = width_deriv[2] * tan_sum + width * tan_func[1];

  return result;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Skew::inv_jacobian(
    const std::array<T, 3>& source_coords, const double time,
    const domain::FunctionsOfTimeMap& functions_of_time) const {
  return determinant_and_inverse(
             jacobian(source_coords, time, functions_of_time))
      .second;
}

template <typename T>
tt::remove_cvref_wrap_t<T> Skew::get_width(
    const std::array<T, 3>& source_coords, const bool ignore_error) const {
  using ResultT = tt::remove_cvref_wrap_t<T>;
  // Will be reused for result
  ResultT lambda =
      one_over_outer_radius_squared_ * dot(source_coords, source_coords);

  ResultT& result = lambda;

  // Go point by point because we have to check roundoff
  for (size_t i = 0; i < get_size(result); i++) {
    if (equal_within_roundoff(get_element(lambda, i), 0.0)) {
      get_element(result, i) = 1.0;
    } else if (equal_within_roundoff(get_element(lambda, i), 1.0)) {
      get_element(result, i) = 0.0;
    } else if (ignore_error or (get_element(lambda, i) > 0.0 and
                                get_element(lambda, i) < 1.0)) {
      get_element(result, i) = 0.5 * (1.0 + cos(M_PI * get_element(lambda, i)));
    } else {
      using ::operator<<;
      const std::vector<double> bad_point{
          get_element(dereference_wrapper(source_coords[0]), i),
          get_element(dereference_wrapper(source_coords[1]), i),
          get_element(dereference_wrapper(source_coords[2]), i)};
      ERROR("Skew map: Source coordinate "
            << bad_point << " not in valid region. Lambda is "
            << get_element(lambda, i) << ". Center is " << center_
            << ". Outer radius is " << outer_radius_);
    }
  }

  return result;
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Skew::get_width_deriv(
    const std::array<T, 3>& source_coords) const {
  using ResultT = tt::remove_cvref_wrap_t<T>;

  const ResultT lambda =
      one_over_outer_radius_squared_ * dot(source_coords, source_coords);

  std::array<ResultT, 3> grad_width{};
  for (size_t i = 0; i < 3; i++) {
    gsl::at(grad_width, i) = gsl::at(source_coords, i);
  }
  // Factors of two cancel from grad lambda and pi/2
  grad_width *= -one_over_outer_radius_squared_ * M_PI * sin(M_PI * lambda);

  for (size_t i = 0; i < get_size(lambda); i++) {
    // sin(lambda * M_PI) is zero at both lambda=0 and lambda=1
    if (equal_within_roundoff(get_element(lambda, i), 0.0) or
        equal_within_roundoff(get_element(lambda, i), 1.0)) {
      for (size_t j = 0; j < 3; j++) {
        get_element(gsl::at(grad_width, j), i) = 0.0;
      }
    } else if (get_element(lambda, i) < 0.0 or get_element(lambda, i) > 1.0) {
      using ::operator<<;
      const std::vector<double> bad_point{
          get_element(dereference_wrapper(source_coords[0]), i),
          get_element(dereference_wrapper(source_coords[1]), i),
          get_element(dereference_wrapper(source_coords[2]), i)};
      ERROR("Skew map: Source coordinate "
            << bad_point << " not in valid region. Lambda is "
            << get_element(lambda, i) << ". Center is " << center_
            << ". Outer radius is " << outer_radius_);
    }
  }

  return grad_width;
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Skew::map_and_velocity_helper(
    const std::array<T, 3>& source_coords, const double time,
    const FunctionsOfTimeMap& functions_of_time,
    const bool return_velocity) const {
  using ResultT = tt::remove_cvref_wrap_t<T>;

  check_for_singular_map(source_coords, time, functions_of_time);

  std::array<ResultT, 3> result{};
  if (return_velocity) {
    result = make_array<3>(
        make_with_value<ResultT>(dereference_wrapper(source_coords[0]), 0.0));
  } else {
    for (size_t i = 0; i < 3; i++) {
      gsl::at(result, i) = gsl::at(source_coords, i);
    }
  }

  // Currently result is the source coords, but with the proper return type
  const ResultT width = get_width(source_coords);

  const auto& function_of_time = functions_of_time.at(f_of_t_name_);
  const auto func_and_deriv = function_of_time->func_and_deriv(time);
  ASSERT(func_and_deriv[0].size() == 2,
         "Expected a function of time with size 2, not "
             << func_and_deriv[0].size() << " in the Skew map.");

  if (return_velocity) {
    const auto& func = func_and_deriv[0];
    const auto& deriv = func_and_deriv[1];

    result[0] = -1.0 * width *
                (deriv[0] * (1.0 + square(tan(func[0]))) *
                     (source_coords[1] - center_[1]) +
                 deriv[1] * (1.0 + square(tan(func[1]))) *
                     (source_coords[2] - center_[2]));
  } else {
    result[0] -=
        width * (tan(func_and_deriv[0][0]) * (source_coords[1] - center_[1]) +
                 tan(func_and_deriv[0][1]) * (source_coords[2] - center_[2]));
  }

  return result;
}

void Skew::pup(PUP::er& p) {
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | f_of_t_name_;
    p | center_;
    p | outer_radius_;
  }

  // No need to pup these because they are uniquely determined
  if (p.isUnpacking()) {
    one_over_outer_radius_squared_ = 1.0 / square(outer_radius_);
    f_of_t_names_.clear();
    f_of_t_names_.insert(f_of_t_name_);
  }
}

template <typename T>
void Skew::check_for_singular_map(
    const std::array<T, 3>& source_coords, const double time,
    const FunctionsOfTimeMap& functions_of_time) const {
  (void)source_coords;
  (void)time;
  (void)functions_of_time;

#ifdef SPECTRE_DEBUG
  using ResultT = tt::remove_cvref_wrap_t<T>;
  const ResultT lambda =
      one_over_outer_radius_squared_ * dot(source_coords, source_coords);

  const auto& function_of_time = functions_of_time.at(f_of_t_name_);
  const auto func = function_of_time->func_and_deriv(time)[0];
  ASSERT(func.size() == 2, "Expected a function of time with size 2, not "
                               << func.size() << " in the Skew map.");
  const double tan_func0 = tan(func[0]);
  const double tan_func1 = tan(func[1]);

  CAPTURE_FOR_ERROR(tan_func0);
  CAPTURE_FOR_ERROR(tan_func1);

  const ResultT tan_sum =
      -1.0 * (tan(func[0]) * (source_coords[1] - center_[1]) +
              tan(func[1]) * (source_coords[2] - center_[2]));

  std::array<double, 3> temporary_point{
      get_element(dereference_wrapper(source_coords[0]), 0),
      get_element(dereference_wrapper(source_coords[1]), 0),
      get_element(dereference_wrapper(source_coords[2]), 0)};

  // We can check if the map is singular by checking if the x-deriv of the map
  // is negative. If it is, then it means the map is not one-to-one and our grid
  // is being stretched over itself. In order to check this, we use a property
  // of the analytic functions used in the map that the second x-deriv of the
  // map will cross zero exactly once for +x and once for -x. We use the
  // x-values of this zero-crossing to evaluate the first x-deriv and check if
  // it is negative. This is not a global check as it only checks along the
  // y=z=const line, but if there is an issue, some point within the domain
  // should error.
  for (size_t i = 0; i < get_size(lambda); i++) {
    // No need to check this point if the map is the identity
    if (equal_within_roundoff(get_element(tan_sum, i), 0.0)) {
      continue;
    }

    temporary_point[0] = get_element(dereference_wrapper(source_coords[0]), i);
    temporary_point[1] = get_element(dereference_wrapper(source_coords[1]), i);
    temporary_point[2] = get_element(dereference_wrapper(source_coords[2]), i);

    // Can't check if x=0 and we are at the outer boundary, because then the
    // largest_x below is just zero and we won't be able to find roots.
    if (equal_within_roundoff(temporary_point[0], 0.0) and
        equal_within_roundoff(magnitude(temporary_point), outer_radius_)) {
      continue;
    }

    CAPTURE_FOR_ERROR(get_element(tan_sum, i));

    const auto first_deriv = [&](const double x) {
      temporary_point[0] = x;
      const double temporary_lambda = one_over_outer_radius_squared_ *
                                      dot(temporary_point, temporary_point);
      return 1.0 + M_PI * x * one_over_outer_radius_squared_ *
                       get_element(tan_sum, i) * sin(M_PI * temporary_lambda);
    };
    const auto second_deriv = [&](const double x) -> double {
      temporary_point[0] = x;
      const double temporary_lambda = one_over_outer_radius_squared_ *
                                      dot(temporary_point, temporary_point);
      return M_PI * one_over_outer_radius_squared_ * get_element(tan_sum, i) *
             (sin(M_PI * temporary_lambda) +
              2.0 * M_PI * one_over_outer_radius_squared_ * square(x) *
                  cos(M_PI * temporary_lambda));
    };

    const double eps = std::numeric_limits<double>::epsilon() * 100.0;
    // The endpoint of the y=z=const line given that the outer boundary is a
    // sphere.
    const double largest_x =
        sqrt(square(outer_radius_) - square(temporary_point[1]) -
             square(temporary_point[2]));

    CAPTURE_FOR_ERROR(temporary_point);
    CAPTURE_FOR_ERROR(largest_x);

    // Positive and negative zeros of the second deriv
    const double negative_candidate = RootFinder::toms748<true>(
        second_deriv, -largest_x, -eps, second_deriv(-largest_x),
        second_deriv(-eps), eps, eps);
    const double positive_candidate = RootFinder::toms748<true>(
        second_deriv, eps, largest_x, second_deriv(eps),
        second_deriv(largest_x), eps, eps);

    CAPTURE_FOR_ERROR(negative_candidate);
    CAPTURE_FOR_ERROR(positive_candidate);

    if (first_deriv(negative_candidate) <= 0.0 or
        first_deriv(positive_candidate) <= 0.0) {
      using ::operator<<;
      ERROR("Skew map is singular. Found for point "
            << temporary_point
            << " (y and z coords are the important ones). Important values "
               "are:\n Outer radius, R = "
            << outer_radius_ << "\n |x|^2/R^2 = " << get_element(lambda, i)
            << "\n tan(F_y)*y + tan(F_z)*z = " << tan_sum
            << "\n F_y = " << func[0] << "\n F_z = " << func[1]
            << "\n tan(F_y) = " << tan(func[0])
            << "\n tan(F_z) = " << tan(func[1]) << "\n");
    }
  }
#endif
}

bool operator==(const Skew& lhs, const Skew& rhs) {
  return lhs.f_of_t_name_ == rhs.f_of_t_name_ and lhs.center_ == rhs.center_ and
         lhs.outer_radius_ == rhs.outer_radius_;
}

bool operator!=(const Skew& lhs, const Skew& rhs) { return not(lhs == rhs); }

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>                 \
  Skew::operator()(const std::array<DTYPE(data), 3>& source_coords,            \
                   double time,                                                \
                   const domain::FunctionsOfTimeMap& functions_of_time) const; \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>                 \
  Skew::frame_velocity(                                                        \
      const std::array<DTYPE(data), 3>& source_coords, const double time,      \
      const domain::FunctionsOfTimeMap& functions_of_time) const;              \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>   \
  Skew::jacobian(const std::array<DTYPE(data), 3>& source_coords, double time, \
                 const domain::FunctionsOfTimeMap& functions_of_time) const;   \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>   \
  Skew::inv_jacobian(                                                          \
      const std::array<DTYPE(data), 3>& source_coords, double time,            \
      const domain::FunctionsOfTimeMap& functions_of_time) const;              \
  template tt::remove_cvref_wrap_t<DTYPE(data)> Skew::get_width(               \
      const std::array<DTYPE(data), 3>& source_coords,                         \
      const bool ignore_error) const;                                          \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>                 \
  Skew::get_width_deriv(const std::array<DTYPE(data), 3>& source_coords)       \
      const;                                                                   \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>                 \
  Skew::map_and_velocity_helper(                                               \
      const std::array<DTYPE(data), 3>& source_coords, double time,            \
      const domain::FunctionsOfTimeMap& functions_of_time,                     \
      bool return_velocity) const;                                             \
  template void Skew::check_for_singular_map(                                  \
      const std::array<DTYPE(data), 3>& source_coords, double time,            \
      const domain::FunctionsOfTimeMap& functions_of_time) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE
}  // namespace domain::CoordinateMaps::TimeDependent
