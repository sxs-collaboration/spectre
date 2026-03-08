// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/TimeDependent/Shape.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <new>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <type_traits>
#include <unordered_set>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain::CoordinateMaps::TimeDependent {

using ylm::Spherepack;
using ylm::SpherepackIterator;

size_t lmax_from_coefs(const DataVector& coefs) {
  const size_t num_coefs = coefs.size();
  const auto l_max =
      static_cast<size_t>(sqrt(static_cast<double>(num_coefs) / 2) - 1);
  ASSERT(square(l_max + 1) * 2 == num_coefs,
         "Number of shape coefficients ("
             << num_coefs << ") is not of valid size 2(l_max+1)(l_max+1).");
  ASSERT(l_max >= 2, "l_max must be at least 2.");
  return l_max;
}

DataVector truncate_coefs(const DataVector& coefs, SpherepackIterator iterator,
                          SpherepackIterator truncated_iterator) {
  DataVector truncated_coefs(truncated_iterator.spherepack_array_size(), 0.0);
  for (truncated_iterator.reset(); truncated_iterator; ++truncated_iterator) {
    const size_t l = truncated_iterator.l();
    const size_t m = truncated_iterator.m();
    // If the requested truncation order exceeds the available coefficients,
    // leave the entry at zero.
    if (l > iterator.l_max() or m > iterator.m_max()) {
      continue;
    }
    iterator.set(l, m, truncated_iterator.coefficient_array());
    truncated_coefs[truncated_iterator()] = coefs[iterator()];
  }
  return truncated_coefs;
}

template <typename T>
std::array<T, 2> cartesian_to_spherical(const std::array<T, 3>& cartesian) {
  const auto& [x, y, z] = cartesian;
  return {atan2(hypot(x, y), z), atan2(y, x)};
}
template <typename T>
void cartesian_to_spherical(gsl::not_null<std::array<T, 2>*> result,
                            const std::array<T, 3>& cartesian) {
  const auto& [x, y, z] = cartesian;
  gsl::at(*result, 0) = atan2(hypot(x, y), z);
  gsl::at(*result, 1) = atan2(y, x);
}

template <typename T>
void Shape::jacobian_helper(
    gsl::not_null<tnsr::Ij<T, 3, Frame::NoFrame>*> result,
    const ylm::Spherepack::InterpolationInfo<T>& interpolation_info,
    const DataVector& extended_coefs, const std::array<T, 3>& centered_coords,
    const T& radial_distortion, const T& transition_func,
    const Spherepack& ylm) const {
  const auto angular_gradient = ylm.gradient_from_coefs(extended_coefs);
  tnsr::i<DataVector, 3, Frame::Inertial> cartesian_gradient(
      ylm.physical_size());

  std::array<DataVector, 2> collocation_theta_phis{};
  collocation_theta_phis[0].set_data_ref(&get<2>(cartesian_gradient));
  collocation_theta_phis[1].set_data_ref(&get<1>(cartesian_gradient));
  collocation_theta_phis = ylm.theta_phi_points();

  const auto& col_thetas = collocation_theta_phis[0];
  const auto& col_phis = collocation_theta_phis[1];

  // The Cartesian derivative is the Pfaffian derivative multiplied by the
  // inverse Jacobian matrix. Some optimizations here may be possible by
  // introducing temporaries for some of the sin/cos which are computed twice,
  // if the compiler CSE doesn't take care of it.
  get<0>(cartesian_gradient) =
      (cos(col_thetas) * cos(col_phis) * get<0>(angular_gradient) -
       sin(col_phis) * get<1>(angular_gradient));

  get<1>(cartesian_gradient) =
      (cos(col_thetas) * sin(col_phis) * get<0>(angular_gradient) +
       cos(col_phis) * get<1>(angular_gradient));

  get<2>(cartesian_gradient) = -sin(col_thetas) * get<0>(angular_gradient);

  // re-use allocations. The specific buffers that are reused are important to
  // avoid overwriting anything
  std::array<T, 3> target_gradient{};
  for (size_t i = 0; i < 3; i++) {
    if constexpr (std::is_same_v<T, DataVector>) {
      gsl::at(target_gradient, i)
          .set_data_ref(make_not_null(&result->get(2, i)));
    } else {
      gsl::at(target_gradient, i) = result->get(2, i);
    }

    // interpolate the cartesian gradient to the thetas and phis of the
    // `source_coords`
    ylm.interpolate(make_not_null(&gsl::at(target_gradient, i)),
                    cartesian_gradient.get(i).data(), interpolation_info);
  }

  // G / r
  auto transition_func_over_radius =
      transition_func_->operator()(centered_coords, {1});
  auto transition_func_gradient_times_distortion =
      transition_func_->gradient(centered_coords) * radial_distortion;

  auto& target_gradient_times_spatial_part = target_gradient;
  target_gradient_times_spatial_part *= transition_func_over_radius;

  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      result->get(i, j) =
          -gsl::at(centered_coords, i) *
          (gsl::at(transition_func_gradient_times_distortion, j) +
           gsl::at(target_gradient_times_spatial_part, j));
    }

    result->get(i, i) += 1.0 - radial_distortion * transition_func;
  }
}

Shape::Shape(
    const std::array<double, 3>& center, const double truncation_limit,
    std::unique_ptr<ShapeMapTransitionFunctions::ShapeMapTransitionFunction>
        transition_func,
    std::string shape_function_of_time_name,
    std::optional<std::string> size_function_of_time_name)
    : shape_f_of_t_name_(std::move(shape_function_of_time_name)),
      size_f_of_t_name_(std::move(size_function_of_time_name)),
      center_(center),
      truncation_limit_(truncation_limit),
      transition_func_(std::move(transition_func)) {
  f_of_t_names_.insert(shape_f_of_t_name_);
  if (size_f_of_t_name_.has_value()) {
    f_of_t_names_.insert(size_f_of_t_name_.value());
  }
}

Shape& Shape::operator=(const Shape& rhs) {
  if (*this != rhs) {
    shape_f_of_t_name_ = rhs.shape_f_of_t_name_;
    size_f_of_t_name_ = rhs.size_f_of_t_name_;
    f_of_t_names_ = rhs.f_of_t_names_;
    center_ = rhs.center_;
    truncation_limit_ = rhs.truncation_limit_;
    transition_func_ = rhs.transition_func_ != nullptr
                           ? rhs.transition_func_->get_clone()
                           : nullptr;
    // we manually call the destructor and constructor here in case the cache
    // needs to be resized. It is up to the user to guarantee that no thread is
    // accessing the cache at this point.
    spherepack_cache_.~SpherepackCache();
    new (&spherepack_cache_) SpherepackCache{cache_capacity_};
  }
  return *this;
}

Shape::Shape(const Shape& rhs)
    : shape_f_of_t_name_(rhs.shape_f_of_t_name_),
      size_f_of_t_name_(rhs.size_f_of_t_name_),
      f_of_t_names_(rhs.f_of_t_names_),
      center_(rhs.center_),
      truncation_limit_(rhs.truncation_limit_),
      transition_func_(rhs.transition_func_ != nullptr
                           ? rhs.transition_func_->get_clone()
                           : nullptr),
      spherepack_cache_(cache_capacity_) {}

Shape::Shape(Shape&& rhs)
    : shape_f_of_t_name_(std::move(rhs.shape_f_of_t_name_)),
      size_f_of_t_name_(std::move(rhs.size_f_of_t_name_)),
      f_of_t_names_(std::move(rhs.f_of_t_names_)),
      center_(std::move(rhs.center_)),
      truncation_limit_(rhs.truncation_limit_),
      transition_func_(std::move(rhs.transition_func_)),
      spherepack_cache_(cache_capacity_) {}

Shape& Shape::operator=(Shape&& rhs) {
  if (this != &rhs) {
    shape_f_of_t_name_ = std::move(rhs.shape_f_of_t_name_);
    size_f_of_t_name_ = std::move(rhs.size_f_of_t_name_);
    f_of_t_names_ = std::move(rhs.f_of_t_names_);
    center_ = std::move(rhs.center_);
    truncation_limit_ = rhs.truncation_limit_;
    transition_func_ = std::move(rhs.transition_func_);
    // we manually call the destructor and constructor here in case the cache
    // needs to be resized. It is up to the user to guarantee that no thread is
    // accessing the cache at this point.
    spherepack_cache_.~SpherepackCache();
    new (&spherepack_cache_) SpherepackCache{cache_capacity_};
  }
  return *this;
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Shape::operator()(
    const std::array<T, 3>& source_coords, const double time,
    const FunctionsOfTimeMap& functions_of_time) const {
  const auto centered_coords = center_coordinates(source_coords);
  auto theta_phis = cartesian_to_spherical(centered_coords);
  const auto [coefs, coef_derivs, coef_dderivs] =
      functions_of_time.at(shape_f_of_t_name_)->func_and_2_derivs(time);
  const size_t l_max = lmax_from_coefs(coefs);
  const SpherepackIterator full_iterator{l_max, l_max};
  const size_t truncated_l_max =
      find_truncated_l_max(coefs, coef_derivs, coef_dderivs, full_iterator);
  const auto cached_item = get_spherepack_cache_entry(truncated_l_max);
  const auto& [truncated_iterator, ylm] = *cached_item.value();
  DataVector truncated_coefs =
      truncate_coefs(coefs, full_iterator, truncated_iterator);
  check_size(make_not_null(&truncated_coefs), functions_of_time, time, false);
  // re-use allocation
  const auto interpolation_info = ylm.set_up_interpolation_info(theta_phis);
  auto& radial_distortion = get<0>(theta_phis);
  // evaluate the spherical harmonic expansion at the angles of
  // `source_coords`
  ylm.interpolate_from_coefs(make_not_null(&radial_distortion), truncated_coefs,
                             interpolation_info);

  // this should be taken care of by the control system but is very hard to
  // debug
#ifdef SPECTRE_DEBUG
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType shift_radii =
      radial_distortion *
      transition_func_->operator()(centered_coords, std::nullopt);
  if constexpr (std::is_same_v<ReturnType, double>) {
    ASSERT(shift_radii < 1., "Coordinates mapped through the center!");
  } else {
    for (const auto& radius : shift_radii) {
      ASSERT(radius < 1., "Coordinates mapped through the center!");
    }
  }
#endif  // SPECTRE_DEBUG

  return center_ +
         centered_coords *
             (1. - radial_distortion * transition_func_->operator()(
                                           centered_coords, std::nullopt));
}

std::optional<std::array<double, 3>> Shape::inverse(
    const std::array<double, 3>& target_coords, const double time,
    const FunctionsOfTimeMap& functions_of_time) const {
  const std::array<double, 3> centered_coords =
      center_coordinates(target_coords);
  const std::array<double, 2> theta_phis =
      cartesian_to_spherical(centered_coords);
  const auto [coefs, coef_derivs, coef_dderivs] =
      functions_of_time.at(shape_f_of_t_name_)->func_and_2_derivs(time);
  const size_t l_max = lmax_from_coefs(coefs);
  const SpherepackIterator full_iterator{l_max, l_max};
  const size_t truncated_l_max =
      find_truncated_l_max(coefs, coef_derivs, coef_dderivs, full_iterator);
  double radial_distortion = 0.0;
  // extra guard to minimize lifetime of cached_item which has a lock
  {
    const auto cached_item = get_spherepack_cache_entry(truncated_l_max);
    const auto& [truncated_iterator, ylm] = *cached_item.value();
    DataVector truncated_coefs =
        truncate_coefs(coefs, full_iterator, truncated_iterator);
    check_size(make_not_null(&truncated_coefs), functions_of_time, time, false);
    radial_distortion = ylm.interpolate_from_coefs(truncated_coefs, theta_phis);
  }
  const std::optional<double> original_radius_over_radius =
      transition_func_->original_radius_over_radius(centered_coords,
                                                    radial_distortion);
  if (not original_radius_over_radius.has_value()) {
    return std::nullopt;
  }
  return center_ + centered_coords * original_radius_over_radius.value();
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Shape::frame_velocity(
    const std::array<T, 3>& source_coords, const double time,
    const FunctionsOfTimeMap& functions_of_time) const {
  const auto centered_coords = center_coordinates(source_coords);
  auto theta_phis = cartesian_to_spherical(centered_coords);
  const auto [coefs, coef_derivs, coef_dderivs] =
      functions_of_time.at(shape_f_of_t_name_)->func_and_2_derivs(time);
  const size_t l_max = lmax_from_coefs(coefs);
  const SpherepackIterator full_iterator{l_max, l_max};
  const size_t truncated_l_max =
      find_truncated_l_max(coefs, coef_derivs, coef_dderivs, full_iterator);
  const auto cached_item = get_spherepack_cache_entry(truncated_l_max);
  const auto& [truncated_iterator, ylm] = *cached_item.value();
  DataVector truncated_coef_derivs =
      truncate_coefs(coef_derivs, full_iterator, truncated_iterator);
  check_size(make_not_null(&truncated_coef_derivs), functions_of_time, time,
             true);
  const auto interpolation_info = ylm.set_up_interpolation_info(theta_phis);
  // re-use allocation
  auto& radii_velocities = get<0>(theta_phis);
  ylm.interpolate_from_coefs(make_not_null(&radii_velocities),
                             truncated_coef_derivs, interpolation_info);
  return -centered_coords * radii_velocities *
         transition_func_->operator()(centered_coords, std::nullopt);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Shape::jacobian(
    const std::array<T, 3>& source_coords, const double time,
    const FunctionsOfTimeMap& functions_of_time) const {
  const auto centered_coords = center_coordinates(source_coords);

  // The distorted radii are calculated analogously to the call operator
  auto theta_phis = cartesian_to_spherical(centered_coords);
  const auto [coefs, coef_derivs, coef_dderivs] =
      functions_of_time.at(shape_f_of_t_name_)->func_and_2_derivs(time);
  const size_t l_max = lmax_from_coefs(coefs);
  const SpherepackIterator full_iterator{l_max, l_max};
  size_t truncated_l_max =
      find_truncated_l_max(coefs, coef_derivs, coef_dderivs, full_iterator);
  // we need an additional l_max to compute the gradient without aliasing error
  truncated_l_max += 1;
  const auto cached_item = get_spherepack_cache_entry(truncated_l_max);
  const auto& [truncated_iterator, ylm] = *cached_item.value();

  DataVector truncated_coefs =
      truncate_coefs(coefs, full_iterator, truncated_iterator);
  check_size(make_not_null(&truncated_coefs), functions_of_time, time, false);
  const auto interpolation_info = ylm.set_up_interpolation_info(theta_phis);

  // Re-use allocation
  auto& radial_distortion = get<0>(theta_phis);
  ylm.interpolate_from_coefs(make_not_null(&radial_distortion), truncated_coefs,
                             interpolation_info);

  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType transition_func =
      transition_func_->operator()(centered_coords, std::nullopt);
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> result(
      get_size(centered_coords[0]));

  jacobian_helper(make_not_null(&result), interpolation_info, truncated_coefs,
                  centered_coords, radial_distortion, transition_func, ylm);
  return result;
}

void Shape::coords_frame_velocity_jacobian(
    gsl::not_null<std::array<DataVector, 3>*> source_and_target_coords,
    gsl::not_null<std::array<DataVector, 3>*> frame_vel,
    gsl::not_null<tnsr::Ij<DataVector, 3, Frame::NoFrame>*> jac, double time,
    const FunctionsOfTimeMap& functions_of_time) const {
  const size_t size = get<0>(*source_and_target_coords).size();
  ASSERT(size > 0,
         "The source coords have size 0 but the argument requires you to pass "
         "in the coordinates.");
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(*frame_vel, i).destructive_resize(size);
    for (size_t j = 0; j < 3; ++j) {
      jac->get(i, j).destructive_resize(size);
    }
  }
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempI<0, 3, Frame::Inertial>>>
      temps(size);

  std::array<DataVector, 3> centered_coords{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(centered_coords, i)
        .set_data_ref(&get<::Tags::TempI<0, 3, Frame::Inertial>>(temps).get(i));
  }
  center_coordinates(make_not_null(&centered_coords),
                     *source_and_target_coords);

  std::array<DataVector, 2> theta_phis{};
  theta_phis[0].set_data_ref(&get<0, 0>(*jac));
  theta_phis[1].set_data_ref(&get<0, 1>(*jac));
  cartesian_to_spherical(make_not_null(&theta_phis), centered_coords);

  const auto [coefs, coef_derivs, coef_dderivs] =
      functions_of_time.at(shape_f_of_t_name_)->func_and_2_derivs(time);
  const size_t l_max = lmax_from_coefs(coefs);
  const SpherepackIterator full_iterator{l_max, l_max};
  size_t truncated_l_max =
      find_truncated_l_max(coefs, coef_derivs, coef_dderivs, full_iterator);
  // we need an additional l_max to compute the gradient without aliasing error
  truncated_l_max += 1;
  const auto cached_item = get_spherepack_cache_entry(truncated_l_max);
  const auto& [truncated_iterator, ylm] = *cached_item.value();
  DataVector truncated_coefs =
      truncate_coefs(coefs, full_iterator, truncated_iterator);
  DataVector truncated_coef_derivs =
      truncate_coefs(coef_derivs, full_iterator, truncated_iterator);
  check_size(make_not_null(&truncated_coefs), functions_of_time, time, false);
  check_size(make_not_null(&truncated_coef_derivs), functions_of_time, time,
             true);
  const auto interpolation_info = ylm.set_up_interpolation_info(theta_phis);
  auto& radial_distortion = get(get<::Tags::TempScalar<0>>(temps));
  // evaluate the spherical harmonic expansion at the angles of
  // `source_coords`
  ylm.interpolate_from_coefs(make_not_null(&radial_distortion), truncated_coefs,
                             interpolation_info);

  auto& transition_func = get(get<::Tags::TempScalar<1>>(temps));
  transition_func = transition_func_->operator()(centered_coords, std::nullopt);
  *source_and_target_coords =
      center_ + centered_coords * (1. - radial_distortion * transition_func);

  auto& radii_velocities = get<0, 1>(*jac);
  ylm.interpolate_from_coefs(make_not_null(&radii_velocities),
                             truncated_coef_derivs, interpolation_info);
  *frame_vel = -centered_coords * radii_velocities * transition_func;

  jacobian_helper<DataVector>(jac, interpolation_info, truncated_coefs,
                              centered_coords, radial_distortion,
                              transition_func, ylm);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Shape::inv_jacobian(
    const std::array<T, 3>& source_coords, const double time,
    const FunctionsOfTimeMap& functions_of_time) const {
  return determinant_and_inverse(
             jacobian(source_coords, time, functions_of_time))
      .second;
}

size_t Shape::find_truncated_l_max(const DataVector& coefs,
                                   const DataVector& coef_derivs,
                                   const DataVector& coef_dderivs,
                                   SpherepackIterator iterator) const {
  const size_t l_max = iterator.l_max();
  // we require at least l=2 because l=0 is size and l=1 is translation
  for (size_t l = l_max; l > 2; --l) {
    for (int m = static_cast<int>(l); m >= -static_cast<int>(l); --m) {
      iterator.set(l, m);
      const size_t current_index = iterator();
      if (abs(coefs[current_index]) > truncation_limit_ or
          abs(coef_derivs[current_index]) > truncation_limit_ or
          abs(coef_dderivs[current_index]) > truncation_limit_) {
        return l;
      }
    }
  }
  return 2;
}

void Shape::check_size(const gsl::not_null<DataVector*>& coefs,
                       const FunctionsOfTimeMap& functions_of_time,
                       const double time, const bool use_deriv) const {
  if (size_f_of_t_name_.has_value()) {
    ASSERT((*coefs)[0] == 0.0,
           "When using a size function of time, the l=0 "
               << (use_deriv ? "derivative" : "component")
               << " of the shape "
                  "function of time must be zero. Currently it is "
               << (*coefs)[0]);

    double l0m0_spherical_harmonic_coef =
        std::numeric_limits<double>::signaling_NaN();
    if (use_deriv) {
      l0m0_spherical_harmonic_coef =
          functions_of_time.at(size_f_of_t_name_.value())
              ->func_and_deriv(time)[1][0];
    } else {
      l0m0_spherical_harmonic_coef =
          functions_of_time.at(size_f_of_t_name_.value())->func(time)[0][0];
    }

    // Size holds the *actual* \lambda_00 spherical harmonic coefficient, but
    // shape holds Spherepack coefficients so we must convert between the two.
    // Need to multiply lambda_00 by sqrt(2/pi)
    (*coefs)[0] = M_SQRT1_2 * M_2_SQRTPI * l0m0_spherical_harmonic_coef;
  }
}

Shape::CachedItem Shape::get_spherepack_cache_entry(const size_t l_max) const {
  auto predicate =
      [&l_max](
          const std::unique_ptr<Shape::SpherepackEntry>& cached_entry) -> bool {
    return cached_entry->first.l_max() == l_max;
  };
  // we expect few cache misses, so use find first to avoid locking the cache
  auto cached_entry = spherepack_cache_.find(predicate);
  if (cached_entry.has_value()) {
    return cached_entry;
  }
  auto compute_new_entry = [&l_max]() {
    return std::make_unique<Shape::SpherepackEntry>(
        Shape::SpherepackEntry{{l_max, l_max}, {l_max, l_max}});
  };
  auto new_entry = spherepack_cache_.push(compute_new_entry, predicate);
  ASSERT(new_entry.has_value(), "FifoCache push failed to return a value.");
  return new_entry;
}

bool operator==(const Shape& lhs, const Shape& rhs) {
  return lhs.shape_f_of_t_name_ == rhs.shape_f_of_t_name_ and
         lhs.size_f_of_t_name_ == rhs.size_f_of_t_name_ and
         lhs.center_ == rhs.center_ and
         lhs.truncation_limit_ == rhs.truncation_limit_ and
         (lhs.transition_func_ == nullptr) ==
             (rhs.transition_func_ == nullptr) and
         ((lhs.transition_func_ == nullptr and
           rhs.transition_func_ == nullptr) or
          *lhs.transition_func_ == *rhs.transition_func_);
}

bool operator!=(const Shape& lhs, const Shape& rhs) { return not(lhs == rhs); }

void Shape::pup(PUP::er& p) {
  size_t version = 1;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version == 0) {
    size_t old_l_max = 0;
    size_t old_m_max = 0;
    p | old_l_max;
    p | old_m_max;
  }
  p | center_;
  p | shape_f_of_t_name_;
  p | size_f_of_t_name_;
  p | transition_func_;

  if (version >= 1) {
    p | truncation_limit_;
  }

  // No need to pup these because they are uniquely determined by other
  // members
  if (p.isUnpacking()) {
    if (version == 0) {
      truncation_limit_ = 0.;
    }
    f_of_t_names_.clear();
    f_of_t_names_.insert(shape_f_of_t_name_);
    if (size_f_of_t_name_.has_value()) {
      f_of_t_names_.insert(size_f_of_t_name_.value());
    }
  }
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                                  \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>                \
  Shape::operator()(const std::array<DTYPE(data), 3>& source_coords,          \
                    double time, const FunctionsOfTimeMap& functions_of_time) \
      const;                                                                  \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 3>                \
  Shape::frame_velocity(const std::array<DTYPE(data), 3>& source_coords,      \
                        double time,                                          \
                        const FunctionsOfTimeMap& functions_of_time) const;   \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>  \
  Shape::jacobian(const std::array<DTYPE(data), 3>& source_coords,            \
                  double time, const FunctionsOfTimeMap& functions_of_time)   \
      const;                                                                  \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 3, Frame::NoFrame>  \
  Shape::inv_jacobian(const std::array<DTYPE(data), 3>& source_coords,        \
                      double time,                                            \
                      const FunctionsOfTimeMap& functions_of_time) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))
#undef DTYPE
#undef INSTANTIATE

}  // namespace domain::CoordinateMaps::TimeDependent
