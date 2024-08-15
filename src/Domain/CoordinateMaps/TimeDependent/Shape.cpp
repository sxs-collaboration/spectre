// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/TimeDependent/Shape.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <unordered_set>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/StdHelpers.hpp"

namespace domain::CoordinateMaps::TimeDependent {

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
    const T& distorted_radii, const T& one_over_radius,
    const T& transition_func_over_radius) const {
  const auto angular_gradient =
      extended_ylm_.gradient_from_coefs(extended_coefs);

  tnsr::i<DataVector, 3, Frame::Inertial> cartesian_gradient(
      extended_ylm_.physical_size());

  std::array<DataVector, 2> collocation_theta_phis{};
  collocation_theta_phis[0].set_data_ref(&get<2>(cartesian_gradient));
  collocation_theta_phis[1].set_data_ref(&get<1>(cartesian_gradient));
  collocation_theta_phis = extended_ylm_.theta_phi_points();

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

  // re-use allocation
  auto& target_gradient_x = get<2, 0>(*result);
  auto& target_gradient_y = get<2, 1>(*result);
  auto& target_gradient_z = get<2, 2>(*result);

  // interpolate the cartesian gradient to the thetas and phis of the
  // `source_coords`
  extended_ylm_.interpolate(make_not_null(&target_gradient_x),
                            get<0>(cartesian_gradient).data(),
                            interpolation_info);
  extended_ylm_.interpolate(make_not_null(&target_gradient_y),
                            get<1>(cartesian_gradient).data(),
                            interpolation_info);
  extended_ylm_.interpolate(make_not_null(&target_gradient_z),
                            get<2>(cartesian_gradient).data(),
                            interpolation_info);

  auto transition_func_over_square_radius =
      transition_func_over_radius * one_over_radius;
  auto transition_func_over_cube_radius =
      transition_func_over_square_radius * one_over_radius;
  auto transition_func_gradient_over_radius =
      transition_func_->gradient(centered_coords) * one_over_radius;

  auto& target_gradient_x_times_spatial_part = target_gradient_x;
  auto& target_gradient_y_times_spatial_part = target_gradient_y;
  auto& target_gradient_z_times_spatial_part = target_gradient_z;
  target_gradient_x_times_spatial_part *= transition_func_over_square_radius;
  target_gradient_y_times_spatial_part *= transition_func_over_square_radius;
  target_gradient_z_times_spatial_part *= transition_func_over_square_radius;

  const auto& [x_transition_gradient_over_radius,
               y_transition_gradient_over_radius,
               z_transition_gradient_over_radius] =
      transition_func_gradient_over_radius;
  const auto& [x_centered, y_centered, z_centered] = centered_coords;

  get<0, 0>(*result) =
      -x_centered * ((x_transition_gradient_over_radius -
                      x_centered * transition_func_over_cube_radius) *
                         distorted_radii +
                     target_gradient_x_times_spatial_part);
  get<0, 1>(*result) =
      -x_centered * ((y_transition_gradient_over_radius -
                      y_centered * transition_func_over_cube_radius) *
                         distorted_radii +
                     target_gradient_y_times_spatial_part);
  get<0, 2>(*result) =
      -x_centered * ((z_transition_gradient_over_radius -
                      z_centered * transition_func_over_cube_radius) *
                         distorted_radii +
                     target_gradient_z_times_spatial_part);
  get<1, 0>(*result) =
      -y_centered * ((x_transition_gradient_over_radius -
                      x_centered * transition_func_over_cube_radius) *
                         distorted_radii +
                     target_gradient_x_times_spatial_part);
  get<1, 1>(*result) =
      -y_centered * ((y_transition_gradient_over_radius -
                      y_centered * transition_func_over_cube_radius) *
                         distorted_radii +
                     target_gradient_y_times_spatial_part);
  get<1, 2>(*result) =
      -y_centered * ((z_transition_gradient_over_radius -
                      z_centered * transition_func_over_cube_radius) *
                         distorted_radii +
                     target_gradient_z_times_spatial_part);
  get<2, 0>(*result) =
      -z_centered * ((x_transition_gradient_over_radius -
                      x_centered * transition_func_over_cube_radius) *
                         distorted_radii +
                     target_gradient_x_times_spatial_part);
  get<2, 1>(*result) =
      -z_centered * ((y_transition_gradient_over_radius -
                      y_centered * transition_func_over_cube_radius) *
                         distorted_radii +
                     target_gradient_y_times_spatial_part);
  get<2, 2>(*result) =
      -z_centered * ((z_transition_gradient_over_radius -
                      z_centered * transition_func_over_cube_radius) *
                         distorted_radii +
                     target_gradient_z_times_spatial_part);

  get<0, 0>(*result) += 1. - distorted_radii * transition_func_over_radius;
  get<1, 1>(*result) += 1. - distorted_radii * transition_func_over_radius;
  get<2, 2>(*result) += 1. - distorted_radii * transition_func_over_radius;
}

Shape::Shape(
    const std::array<double, 3>& center, const size_t l_max, const size_t m_max,
    std::unique_ptr<ShapeMapTransitionFunctions::ShapeMapTransitionFunction>
        transition_func,
    std::string shape_function_of_time_name,
    std::optional<std::string> size_function_of_time_name)
    : shape_f_of_t_name_(std::move(shape_function_of_time_name)),
      size_f_of_t_name_(std::move(size_function_of_time_name)),
      center_(center),
      l_max_(l_max),
      m_max_(m_max),
      ylm_(l_max, m_max),
      extended_ylm_(l_max + 1, m_max + 1),
      transition_func_(std::move(transition_func)) {
  f_of_t_names_.insert(shape_f_of_t_name_);
  if (size_f_of_t_name_.has_value()) {
    f_of_t_names_.insert(size_f_of_t_name_.value());
  }
  ASSERT(l_max >= 2, "The shape map requires l_max >= 2 but l_max = " << l_max);
  ASSERT(m_max >= 2, "The shape map requires m_max >= 2 but m_max = " << m_max);
  ASSERT(l_max >= m_max, "The shape map requires l_max >= m_max but l_max = "
                             << l_max << ", m_max = " << m_max);
}

Shape& Shape::operator=(const Shape& rhs) {
  if (*this != rhs) {
    shape_f_of_t_name_ = rhs.shape_f_of_t_name_;
    size_f_of_t_name_ = rhs.size_f_of_t_name_;
    f_of_t_names_ = rhs.f_of_t_names_;
    center_ = rhs.center_;
    l_max_ = rhs.l_max_;
    m_max_ = rhs.m_max_;
    ylm_ = rhs.ylm_;
    extended_ylm_ = rhs.extended_ylm_;
    transition_func_ = rhs.transition_func_->get_clone();
  }
  return *this;
}

Shape::Shape(const Shape& rhs) { *this = rhs; }

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 3> Shape::operator()(
    const std::array<T, 3>& source_coords, const double time,
    const FunctionsOfTimeMap& functions_of_time) const {
  const auto centered_coords = center_coordinates(source_coords);
  auto theta_phis = cartesian_to_spherical(centered_coords);
  const auto interpolation_info = ylm_.set_up_interpolation_info(theta_phis);
  DataVector coefs = functions_of_time.at(shape_f_of_t_name_)->func(time)[0];
  check_size(make_not_null(&coefs), functions_of_time, time, false);
  check_coefficients(coefs);
  // re-use allocation
  auto& distorted_radii = get<0>(theta_phis);
  // evaluate the spherical harmonic expansion at the angles of `source_coords`
  ylm_.interpolate_from_coefs(make_not_null(&distorted_radii), coefs,
                              interpolation_info);

  // this should be taken care of by the control system but is very hard to
  // debug
#ifdef SPECTRE_DEBUG
  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType shift_radii =
      distorted_radii * transition_func_->operator()(centered_coords) *
      check_and_compute_one_over_radius(centered_coords);
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
             (1. - distorted_radii *
                       transition_func_->operator()(centered_coords) *
                       check_and_compute_one_over_radius(centered_coords));
}

std::optional<std::array<double, 3>> Shape::inverse(
    const std::array<double, 3>& target_coords, const double time,
    const FunctionsOfTimeMap& functions_of_time) const {
  const std::array<double, 3> centered_coords =
      center_coordinates(target_coords);
  const std::array<double, 2> theta_phis =
      cartesian_to_spherical(centered_coords);
  DataVector coefs = functions_of_time.at(shape_f_of_t_name_)->func(time)[0];
  check_size(make_not_null(&coefs), functions_of_time, time, false);
  check_coefficients(coefs);
  const double distorted_radii = ylm_.interpolate_from_coefs(coefs, theta_phis);
  const std::optional<double> original_radius_over_radius =
      transition_func_->original_radius_over_radius(centered_coords,
                                                    distorted_radii);
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
  const auto interpolation_info = ylm_.set_up_interpolation_info(theta_phis);
  DataVector coef_derivs =
      functions_of_time.at(shape_f_of_t_name_)->func_and_deriv(time)[1];
  check_size(make_not_null(&coef_derivs), functions_of_time, time, true);
  check_coefficients(coef_derivs);
  // re-use allocation
  auto& radii_velocities = get<0>(theta_phis);
  ylm_.interpolate_from_coefs(make_not_null(&radii_velocities), coef_derivs,
                              interpolation_info);
  return -centered_coords * radii_velocities *
         transition_func_->operator()(centered_coords) *
         check_and_compute_one_over_radius(centered_coords);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Shape::jacobian(
    const std::array<T, 3>& source_coords, const double time,
    const FunctionsOfTimeMap& functions_of_time) const {
  const auto centered_coords = center_coordinates(source_coords);

  // The distorted radii are calculated analogously to the call operator
  auto theta_phis = cartesian_to_spherical(centered_coords);

  // The Cartesian gradient cannot be represented exactly by `l_max_` and
  // `m_max_` which causes an aliasing error. We need an additional order to
  // represent it. This is in theory not needed for the distorted_radii
  // calculation but saves calculating the `interpolation_info` twice.
  const auto interpolation_info =
      extended_ylm_.set_up_interpolation_info(theta_phis);

  const DataVector coefs =
      functions_of_time.at(shape_f_of_t_name_)->func(time)[0];
  check_coefficients(coefs);
  DataVector extended_coefs(extended_ylm_.spectral_size(), 0.);

  // Copy over the coefficients. The additional coefficients of order `l_max_
  // +1` are zero and will only have an effect in the interpolation of the
  // cartesian gradient.
  ylm::SpherepackIterator extended_iter(l_max_ + 1, m_max_ + 1);
  ylm::SpherepackIterator iter(l_max_, m_max_);
  for (size_t l = 0; l <= l_max_; ++l) {
    const int m_max = std::min(l, m_max_);
    for (int m = -m_max; m <= m_max; ++m) {
      iter.set(l, m);
      extended_iter.set(l, m);
      extended_coefs[extended_iter()] = coefs[iter()];
    }
  }
  check_size(make_not_null(&extended_coefs), functions_of_time, time, false);

  // Re-use allocation
  auto& distorted_radii = get<0>(theta_phis);
  extended_ylm_.interpolate_from_coefs(make_not_null(&distorted_radii),
                                       extended_coefs, interpolation_info);

  using ReturnType = tt::remove_cvref_wrap_t<T>;
  const ReturnType one_over_radius =
      check_and_compute_one_over_radius(centered_coords);
  const ReturnType transition_func_over_radius =
      transition_func_->operator()(centered_coords) * one_over_radius;
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> result(
      get_size(centered_coords[0]));

  jacobian_helper(make_not_null(&result), interpolation_info, extended_coefs,
                  centered_coords, distorted_radii, one_over_radius,
                  transition_func_over_radius);
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
  Variables<
      tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                 ::Tags::TempScalar<2>, ::Tags::TempI<0, 3, Frame::Inertial>>>
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
  const auto interpolation_info =
      extended_ylm_.set_up_interpolation_info(theta_phis);

  const auto [coefs, coef_derivs] =
      functions_of_time.at(shape_f_of_t_name_)->func_and_deriv(time);
  DataVector extended_coefs_derivs(extended_ylm_.spectral_size(), 0.);
  DataVector extended_coefs(extended_ylm_.spectral_size(), 0.);

  // Copy over the coefficients. The additional coefficients of order `l_max_
  // +1` are zero and will only have an effect in the interpolation of the
  // cartesian gradient.
  ylm::SpherepackIterator extended_iter(l_max_ + 1, m_max_ + 1);
  ylm::SpherepackIterator iter(l_max_, m_max_);
  for (size_t l = 0; l <= l_max_; ++l) {
    const int m_max = static_cast<int>(std::min(l, m_max_));
    for (int m = -m_max; m <= m_max; ++m) {
      iter.set(l, m);
      extended_iter.set(l, m);
      extended_coefs[extended_iter()] = coefs[iter()];
      extended_coefs_derivs[extended_iter()] = coef_derivs[iter()];
    }
  }
  check_size(make_not_null(&extended_coefs), functions_of_time, time, false);
  check_size(make_not_null(&extended_coefs_derivs), functions_of_time, time,
             true);
  auto& distorted_radii = get(get<::Tags::TempScalar<0>>(temps));
  // evaluate the spherical harmonic expansion at the angles of
  // `source_coords`
  extended_ylm_.interpolate_from_coefs(make_not_null(&distorted_radii),
                                       extended_coefs, interpolation_info);

  auto& one_over_radius = get(get<::Tags::TempScalar<1>>(temps));
  one_over_radius = check_and_compute_one_over_radius(centered_coords);
  auto& transition_func_over_radius = get(get<::Tags::TempScalar<2>>(temps));
  transition_func_over_radius =
      transition_func_->operator()(centered_coords) * one_over_radius;
  *source_and_target_coords =
      center_ +
      centered_coords * (1. - distorted_radii * transition_func_over_radius);

  auto& radii_velocities = get<0, 1>(*jac);
  extended_ylm_.interpolate_from_coefs(make_not_null(&radii_velocities),
                                       extended_coefs_derivs,
                                       interpolation_info);
  *frame_vel =
      -centered_coords * radii_velocities * transition_func_over_radius;

  jacobian_helper<DataVector>(jac, interpolation_info, extended_coefs,
                              centered_coords, distorted_radii, one_over_radius,
                              transition_func_over_radius);
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> Shape::inv_jacobian(
    const std::array<T, 3>& source_coords, const double time,
    const FunctionsOfTimeMap& functions_of_time) const {
  return determinant_and_inverse(
             jacobian(source_coords, time, functions_of_time))
      .second;
}

void Shape::check_coefficients([[maybe_unused]] const DataVector& coefs) const {
#ifdef SPECTRE_DEBUG
  // The expected format of the coefficients passed from the control system can
  // be changed depending on what turns out to be most convenient for the
  // control system
  ASSERT(coefs.size() == ylm_.spectral_size(),
         "Spectral coefficients are expected to be in ylm::Spherepack format "
         "with size 2 * (l_max + 1) * (m_max + 1) = "
             << ylm_.spectral_size() << ", but have size " << coefs.size());
#endif  // SPECTRE_DEBUG
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

template <typename T>
T Shape::check_and_compute_one_over_radius(
    const std::array<T, 3>& centered_coords) const {
  const T radius = magnitude(centered_coords);
#ifdef SPECTRE_DEBUG
  for (size_t i = 0; i < get_size(radius); ++i) {
    if (get_element(radius, i) == 0.0) {
      ERROR(
          "The shape map does not support a (centered) point with radius zero. "
          "All centered coordinates: "
          << centered_coords);
    }
  }
#endif  // SPECTRE_DEBUG

  return 1.0 / radius;
}

bool operator==(const Shape& lhs, const Shape& rhs) {
  return lhs.shape_f_of_t_name_ == rhs.shape_f_of_t_name_ and
         lhs.size_f_of_t_name_ == rhs.size_f_of_t_name_ and
         lhs.center_ == rhs.center_ and lhs.l_max_ == rhs.l_max_ and
         lhs.m_max_ == rhs.m_max_ and
         (lhs.transition_func_ == nullptr) ==
             (rhs.transition_func_ == nullptr) and
         ((lhs.transition_func_ == nullptr and
           rhs.transition_func_ == nullptr) or
          *lhs.transition_func_ == *rhs.transition_func_);
}

bool operator!=(const Shape& lhs, const Shape& rhs) { return not(lhs == rhs); }

void Shape::pup(PUP::er& p) {
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | l_max_;
    p | m_max_;
    p | center_;
    p | shape_f_of_t_name_;
    p | size_f_of_t_name_;
    p | transition_func_;
  }

  // No need to pup these because they are uniquely determined by other members
  if (p.isUnpacking()) {
    ylm_ = ylm::Spherepack(l_max_, m_max_);
    extended_ylm_ = ylm::Spherepack(l_max_ + 1, m_max_ + 1);
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
