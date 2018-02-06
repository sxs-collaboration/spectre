// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/BulgedCube.hpp"

#include <pup.h>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/RootFinding/RootFinder.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace {
template <typename DType>
class RootFunction {
 public:
  RootFunction(const double radius, const double sphericity,
               const DType& physical_r_squared, const DType& x_sq,
               const DType& y_sq, const DType& z_sq) noexcept
      : radius_(radius),
        sphericity_(sphericity),
        physical_r_squared_(physical_r_squared),
        x_sq_(x_sq),
        y_sq_(y_sq),
        z_sq_(z_sq) {}

  double operator()(const double rho, const size_t i = 0) const noexcept {
    if (not(equal_within_roundoff(get_element(physical_r_squared_, i), 0.0))) {
      const double x_sq_over_r_sq =
          get_element(x_sq_, i) / get_element(physical_r_squared_, i);
      const double y_sq_over_r_sq =
          get_element(y_sq_, i) / get_element(physical_r_squared_, i);
      const double z_sq_over_r_sq =
          get_element(z_sq_, i) / get_element(physical_r_squared_, i);
      return sqrt(get_element(physical_r_squared_, i)) -
             radius_ * rho *
                 (1.0 / sqrt(3.0) +
                  sphericity_ *
                      (1.0 / sqrt(1.0 + square(rho) *
                                            (x_sq_over_r_sq + y_sq_over_r_sq)) +
                       1.0 / sqrt(1.0 + square(rho) *
                                            (x_sq_over_r_sq + z_sq_over_r_sq)) +
                       1.0 / sqrt(1.0 + square(rho) *
                                            (y_sq_over_r_sq + z_sq_over_r_sq)) -
                       1.0 / sqrt(2.0 + square(rho) * x_sq_over_r_sq) -
                       1.0 / sqrt(2.0 + square(rho) * y_sq_over_r_sq) -
                       1.0 / sqrt(2.0 + square(rho) * z_sq_over_r_sq)));
    } else {
      return 0.0;
    }
  }
  const DType& get_x_sq() noexcept { return x_sq_; }
  const DType& get_r_sq() noexcept { return physical_r_squared_; }

 private:
  const double radius_;
  const double sphericity_;
  const DType& physical_r_squared_;
  const DType& x_sq_;
  const DType& y_sq_;
  const DType& z_sq_;
};

template <typename DType>
DType scaling_factor(RootFunction<DType>&& rootfunction) noexcept {
  const DType& x_sq = rootfunction.get_x_sq();
  const DType& physical_r_squared = rootfunction.get_r_sq();
  DType rho = find_root_of_function(
      rootfunction, make_with_value<DType>(x_sq, 0.0),
      make_with_value<DType>(x_sq, sqrt(3.0)), 1.0e-16, 1.0e-16);
  for (size_t i = 0; i < get_size(rho); i++) {
    if (not(equal_within_roundoff(get_element(physical_r_squared, i), 0.0))) {
      get_element(rho, i) /= sqrt(get_element(physical_r_squared, i));
    } else {
      ASSERT(equal_within_roundoff(get_element(rho, i), 0.0),
             "r == 0 must imply rho == 0. This has failed.");
    }
  }
  return rho;
}
}  // namespace

namespace CoordinateMaps {
BulgedCube::BulgedCube(const double radius, const double sphericity,
                       const bool use_equiangular_map) noexcept
    : radius_(radius),
      sphericity_(sphericity),
      use_equiangular_map_(use_equiangular_map) {
  ASSERT(radius > 0.0, "The radius of the cube must be greater than zero");
  ASSERT(sphericity >= 0.0 and sphericity < 1.0,
         "The sphericity must be strictly less than one.");
}

template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 3> BulgedCube::
operator()(const std::array<T, 3>& source_coords) const noexcept {
  using ReturnType = std::decay_t<tt::remove_reference_wrapper_t<T>>;
  const auto physical_coordinates = [this](
      const ReturnType& cap_xi, const ReturnType& cap_eta,
      const ReturnType& cap_zeta) noexcept {
    const auto one_over_rho_xi = 1.0 / sqrt(2.0 + square(cap_xi));
    const auto one_over_rho_eta = 1.0 / sqrt(2.0 + square(cap_eta));
    const auto one_over_rho_zeta = 1.0 / sqrt(2.0 + square(cap_zeta));
    const auto one_over_rho_xi_eta =
        1.0 / sqrt(1.0 + square(cap_xi) + square(cap_eta));
    const auto one_over_rho_xi_zeta =
        1.0 / sqrt(1.0 + square(cap_xi) + square(cap_zeta));
    const auto one_over_rho_eta_zeta =
        1.0 / sqrt(1.0 + square(cap_eta) + square(cap_zeta));
    const ReturnType radial_scaling_factor =
        radius_ * (1.0 / sqrt(3.0) +
                   sphericity_ * (one_over_rho_eta_zeta + one_over_rho_xi_zeta +
                                  one_over_rho_xi_eta - one_over_rho_xi -
                                  one_over_rho_eta - one_over_rho_zeta));

    ReturnType physical_x = radial_scaling_factor * cap_xi;
    ReturnType physical_y = radial_scaling_factor * cap_eta;
    ReturnType physical_z = radial_scaling_factor * cap_zeta;
    return std::array<ReturnType, 3>{
        {std::move(physical_x), std::move(physical_y), std::move(physical_z)}};
  };

  if (use_equiangular_map_) {
    return physical_coordinates(
        tan(M_PI_4 * dereference_wrapper(source_coords[0])),
        tan(M_PI_4 * dereference_wrapper(source_coords[1])),
        tan(M_PI_4 * dereference_wrapper(source_coords[2])));
  }
  return physical_coordinates(dereference_wrapper(source_coords[0]),
                              dereference_wrapper(source_coords[1]),
                              dereference_wrapper(source_coords[2]));
}

template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 3>
BulgedCube::inverse(const std::array<T, 3>& target_coords) const noexcept {
  using ReturnType = std::decay_t<tt::remove_reference_wrapper_t<T>>;
  const ReturnType& physical_x = target_coords[0];
  const ReturnType& physical_y = target_coords[1];
  const ReturnType& physical_z = target_coords[2];
  const ReturnType x_sq = square(physical_x);
  const ReturnType y_sq = square(physical_y);
  const ReturnType z_sq = square(physical_z);
  const ReturnType physical_r_squared = x_sq + y_sq + z_sq;
  const auto scaling_factor =
      ::scaling_factor<ReturnType>(RootFunction<ReturnType>{
          radius_, sphericity_, physical_r_squared, x_sq, y_sq, z_sq});
  if (use_equiangular_map_) {
    return {{2.0 * M_2_PI * atan(physical_x * scaling_factor),
             2.0 * M_2_PI * atan(physical_y * scaling_factor),
             2.0 * M_2_PI * atan(physical_z * scaling_factor)}};
  }
  return {{physical_x * scaling_factor, physical_y * scaling_factor,
           physical_z * scaling_factor}};
}

template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, 3>
BulgedCube::xi_derivative(const std::array<T, 3>& source_coords) const
    noexcept {
  using ReturnType = std::decay_t<tt::remove_reference_wrapper_t<T>>;
  const auto derivative_lambda = [this](const ReturnType& cap_xi,
                                        const ReturnType& cap_eta,
                                        const ReturnType& cap_zeta,
                                        const auto& cap_xi_deriv) {
    const ReturnType one_over_rho_xi_cubed =
        pow<3>(1.0 / sqrt(2.0 + square(cap_xi)));
    const ReturnType one_over_rho_eta = 1.0 / sqrt(2.0 + square(cap_eta));
    const ReturnType one_over_rho_zeta = 1.0 / sqrt(2.0 + square(cap_zeta));
    const ReturnType one_over_rho_xi_eta_cubed =
        pow<3>(1.0 / sqrt(1.0 + square(cap_xi) + square(cap_eta)));
    const ReturnType one_over_rho_xi_zeta_cubed =
        pow<3>(1.0 / sqrt(1.0 + square(cap_xi) + square(cap_zeta)));
    const ReturnType one_over_rho_eta_zeta =
        1.0 / sqrt(1.0 + square(cap_eta) + square(cap_zeta));
    const ReturnType common_factor =
        sphericity_ * radius_ * cap_xi * cap_xi_deriv *
        (one_over_rho_xi_cubed - one_over_rho_xi_eta_cubed -
         one_over_rho_xi_zeta_cubed);

    const ReturnType physical_x =
        radius_ * cap_xi_deriv *
        (1.0 / sqrt(3.0) +
         sphericity_ *
             (((1.0 + square(cap_eta)) * one_over_rho_xi_eta_cubed +
               (1.0 + square(cap_zeta)) * one_over_rho_xi_zeta_cubed -
               2.0 * one_over_rho_xi_cubed) +
              one_over_rho_eta_zeta - one_over_rho_eta - one_over_rho_zeta));
    const ReturnType physical_y = cap_eta * common_factor;
    const ReturnType physical_z = cap_zeta * common_factor;

    return std::array<ReturnType, 3>{{physical_x, physical_y, physical_z}};
  };
  if (use_equiangular_map_) {
    return derivative_lambda(
        tan(M_PI_4 * dereference_wrapper(source_coords[0])),
        tan(M_PI_4 * dereference_wrapper(source_coords[1])),
        tan(M_PI_4 * dereference_wrapper(source_coords[2])),
        ReturnType{M_PI_4 *
                   (1.0 + square(tan(M_PI_4 *
                                     dereference_wrapper(source_coords[0]))))});
  }
  return derivative_lambda(dereference_wrapper(source_coords[0]),
                           dereference_wrapper(source_coords[1]),
                           dereference_wrapper(source_coords[2]), 1.0);
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
BulgedCube::jacobian(const std::array<T, 3>& source_coords) const noexcept {
  const auto dX_dxi = xi_derivative(source_coords);
  const auto dX_deta = xi_derivative(
      std::array<std::reference_wrapper<
                     const std::decay_t<tt::remove_reference_wrapper_t<T>>>,
                 3>{{std::cref(dereference_wrapper(source_coords[1])),
                     std::cref(dereference_wrapper(source_coords[0])),
                     std::cref(dereference_wrapper(source_coords[2]))}});
  const auto dX_dzeta = xi_derivative(
      std::array<std::reference_wrapper<
                     const std::decay_t<tt::remove_reference_wrapper_t<T>>>,
                 3>{{std::cref(dereference_wrapper(source_coords[2])),
                     std::cref(dereference_wrapper(source_coords[1])),
                     std::cref(dereference_wrapper(source_coords[0]))}});
  auto jacobian_matrix = make_with_value<
      Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
             tmpl::integral_list<std::int32_t, 2, 1>,
             index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                        SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>>(
      dereference_wrapper(source_coords[0]), 0.0);

  get<0, 0>(jacobian_matrix) = dX_dxi[0];
  get<0, 1>(jacobian_matrix) = dX_deta[1];
  get<0, 2>(jacobian_matrix) = dX_dzeta[2];
  get<1, 0>(jacobian_matrix) = dX_dxi[1];
  get<1, 1>(jacobian_matrix) = dX_deta[0];
  get<1, 2>(jacobian_matrix) = dX_dzeta[1];
  get<2, 0>(jacobian_matrix) = dX_dxi[2];
  get<2, 1>(jacobian_matrix) = dX_deta[2];
  get<2, 2>(jacobian_matrix) = dX_dzeta[0];
  return jacobian_matrix;
}

void BulgedCube::pup(PUP::er& p) noexcept {
  p | radius_;
  p | sphericity_;
  p | use_equiangular_map_;
}

template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>
BulgedCube::inv_jacobian(const std::array<T, 3>& source_coords) const noexcept {
  const auto jac = jacobian(source_coords);
  return determinant_and_inverse(jac).second;
}

bool operator==(const BulgedCube& lhs, const BulgedCube& rhs) noexcept {
  return lhs.radius_ == rhs.radius_ and lhs.sphericity_ == rhs.sphericity_ and
         lhs.use_equiangular_map_ == rhs.use_equiangular_map_;
}

bool operator!=(const BulgedCube& lhs, const BulgedCube& rhs) noexcept {
  return not(lhs == rhs);
}

// Explicit instantiations
/// \cond
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                \
  template std::array<DTYPE(data), 3> BulgedCube::operator()(               \
      const std::array<std::reference_wrapper<const DTYPE(data)>, 3>&       \
          source_coords) const noexcept;                                    \
  template std::array<DTYPE(data), 3> BulgedCube::operator()(               \
      const std::array<DTYPE(data), 3>& source_coords) const noexcept;      \
  template std::array<DTYPE(data), 3> BulgedCube::inverse(                  \
      const std::array<std::reference_wrapper<const DTYPE(data)>, 3>&       \
          target_coords) const noexcept;                                    \
  template std::array<DTYPE(data), 3> BulgedCube::inverse(                  \
      const std::array<DTYPE(data), 3>& target_coords) const noexcept;      \
  template Tensor<DTYPE(data), tmpl::integral_list<std::int32_t, 2, 1>,     \
                  index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,     \
                             SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>    \
  BulgedCube::jacobian(                                                     \
      const std::array<std::reference_wrapper<const DTYPE(data)>, 3>&       \
          source_coords) const noexcept;                                    \
  template Tensor<DTYPE(data), tmpl::integral_list<std::int32_t, 2, 1>,     \
                  index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,     \
                             SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>    \
  BulgedCube::jacobian(const std::array<DTYPE(data), 3>& source_coords)     \
      const noexcept;                                                       \
  template Tensor<DTYPE(data), tmpl::integral_list<std::int32_t, 2, 1>,     \
                  index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,     \
                             SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>    \
  BulgedCube::inv_jacobian(                                                 \
      const std::array<std::reference_wrapper<const DTYPE(data)>, 3>&       \
          source_coords) const noexcept;                                    \
  template Tensor<DTYPE(data), tmpl::integral_list<std::int32_t, 2, 1>,     \
                  index_list<SpatialIndex<3, UpLo::Up, Frame::NoFrame>,     \
                             SpatialIndex<3, UpLo::Lo, Frame::NoFrame>>>    \
  BulgedCube::inv_jacobian(const std::array<DTYPE(data), 3>& source_coords) \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
/// \endcond
}  // namespace CoordinateMaps
