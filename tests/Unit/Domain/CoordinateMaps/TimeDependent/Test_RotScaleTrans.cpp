// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Identity.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/RotScaleTrans.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"
#include "Domain/CoordinateMaps/TimeDependent/RotationMatrixHelpers.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GetOutput.hpp"

template <size_t Dim>
void test_RotScaleTrans() {
  MAKE_GENERATOR(gen);
  // define vars for FunctionOfTime::PiecewisePolynomial f(t) = t**2.
  double initial_t = -1.0;
  double t = -0.9;
  const double dt = 0.3;
  const double final_time = 2.0;
  constexpr size_t deriv_order = 3;
  const double inner_radius = 1.0;
  const double outer_radius = 50.0;
  const double angle = 1.0;
  const double omega = -2.0;
  const double dtomega = 2.0;
  const double d2tomega = 0.0;

  using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>;
  using FoftPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;
  using QuatFoT =
      domain::FunctionsOfTime::QuaternionFunctionOfTime<deriv_order>;
  using BlockRegion =
      typename domain::CoordinateMaps::TimeDependent::RotScaleTrans<
          Dim>::BlockRegion;
  std::unordered_map<std::string, FoftPtr> f_of_t_list{};
  std::array<DataVector, deriv_order + 1> init_func{};
  const std::array<DataVector, deriv_order + 1> init_func_trans{
      {{Dim, 1.0}, {Dim, -2.0}, {Dim, 2.0}, {Dim, 0.0}}};
  const std::array<DataVector, deriv_order + 1> init_func_a{
      {{0.58}, {0.0}, {0.0}, {0.0}}};
  const std::array<DataVector, deriv_order + 1> init_func_b{
      {{0.96}, {0.0}, {0.0}, {0.0}}};
  const std::string rot_f_of_t_name{"rotation_angle"};
  // custom_approx will be redifined multiple times below.
  Approx custom_approx = Approx::custom().epsilon(1.0).scale(1.0);

  if constexpr (Dim == 2) {
    init_func = {{{angle}, {omega}, {dtomega}, {d2tomega}}};
    f_of_t_list[rot_f_of_t_name] =
        std::make_unique<Polynomial>(initial_t, init_func, final_time + dt);
    custom_approx = Approx::custom().epsilon(5.0e-13).scale(1.0);
  } else {
    // Axis of rotation nhat = (1.0, -1.0, 1.0) / sqrt(3.0)
    DataVector axis{{1.0, -1.0, 1.0}};
    axis /= sqrt(3.0);
    init_func = {axis * angle, axis * omega, axis * dtomega, axis * d2tomega};
    // initial quaternion is (cos(angle/2), nhat*sin(angle/2))
    const std::array<DataVector, 1> init_quat{
        DataVector{{cos(angle / 2.0), axis[0] * sin(angle / 2.0),
                    axis[1] * sin(angle / 2.0), axis[2] * sin(angle / 2.0)}}};
    f_of_t_list[rot_f_of_t_name] = std::make_unique<QuatFoT>(
        initial_t, init_quat, init_func, final_time + dt);
    custom_approx = Approx::custom().epsilon(5.0e-11).scale(1.0);
  }
  f_of_t_list["expansion_a"] =
      std::make_unique<Polynomial>(initial_t, init_func_a, final_time);
  f_of_t_list["expansion_b"] =
      std::make_unique<Polynomial>(initial_t, init_func_b, final_time);
  f_of_t_list["translation"] =
      std::make_unique<Polynomial>(initial_t, init_func_trans, final_time + dt);

  const FoftPtr& scale_a_of_t = f_of_t_list.at("expansion_a");
  const FoftPtr& scale_b_of_t = f_of_t_list.at("expansion_b");
  const std::pair<std::string, std::string> scale_pair{"expansion_a",
                                                       "expansion_b"};

  const auto map_serialize_and_deserialize =
      [&](const auto scaling, const auto rotation, const auto translation,
          const auto region) {
        domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim> map{
            scaling, rotation, translation, inner_radius, outer_radius, region};
        return serialize_and_deserialize(map);
      };
  // Rotation, Scaling, Translation Inner
  domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim>
      rot_scale_trans_map_inner = map_serialize_and_deserialize(
          scale_pair, "rotation_angle", "translation", BlockRegion::Inner);

  // Rotation, Scaling, Translation Transition
  domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim>
      rot_scale_trans_map_transition = map_serialize_and_deserialize(
          scale_pair, "rotation_angle", "translation", BlockRegion::Transition);

  // Rotation, Scaling, Translation Outer
  domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim>
      rot_scale_trans_map_outer = map_serialize_and_deserialize(
          scale_pair, "rotation_angle", "translation", BlockRegion::Outer);

  // Rotation, Scaling Inner
  domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim>
      rot_scale_map_inner = map_serialize_and_deserialize(
          scale_pair, "rotation_angle", std::nullopt, BlockRegion::Inner);

  // Rotation, Scaling Transition
  domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim>
      rot_scale_map_transition = map_serialize_and_deserialize(
          scale_pair, "rotation_angle", std::nullopt, BlockRegion::Transition);

  // Rotation, Scaling Outer
  domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim>
      rot_scale_map_outer = map_serialize_and_deserialize(
          scale_pair, "rotation_angle", std::nullopt, BlockRegion::Outer);

  // Rotation, Translation Inner
  domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim>
      rot_trans_map_inner = map_serialize_and_deserialize(
          std::nullopt, "rotation_angle", "translation", BlockRegion::Inner);

  // Rotation, Translation Transition
  domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim>
      rot_trans_map_transition =
          map_serialize_and_deserialize(std::nullopt, "rotation_angle",
                                        "translation", BlockRegion::Transition);

  // Rotation, Translation Outer
  domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim>
      rot_trans_map_outer = map_serialize_and_deserialize(
          std::nullopt, "rotation_angle", "translation", BlockRegion::Outer);

  // Scaling, Translation Inner
  domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim>
      scale_trans_map_inner = map_serialize_and_deserialize(
          scale_pair, std::nullopt, "translation", BlockRegion::Inner);

  // Scaling, Translation Transition
  domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim>
      scale_trans_map_transition = map_serialize_and_deserialize(
          scale_pair, std::nullopt, "translation", BlockRegion::Transition);

  // Scaling, Translation Outer
  domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim>
      scale_trans_map_outer = map_serialize_and_deserialize(
          scale_pair, std::nullopt, "translation", BlockRegion::Outer);

  // Rotation (doesn't matter which region you use)
  domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim> rot_map =
      map_serialize_and_deserialize(std::nullopt, "rotation_angle",
                                    std::nullopt, BlockRegion::Inner);

  // Scaling Inner
  domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim> scale_map_inner =
      map_serialize_and_deserialize(scale_pair, std::nullopt, std::nullopt,
                                    BlockRegion::Inner);

  // Scaling Transition
  domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim>
      scale_map_transition = map_serialize_and_deserialize(
          scale_pair, std::nullopt, std::nullopt, BlockRegion::Transition);

  // Scaling Outer
  domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim> scale_map_outer =
      map_serialize_and_deserialize(scale_pair, std::nullopt, std::nullopt,
                                    BlockRegion::Outer);

  // Translation Inner
  domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim> trans_map_inner =
      map_serialize_and_deserialize(std::nullopt, std::nullopt, "translation",
                                    BlockRegion::Inner);

  // Translation Transition
  domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim>
      trans_map_transition = map_serialize_and_deserialize(
          std::nullopt, std::nullopt, "translation", BlockRegion::Transition);

  // Translation Outer
  domain::CoordinateMaps::TimeDependent::RotScaleTrans<Dim> trans_map_outer =
      map_serialize_and_deserialize(std::nullopt, std::nullopt, "translation",
                                    BlockRegion::Outer);

  const double far_double_1 = Dim == 2 ? 30.0 : 25.0;
  const double far_double_2 = Dim == 2 ? 35.36 : 28.87;
  UniformCustomDistribution<double> dist_double{-10.0, 10.0};
  UniformCustomDistribution<double> far_dist_double{far_double_1, far_double_2};
  std::array<double, Dim> point_xi{};
  std::array<DataVector, Dim> point_xi_dv{};
  std::array<double, Dim> far_point_xi{};
  fill_with_random_values(make_not_null(&point_xi), make_not_null(&gen),
                          make_not_null(&dist_double));
  fill_with_random_values(make_not_null(&far_point_xi), make_not_null(&gen),
                          make_not_null(&far_dist_double));
  for (size_t i = 0; i < Dim; i++) {
    auto points = make_with_random_values<DataVector>(
        make_not_null(&gen), make_not_null(&dist_double), DataVector(5));
    gsl::at(point_xi_dv, i) = points;
  }

  while (t < final_time) {
    std::array<double, Dim> translation{};
    std::array<double, Dim> expected_rotation{};
    std::array<DataVector, Dim> expected_rotation_dv{};
    std::array<double, Dim> far_expected_rotation{};
    const double radius = magnitude(point_xi);
    const DataVector radius_dv = magnitude(point_xi_dv);
    const double far_radius = magnitude(far_point_xi);
    const Matrix rot_matrix =
        rotation_matrix<Dim>(t, *(f_of_t_list[rot_f_of_t_name]));
    const Matrix deriv_rot_matrix =
        rotation_matrix_deriv<Dim>(t, *(f_of_t_list[rot_f_of_t_name]));
    const double scale_a = scale_a_of_t->func_and_deriv(t)[0][0];
    const double scale_b = scale_b_of_t->func_and_deriv(t)[0][0];
    double radial_scaling_factor = 0.0;
    double radial_translation_factor = 0.0;
    custom_approx = Approx::custom().epsilon(1e-9).scale(1.0);
    for (size_t i = 0; i < Dim; i++) {
      gsl::at(translation, i) = square(t);
      for (size_t j = 0; j < Dim; j++) {
        gsl::at(expected_rotation, i) +=
            rot_matrix(i, j) * gsl::at(point_xi, j);
        gsl::at(far_expected_rotation, i) +=
            rot_matrix(i, j) * gsl::at(far_point_xi, j);
      }
    }
    for (size_t i = 0; i < Dim; i++) {
      gsl::at(expected_rotation_dv, i) = rot_matrix(i, 0) * point_xi_dv[0];
      for (size_t j = 1; j < Dim; j++) {
        gsl::at(expected_rotation_dv, i) +=
            rot_matrix(i, j) * gsl::at(point_xi_dv, j);
      }
    }
    // Operator
    CHECK_ITERABLE_APPROX(trans_map_inner(point_xi, t, f_of_t_list),
                          point_xi + translation);
    CHECK_ITERABLE_APPROX(scale_map_inner(point_xi, t, f_of_t_list),
                          point_xi * scale_a);
    CHECK_ITERABLE_APPROX(rot_map(point_xi, t, f_of_t_list), expected_rotation);
    CHECK_ITERABLE_APPROX(rot_scale_map_inner(point_xi, t, f_of_t_list),
                          expected_rotation * scale_a);
    CHECK_ITERABLE_APPROX(rot_trans_map_inner(point_xi, t, f_of_t_list),
                          expected_rotation + translation);
    CHECK_ITERABLE_APPROX(scale_trans_map_inner(point_xi, t, f_of_t_list),
                          point_xi * scale_a + translation);
    CHECK_ITERABLE_APPROX(rot_scale_trans_map_inner(point_xi, t, f_of_t_list),
                          expected_rotation * scale_a + translation);
    // Operator DataVector
    CHECK_ITERABLE_APPROX(
        rot_scale_trans_map_inner(point_xi_dv, t, f_of_t_list),
        expected_rotation_dv * scale_a + translation);

    // Testing points close to inner radius
    radial_scaling_factor =
        ((inner_radius - radius) * (scale_a - scale_b) * outer_radius) /
        ((outer_radius - inner_radius) * radius);
    radial_translation_factor =
        (inner_radius - radius) / (outer_radius - inner_radius);
    CHECK_ITERABLE_APPROX(scale_map_transition(point_xi, t, f_of_t_list),
                          point_xi * (radial_scaling_factor + scale_a));
    CHECK_ITERABLE_APPROX(
        trans_map_transition(point_xi, t, f_of_t_list),
        point_xi + translation + translation * radial_translation_factor);
    CHECK_ITERABLE_APPROX(
        rot_scale_map_transition(point_xi, t, f_of_t_list),
        expected_rotation * (radial_scaling_factor + scale_a));
    CHECK_ITERABLE_APPROX(rot_trans_map_transition(point_xi, t, f_of_t_list),
                          expected_rotation + translation +
                              translation * radial_translation_factor);
    CHECK_ITERABLE_APPROX(scale_trans_map_transition(point_xi, t, f_of_t_list),
                          point_xi * (radial_scaling_factor + scale_a) +
                              translation +
                              translation * radial_translation_factor);
    CHECK_ITERABLE_APPROX(
        rot_scale_trans_map_transition(point_xi, t, f_of_t_list),
        expected_rotation * (radial_scaling_factor + scale_a) + translation +
            translation * radial_translation_factor);
    // Testing far points close to outer radius.
    if (far_radius < outer_radius) {
      radial_scaling_factor =
          ((outer_radius - far_radius) * (scale_a - scale_b) * inner_radius) /
          ((outer_radius - inner_radius) * far_radius);
      radial_translation_factor =
          (outer_radius - far_radius) / (outer_radius - inner_radius);
      CHECK_ITERABLE_APPROX(scale_map_transition(far_point_xi, t, f_of_t_list),
                            far_point_xi * (radial_scaling_factor + scale_b));
      CHECK_ITERABLE_APPROX(
          trans_map_transition(far_point_xi, t, f_of_t_list),
          far_point_xi + translation * radial_translation_factor);
      CHECK_ITERABLE_APPROX(
          rot_scale_map_transition(far_point_xi, t, f_of_t_list),
          far_expected_rotation * (radial_scaling_factor + scale_b));
      CHECK_ITERABLE_APPROX(
          rot_trans_map_transition(far_point_xi, t, f_of_t_list),
          far_expected_rotation + translation * radial_translation_factor);
      CHECK_ITERABLE_APPROX(
          scale_trans_map_transition(far_point_xi, t, f_of_t_list),
          far_point_xi * (radial_scaling_factor + scale_b) +
              translation * radial_translation_factor);
      CHECK_ITERABLE_APPROX(
          rot_scale_trans_map_transition(far_point_xi, t, f_of_t_list),
          far_expected_rotation * (radial_scaling_factor + scale_b) +
              translation * radial_translation_factor);
    } else {
      CHECK_ITERABLE_APPROX(scale_map_outer(far_point_xi, t, f_of_t_list),
                            far_point_xi * scale_b);
      CHECK_ITERABLE_APPROX(trans_map_outer(far_point_xi, t, f_of_t_list),
                            far_point_xi);
      CHECK_ITERABLE_APPROX(rot_scale_map_outer(far_point_xi, t, f_of_t_list),
                            far_expected_rotation * scale_b);
      CHECK_ITERABLE_APPROX(rot_trans_map_outer(far_point_xi, t, f_of_t_list),
                            far_expected_rotation);
      CHECK_ITERABLE_APPROX(scale_trans_map_outer(far_point_xi, t, f_of_t_list),
                            far_point_xi * scale_b);
      CHECK_ITERABLE_APPROX(
          rot_scale_trans_map_outer(far_point_xi, t, f_of_t_list),
          expected_rotation * scale_b);
    }

    const auto check_inner_maps_inverse = [&](const auto& point_to_check) {
      test_inverse_map(rot_map, point_to_check, t, f_of_t_list);
      test_inverse_map(scale_map_inner, point_to_check, t, f_of_t_list);
      test_inverse_map(trans_map_inner, point_to_check, t, f_of_t_list);
      test_inverse_map(rot_scale_map_inner, point_to_check, t, f_of_t_list);
      test_inverse_map(rot_trans_map_inner, point_to_check, t, f_of_t_list);
      test_inverse_map(scale_trans_map_inner, point_to_check, t, f_of_t_list);
      test_inverse_map(rot_scale_trans_map_inner, point_to_check, t,
                       f_of_t_list);
    };
    const auto check_transition_maps_inverse = [&](const auto& point_to_check) {
      test_inverse_map(rot_map, point_to_check, t, f_of_t_list);
      test_inverse_map(scale_map_transition, point_to_check, t, f_of_t_list);
      test_inverse_map(trans_map_transition, point_to_check, t, f_of_t_list);
      test_inverse_map(rot_scale_map_transition, point_to_check, t,
                       f_of_t_list);
      test_inverse_map(rot_trans_map_transition, point_to_check, t,
                       f_of_t_list);
      test_inverse_map(scale_trans_map_transition, point_to_check, t,
                       f_of_t_list);
      test_inverse_map(rot_scale_trans_map_transition, point_to_check, t,
                       f_of_t_list);
    };
    const auto check_outer_maps_inverse = [&](const auto& point_to_check) {
      test_inverse_map(rot_map, point_to_check, t, f_of_t_list);
      test_inverse_map(scale_map_outer, point_to_check, t, f_of_t_list);
      test_inverse_map(trans_map_outer, point_to_check, t, f_of_t_list);
      test_inverse_map(rot_scale_map_outer, point_to_check, t, f_of_t_list);
      test_inverse_map(rot_trans_map_outer, point_to_check, t, f_of_t_list);
      test_inverse_map(scale_trans_map_outer, point_to_check, t, f_of_t_list);
      test_inverse_map(rot_scale_trans_map_outer, point_to_check, t,
                       f_of_t_list);
    };
    const auto check_all_maps_frame_velocity = [&](const auto& point_to_check) {
      test_frame_velocity(rot_map, point_to_check, t, f_of_t_list,
                          custom_approx);
      test_frame_velocity(scale_map_inner, point_to_check, t, f_of_t_list);
      test_frame_velocity(trans_map_inner, point_to_check, t, f_of_t_list);
      test_frame_velocity(rot_scale_map_inner, point_to_check, t, f_of_t_list,
                          custom_approx);
      test_frame_velocity(rot_trans_map_inner, point_to_check, t, f_of_t_list,
                          custom_approx);
      test_frame_velocity(scale_trans_map_inner, point_to_check, t,
                          f_of_t_list);
      test_frame_velocity(rot_scale_trans_map_inner, point_to_check, t,
                          f_of_t_list, custom_approx);
      test_frame_velocity(scale_map_transition, point_to_check, t, f_of_t_list);
      test_frame_velocity(trans_map_transition, point_to_check, t, f_of_t_list);
      test_frame_velocity(rot_scale_map_transition, point_to_check, t,
                          f_of_t_list, custom_approx);
      test_frame_velocity(rot_trans_map_transition, point_to_check, t,
                          f_of_t_list, custom_approx);
      test_frame_velocity(scale_trans_map_transition, point_to_check, t,
                          f_of_t_list);
      test_frame_velocity(rot_scale_trans_map_transition, point_to_check, t,
                          f_of_t_list, custom_approx);
      test_frame_velocity(scale_map_outer, point_to_check, t, f_of_t_list);
      test_frame_velocity(trans_map_outer, point_to_check, t, f_of_t_list);
      test_frame_velocity(rot_scale_map_outer, point_to_check, t, f_of_t_list,
                          custom_approx);
      test_frame_velocity(rot_trans_map_outer, point_to_check, t, f_of_t_list,
                          custom_approx);
      test_frame_velocity(scale_trans_map_outer, point_to_check, t,
                          f_of_t_list);
      test_frame_velocity(rot_scale_trans_map_outer, point_to_check, t,
                          f_of_t_list, custom_approx);
    };
    const auto check_all_maps_jacobian = [&](const auto& point_to_check) {
      test_jacobian(rot_map, point_to_check, t, f_of_t_list, custom_approx);
      test_inv_jacobian(rot_map, point_to_check, t, f_of_t_list);
      test_jacobian(scale_map_inner, point_to_check, t, f_of_t_list,
                    custom_approx);
      test_inv_jacobian(scale_map_inner, point_to_check, t, f_of_t_list);
      test_jacobian(scale_map_transition, point_to_check, t, f_of_t_list,
                    custom_approx);
      test_inv_jacobian(scale_map_transition, point_to_check, t, f_of_t_list);
      test_jacobian(scale_map_outer, point_to_check, t, f_of_t_list,
                    custom_approx);
      test_inv_jacobian(scale_map_outer, point_to_check, t, f_of_t_list);
      test_jacobian(trans_map_inner, point_to_check, t, f_of_t_list,
                    custom_approx);
      test_inv_jacobian(trans_map_inner, point_to_check, t, f_of_t_list);
      test_jacobian(trans_map_transition, point_to_check, t, f_of_t_list,
                    custom_approx);
      test_inv_jacobian(trans_map_transition, point_to_check, t, f_of_t_list);
      test_jacobian(trans_map_outer, point_to_check, t, f_of_t_list,
                    custom_approx);
      test_inv_jacobian(trans_map_outer, point_to_check, t, f_of_t_list);
      test_jacobian(rot_scale_map_inner, point_to_check, t, f_of_t_list,
                    custom_approx);
      test_inv_jacobian(rot_scale_map_inner, point_to_check, t, f_of_t_list);
      test_jacobian(rot_scale_map_transition, point_to_check, t, f_of_t_list,
                    custom_approx);
      test_inv_jacobian(rot_scale_map_transition, point_to_check, t,
                        f_of_t_list);
      test_jacobian(rot_scale_map_outer, point_to_check, t, f_of_t_list,
                    custom_approx);
      test_inv_jacobian(rot_scale_map_outer, point_to_check, t, f_of_t_list);
      test_jacobian(rot_trans_map_inner, point_to_check, t, f_of_t_list,
                    custom_approx);
      test_inv_jacobian(rot_trans_map_inner, point_to_check, t, f_of_t_list);
      test_jacobian(rot_trans_map_transition, point_to_check, t, f_of_t_list,
                    custom_approx);
      test_inv_jacobian(rot_trans_map_transition, point_to_check, t,
                        f_of_t_list);
      test_jacobian(rot_trans_map_outer, point_to_check, t, f_of_t_list,
                    custom_approx);
      test_inv_jacobian(rot_trans_map_outer, point_to_check, t, f_of_t_list);
      test_jacobian(scale_trans_map_inner, point_to_check, t, f_of_t_list,
                    custom_approx);
      test_inv_jacobian(scale_trans_map_inner, point_to_check, t, f_of_t_list);
      test_jacobian(scale_trans_map_transition, point_to_check, t, f_of_t_list,
                    custom_approx);
      test_inv_jacobian(scale_trans_map_transition, point_to_check, t,
                        f_of_t_list);
      test_jacobian(scale_trans_map_outer, point_to_check, t, f_of_t_list,
                    custom_approx);
      test_inv_jacobian(scale_trans_map_outer, point_to_check, t, f_of_t_list);
      test_jacobian(rot_scale_trans_map_inner, point_to_check, t, f_of_t_list,
                    custom_approx);
      test_inv_jacobian(rot_scale_trans_map_inner, point_to_check, t,
                        f_of_t_list);
      test_jacobian(rot_scale_trans_map_transition, point_to_check, t,
                    f_of_t_list, custom_approx);
      test_inv_jacobian(rot_scale_trans_map_transition, point_to_check, t,
                        f_of_t_list);
      test_jacobian(rot_scale_trans_map_outer, point_to_check, t, f_of_t_list,
                    custom_approx);
      test_inv_jacobian(rot_scale_trans_map_outer, point_to_check, t,
                        f_of_t_list);
    };

    if (radius <= inner_radius) {
      check_inner_maps_inverse(point_xi);
    } else {
      check_transition_maps_inverse(point_xi);
    }
    check_all_maps_frame_velocity(point_xi);
    check_all_maps_jacobian(point_xi);
    check_all_maps_jacobian(point_xi_dv);

    if (far_radius <= outer_radius) {
      check_transition_maps_inverse(far_point_xi);
    } else {
      check_outer_maps_inverse(far_point_xi);
    }
    check_all_maps_frame_velocity(far_point_xi);
    check_all_maps_jacobian(far_point_xi);

    t += dt;
  }

  // test serialized/deserialized map and names
  const auto rot_map_deserialized = serialize_and_deserialize(rot_map);
  const auto scale_map_inner_deserialized =
      serialize_and_deserialize(scale_map_inner);
  const auto scale_map_transition_deserialized =
      serialize_and_deserialize(scale_map_transition);
  const auto scale_map_outer_deserialized =
      serialize_and_deserialize(scale_map_outer);
  const auto trans_map_inner_deserialized =
      serialize_and_deserialize(trans_map_inner);
  const auto trans_map_transition_deserialized =
      serialize_and_deserialize(trans_map_transition);
  const auto trans_map_outer_deserialized =
      serialize_and_deserialize(trans_map_outer);
  const auto rot_scale_map_inner_deserialized =
      serialize_and_deserialize(rot_scale_map_inner);
  const auto rot_scale_map_transition_deserialized =
      serialize_and_deserialize(rot_scale_map_transition);
  const auto rot_scale_map_outer_deserialized =
      serialize_and_deserialize(rot_scale_map_outer);
  const auto rot_trans_map_inner_deserialized =
      serialize_and_deserialize(rot_trans_map_inner);
  const auto rot_trans_map_transition_deserialized =
      serialize_and_deserialize(rot_trans_map_transition);
  const auto rot_trans_map_outer_deserialized =
      serialize_and_deserialize(rot_trans_map_outer);
  const auto scale_trans_map_inner_deserialized =
      serialize_and_deserialize(scale_trans_map_inner);
  const auto scale_trans_map_transition_deserialized =
      serialize_and_deserialize(scale_trans_map_transition);
  const auto scale_trans_map_outer_deserialized =
      serialize_and_deserialize(scale_trans_map_outer);
  const auto rot_scale_trans_map_inner_deserialized =
      serialize_and_deserialize(rot_scale_trans_map_inner);
  const auto rot_scale_trans_map_transition_deserialized =
      serialize_and_deserialize(rot_scale_trans_map_transition);
  const auto rot_scale_trans_map_outer_deserialized =
      serialize_and_deserialize(rot_scale_trans_map_outer);

  CHECK(rot_map == rot_map_deserialized);
  CHECK(scale_map_inner == scale_map_inner_deserialized);
  CHECK(scale_map_transition == scale_map_transition_deserialized);
  CHECK(scale_map_outer == scale_map_outer_deserialized);
  CHECK(trans_map_inner == trans_map_inner_deserialized);
  CHECK(trans_map_transition == trans_map_transition_deserialized);
  CHECK(trans_map_outer == trans_map_outer_deserialized);
  CHECK(rot_scale_map_inner == rot_scale_map_inner_deserialized);
  CHECK(rot_scale_map_transition == rot_scale_map_transition_deserialized);
  CHECK(rot_scale_map_outer == rot_scale_map_outer_deserialized);
  CHECK(rot_trans_map_inner == rot_trans_map_inner_deserialized);
  CHECK(rot_trans_map_transition == rot_trans_map_transition_deserialized);
  CHECK(rot_trans_map_outer == rot_trans_map_outer_deserialized);
  CHECK(scale_trans_map_inner == scale_trans_map_inner_deserialized);
  CHECK(scale_trans_map_transition == scale_trans_map_transition_deserialized);
  CHECK(scale_trans_map_outer == scale_trans_map_outer_deserialized);
  CHECK(rot_scale_trans_map_inner == rot_scale_trans_map_inner_deserialized);
  CHECK(rot_scale_trans_map_transition ==
        rot_scale_trans_map_transition_deserialized);
  CHECK(rot_scale_trans_map_outer == rot_scale_trans_map_outer_deserialized);

  const auto check_names1 = [](const auto& names) {
    CHECK(names.size() == 1);
    CHECK((names.count("rotation_angle") == 1 or
           names.count("translation") == 1));
  };
  const auto check_names2 = [](const auto& names) {
    CHECK(names.size() == 2);
    CHECK((
        (names.count("rotation_angle") == 1 and
         names.count("translation") == 1) or
        (names.count("expansion_a") == 1 and names.count("expansion_b") == 1)));
  };
  const auto check_names3 = [](const auto& names) {
    CHECK(names.size() == 3);
    CHECK((
        (names.count("rotation_angle") == 1 and
         names.count("expansion_a") == 1 and names.count("expansion_b") == 1) or
        (names.count("translation") == 1 and names.count("expansion_a") == 1 and
         names.count("expansion_b") == 1)));
  };
  const auto check_names4 = [](const auto& names) {
    CHECK(names.size() == 4);
    CHECK((names.count("rotation_angle") == 1 and
           names.count("expansion_a") == 1 and
           names.count("expansion_b") == 1 and
           names.count("translation") == 1));
  };
  check_names1(rot_map.function_of_time_names());
  check_names1(trans_map_inner.function_of_time_names());
  check_names1(trans_map_transition.function_of_time_names());
  check_names1(trans_map_outer.function_of_time_names());
  check_names2(scale_map_inner.function_of_time_names());
  check_names2(scale_map_transition.function_of_time_names());
  check_names2(scale_map_outer.function_of_time_names());
  check_names2(rot_trans_map_inner.function_of_time_names());
  check_names2(rot_trans_map_transition.function_of_time_names());
  check_names2(rot_trans_map_outer.function_of_time_names());
  check_names3(rot_scale_map_inner.function_of_time_names());
  check_names3(rot_scale_map_transition.function_of_time_names());
  check_names3(rot_scale_map_outer.function_of_time_names());
  check_names3(scale_trans_map_inner.function_of_time_names());
  check_names3(scale_trans_map_transition.function_of_time_names());
  check_names3(scale_trans_map_outer.function_of_time_names());
  check_names4(rot_scale_trans_map_inner.function_of_time_names());
  check_names4(rot_scale_trans_map_transition.function_of_time_names());
  check_names4(rot_scale_trans_map_outer.function_of_time_names());
}
namespace domain {
// [[Timeout, 45]]
SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.TimeDependent.RotScaleTrans",
                  "[Domain][Unit]") {
  test_RotScaleTrans<2>();
  test_RotScaleTrans<3>();
}
}  // namespace domain
