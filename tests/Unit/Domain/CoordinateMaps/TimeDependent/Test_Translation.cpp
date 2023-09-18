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
#include "DataStructures/Tensor/Identity.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/RegisterDerivedWithCharm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

namespace domain {

namespace {
template <size_t Dim>
void test_translation() {
  MAKE_GENERATOR(gen);
  // define vars for FunctionOfTime::PiecewisePolynomial f(t) = t**2.
  double t = -1.0;
  const double dt = 0.6;
  const double final_time = 4.0;
  constexpr size_t deriv_order = 3;
  const double amplitude = 1.0;
  const double width = 100.0;
  const double inner_radius = 1.0;
  const double outer_radius = 40.0;
  std::array<double, 1> gauss_center{0.};
  const MathFunctions::Gaussian<1, Frame::Inertial> gaussian{amplitude, width,
                                                             gauss_center};

  const std::array<DataVector, deriv_order + 1> init_func{
      {{Dim, 1.0}, {Dim, -2.0}, {Dim, 2.0}, {Dim, 0.0}}};

  using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>;
  using FoftPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;
  std::unordered_map<std::string, FoftPtr> f_of_t_list{};
  f_of_t_list["translation"] =
      std::make_unique<Polynomial>(t, init_func, final_time + dt);

  const FoftPtr& f_of_t = f_of_t_list.at("translation");

  UniformCustomDistribution<double> dist_double{-5.0, 5.0};
  UniformCustomDistribution<double> far_dist_double{40.0, 50.0};
  std::array<double, Dim> point_xi{};
  std::array<DataVector, Dim> point_xi_dv{};
  std::array<double, Dim> center{};
  std::array<double, Dim> far_point_xi{};
  fill_with_random_values(make_not_null(&point_xi), make_not_null(&gen),
                          make_not_null(&dist_double));
  fill_with_random_values(make_not_null(&center), make_not_null(&gen),
                          make_not_null(&dist_double));
  fill_with_random_values(make_not_null(&far_point_xi), make_not_null(&gen),
                          make_not_null(&far_dist_double));
  for (size_t i = 0; i < Dim; i++) {
    auto points = make_with_random_values<DataVector>(
        make_not_null(&gen), make_not_null(&dist_double), DataVector(5));
    gsl::at(point_xi_dv, i) = points;
  }
  const CoordinateMaps::TimeDependent::Translation<Dim> translation_map{
      "translation"};
  const CoordinateMaps::TimeDependent::Translation<Dim>
      piecewise_translation_map{"translation", inner_radius, outer_radius};
  const CoordinateMaps::TimeDependent::Translation<Dim> radial_translation_map{
      "translation",
      std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(
          amplitude, width, gauss_center),
      center};
  const auto check_names = [](const auto& names) {
    CHECK(names.size() == 1);
    CHECK(names.count("translation") == 1);
  };
  check_names(translation_map.function_of_time_names());
  check_names(radial_translation_map.function_of_time_names());
  // test serialized/deserialized map
  MathFunctions::register_derived_with_charm();
  const auto translation_map_deserialized =
      serialize_and_deserialize(translation_map);
  const auto radial_translation_map_deserialized =
      serialize_and_deserialize(radial_translation_map);
  const auto piecewise_translation_map_deserialized =
      serialize_and_deserialize(piecewise_translation_map);
  check_names(translation_map_deserialized.function_of_time_names());
  check_names(radial_translation_map_deserialized.function_of_time_names());

  while (t < final_time) {
    std::array<double, Dim> translation{};
    std::array<double, Dim> distance_to_center{};
    double radius = 0;
    for (size_t i = 0; i < Dim; i++) {
      gsl::at(translation, i) = square(t);
      distance_to_center[i] = point_xi[i] - center[i];
      radius += square(distance_to_center[i]);
    }
    radius = sqrt(radius);
    const double piecewise_radius = magnitude(point_xi);
    const DataVector piecewise_radius_dv = magnitude(point_xi_dv);
    const double far_piecewise_radius = magnitude(far_point_xi);
    const double radial_falloff_factor =
        (outer_radius - piecewise_radius) / (outer_radius - inner_radius);
    std::array<double, Dim> frame_vel{};
    std::array<double, Dim> radial_frame_vel{};
    std::array<double, Dim> inspiral_frame_vel{};
    std::array<double, Dim> far_inspiral_frame_vel{0.};
    for (size_t i = 0; i < Dim; i++) {
      gsl::at(frame_vel, i) = f_of_t->func_and_deriv(t)[1][i];
      gsl::at(radial_frame_vel, i) =
          f_of_t->func_and_deriv(t)[1][i] * gaussian(radius);
      if (piecewise_radius <= inner_radius) {
        gsl::at(inspiral_frame_vel, i) = f_of_t->func_and_deriv(t)[1][i];
      } else {
        gsl::at(inspiral_frame_vel, i) =
            f_of_t->func_and_deriv(t)[1][i] * radial_falloff_factor;
      }
    }
    Approx custom_approx = Approx::custom().epsilon(1.e-9);
    CHECK_ITERABLE_APPROX(translation_map(point_xi, t, f_of_t_list),
                          point_xi + translation);
    if (piecewise_radius <= inner_radius) {
      CHECK_ITERABLE_APPROX(piecewise_translation_map(point_xi, t, f_of_t_list),
                            point_xi + translation);
      CHECK_ITERABLE_APPROX(piecewise_translation_map
                                .inverse(point_xi + translation, t, f_of_t_list)
                                .value(),
                            point_xi);
      CHECK_ITERABLE_APPROX(
          piecewise_translation_map_deserialized(point_xi, t, f_of_t_list),
          point_xi + translation);
      CHECK_ITERABLE_APPROX(piecewise_translation_map_deserialized
                                .inverse(point_xi + translation, t, f_of_t_list)
                                .value(),
                            point_xi);
    } else if (piecewise_radius > inner_radius and
               piecewise_radius < outer_radius) {
      CHECK_ITERABLE_APPROX(piecewise_translation_map(point_xi, t, f_of_t_list),
                            point_xi + translation * radial_falloff_factor);
      CHECK_ITERABLE_CUSTOM_APPROX(
          piecewise_translation_map
              .inverse(point_xi + translation * radial_falloff_factor, t,
                       f_of_t_list)
              .value(),
          point_xi, custom_approx);
      CHECK_ITERABLE_APPROX(
          piecewise_translation_map_deserialized(point_xi, t, f_of_t_list),
          point_xi + translation * radial_falloff_factor);
      CHECK_ITERABLE_CUSTOM_APPROX(
          piecewise_translation_map_deserialized
              .inverse(point_xi + translation * radial_falloff_factor, t,
                       f_of_t_list)
              .value(),
          point_xi, custom_approx);
    }

    CHECK_ITERABLE_APPROX(
        piecewise_translation_map(far_point_xi, t, f_of_t_list), far_point_xi);
    CHECK_ITERABLE_APPROX(
        piecewise_translation_map.inverse(far_point_xi, t, f_of_t_list).value(),
        far_point_xi);
    CHECK_ITERABLE_APPROX(
        piecewise_translation_map_deserialized(far_point_xi, t, f_of_t_list),
        far_point_xi);
    CHECK_ITERABLE_APPROX(piecewise_translation_map_deserialized
                              .inverse(far_point_xi, t, f_of_t_list)
                              .value(),
                          far_point_xi);
    CHECK_ITERABLE_APPROX(radial_translation_map(point_xi, t, f_of_t_list),
                          point_xi + (translation * gaussian(radius)));
    CHECK_ITERABLE_APPROX(
        translation_map.inverse(point_xi + translation, t, f_of_t_list).value(),
        point_xi);
    CHECK_ITERABLE_CUSTOM_APPROX(
        radial_translation_map
            .inverse(point_xi + (translation * gaussian(radius)), t,
                     f_of_t_list)
            .value(),
        point_xi, custom_approx);
    CHECK_ITERABLE_APPROX(
        translation_map.frame_velocity(point_xi, t, f_of_t_list), frame_vel);
    CHECK_ITERABLE_APPROX(
        radial_translation_map.frame_velocity(point_xi, t, f_of_t_list),
        radial_frame_vel);
    CHECK_ITERABLE_APPROX(
        piecewise_translation_map.frame_velocity(point_xi, t, f_of_t_list),
        inspiral_frame_vel);
    CHECK_ITERABLE_APPROX(
        piecewise_translation_map.frame_velocity(far_point_xi, t, f_of_t_list),
        far_inspiral_frame_vel);
    CHECK_ITERABLE_APPROX(piecewise_translation_map_deserialized.frame_velocity(
                              point_xi, t, f_of_t_list),
                          inspiral_frame_vel);
    CHECK_ITERABLE_APPROX(piecewise_translation_map_deserialized.frame_velocity(
                              far_point_xi, t, f_of_t_list),
                          far_inspiral_frame_vel);
    CHECK_ITERABLE_APPROX(
        radial_translation_map_deserialized(point_xi, t, f_of_t_list),
        point_xi + (translation * gaussian(radius)));
    CHECK_ITERABLE_APPROX(translation_map_deserialized
                              .inverse(point_xi + translation, t, f_of_t_list)
                              .value(),
                          point_xi);
    CHECK_ITERABLE_CUSTOM_APPROX(
        radial_translation_map_deserialized
            .inverse(point_xi + (translation * gaussian(radius)), t,
                     f_of_t_list)
            .value(),
        point_xi, custom_approx);
    CHECK_ITERABLE_APPROX(
        translation_map_deserialized.frame_velocity(point_xi, t, f_of_t_list),
        frame_vel);
    CHECK_ITERABLE_APPROX(radial_translation_map_deserialized.frame_velocity(
                              point_xi, t, f_of_t_list),
                          radial_frame_vel);

    test_jacobian(translation_map, point_xi, t, f_of_t_list);
    test_inv_jacobian(translation_map, point_xi, t, f_of_t_list);
    test_jacobian(translation_map_deserialized, point_xi, t, f_of_t_list);
    test_inv_jacobian(translation_map_deserialized, point_xi, t, f_of_t_list);
    test_jacobian(radial_translation_map, point_xi, t, f_of_t_list);
    test_inv_jacobian(radial_translation_map, point_xi, t, f_of_t_list);
    test_jacobian(radial_translation_map_deserialized, point_xi, t,
                  f_of_t_list);
    test_inv_jacobian(radial_translation_map_deserialized, point_xi, t,
                      f_of_t_list);
    // This is for an edge case numerical derivative function used in the
    // jacobian, it calls the operator with tiny steps forward and backward on
    // the order of x1 = x + dx, x2 = x1 + dx and x3 = x2 + dx where dx = 1e-4
    // so it moves the point_xi up to 3e-4 forward and backward and calls the
    // operator with each different value. When it does that, it can cross the
    // inner radius where the translation is different causing the numerical
    // jacobian to be computed in the wrong region. However, when it calls the
    // jacobian, it still uses the original point_xi so there's a mixup between
    // which region it's in.
    if (piecewise_radius <= inner_radius * .99 or
        piecewise_radius >= inner_radius * 1.01) {
      test_jacobian(piecewise_translation_map, point_xi, t, f_of_t_list);
      test_inv_jacobian(piecewise_translation_map, point_xi, t, f_of_t_list);
      test_jacobian(piecewise_translation_map_deserialized, point_xi, t,
                    f_of_t_list);
      test_inv_jacobian(piecewise_translation_map_deserialized, point_xi, t,
                        f_of_t_list);
    }
    // Checking if any radius in the datavector is within the range above.
    bool jac_flag = true;
    for (size_t i = 0; i < gsl::at(point_xi_dv, 0).size(); i++) {
      if (gsl::at(piecewise_radius_dv, i) > inner_radius * .99 and
          gsl::at(piecewise_radius_dv, i) < inner_radius * 1.01) {
        jac_flag = false;
      }
    }
    if (jac_flag) {
      test_jacobian(piecewise_translation_map, point_xi_dv, t, f_of_t_list);
      test_inv_jacobian(piecewise_translation_map, point_xi_dv, t, f_of_t_list);
      test_jacobian(piecewise_translation_map_deserialized, point_xi_dv, t,
                    f_of_t_list);
      test_inv_jacobian(piecewise_translation_map_deserialized, point_xi_dv, t,
                        f_of_t_list);
    }
    // Same issue as above, but for the outer boundary.
    if (far_piecewise_radius >= outer_radius * 1.01) {
      test_jacobian(piecewise_translation_map, far_point_xi, t, f_of_t_list);
      test_inv_jacobian(piecewise_translation_map, far_point_xi, t,
                        f_of_t_list);
      test_jacobian(piecewise_translation_map_deserialized, far_point_xi, t,
                    f_of_t_list);
      test_inv_jacobian(piecewise_translation_map_deserialized, far_point_xi, t,
                        f_of_t_list);
    }
    t += dt;
  }

  // Check inequivalence operator
  CHECK_FALSE(translation_map != translation_map);
  CHECK_FALSE(translation_map_deserialized != translation_map_deserialized);
  CHECK_FALSE(radial_translation_map != radial_translation_map);
  CHECK_FALSE(radial_translation_map_deserialized !=
              radial_translation_map_deserialized);
  CHECK_FALSE(piecewise_translation_map != piecewise_translation_map);
  CHECK_FALSE(piecewise_translation_map_deserialized !=
              piecewise_translation_map_deserialized);

  // Check serialization
  CHECK(translation_map == translation_map_deserialized);
  CHECK_FALSE(translation_map != translation_map_deserialized);

  test_coordinate_map_argument_types(translation_map, point_xi, t, f_of_t_list);
  CHECK(radial_translation_map == radial_translation_map_deserialized);
  CHECK_FALSE(radial_translation_map != radial_translation_map_deserialized);

  test_coordinate_map_argument_types(radial_translation_map, point_xi, t,
                                     f_of_t_list);
  CHECK(piecewise_translation_map == piecewise_translation_map_deserialized);
  CHECK_FALSE(piecewise_translation_map !=
              piecewise_translation_map_deserialized);
  test_coordinate_map_argument_types(piecewise_translation_map, point_xi, t,
                                     f_of_t_list);
  CHECK(not CoordinateMaps::TimeDependent::Translation<Dim>{}.is_identity());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.TimeDependent.Translation",
                  "[Domain][Unit]") {
  test_translation<1>();
  test_translation<2>();
  test_translation<3>();
}
}  // namespace domain
