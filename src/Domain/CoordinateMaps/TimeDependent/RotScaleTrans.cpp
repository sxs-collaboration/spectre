// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/TimeDependent/RotScaleTrans.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Identity.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/TimeDependent/RotationMatrixHelpers.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

namespace domain::CoordinateMaps::TimeDependent {

template <size_t Dim>
RotScaleTrans<Dim>::RotScaleTrans(
    std::optional<std::pair<std::string, std::string>> scale_f_of_t_names,
    std::optional<std::string> rot_f_of_t_name,
    std::optional<std::string> trans_f_of_t_name, double inner_radius,
    double outer_radius, BlockRegion region)
    : inner_radius_(inner_radius),
      outer_radius_(outer_radius),
      region_(region) {
  if (scale_f_of_t_names.has_value()) {
    scale_f_of_t_a_ = std::move(scale_f_of_t_names.value().first);
    scale_f_of_t_b_ = std::move(scale_f_of_t_names.value().second);
    f_of_t_names_.insert(scale_f_of_t_a_.value());
    f_of_t_names_.insert(scale_f_of_t_b_.value());
  }
  if (rot_f_of_t_name.has_value()) {
    rot_f_of_t_ = std::move(rot_f_of_t_name.value());
    f_of_t_names_.insert(rot_f_of_t_.value());
  }
  if (trans_f_of_t_name.has_value()) {
    trans_f_of_t_ = std::move(trans_f_of_t_name.value());
    f_of_t_names_.insert(trans_f_of_t_.value());
  }
}

template <size_t Dim>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, Dim> RotScaleTrans<Dim>::operator()(
    const std::array<T, Dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  std::array<tt::remove_cvref_wrap_t<T>, Dim> result{};
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = gsl::at(source_coords, i);
  }
  const tt::remove_cvref_wrap_t<T> radius = magnitude(result);
  // Rotation Map
  if (rot_f_of_t_.has_value()) {
    const Matrix rot_matrix = rotation_matrix<Dim>(
        time, *(functions_of_time.at(rot_f_of_t_.value())));
    for (size_t i = 0; i < Dim; i++) {
      gsl::at(result, i) = rot_matrix(i, 0) * source_coords[0];
      for (size_t j = 1; j < Dim; j++) {
        gsl::at(result, i) += rot_matrix(i, j) * gsl::at(source_coords, j);
      }
    }
  }
  // Expansion Map
  if (scale_f_of_t_a_.has_value()) {
    const double scale_a_of_t =
        functions_of_time.at(scale_f_of_t_a_.value())->func(time)[0][0];
    const double scale_b_of_t =
        functions_of_time.at(scale_f_of_t_b_.value())->func(time)[0][0];
    if (region_ == BlockRegion::Inner) {
      for (size_t i = 0; i < Dim; i++) {
        gsl::at(result, i) *= scale_a_of_t;
      }
    } else if (region_ == BlockRegion::Transition) {
      for (size_t k = 0; k < get_size(radius); k++) {
        // Optimization from SpEC to reduce roundoff.
        // Closer to outer radius
        if (1.0 - get_element(radius, k) / (inner_radius_ + outer_radius_) <
            .5) {
          // Expansion falloff factor w_E in the documentation of the form
          // w_E = \frac{R_{in}(R_{out} - r)(E_{a}(t) - E_{b}(t))}{r(R_{out} -
          // R_{in})}
          double radial_scaling_factor =
              ((outer_radius_ - get_element(radius, k)) *
               (scale_a_of_t - scale_b_of_t) * inner_radius_) /
              ((outer_radius_ - inner_radius_) * get_element(radius, k));
          for (size_t i = 0; i < Dim; i++) {
            get_element(gsl::at(result, i), k) *=
                (scale_b_of_t + radial_scaling_factor);
          }
          // Closer to inner radius
        } else {
          // Expansion falloff factor w_E in the documentation of the form
          // w_E = \frac{R_{out}(R_{in} - r)(E_{a}(t) - E_{b}(t))}{r(R_{out} -
          // R_{in})}
          double radial_scaling_factor =
              ((inner_radius_ - get_element(radius, k)) *
               (scale_a_of_t - scale_b_of_t) * outer_radius_) /
              ((outer_radius_ - inner_radius_) * get_element(radius, k));
          for (size_t i = 0; i < Dim; i++) {
            get_element(gsl::at(result, i), k) *=
                (scale_a_of_t + radial_scaling_factor);
          }
        }
      }
    } else {
      for (size_t i = 0; i < Dim; i++) {
        gsl::at(result, i) *= scale_b_of_t;
      }
    }
  }
  // Translation map
  if (trans_f_of_t_.has_value()) {
    const DataVector trans_func_of_time =
        functions_of_time.at(trans_f_of_t_.value())->func(time)[0];
    ASSERT(trans_func_of_time.size() == Dim,
           "The dimension of the function of time ("
               << trans_func_of_time.size()
               << ") does not match the dimension of the map (" << Dim << ").");
    if (region_ == BlockRegion::Inner) {
      for (size_t i = 0; i < Dim; i++) {
        gsl::at(result, i) += gsl::at(trans_func_of_time, i);
      }
    } else if (region_ == BlockRegion::Transition) {
      for (size_t k = 0; k < get_size(radius); k++) {
        // closer to outer radius
        if (1.0 - get_element(radius, k) / (inner_radius_ + outer_radius_) <
            .5) {
          // Translation falloff factor w_T in the documentation of the
          // form w_T = (R_{out} - r) / (R_{out} - R_{in})
          const double radial_translation_factor =
              (outer_radius_ - get_element(radius, k)) /
              (outer_radius_ - inner_radius_);
          for (size_t i = 0; i < Dim; i++) {
            get_element(gsl::at(result, i), k) +=
                gsl::at(trans_func_of_time, i) * radial_translation_factor;
          }
        } else {
          // Translation falloff factor w_T in the documentation of the
          // form w_T = (R_{in} - r) / (R_{out} - R_{in})
          const double radial_translation_factor =
              (inner_radius_ - get_element(radius, k)) /
                  (outer_radius_ - inner_radius_) +
              1.0;
          for (size_t i = 0; i < Dim; i++) {
            get_element(gsl::at(result, i), k) +=
                gsl::at(trans_func_of_time, i) * radial_translation_factor;
          }
        }
      }
      ASSERT(max(magnitude(result)) <= outer_radius_ or
                 equal_within_roundoff(max(magnitude(result)), outer_radius_),
             "Coordinates translated outside outer radius, this should "
             "not happen: "
                 << max(magnitude(result))
                 << " outer radius: " << outer_radius_);
    }
  }
  return result;
}

template <size_t Dim>
std::optional<std::array<double, Dim>> RotScaleTrans<Dim>::inverse(
    const std::array<double, Dim>& target_coords, double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  std::array<double, Dim> result{};
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = gsl::at(target_coords, i);
  }
  const double radius = magnitude(result);
  // This variable is used when taking the inverse of rotation which needs the
  // source coords after the inverse expansion and translation have been done.
  std::array<double, Dim> pre_rotation_result = result;

  // Inverse translation without expansion
  if (trans_f_of_t_.has_value() and not scale_f_of_t_a_.has_value()) {
    const DataVector trans_func_of_time =
        functions_of_time.at(trans_f_of_t_.value())->func(time)[0];
    ASSERT(trans_func_of_time.size() == Dim,
           "The dimension of the function of time ("
               << trans_func_of_time.size()
               << ") does not match the dimension of the map (" << Dim << ").");
      double non_translated_radius_squared = 0.;
      for (size_t i = 0; i < Dim; i++) {
        non_translated_radius_squared +=
            square(gsl::at(target_coords, i) - gsl::at(trans_func_of_time, i));
      }
      if (radius >= outer_radius_) {
        // no translation applied
      } else if (non_translated_radius_squared <= square(inner_radius_)) {
        for (size_t i = 0; i < Dim; i++) {
          gsl::at(pre_rotation_result, i) -= gsl::at(trans_func_of_time, i);
        }
      } else {
        // We need to solve a quadratic of the form w^2a + wb + c where
        // a, b, and c change if you're closer to the inner radius. This
        // is an optimization to reduce roundoff from SpEC.
        double a = 0.;
        double b = 0.;
        double c = 0.;
        double radial_translation_factor = 0.;
        for (size_t i = 0; i < Dim; i++) {
          a += square(gsl::at(trans_func_of_time, i));
        }
        a -= square(outer_radius_ - inner_radius_);
        // When closer to the outer radius the quadratic has the form
        // a = T(t)^2 - (R_{out} - R_{in})^2,
        // b = 2(R_{out}(R_{out} - R_{in}) - T(t)\vec{\bar{\xi}}), and
        // c = \vec{\bar{\xi}}^2 - R_{out}^2
        if (1.0 - radius / (inner_radius_ + outer_radius_) < .5) {
          c = square(radius) - square(outer_radius_);
          for (size_t i = 0; i < Dim; i++) {
            b -= 2.0 * gsl::at(trans_func_of_time, i) *
                 gsl::at(target_coords, i);
          }
          b += 2.0 * outer_radius_ * (outer_radius_ - inner_radius_);
          std::optional<std::array<double, 2>> roots = real_roots(a, b, c);
          radial_translation_factor = root_helper(roots);
          for (size_t i = 0; i < Dim; i++) {
            gsl::at(pre_rotation_result, i) -=
                gsl::at(trans_func_of_time, i) * radial_translation_factor;
          }
        }
        // When closer to the inner radius, the quadratic has the form
        // a = T(t)^2 - (R_{out} - R_{in})^2,
        // b = -2 (T(t)(T(t) - \vec{\bar{\xi}}) + R_{in}(R_{out} - R_{in}),
        // c = (T(t) - \vec{\bar{\xi}})^2 - R_{in}^2
        else {
          c = -square(inner_radius_);
          for (size_t i = 0; i < Dim; i++) {
            b -= 2.0 * gsl::at(trans_func_of_time, i) *
                 (gsl::at(trans_func_of_time, i) - gsl::at(target_coords, i));
            c += square(gsl::at(trans_func_of_time, i) -
                        gsl::at(target_coords, i));
          }
          b -= 2.0 * inner_radius_ * (outer_radius_ - inner_radius_);
          std::optional<std::array<double, 2>> roots = real_roots(a, b, c);
          radial_translation_factor = root_helper(roots);
          for (size_t i = 0; i < Dim; i++) {
            gsl::at(pre_rotation_result, i) -=
                gsl::at(trans_func_of_time, i) *
                (1.0 - radial_translation_factor);
          }
        }
      }
  }
  // Inverse expansion without translation
  else if (scale_f_of_t_a_.has_value() and not trans_f_of_t_.has_value()) {
    const double scale_a_of_t =
        functions_of_time.at(scale_f_of_t_a_.value())->func(time)[0][0];
    const double scale_b_of_t =
        functions_of_time.at(scale_f_of_t_b_.value())->func(time)[0][0];
    ASSERT(scale_a_of_t != 0.0 and scale_b_of_t != 0.0,
           "An expansion map "
           "value was set to 0.0, this will cause an FPE. Expansion a: "
               << scale_a_of_t << " expansion b: " << scale_b_of_t);
    if (radius >= scale_b_of_t * outer_radius_) {
      for (size_t i = 0; i < Dim; i++) {
        gsl::at(pre_rotation_result, i) /= scale_b_of_t;
      }
    } else if (radius <= scale_a_of_t * inner_radius_) {
      for (size_t i = 0; i < Dim; i++) {
        gsl::at(pre_rotation_result, i) /= scale_a_of_t;
      }
    } else {
      // We need to solve a quadratic of the form w^2a + wb + c where
      // a, b, and c change if you're closer to the inner radius. This
      // is an optimization to reduce roundoff from SpEC.
      double a =
          square(scale_a_of_t * inner_radius_ - scale_b_of_t * outer_radius_);
      double b = 0.;
      double c = 0.;
      double radial_scaling_factor = 0.;
      double scaled_radius = 0.;
      double root = 0.;
      double radius_squared = square(radius);
      // When closer to the inner radius the quadratic has the form
      // a = (E_{a}(t)R_{in} - E_{b}(t)R_{out})^2,
      // b = 2E{b}(t)R_{out}(E_{a}(t)R_{in} - E_{b}(t)R_{out}),
      // c = E_{b}(t)^2 R_{out}^2 - \bar{\xi}^2
      if (1.0 - radius / (inner_radius_ * scale_a_of_t +
                          outer_radius_ * scale_b_of_t) <
          .5) {
        scaled_radius = scale_b_of_t * outer_radius_;
        b = 2.0 * scaled_radius *
            (scale_a_of_t * inner_radius_ - scaled_radius);
        c = square(scale_b_of_t * outer_radius_) - radius_squared;
        std::optional<std::array<double, 2>> roots = real_roots(a, b, c);
        root = root_helper(roots);
        radial_scaling_factor =
            (root * inner_radius_ * scale_a_of_t +
             (1.0 - root) * outer_radius_ * scale_b_of_t) /
            (root * inner_radius_ + (1.0 - root) * outer_radius_);
      }
      // When closer to the inner radius the quadratic has the form
      // a = (E_{a}(t)R_{in} - E_{b}(t)R_{out})^2,
      // b = 2E{a}(t)R_{in}(E_{b}(t)R_{out} - E_{a}(t)R_{in}),
      // c = E_{a}(t)^2 R_{in}^2 - \bar{\xi}^2
      else {
        scaled_radius = scale_b_of_t * outer_radius_;
        b = 2.0 * scale_a_of_t * inner_radius_ *
            (scaled_radius - scale_a_of_t * inner_radius_);
        c = square(scale_a_of_t * inner_radius_) - radius_squared;
        std::optional<std::array<double, 2>> roots = real_roots(a, b, c);
        root = root_helper(roots);
        radial_scaling_factor =
            ((1.0 - root) * inner_radius_ * scale_a_of_t +
             root * outer_radius_ * scale_b_of_t) /
            ((1.0 - root) * inner_radius_ + root * outer_radius_);
      }
      pre_rotation_result /= radial_scaling_factor;
    }
  }

  // Inverse expansion and translation
  else if (trans_f_of_t_.has_value() and scale_f_of_t_a_.has_value()) {
    const DataVector trans_func_of_time =
        functions_of_time.at(trans_f_of_t_.value())->func(time)[0];
    ASSERT(trans_func_of_time.size() == Dim,
           "The dimension of the function of time ("
               << trans_func_of_time.size()
               << ") does not match the dimension of the map (" << Dim << ").");
    const double scale_a_of_t =
        functions_of_time.at(scale_f_of_t_a_.value())->func(time)[0][0];
    const double scale_b_of_t =
        functions_of_time.at(scale_f_of_t_b_.value())->func(time)[0][0];
    ASSERT(scale_a_of_t != 0.0 and scale_b_of_t != 0.0,
           "An expansion map "
           "value was set to 0.0, this will cause an FPE. Expansion a: "
               << scale_a_of_t << " expansion b: " << scale_b_of_t);
      double non_translated_radius_squared = 0.;
      for (size_t i = 0; i < Dim; i++) {
        non_translated_radius_squared +=
            square(gsl::at(target_coords, i) - gsl::at(trans_func_of_time, i));
      }
      if (radius >= scale_b_of_t * outer_radius_) {
        pre_rotation_result /= scale_b_of_t;
      } else if (non_translated_radius_squared <=
                 square(scale_a_of_t * inner_radius_)) {
        for (size_t i = 0; i < Dim; i++) {
          gsl::at(pre_rotation_result, i) -= gsl::at(trans_func_of_time, i);
        }
        pre_rotation_result /= scale_a_of_t;
      } else {
        // We need to solve a quadratic of the form w^2a + wb + c where
        // a, b, and c change if you're closer to the inner radius. This
        // is an optimization to reduce roundoff from SpEC.
        double a =
            square(scale_a_of_t * inner_radius_ - scale_b_of_t * outer_radius_);
        for (size_t i = 0; i < Dim; i++) {
          a -= square(gsl::at(trans_func_of_time, i));
        }
        double b = 0.;
        double c = 0.;
        double radial_translation_factor = 0.;
        double radial_scaling_factor = 0.;
        // When closer to the outer radius, the quadratic has the form
        // a = (E_{a}(t)R_{in} - E_{b}(t)R_{out})^2 - T(t)^2,
        // b = 2(E_{b}(t)R_{out}(E_{a}(t)R_{in} - E_{b}(t)R_{out}) +
        // T(t)\vec{\bar{\xi}}), and
        // c = E_{b}(t)^2 R_{out}^2 - \vec{\bar{\xi}}^2
        if (1.0 - radius / (scale_a_of_t * inner_radius_ +
                            scale_b_of_t * outer_radius_) <
            .5) {
          b = 2.0 *
              (scale_b_of_t * outer_radius_ *
               (scale_a_of_t * inner_radius_ - scale_b_of_t * outer_radius_));
          for (size_t i = 0; i < Dim; i++) {
            b += 2.0 * gsl::at(trans_func_of_time, i) *
                 gsl::at(target_coords, i);
          }
          c = square(scale_b_of_t * outer_radius_) - square(radius);
          std::optional<std::array<double, 2>> roots = real_roots(a, b, c);
          radial_translation_factor = root_helper(roots);
          for (size_t i = 0; i < Dim; i++) {
            gsl::at(pre_rotation_result, i) -=
                gsl::at(trans_func_of_time, i) * radial_translation_factor;
          }
          radial_scaling_factor =
              (radial_translation_factor * inner_radius_ * scale_a_of_t +
               (1.0 - radial_translation_factor) * outer_radius_ *
                   scale_b_of_t) /
              (radial_translation_factor * inner_radius_ +
               (1.0 - radial_translation_factor) * outer_radius_);
        }
        // When closer to the inner radius, the quadratic has the form
        // a = (E_{a}(t)R_{in} - E_{b}(t)R_{out})^2 - T(t)^2,
        // b = 2(E_{a}(t)R_{in}(E_{b}(t)R_{out} - E_{a}(t)R_{in}) +
        // T(t)(T(t) - \vec{\bar{\xi}})), and
        // c = E_{a}(t)^2 R_{in}^2 - (\vec{\bar{\xi}} - T(t))^2
        else {
          b = 2.0 *
              (scale_a_of_t * inner_radius_ *
               (scale_b_of_t * outer_radius_ - scale_a_of_t * inner_radius_));
          c = square(scale_a_of_t * inner_radius_) -
              non_translated_radius_squared;
          for (size_t i = 0; i < Dim; i++) {
            b += 2.0 * gsl::at(trans_func_of_time, i) *
                 (gsl::at(trans_func_of_time, i) - gsl::at(target_coords, i));
          }
          std::optional<std::array<double, 2>> roots = real_roots(a, b, c);
          radial_translation_factor = root_helper(roots);
          for (size_t i = 0; i < Dim; i++) {
            gsl::at(pre_rotation_result, i) -=
                gsl::at(trans_func_of_time, i) *
                (1.0 - radial_translation_factor);
          }
          radial_scaling_factor =
              ((1.0 - radial_translation_factor) * inner_radius_ *
                   scale_a_of_t +
               radial_translation_factor * outer_radius_ * scale_b_of_t) /
              ((1.0 - radial_translation_factor) * inner_radius_ +
               radial_translation_factor * outer_radius_);
        }
        pre_rotation_result /= radial_scaling_factor;
      }
  }

  // Inverse rotation is the transpose of the original rotation matrix.
  if (rot_f_of_t_.has_value()) {
    const Matrix rot_matrix = rotation_matrix<Dim>(
        time, *(functions_of_time.at(rot_f_of_t_.value())));
    for (size_t i = 0; i < Dim; i++) {
      gsl::at(result, i) = rot_matrix(0, i) * gsl::at(pre_rotation_result, 0);
      for (size_t j = 1; j < Dim; j++) {
        gsl::at(result, i) +=
            rot_matrix(j, i) * gsl::at(pre_rotation_result, j);
      }
    }
    return result;
  } else {
    return pre_rotation_result;
  }
}

template <size_t Dim>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, Dim> RotScaleTrans<Dim>::frame_velocity(
    const std::array<T, Dim>& source_coords, double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  std::array<tt::remove_cvref_wrap_t<T>, Dim> result =
      make_with_value<std::array<tt::remove_cvref_wrap_t<T>, Dim>>(
          dereference_wrapper(source_coords[0]), 0.0);
  tt::remove_cvref_wrap_t<T> radius =
      square(dereference_wrapper(source_coords[0]));
  for (size_t i = 1; i < Dim; ++i) {
    radius += square(dereference_wrapper(gsl::at(source_coords, i)));
  }
  radius = sqrt(radius);
  // Rotation map with no expansion
  if (rot_f_of_t_.has_value() and not scale_f_of_t_a_.has_value()) {
    const Matrix rot_matrix_deriv = rotation_matrix_deriv<Dim>(
        time, *(functions_of_time.at(rot_f_of_t_.value())));
    for (size_t i = 0; i < Dim; i++) {
      for (size_t j = 0; j < Dim; j++) {
        gsl::at(result, i) +=
            rot_matrix_deriv(i, j) * gsl::at(source_coords, j);
      }
    }
  }
  // Expansion map with no rotation
  else if (scale_f_of_t_a_.has_value() and not rot_f_of_t_.has_value()) {
    const double dt_a_of_t = functions_of_time.at(scale_f_of_t_a_.value())
                                 ->func_and_deriv(time)[1][0];
    const double dt_b_of_t = functions_of_time.at(scale_f_of_t_b_.value())
                                 ->func_and_deriv(time)[1][0];
    if (region_ == BlockRegion::Inner) {
      for (size_t i = 0; i < Dim; i++) {
        gsl::at(result, i) +=
            dereference_wrapper(gsl::at(source_coords, i)) * (dt_a_of_t);
      }
    } else if (region_ == BlockRegion::Transition) {
      for (size_t k = 0; k < get_size(radius); k++) {
        double deriv_radial_scaling_factor = 0;
        // Optimization from SpEC to reduce roundoff.
        // Closer to outer radius
        if (1.0 - get_element(radius, k) / (inner_radius_ + outer_radius_) <
            .5) {
          deriv_radial_scaling_factor =
              ((outer_radius_ - get_element(radius, k)) *
               (dt_a_of_t - dt_b_of_t) * inner_radius_) /
              ((outer_radius_ - inner_radius_) * get_element(radius, k));
          for (size_t i = 0; i < Dim; i++) {
            get_element(gsl::at(result, i), k) +=
                get_element(dereference_wrapper(gsl::at(source_coords, i)), k) *
                (dt_b_of_t + deriv_radial_scaling_factor);
          }
          // Closer to inner radius
        } else {
          deriv_radial_scaling_factor =
              ((inner_radius_ - get_element(radius, k)) *
               (dt_a_of_t - dt_b_of_t) * outer_radius_) /
              ((outer_radius_ - inner_radius_) * get_element(radius, k));
          for (size_t i = 0; i < Dim; i++) {
            get_element(gsl::at(result, i), k) +=
                get_element(dereference_wrapper(gsl::at(source_coords, i)), k) *
                (dt_a_of_t + deriv_radial_scaling_factor);
          }
        }
      }
    } else {
      for (size_t i = 0; i < Dim; i++) {
        gsl::at(result, i) +=
            dereference_wrapper(gsl::at(source_coords, i)) * (dt_b_of_t);
      }
    }
  }
  // Rotation and expansion map
  else if (scale_f_of_t_a_.has_value() and rot_f_of_t_.has_value()) {
    const Matrix rot_matrix = rotation_matrix<Dim>(
        time, *(functions_of_time.at(rot_f_of_t_.value())));
    const Matrix rot_matrix_deriv = rotation_matrix_deriv<Dim>(
        time, *(functions_of_time.at(rot_f_of_t_.value())));
    std::array<DataVector, 2> scale_a_func_and_deriv =
        functions_of_time.at(scale_f_of_t_a_.value())->func_and_deriv(time);
    std::array<DataVector, 2> scale_b_func_and_deriv =
        functions_of_time.at(scale_f_of_t_b_.value())->func_and_deriv(time);
    const double scale_a_of_t = scale_a_func_and_deriv[0][0];
    const double scale_b_of_t = scale_b_func_and_deriv[0][0];
    const double dt_a_of_t = scale_a_func_and_deriv[1][0];
    const double dt_b_of_t = scale_b_func_and_deriv[1][0];
    if (region_ == BlockRegion::Inner) {
      for (size_t i = 0; i < Dim; i++) {
        for (size_t j = 0; j < Dim; j++) {
          gsl::at(result, i) += dereference_wrapper(gsl::at(source_coords, j)) *
                                (scale_a_of_t * rot_matrix_deriv(i, j) +
                                 dt_a_of_t * rot_matrix(i, j));
        }
      }
    } else if (region_ == BlockRegion::Transition) {
      for (size_t k = 0; k < get_size(radius); k++) {
        double radial_scaling_factor = 0;
        double deriv_radial_scaling_factor = 0;
        // Optimization from SpEC to reduce roundoff.
        // Closer to outer radius
        if (1.0 - get_element(radius, k) / (inner_radius_ + outer_radius_) <
            .5) {
          radial_scaling_factor =
              ((outer_radius_ - get_element(radius, k)) *
               (scale_a_of_t - scale_b_of_t) * inner_radius_) /
              ((outer_radius_ - inner_radius_) * get_element(radius, k));
          deriv_radial_scaling_factor =
              ((outer_radius_ - get_element(radius, k)) *
               (dt_a_of_t - dt_b_of_t) * inner_radius_) /
              ((outer_radius_ - inner_radius_) * get_element(radius, k));
          for (size_t i = 0; i < Dim; i++) {
            for (size_t j = 0; j < Dim; j++) {
              get_element(gsl::at(result, i), k) +=
                  get_element(dereference_wrapper(gsl::at(source_coords, j)),
                              k) *
                  (rot_matrix_deriv(i, j) *
                       (scale_b_of_t + radial_scaling_factor) +
                   rot_matrix(i, j) *
                       (dt_b_of_t + deriv_radial_scaling_factor));
            }
          }
          // Closer to inner radius
        } else {
          radial_scaling_factor =
              ((inner_radius_ - get_element(radius, k)) *
               (scale_a_of_t - scale_b_of_t) * outer_radius_) /
              ((outer_radius_ - inner_radius_) * get_element(radius, k));
          deriv_radial_scaling_factor =
              ((inner_radius_ - get_element(radius, k)) *
               (dt_a_of_t - dt_b_of_t) * outer_radius_) /
              ((outer_radius_ - inner_radius_) * get_element(radius, k));
          for (size_t i = 0; i < Dim; i++) {
            for (size_t j = 0; j < Dim; j++) {
              get_element(gsl::at(result, i), k) +=
                  get_element(dereference_wrapper(gsl::at(source_coords, j)),
                              k) *
                  (rot_matrix_deriv(i, j) *
                       (scale_a_of_t + radial_scaling_factor) +
                   rot_matrix(i, j) *
                       (dt_a_of_t + deriv_radial_scaling_factor));
            }
          }
        }
      }
    } else {
      for (size_t i = 0; i < Dim; i++) {
        for (size_t j = 0; j < Dim; j++) {
          gsl::at(result, i) += dereference_wrapper(gsl::at(source_coords, j)) *
                                (scale_b_of_t * rot_matrix_deriv(i, j) +
                                 dt_b_of_t * rot_matrix(i, j));
        }
      }
    }
  }
  // Translation map
  if (trans_f_of_t_.has_value()) {
    const DataVector deriv_trans_func_of_time =
        functions_of_time.at(trans_f_of_t_.value())->func_and_deriv(time)[1];
    ASSERT(deriv_trans_func_of_time.size() == Dim,
           "The dimension of the function of time ("
               << deriv_trans_func_of_time.size()
               << ") does not match the dimension of the map (" << Dim << ").");
    if (region_ == BlockRegion::Inner) {
      for (size_t i = 0; i < Dim; i++) {
        gsl::at(result, i) += gsl::at(deriv_trans_func_of_time, i);
      }
    } else if (region_ == BlockRegion::Transition) {
      for (size_t k = 0; k < get_size(radius); k++) {
        // this is the linear falloff factor w in the documentation of the
        // form w = (R_{out} - r) / (R_{out} - R_{in})
        if (1.0 - get_element(radius, k) / (inner_radius_ + outer_radius_) <
            .5) {
          const double radial_translation_factor =
              (outer_radius_ - get_element(radius, k)) /
              (outer_radius_ - inner_radius_);
          for (size_t i = 0; i < Dim; i++) {
            get_element(gsl::at(result, i), k) +=
                gsl::at(deriv_trans_func_of_time, i) *
                radial_translation_factor;
          }
        } else {
          const double radial_translation_factor =
              (inner_radius_ - get_element(radius, k)) /
                  (outer_radius_ - inner_radius_) +
              1.0;
          for (size_t i = 0; i < Dim; i++) {
            get_element(gsl::at(result, i), k) +=
                gsl::at(deriv_trans_func_of_time, i) *
                radial_translation_factor;
          }
        }
      }
    }
  }
  return result;
}

template <size_t Dim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>
RotScaleTrans<Dim>::jacobian(
    const std::array<T, Dim>& source_coords, double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  auto result = make_with_value<
      tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>>(
      dereference_wrapper(source_coords[0]), 0.0);
  // Making the identity in case rotation isn't specified.
  for (size_t i = 0; i < Dim; i++) {
    result.get(i, i) = 1;
  }
  const tt::remove_cvref_wrap_t<T> radius = magnitude(source_coords);
  // Rotation map with no expansion
  if (rot_f_of_t_.has_value() and not scale_f_of_t_a_.has_value()) {
    const Matrix rot_matrix = rotation_matrix<Dim>(
        time, *(functions_of_time.at(rot_f_of_t_.value())));

    for (size_t i = 0; i < Dim; i++) {
      for (size_t j = 0; j < Dim; j++) {
        result.get(i, j) = rot_matrix(i, j);
      }
    }
  }
  // Expansion map with no rotation
  else if (scale_f_of_t_a_.has_value() and not rot_f_of_t_.has_value()) {
    const double scale_a_of_t =
        functions_of_time.at(scale_f_of_t_a_.value())->func(time)[0][0];
    const double scale_b_of_t =
        functions_of_time.at(scale_f_of_t_b_.value())->func(time)[0][0];
    if (region_ == BlockRegion::Inner) {
      for (size_t i = 0; i < Dim; i++) {
        result.get(i, i) = scale_a_of_t;
      }
    } else if (region_ == BlockRegion::Transition) {
      for (size_t k = 0; k < get_size(radius); k++) {
        const double alpha = inner_radius_ * outer_radius_ /
                             (square((get_element(radius, k))) *
                              (inner_radius_ - outer_radius_));
        // Optimization from SpEC to reduce roundoff.
        // Closer to outer radius
        if (1.0 - get_element(radius, k) / (inner_radius_ + outer_radius_) <
            .5) {
          double radial_scaling_factor =
              ((outer_radius_ - get_element(radius, k)) *
               (scale_a_of_t - scale_b_of_t) * inner_radius_) /
              ((outer_radius_ - inner_radius_) * get_element(radius, k));
          for (size_t i = 0; i < Dim; i++) {
            for (size_t j = 0; j < Dim; j++) {
              get_element(result.get(i, j), k) =
                  alpha *
                  get_element(dereference_wrapper(gsl::at(source_coords, i)),
                              k) *
                  (scale_a_of_t - scale_b_of_t) *
                  get_element(dereference_wrapper(gsl::at(source_coords, j)),
                              k) /
                  get_element(radius, k);
            }
            get_element(result.get(i, i), k) +=
                scale_b_of_t + radial_scaling_factor;
          }
          // Closer to inner radius
        } else {
          double radial_scaling_factor =
              ((inner_radius_ - get_element(radius, k)) *
               (scale_a_of_t - scale_b_of_t) * outer_radius_) /
              ((outer_radius_ - inner_radius_) * get_element(radius, k));
          for (size_t i = 0; i < Dim; i++) {
            for (size_t j = 0; j < Dim; j++) {
              get_element(result.get(i, j), k) =
                  alpha *
                  get_element(dereference_wrapper(gsl::at(source_coords, i)),
                              k) *
                  (scale_a_of_t - scale_b_of_t) *
                  get_element(dereference_wrapper(gsl::at(source_coords, j)),
                              k) /
                  get_element(radius, k);
            }
            get_element(result.get(i, i), k) +=
                scale_a_of_t + radial_scaling_factor;
          }
        }
      }
    } else {
      for (size_t i = 0; i < Dim; i++) {
        result.get(i, i) = scale_b_of_t;
      }
    }
  }
  // Rotation and expansion map
  else if (scale_f_of_t_a_.has_value() and rot_f_of_t_.has_value()) {
    const Matrix rot_matrix = rotation_matrix<Dim>(
        time, *(functions_of_time.at(rot_f_of_t_.value())));
    const double scale_a_of_t =
        functions_of_time.at(scale_f_of_t_a_.value())->func(time)[0][0];
    const double scale_b_of_t =
        functions_of_time.at(scale_f_of_t_b_.value())->func(time)[0][0];
    if (region_ == BlockRegion::Inner) {
      for (size_t i = 0; i < Dim; i++) {
        for (size_t j = 0; j < Dim; j++) {
          result.get(i, j) = scale_a_of_t * rot_matrix(i, j);
        }
      }
    } else if (region_ == BlockRegion::Transition) {
      for (size_t k = 0; k < get_size(radius); k++) {
        const double alpha = inner_radius_ * outer_radius_ /
                             (square((get_element(radius, k))) *
                              (inner_radius_ - outer_radius_));
        // Optimization from SpEC to reduce roundoff.
        // Closer to outer radius
        if (1.0 - get_element(radius, k) / (inner_radius_ + outer_radius_) <
            .5) {
          double radial_scaling_factor =
              ((outer_radius_ - get_element(radius, k)) *
               (scale_a_of_t - scale_b_of_t) * inner_radius_) /
              ((outer_radius_ - inner_radius_) * get_element(radius, k));
          for (size_t i = 0; i < Dim; i++) {
            double rotated_coords = 0;
            for (size_t l = 0; l < Dim; l++) {
              rotated_coords +=
                  rot_matrix(i, l) *
                  get_element(dereference_wrapper(gsl::at(source_coords, l)),
                              k);
            }
            for (size_t j = 0; j < Dim; j++) {
              get_element(result.get(i, j), k) =
                  scale_b_of_t * rot_matrix(i, j) +
                  alpha * rotated_coords * (scale_a_of_t - scale_b_of_t) *
                      get_element(
                          dereference_wrapper(gsl::at(source_coords, j)), k) /
                      get_element(radius, k) +
                  rot_matrix(i, j) * radial_scaling_factor;
            }
          }
          // Closer to inner radius
        } else {
          double radial_scaling_factor =
              ((inner_radius_ - get_element(radius, k)) *
               (scale_a_of_t - scale_b_of_t) * outer_radius_) /
              ((outer_radius_ - inner_radius_) * get_element(radius, k));
          for (size_t i = 0; i < Dim; i++) {
            double rotated_coords = 0;
            for (size_t l = 0; l < Dim; l++) {
              rotated_coords +=
                  rot_matrix(i, l) *
                  get_element(dereference_wrapper(gsl::at(source_coords, l)),
                              k);
            }
            for (size_t j = 0; j < Dim; j++) {
              get_element(result.get(i, j), k) =
                  scale_a_of_t * rot_matrix(i, j) +
                  alpha * rotated_coords * (scale_a_of_t - scale_b_of_t) *
                      get_element(
                          dereference_wrapper(gsl::at(source_coords, j)), k) /
                      get_element(radius, k) +
                  rot_matrix(i, j) * radial_scaling_factor;
            }
          }
        }
      }
    } else {
      for (size_t i = 0; i < Dim; i++) {
        for (size_t j = 0; j < Dim; j++) {
          result.get(i, j) = scale_b_of_t * rot_matrix(i, j);
        }
      }
    }
  }
  // Translation map
  if (trans_f_of_t_.has_value()) {
      const DataVector trans_func_of_time =
          functions_of_time.at(trans_f_of_t_.value())->func_and_deriv(time)[0];
      for (size_t i = 0; i < Dim; i++) {
        const double deriv_translation_factor =
            (-gsl::at(trans_func_of_time, i) / (outer_radius_ - inner_radius_));
        for (size_t j = 0; j < Dim; j++) {
          for (size_t k = 0; k < get_size(radius); k++) {
            // The jacobian is the identity for the translation map in regions
            // not between the inner and outer radius.
            if (region_ == BlockRegion::Transition) {
              // using the derivative of the radial falloff factor as
              // \frac{dw}{dr} = \frac{-1.0}{R_{out} - R{in}}
              get_element(result.get(i, j), k) +=
                  deriv_translation_factor *
                  get_element(dereference_wrapper(gsl::at(source_coords, j)),
                              k) /
                  get_element(radius, k);
            }
          }
        }
      }
  }
  return result;
}

template <size_t Dim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame>
RotScaleTrans<Dim>::inv_jacobian(
    const std::array<T, Dim>& source_coords, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) const {
  return determinant_and_inverse(
             jacobian(source_coords, time, functions_of_time))
      .second;
}

template <size_t Dim>
double RotScaleTrans<Dim>::root_helper(
    const std::optional<std::array<double, 2>> roots) const {
  ASSERT(roots.has_value(), "No roots found");
  if (roots.value()[0] >= 0 and roots.value()[0] <= 1.0) {
    ASSERT(roots.value()[1] > 1.0 or roots.value()[1] < 0.0,
           "Singular map: two solutions between 0 and 1");
    return roots.value()[0];
  } else if (roots.value()[1] >= 0 and roots.value()[1] <= 1.0) {
    return roots.value()[1];
  } else if (equal_within_roundoff(roots.value()[0], 1.0) or

             equal_within_roundoff(roots.value()[1], 1.0)) {
    return 1.0;
  } else if (equal_within_roundoff(roots.value()[0], 0.0) or

             equal_within_roundoff(roots.value()[1], 0.0)) {
    return 0.0;
  }
  ERROR("Root helper couldn't find the correct root");
  return 0.0;
}

template <size_t Dim>
void RotScaleTrans<Dim>::pup(PUP::er& p) {
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | scale_f_of_t_a_;
    p | scale_f_of_t_b_;
    p | rot_f_of_t_;
    p | trans_f_of_t_;
    p | inner_radius_;
    p | outer_radius_;
    p | region_;
  }
  // No need to pup this because it is uniquely determined by f_of_t_names_
  if (p.isUnpacking()) {
    f_of_t_names_.clear();
    if (rot_f_of_t_.has_value()) {
      f_of_t_names_.insert(rot_f_of_t_.value());
    }
    if (scale_f_of_t_a_.has_value() and scale_f_of_t_b_.has_value()) {
      f_of_t_names_.insert(scale_f_of_t_a_.value());
      f_of_t_names_.insert(scale_f_of_t_b_.value());
    }
    if (trans_f_of_t_.has_value()) {
      f_of_t_names_.insert(trans_f_of_t_.value());
    }
  }
}

template <size_t Dim>
bool operator==(const RotScaleTrans<Dim>& lhs, const RotScaleTrans<Dim>& rhs) {
  return lhs.scale_f_of_t_a_ == rhs.scale_f_of_t_a_ and
         lhs.scale_f_of_t_b_ == rhs.scale_f_of_t_b_ and
         lhs.rot_f_of_t_ == rhs.rot_f_of_t_ and
         lhs.trans_f_of_t_ == rhs.trans_f_of_t_ and
         lhs.inner_radius_ == rhs.inner_radius_ and
         lhs.outer_radius_ == rhs.outer_radius_ and lhs.region_ == rhs.region_;
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                \
  template class RotScaleTrans<DIM(data)>;                  \
  template bool operator==(const RotScaleTrans<DIM(data)>&, \
                           const RotScaleTrans<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

#undef INSTANTIATE

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data)>      \
  RotScaleTrans<DIM(data)>::operator()(                                     \
      const std::array<DTYPE(data), DIM(data)>& source_coords, double time, \
      const std::unordered_map<                                             \
          std::string,                                                      \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&        \
          functions_of_time) const;                                         \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data)>      \
  RotScaleTrans<DIM(data)>::frame_velocity(                                 \
      const std::array<DTYPE(data), DIM(data)>& source_coords, double time, \
      const std::unordered_map<                                             \
          std::string,                                                      \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&        \
          functions_of_time) const;                                         \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),        \
                    Frame::NoFrame>                                         \
  RotScaleTrans<DIM(data)>::jacobian(                                       \
      const std::array<DTYPE(data), DIM(data)>& source_coords, double time, \
      const std::unordered_map<                                             \
          std::string,                                                      \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&        \
          functions_of_time) const;                                         \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),        \
                    Frame::NoFrame>                                         \
  RotScaleTrans<DIM(data)>::inv_jacobian(                                   \
      const std::array<DTYPE(data), DIM(data)>& source_coords, double time, \
      const std::unordered_map<                                             \
          std::string,                                                      \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&        \
          functions_of_time) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3),
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))
#undef DIM
#undef DTYPE
#undef INSTANTIATE
}  // namespace domain::CoordinateMaps::TimeDependent
