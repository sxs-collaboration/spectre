// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/Wedge.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/IntegratedFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "Options/ParseError.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace domain::creators::bco {
template <bool IsCylindrical>
TimeDependentMapOptions<IsCylindrical>::TimeDependentMapOptions(
    double initial_time,
    std::optional<ExpansionMapOptions> expansion_map_options,
    std::optional<RotationMapOptions> rotation_map_options,
    std::optional<TranslationMapOptions> translation_map_options,
    std::optional<ShapeMapOptions<domain::ObjectLabel::A>> shape_options_A,
    std::optional<ShapeMapOptions<domain::ObjectLabel::B>> shape_options_B,
    const Options::Context& context)
    : initial_time_(initial_time),
      expansion_map_options_(expansion_map_options),
      rotation_map_options_(rotation_map_options),
      translation_map_options_(translation_map_options),
      shape_options_A_(shape_options_A),
      shape_options_B_(shape_options_B) {
  if (not(expansion_map_options_.has_value() or
          rotation_map_options_.has_value() or
          translation_map_options_.has_value() or
          shape_options_A_.has_value() or shape_options_B_.has_value())) {
    PARSE_ERROR(context,
                "Time dependent map options were specified, but all options "
                "were 'None'. If you don't want time dependent maps, specify "
                "'None' for the TimeDependentMapOptions. If you want time "
                "dependent maps, specify options for at least one map.");
  }

  const auto check_l_max = [&context](const auto& shape_option,
                                      const domain::ObjectLabel label) {
    if (shape_option.has_value() and shape_option.value().l_max <= 1) {
      PARSE_ERROR(context, "Initial LMax for object "
                               << label << " must be 2 or greater but is "
                               << shape_option.value().l_max << " instead.");
    }
  };

  check_l_max(shape_options_A_, domain::ObjectLabel::A);
  check_l_max(shape_options_B_, domain::ObjectLabel::B);
}

template <bool IsCylindrical>
std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
TimeDependentMapOptions<IsCylindrical>::create_worldtube_functions_of_time()
    const {
  if (translation_map_options_.has_value()) {
    ERROR("Translation map is not implemented for worldtube evolutions.");
  }
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};
  // The functions of time need to be valid only for the very first time step,
  // after that they need to be updated by the worldtube singleton.
  const double initial_expiration_time = initial_time_ + 1e-10;
  if (not expansion_map_options_.has_value()) {
    ERROR("Initial values for the expansion map need to be provided.");
  }
  result[expansion_name] =
      std::make_unique<FunctionsOfTime::IntegratedFunctionOfTime>(
          initial_time_,
          std::array<double, 2>{
              {{gsl::at(expansion_map_options_.value().initial_values, 0)},
               {gsl::at(expansion_map_options_.value().initial_values, 1)}}},
          initial_expiration_time, false);
  result[expansion_outer_boundary_name] =
      std::make_unique<FunctionsOfTime::FixedSpeedCubic>(
          1.0, initial_time_,
          expansion_map_options_.value().outer_boundary_velocity,
          expansion_map_options_.value().outer_boundary_decay_time);
  if (not rotation_map_options_.has_value()) {
    ERROR(
        "Initial values for the rotation map need to be provided when using "
        "the worldtube.");
  }

  result[rotation_name] =
      std::make_unique<FunctionsOfTime::IntegratedFunctionOfTime>(
          initial_time_,
          std::array<double, 2>{
              0.,
              gsl::at(rotation_map_options_.value().initial_angular_velocity,
                      2)},
          initial_expiration_time, true);

  // Size and Shape FunctionOfTime for objects A and B. Only spherical excision
  // spheres are supported currently.
  if (not(shape_options_A_.has_value() and shape_options_B_.has_value())) {
    ERROR(
        "Initial size for both excision spheres need to be provided when using "
        "the worldtube.");
  }
  for (size_t i = 0; i < shape_names.size(); i++) {
    const auto make_initial_size_values = [](const auto& lambda_options) {
      return std::array<double, 2>{
          {gsl::at(lambda_options.value().initial_size_values.value(), 0),
           gsl::at(lambda_options.value().initial_size_values.value(), 1)}};
    };
    const std::array<double, 2> initial_size_values =
        i == 0 ? make_initial_size_values(shape_options_A_)
               : make_initial_size_values(shape_options_B_);
    const size_t initial_l_max = 2;
    const DataVector shape_zeros{
        ylm::Spherepack::spectral_size(initial_l_max, initial_l_max), 0.0};

    result[gsl::at(shape_names, i)] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
            initial_time_,
            std::array<DataVector, 3>{shape_zeros, shape_zeros, shape_zeros},
            std::numeric_limits<double>::infinity());
    result[gsl::at(size_names, i)] =
        std::make_unique<FunctionsOfTime::IntegratedFunctionOfTime>(
            initial_time_, initial_size_values, initial_expiration_time, false);
  }
  return result;
}

template <bool IsCylindrical>
template <bool UseWorldtube>
std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
TimeDependentMapOptions<IsCylindrical>::create_functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  if constexpr (UseWorldtube) {
    static_assert(not IsCylindrical,
                  "Cylindrical map not supported with worldtube");
    if (not initial_expiration_times.empty()) {
      ERROR(
          "Initial expiration times were specified with worldtube functions of "
          "time. This is not supported, as the worldtube singleton has to set "
          "the expiration times each time step");
    }
    return create_worldtube_functions_of_time();
  }
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};

  // Get existing function of time names that are used for the maps and assign
  // their initial expiration time to infinity (i.e. not expiring)
  std::unordered_map<std::string, double> expiration_times{
      {expansion_name, std::numeric_limits<double>::infinity()},
      {rotation_name, std::numeric_limits<double>::infinity()},
      {translation_name, std::numeric_limits<double>::infinity()},
      {gsl::at(size_names, 0), std::numeric_limits<double>::infinity()},
      {gsl::at(size_names, 1), std::numeric_limits<double>::infinity()},
      {gsl::at(shape_names, 0), std::numeric_limits<double>::infinity()},
      {gsl::at(shape_names, 1), std::numeric_limits<double>::infinity()}};

  // If we have control systems, overwrite these expiration times with the ones
  // supplied by the control system
  for (const auto& [name, expr_time] : initial_expiration_times) {
    expiration_times[name] = expr_time;
  }

  // ExpansionMap FunctionOfTime for the function \f$a(t)\f$ in the
  // domain::CoordinateMaps::TimeDependent::RotScaleTrans map
  if (expansion_map_options_.has_value()) {
    result[expansion_name] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
            initial_time_,
            std::array<DataVector, 3>{
                {{gsl::at(expansion_map_options_.value().initial_values, 0)},
                 {gsl::at(expansion_map_options_.value().initial_values, 1)},
                 {0.0}}},
            expiration_times.at(expansion_name));

    // ExpansionMap FunctionOfTime for the function \f$b(t)\f$ in the
    // domain::CoordinateMaps::TimeDependent::RotScaleTrans map
    result[expansion_outer_boundary_name] =
        std::make_unique<FunctionsOfTime::FixedSpeedCubic>(
            1.0, initial_time_,
            expansion_map_options_.value().outer_boundary_velocity,
            expansion_map_options_.value().outer_boundary_decay_time);
  }

  // RotationMap FunctionOfTime for the rotation angles about each
  // axis.  The initial rotation angles don't matter as we never
  // actually use the angles themselves. We only use their derivatives
  // (omega) to determine map parameters. In theory we could determine
  // each initial angle from the input axis-angle representation, but
  // we don't need to.
  if (rotation_map_options_.has_value()) {
    result[rotation_name] = std::make_unique<
        FunctionsOfTime::QuaternionFunctionOfTime<3>>(
        initial_time_,
        std::array<DataVector, 1>{DataVector{1.0, 0.0, 0.0, 0.0}},
        std::array<DataVector, 4>{
            {{3, 0.0},
             {gsl::at(rotation_map_options_.value().initial_angular_velocity,
                      0),
              gsl::at(rotation_map_options_.value().initial_angular_velocity,
                      1),
              gsl::at(rotation_map_options_.value().initial_angular_velocity,
                      2)},
             {3, 0.0},
             {3, 0.0}}},
        expiration_times.at(rotation_name));
  }

  // TranslationMap FunctionOfTime
  if (translation_map_options_.has_value()) {
    result[translation_name] = std::make_unique<
        FunctionsOfTime::PiecewisePolynomial<2>>(
        initial_time_,
        std::array<DataVector, 3>{
            {{gsl::at(translation_map_options_.value().initial_values, 0)[0],
              gsl::at(translation_map_options_.value().initial_values, 0)[1],
              gsl::at(translation_map_options_.value().initial_values, 0)[2]},
             {gsl::at(translation_map_options_.value().initial_values, 1)[0],
              gsl::at(translation_map_options_.value().initial_values, 1)[1],
              gsl::at(translation_map_options_.value().initial_values, 1)[2]},
             {gsl::at(translation_map_options_.value().initial_values, 2)[0],
              gsl::at(translation_map_options_.value().initial_values, 2)[1],
              gsl::at(translation_map_options_.value().initial_values, 2)[2]}}},
        expiration_times.at(translation_name));
  }

  // Size and Shape FunctionOfTime for objects A and B
  const auto build_shape_and_size_fot =
      [&result, &expiration_times, this](
          const auto& shape_options, const double inner_radius,
          const std::string& shape_name, const std::string& size_name) {
        auto [shape_funcs, size_funcs] =
            time_dependent_options::initial_shape_and_size_funcs(shape_options,
                                                                 inner_radius);

        result[shape_name] =
            std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
                initial_time_, std::move(shape_funcs),
                expiration_times.at(shape_name));
        result[size_name] =
            std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
                initial_time_, std::move(size_funcs),
                expiration_times.at(size_name));
      };

  if (shape_options_A_.has_value()) {
    if (not inner_radii_[0].has_value()) {
      ERROR(
          "A shape map was specified for object A, but no inner radius is "
          "available. The object must be enclosed by a sphere.");
    }
    build_shape_and_size_fot(shape_options_A_.value(), *inner_radii_[0],
                             shape_names[0], size_names[0]);
  }
  if (shape_options_B_.has_value()) {
    if (not inner_radii_[1].has_value()) {
      ERROR(
          "A shape map was specified for object B, but no inner radius is "
          "available. The object must be enclosed by a sphere.");
    }
    build_shape_and_size_fot(shape_options_B_.value(), *inner_radii_[1],
                             shape_names[1], size_names[1]);
  }

  return result;
}

template <bool IsCylindrical>
void TimeDependentMapOptions<IsCylindrical>::build_maps(
    const std::array<std::array<double, 3>, 2>& object_centers,
    const std::optional<std::array<double, 3>>& cube_A_center,
    const std::optional<std::array<double, 3>>& cube_B_center,
    const std::optional<std::array<double, IsCylindrical ? 2 : 3>>&
        object_A_radii,
    const std::optional<std::array<double, IsCylindrical ? 2 : 3>>&
        object_B_radii,
    const double envelope_radius, const double domain_outer_radius) {
  if (expansion_map_options_.has_value() or rotation_map_options_.has_value() or
      translation_map_options_.has_value()) {
    rot_scale_trans_map_ = std::make_pair(
        RotScaleTrans{
            expansion_map_options_.has_value()
                ? std::make_pair(expansion_name, expansion_outer_boundary_name)
                : std::optional<std::pair<std::string, std::string>>{},
            rotation_map_options_.has_value() ? rotation_name
                                              : std::optional<std::string>{},
            translation_map_options_.has_value() ? translation_name
                                                 : std::optional<std::string>{},
            envelope_radius, domain_outer_radius,
            domain::CoordinateMaps::TimeDependent::RotScaleTrans<
                3>::BlockRegion::Inner},
        RotScaleTrans{
            expansion_map_options_.has_value()
                ? std::make_pair(expansion_name, expansion_outer_boundary_name)
                : std::optional<std::pair<std::string, std::string>>{},
            rotation_map_options_.has_value() ? rotation_name
                                              : std::optional<std::string>{},
            translation_map_options_.has_value() ? translation_name
                                                 : std::optional<std::string>{},
            envelope_radius, domain_outer_radius,
            domain::CoordinateMaps::TimeDependent::RotScaleTrans<
                3>::BlockRegion::Transition});
  }

  for (size_t i = 0; i < 2; i++) {
    const auto& radii_opt = i == 0 ? object_A_radii : object_B_radii;
    if (radii_opt.has_value()) {
      const auto& radii = radii_opt.value();
      if (not(i == 0 ? shape_options_A_.has_value()
                     : shape_options_B_.has_value())) {
        ERROR_NO_TRACE(
            "Trying to build the shape map for object "
            << (i == 0 ? domain::ObjectLabel::A : domain::ObjectLabel::B)
            << ", but no time dependent map options were specified "
               "for that object.");
      }
      // Store the inner radii for creating functions of time
      gsl::at(inner_radii_, i) = radii[0];

      const size_t initial_l_max = i == 0 ? shape_options_A_.value().l_max
                                          : shape_options_B_.value().l_max;

      std::unique_ptr<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                          ShapeMapTransitionFunction>
          transition_func{};

      // Currently, we don't support different transition functions for the
      // cylindrical domain
      if constexpr (IsCylindrical) {
        if (cube_A_center.has_value() or cube_B_center.has_value()) {
          ERROR_NO_TRACE(
              "When using the CylindricalBinaryCompactObject domain creator, "
              "the excision centers cannot be offset.");
        }
        transition_func =
            std::make_unique<domain::CoordinateMaps::
                                 ShapeMapTransitionFunctions::SphereTransition>(
                radii[0], radii[1]);

        gsl::at(shape_maps_, i) = Shape{gsl::at(object_centers, i),
                                        initial_l_max,
                                        initial_l_max,
                                        std::move(transition_func),
                                        gsl::at(shape_names, i),
                                        gsl::at(size_names, i)};
      } else {
        // These must match the order of orientations_for_sphere_wrappings() in
        // DomainHelpers.hpp. The values must match that of Wedge::Axis
        const std::array<int, 6> axes{3, -3, 2, -2, 1, -1};

        const bool transition_ends_at_cube =
            i == 0 ? shape_options_A_->transition_ends_at_cube
                   : shape_options_B_->transition_ends_at_cube;

        // These centers must take in to account if we have an offset of the
        // center of the object and where the transition ends. The inner center
        // is always the center of the object. The outer center depends on if we
        // have an offset and where the transition ends. If the transition ends
        // at the cube, then if we have an offset we use the cube center, if not
        // it's the same as the object center. If the transition ends at the
        // sphere, then the center is the object center
        const std::optional<std::array<double, 3>>& cube_center =
            i == 0 ? cube_A_center : cube_B_center;
        const std::array<double, 3>& inner_center = gsl::at(object_centers, i);
        const std::array<double, 3>& outer_center =
            transition_ends_at_cube
                ? cube_center.value_or(gsl::at(object_centers, i))
                : gsl::at(object_centers, i);

        const double inner_sphericity = 1.0;
        const double outer_sphericity = transition_ends_at_cube ? 0.0 : 1.0;

        const double inner_radius = radii[0];
        const double outer_radius =
            transition_ends_at_cube ? radii[2] : radii[1];

        using Wedge =
            domain::CoordinateMaps::ShapeMapTransitionFunctions::Wedge;
        for (size_t j = 0; j < axes.size(); j++) {
          transition_func = std::make_unique<Wedge>(
              inner_center, inner_radius, inner_sphericity, outer_center,
              outer_radius, outer_sphericity,
              static_cast<Wedge::Axis>(gsl::at(axes, j)));

          // The shape map should be given the center of the excision always,
          // regardless of if it is offset or not
          gsl::at(gsl::at(shape_maps_, i), j) =
              Shape{inner_center,
                    initial_l_max,
                    initial_l_max,
                    std::move(transition_func),
                    gsl::at(shape_names, i),
                    gsl::at(size_names, i)};
        }
      }

    } else if (i == 0 ? shape_options_A_.has_value()
                      : shape_options_B_.has_value()) {
      ERROR_NO_TRACE(
          "No excision was specified for object "
          << (i == 0 ? domain::ObjectLabel::A : domain::ObjectLabel::B)
          << ", but ShapeMap options were specified for that object.");
    }
  }
}

template <bool IsCylindrical>
bool TimeDependentMapOptions<IsCylindrical>::has_distorted_frame_options(
    domain::ObjectLabel object) const {
  ASSERT(object == domain::ObjectLabel::A or object == domain::ObjectLabel::B,
         "object label for TimeDependentMapOptions must be either A or B, not"
             << object);
  return object == domain::ObjectLabel::A ? shape_options_A_.has_value()
                                          : shape_options_B_.has_value();
}

template <bool IsCylindrical>
template <domain::ObjectLabel Object>
typename TimeDependentMapOptions<IsCylindrical>::template MapType<
    Frame::Distorted, Frame::Inertial>
TimeDependentMapOptions<IsCylindrical>::distorted_to_inertial_map(
    const IncludeDistortedMapType& include_distorted_map,
    const bool use_rigid_map) const {
  bool block_has_shape_map = false;

  if constexpr (IsCylindrical) {
    block_has_shape_map = include_distorted_map;
  } else {
    const bool transition_ends_at_cube =
        Object == domain::ObjectLabel::A
            ? shape_options_A_->transition_ends_at_cube
            : shape_options_B_->transition_ends_at_cube;
    block_has_shape_map =
        include_distorted_map.has_value() and
        (transition_ends_at_cube or include_distorted_map.value() < 6);
  }

  const auto& rot_scale_trans = rot_scale_trans_map_.has_value()
                                    ? use_rigid_map
                                          ? rot_scale_trans_map_->first
                                          : rot_scale_trans_map_->second
                                    : RotScaleTrans{};

  if (block_has_shape_map) {
    if (rot_scale_trans_map_.has_value()) {
      return std::make_unique<detail::di_map<RotScaleTrans>>(rot_scale_trans);
    } else {
      return std::make_unique<detail::di_map<Identity>>(Identity{});
    }
  } else {
    return nullptr;
  }
}

template <bool IsCylindrical>
template <domain::ObjectLabel Object>
typename TimeDependentMapOptions<IsCylindrical>::template MapType<
    Frame::Grid, Frame::Distorted>
TimeDependentMapOptions<IsCylindrical>::grid_to_distorted_map(
    const IncludeDistortedMapType& include_distorted_map) const {
  bool block_has_shape_map = false;

  if constexpr (IsCylindrical) {
    block_has_shape_map = include_distorted_map;
  } else {
    const bool transition_ends_at_cube =
        Object == domain::ObjectLabel::A
            ? shape_options_A_->transition_ends_at_cube
            : shape_options_B_->transition_ends_at_cube;
    block_has_shape_map =
        include_distorted_map.has_value() and
        (transition_ends_at_cube or include_distorted_map.value() < 6);
  }

  if (block_has_shape_map) {
    const size_t index = get_index(Object);
    const std::optional<Shape>* shape{};
    if constexpr (IsCylindrical) {
      shape = &gsl::at(shape_maps_, index);
    } else {
      if (include_distorted_map.value() >= 12) {
        ERROR(
            "Invalid 'include_distorted_map' argument. Max value allowed is "
            "11, but it is "
            << include_distorted_map.value());
      }
      shape = &gsl::at(gsl::at(shape_maps_, index),
                       include_distorted_map.value() % 6);
    }
    if (not shape->has_value()) {
      ERROR(
          "Requesting grid to distorted map with distorted frame but shape map "
          "options were not specified.");
    }
    return std::make_unique<detail::gd_map<Shape>>(shape->value());
  } else {
    return nullptr;
  }
}

template <bool IsCylindrical>
template <domain::ObjectLabel Object>
typename TimeDependentMapOptions<IsCylindrical>::template MapType<
    Frame::Grid, Frame::Inertial>
TimeDependentMapOptions<IsCylindrical>::grid_to_inertial_map(
    const IncludeDistortedMapType& include_distorted_map,
    const bool use_rigid_map) const {
  bool block_has_shape_map = false;

  if constexpr (IsCylindrical) {
    block_has_shape_map = include_distorted_map;
  } else {
    const bool transition_ends_at_cube =
        Object == domain::ObjectLabel::A
            ? shape_options_A_->transition_ends_at_cube
            : shape_options_B_->transition_ends_at_cube;
    block_has_shape_map =
        include_distorted_map.has_value() and
        (transition_ends_at_cube or include_distorted_map.value() < 6);
  }

  const auto& rot_scale_trans = rot_scale_trans_map_.has_value()
                                    ? use_rigid_map
                                          ? rot_scale_trans_map_->first
                                          : rot_scale_trans_map_->second
                                    : RotScaleTrans{};

  if (block_has_shape_map) {
    const size_t index = get_index(Object);
    const std::optional<Shape>* shape{};
    if constexpr (IsCylindrical) {
      shape = &gsl::at(shape_maps_, index);
    } else {
      if (include_distorted_map.value() >= 12) {
        ERROR(
            "Invalid 'include_distorted_map' argument. Max value allowed is "
            "11, but it is "
            << include_distorted_map.value());
      }
      shape = &gsl::at(gsl::at(shape_maps_, index),
                       include_distorted_map.value() % 6);
    }
    if (not shape->has_value()) {
      ERROR(
          "Requesting grid to inertial map with distorted frame but shape map "
          "options were not specified.");
    }
    if (rot_scale_trans_map_.has_value()) {
      return std::make_unique<detail::gi_map<Shape, RotScaleTrans>>(
          shape->value(), rot_scale_trans);
    } else {
      return std::make_unique<detail::gi_map<Shape>>(shape->value());
    }
  } else {
    if (rot_scale_trans_map_.has_value()) {
      return std::make_unique<detail::gi_map<RotScaleTrans>>(rot_scale_trans);
    } else {
      return nullptr;
    }
  }
}

template <bool IsCylindrical>
size_t TimeDependentMapOptions<IsCylindrical>::get_index(
    const domain::ObjectLabel object) {
  ASSERT(object == domain::ObjectLabel::A or object == domain::ObjectLabel::B,
         "object label for TimeDependentMapOptions must be either A or B, not"
             << object);
  return object == domain::ObjectLabel::A ? 0_st : 1_st;
}

template class TimeDependentMapOptions<true>;
template class TimeDependentMapOptions<false>;

#define ISCYL(data) BOOST_PP_TUPLE_ELEM(0, data)
#define OBJECT(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                 \
  template TimeDependentMapOptions<ISCYL(data)>::MapType<Frame::Distorted,   \
                                                         Frame::Inertial>    \
  TimeDependentMapOptions<ISCYL(data)>::distorted_to_inertial_map<OBJECT(    \
      data)>(                                                                \
      const TimeDependentMapOptions<ISCYL(data)>::IncludeDistortedMapType&,  \
      const bool) const;                                                     \
  template TimeDependentMapOptions<ISCYL(data)>::MapType<Frame::Grid,        \
                                                         Frame::Distorted>   \
  TimeDependentMapOptions<ISCYL(data)>::grid_to_distorted_map<OBJECT(data)>( \
      const TimeDependentMapOptions<ISCYL(data)>::IncludeDistortedMapType&)  \
      const;                                                                 \
  template TimeDependentMapOptions<ISCYL(data)>::MapType<Frame::Grid,        \
                                                         Frame::Inertial>    \
  TimeDependentMapOptions<ISCYL(data)>::grid_to_inertial_map<OBJECT(data)>(  \
      const TimeDependentMapOptions<ISCYL(data)>::IncludeDistortedMapType&,  \
      const bool) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (true, false),
                        (domain::ObjectLabel::A, domain::ObjectLabel::B,
                         domain::ObjectLabel::None))

#undef OBJECT
#undef ISCYL
#undef INSTANTIATE

template std::unordered_map<
    std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
TimeDependentMapOptions<false>::create_functions_of_time<true>(
    const std::unordered_map<std::string, double>&) const;
template std::unordered_map<
    std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
TimeDependentMapOptions<false>::create_functions_of_time<false>(
    const std::unordered_map<std::string, double>&) const;
template std::unordered_map<
    std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
TimeDependentMapOptions<true>::create_functions_of_time<false>(
    const std::unordered_map<std::string, double>&) const;
}  // namespace domain::creators::bco
