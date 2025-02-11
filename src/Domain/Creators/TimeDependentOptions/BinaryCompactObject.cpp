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
#include "Domain/Creators/TimeDependentOptions/ExpansionMap.hpp"
#include "Domain/Creators/TimeDependentOptions/RotationMap.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/Creators/TimeDependentOptions/SkewMap.hpp"
#include "Domain/Creators/TimeDependentOptions/TranslationMap.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/IntegratedFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
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
    double initial_time, ExpansionMapOptionType expansion_map_options,
    RotationMapOptionType rotation_map_options,
    TranslationMapOptionType translation_map_options,
    SkewMapOptionType skew_map_options,
    ShapeMapOptionType<domain::ObjectLabel::A> shape_options_A,
    ShapeMapOptionType<domain::ObjectLabel::B> shape_options_B,
    const Options::Context& context)
    : initial_time_(initial_time),
      expansion_map_options_(std::move(expansion_map_options)),
      rotation_map_options_(std::move(rotation_map_options)),
      translation_map_options_(std::move(translation_map_options)),
      skew_map_options_(std::move(skew_map_options)),
      shape_options_A_(std::move(shape_options_A)),
      shape_options_B_(std::move(shape_options_B)) {
  if (not(expansion_map_options_.has_value() or
          rotation_map_options_.has_value() or
          translation_map_options_.has_value() or
          skew_map_options_.has_value() or shape_options_A_.has_value() or
          shape_options_B_.has_value())) {
    PARSE_ERROR(context,
                "Time dependent map options were specified, but all options "
                "were 'None'. If you don't want time dependent maps, specify "
                "'None' for the TimeDependentMapOptions. If you want time "
                "dependent maps, specify options for at least one map.");
  }

  const auto check_l_max = [&context](const auto& shape_option,
                                      const domain::ObjectLabel label) {
    if (shape_option.has_value() and
        time_dependent_options::l_max_from_shape_options(
            shape_option.value()) <= 1) {
      PARSE_ERROR(context,
                  "Initial LMax for object "
                      << label << " must be 2 or greater but is "
                      << time_dependent_options::l_max_from_shape_options(
                             shape_option.value())
                      << " instead.");
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
  if (skew_map_options_.has_value()) {
    ERROR("Skew map is not implemented for worldtube evolutions.");
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

  const auto& expansion_map_opts =
      std::get<time_dependent_options::ExpansionMapOptions<false>>(
          expansion_map_options_.value());
  result[expansion_name] =
      std::make_unique<FunctionsOfTime::IntegratedFunctionOfTime>(
          initial_time_,
          std::array<double, 2>{expansion_map_opts.initial_values[0][0],
                                expansion_map_opts.initial_values[1][0]},
          initial_expiration_time, false);
  result[expansion_outer_boundary_name] =
      std::make_unique<FunctionsOfTime::FixedSpeedCubic>(
          1.0, initial_time_,
          expansion_map_opts.asymptotic_velocity_outer_boundary.value(),
          expansion_map_opts.decay_timescale_outer_boundary);

  if (not rotation_map_options_.has_value()) {
    ERROR(
        "Initial values for the rotation map need to be provided when using "
        "the worldtube.");
  }
  const auto& rotation_map_opts =
      std::get<time_dependent_options::RotationMapOptions<false>>(
          rotation_map_options_.value());

  result[rotation_name] =
      std::make_unique<FunctionsOfTime::IntegratedFunctionOfTime>(
          initial_time_,
          std::array<double, 2>{0., rotation_map_opts.angles[1][2]},
          initial_expiration_time, true);

  // Size and Shape FunctionOfTime for objects A and B. Only spherical excision
  // spheres are supported currently.
  if (not(shape_options_A_.has_value() and shape_options_B_.has_value())) {
    ERROR(
        "Initial size for both excision spheres need to be provided when using "
        "the worldtube.");
  }
  const auto& shape_opts_A = std::get<time_dependent_options::ShapeMapOptions<
      not IsCylindrical, domain::ObjectLabel::A>>(shape_options_A_.value());
  const auto& shape_opts_B = std::get<time_dependent_options::ShapeMapOptions<
      not IsCylindrical, domain::ObjectLabel::B>>(shape_options_B_.value());
  for (size_t i = 0; i < shape_names.size(); i++) {
    const auto make_initial_size_values = [](const auto& lambda_options) {
      return std::array<double, 2>{
          {gsl::at(lambda_options.initial_size_values.value(), 0),
           gsl::at(lambda_options.initial_size_values.value(), 1)}};
    };
    const std::array<double, 2> initial_size_values =
        i == 0 ? make_initial_size_values(shape_opts_A)
               : make_initial_size_values(shape_opts_B);
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
      {skew_name, std::numeric_limits<double>::infinity()},
      {gsl::at(size_names, 0), std::numeric_limits<double>::infinity()},
      {gsl::at(size_names, 1), std::numeric_limits<double>::infinity()},
      {gsl::at(shape_names, 0), std::numeric_limits<double>::infinity()},
      {gsl::at(shape_names, 1), std::numeric_limits<double>::infinity()}};

  // If we have control systems, overwrite these expiration times with the ones
  // supplied by the control system
  for (const auto& [name, expr_time] : initial_expiration_times) {
    expiration_times[name] = expr_time;
  }

  // ExpansionMap FunctionOfTime for the function a(t) and b(t) in the
  // domain::CoordinateMaps::TimeDependent::RotScaleTrans map
  if (expansion_map_options_.has_value()) {
    auto expansion_functions_of_time = time_dependent_options::get_expansion(
        expansion_map_options_.value(), initial_time_,
        expiration_times.at(expansion_name));

    result.merge(expansion_functions_of_time);
  }

  // RotationMap FunctionOfTime for the rotation angles about each
  // axis.
  if (rotation_map_options_.has_value()) {
    result[rotation_name] = time_dependent_options::get_rotation(
        rotation_map_options_.value(), initial_time_,
        expiration_times.at(rotation_name));
  }

  // TranslationMap FunctionOfTime
  if (translation_map_options_.has_value()) {
    result[translation_name] = time_dependent_options::get_translation(
        translation_map_options_.value(), initial_time_,
        expiration_times.at(translation_name));
  }

  // SkewMap FunctionOfTime
  if (skew_map_options_.has_value()) {
    result[skew_name] = time_dependent_options::get_skew(
        skew_map_options_.value(), initial_time_,
        expiration_times.at(skew_name));
  }

  // Size and Shape FunctionOfTime for objects A and B
  if (shape_options_A_.has_value()) {
    if (not deformed_radii_[0].has_value()) {
      ERROR(
          "A shape map was specified for object A, but no inner radius is "
          "available. The object must be enclosed by a sphere.");
    }

    auto shape_and_size = time_dependent_options::get_shape_and_size(
        shape_options_A_.value(), initial_time_,
        expiration_times.at(shape_names[0]), expiration_times.at(size_names[0]),
        *deformed_radii_[0]);

    result.merge(shape_and_size);
  }
  if (shape_options_B_.has_value()) {
    if (not deformed_radii_[1].has_value()) {
      ERROR(
          "A shape map was specified for object B, but no inner radius is "
          "available. The object must be enclosed by a sphere.");
    }

    auto shape_and_size = time_dependent_options::get_shape_and_size(
        shape_options_B_.value(), initial_time_,
        expiration_times.at(shape_names[1]), expiration_times.at(size_names[1]),
        *deformed_radii_[1]);

    result.merge(shape_and_size);
  }

  return result;
}

template <bool IsCylindrical>
void TimeDependentMapOptions<IsCylindrical>::build_maps(
    const std::array<std::array<double, 3>, 2>& object_centers,
    const std::optional<std::array<double, 3>>& cube_A_center,
    const std::optional<std::array<double, 3>>& cube_B_center,
    const std::array<double, 3>& center_of_mass,
    const std::optional<std::array<double, IsCylindrical ? 2 : 3>>&
        object_A_radii,
    const std::optional<std::array<double, IsCylindrical ? 2 : 3>>&
        object_B_radii,
    const bool object_A_filled, const bool object_B_filled,
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
  if (skew_map_options_.has_value()) {
    skew_map_ = Skew{skew_name, center_of_mass, envelope_radius};
  }

  for (size_t i = 0; i < 2; i++) {
    const auto& radii_opt = i == 0 ? object_A_radii : object_B_radii;
    const bool has_shape_map =
        i == 0 ? shape_options_A_.has_value() : shape_options_B_.has_value();
    if (not radii_opt.has_value()) {
      // No radii were specified, so we skip building the shape map. This
      // happens when the object is covered by a Cartesian cube.
      if (has_shape_map) {
        ERROR(
            "A shape map was specified for object "
            << (i == 0 ? "A" : "B")
            << ", but no radii were provided. The object must be enclosed by a "
               "sphere, not covered by a Cartesian cube.");
      }
      continue;
    }
    if (not has_shape_map) {
      // Radii were specified, but no shape map was requested. Skip building the
      // shape map.
      continue;
    }
    const auto& radii = radii_opt.value();
    const bool filled = i == 0 ? object_A_filled : object_B_filled;
    // Store the inner radii for creating functions of time
    gsl::at(deformed_radii_, i) = filled ? radii[1] : radii[0];

    const size_t initial_l_max =
        i == 0 ? time_dependent_options::l_max_from_shape_options(
                     shape_options_A_.value())
               : time_dependent_options::l_max_from_shape_options(
                     shape_options_B_.value());

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
          std::make_unique<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                               SphereTransition>(radii[0], radii[1]);

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
          i == 0 ? time_dependent_options::
                       transition_ends_at_cube_from_shape_options(
                           shape_options_A_.value())
                 : time_dependent_options::
                       transition_ends_at_cube_from_shape_options(
                           shape_options_B_.value());

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

      // These are the inner and outer radii between which the shape map falls
      // off outside the object/excision. If the object is filled, then inside
      // the object there will be an additional reverse transition function from
      // the inner cube to the deformed outer surface of the sphere.
      const double inner_radius = filled ? radii[1] : radii[0];
      const double outer_radius = transition_ends_at_cube ? radii[2] : radii[1];
      const double inner_sphericity = 1.0;
      const double outer_sphericity = transition_ends_at_cube ? 0.0 : 1.0;

      if (filled and not transition_ends_at_cube) {
        ERROR_NO_TRACE(
            "If the object is filled, the transition must end at the cube.");
      }

      using Wedge = domain::CoordinateMaps::ShapeMapTransitionFunctions::Wedge;
      for (size_t j = 0; j < 12; j++) {
        if (filled and j < 6) {
          // Reverse the transition function so the shape map goes to zero at
          // the inner cube
          transition_func = std::make_unique<Wedge>(
              inner_center, radii[0], 0.0, outer_center, radii[1], 1.0,
              static_cast<Wedge::Axis>(gsl::at(axes, j)), true);
        } else {
          transition_func = std::make_unique<Wedge>(
              inner_center, inner_radius, inner_sphericity, outer_center,
              outer_radius, outer_sphericity,
              static_cast<Wedge::Axis>(gsl::at(axes, j % 6)));
        }

        // The shape map should be given the center of the excision always,
        // regardless of if it is offset or not
        gsl::at(gsl::at(shape_maps_, i), j) = Shape{inner_center,
                                                    initial_l_max,
                                                    initial_l_max,
                                                    std::move(transition_func),
                                                    gsl::at(shape_names, i),
                                                    gsl::at(size_names, i)};
      }
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
            ? (shape_options_A_.has_value()
                   ? time_dependent_options::
                         transition_ends_at_cube_from_shape_options(
                             shape_options_A_.value())
                   : false)
            : (shape_options_B_.has_value()
                   ? time_dependent_options::
                         transition_ends_at_cube_from_shape_options(
                             shape_options_B_.value())
                   : false);
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
    // The skew map is only applied within the envelope, which is also where we
    // apply the rigid RotScaleTrans map, so `use_rigid_map` is a sentinel
    // for where we put the skew map (if we have one).
    if (not use_rigid_map) {
      ERROR(
          "'use_rigid_map' must be true when requesting the distorted to "
          "inertial map with a shape map.");
    }
    if (rot_scale_trans_map_.has_value() and skew_map_.has_value()) {
      return std::make_unique<detail::di_map<Skew, RotScaleTrans>>(
          skew_map_.value(), rot_scale_trans);
    } else if (rot_scale_trans_map_.has_value()) {
      return std::make_unique<detail::di_map<RotScaleTrans>>(rot_scale_trans);
    } else if (skew_map_.has_value()) {
      return std::make_unique<detail::di_map<Skew>>(skew_map_.value());
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
  bool block_has_shape_map = Object == domain::ObjectLabel::A
                                 ? shape_options_A_.has_value()
                                 : shape_options_B_.has_value();

  if constexpr (IsCylindrical) {
    block_has_shape_map = block_has_shape_map and include_distorted_map;
  } else if (block_has_shape_map) {
    const bool transition_ends_at_cube =
        Object == domain::ObjectLabel::A
            ? time_dependent_options::
                  transition_ends_at_cube_from_shape_options(
                      shape_options_A_.value())
            : time_dependent_options::
                  transition_ends_at_cube_from_shape_options(
                      shape_options_B_.value());
    block_has_shape_map =
        block_has_shape_map and include_distorted_map.has_value() and
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
      shape =
          &gsl::at(gsl::at(shape_maps_, index), include_distorted_map.value());
    }
    ASSERT(shape->has_value(), "Shape map was requested but not built.");
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
  bool block_has_shape_map = Object == domain::ObjectLabel::A
                                 ? shape_options_A_.has_value()
                                 : shape_options_B_.has_value();

  if constexpr (IsCylindrical) {
    block_has_shape_map = block_has_shape_map and include_distorted_map;
  } else if (block_has_shape_map) {
    const bool transition_ends_at_cube =
        Object == domain::ObjectLabel::A
            ? time_dependent_options::
                  transition_ends_at_cube_from_shape_options(
                      shape_options_A_.value())
            : time_dependent_options::
                  transition_ends_at_cube_from_shape_options(
                      shape_options_B_.value());
    block_has_shape_map =
        block_has_shape_map and include_distorted_map.has_value() and
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
      shape =
          &gsl::at(gsl::at(shape_maps_, index), include_distorted_map.value());
    }
    ASSERT(shape->has_value(), "Shape map was requested but not built.");
    // The skew map is only applied within the envelope, which is also where we
    // apply the rigid RotScaleTrans map, so `use_rigid_map` is a sentinel
    // for where we put the skew map (if we have one).
    if (not use_rigid_map) {
      ERROR(
          "'use_rigid_map' must be true when requesting the grid to inertial "
          "map with a shape map.");
    }
    if (rot_scale_trans_map_.has_value() and skew_map_.has_value()) {
      return std::make_unique<detail::gi_map<Shape, Skew, RotScaleTrans>>(
          shape->value(), skew_map_.value(), rot_scale_trans);
    } else if (rot_scale_trans_map_.has_value()) {
      return std::make_unique<detail::gi_map<Shape, RotScaleTrans>>(
          shape->value(), rot_scale_trans);
    } else if (skew_map_.has_value()) {
      return std::make_unique<detail::gi_map<Shape, Skew>>(shape->value(),
                                                           skew_map_.value());
    } else {
      return std::make_unique<detail::gi_map<Shape>>(shape->value());
    }
  } else {
    // The skew map is only applied within the envelope, which is also where we
    // apply the rigid RotScaleTrans map, so use `use_rigid_map` as a sentinel
    // for where we put the skew map (if we have one).
    if (rot_scale_trans_map_.has_value() and
        (use_rigid_map and skew_map_.has_value())) {
      return std::make_unique<detail::gi_map<Skew, RotScaleTrans>>(
          skew_map_.value(), rot_scale_trans);
    } else if (rot_scale_trans_map_.has_value()) {
      return std::make_unique<detail::gi_map<RotScaleTrans>>(rot_scale_trans);
    } else if (use_rigid_map and skew_map_.has_value()) {
      return std::make_unique<detail::gi_map<Skew>>(skew_map_.value());
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
