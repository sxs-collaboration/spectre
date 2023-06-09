// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/BinaryCompactObjectHelpers.hpp"

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
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain::creators::bco {
TimeDependentMapOptions::TimeDependentMapOptions(
    double initial_time, ExpansionMapOptions expansion_map_options,
    std::array<double, 3> initial_angular_velocity,
    std::array<double, 3> initial_size_values_A,
    std::array<double, 3> initial_size_values_B, const size_t initial_l_max_A,
    const size_t initial_l_max_B, const Options::Context& context)
    : initial_time_(initial_time),
      expansion_map_options_(expansion_map_options),
      initial_angular_velocity_(initial_angular_velocity),
      initial_size_values_(
          std::array{initial_size_values_A, initial_size_values_B}),
      initial_l_max_{initial_l_max_A, initial_l_max_B} {
  const auto check_l_max = [&context](const size_t l_max,
                                      const domain::ObjectLabel label) {
    if (l_max <= 1) {
      PARSE_ERROR(context, "Initial LMax for object "
                               << label << " must be 2 or greater but is "
                               << l_max << " instead.");
    }
  };

  check_l_max(initial_l_max_A, domain::ObjectLabel::A);
  check_l_max(initial_l_max_B, domain::ObjectLabel::B);
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
TimeDependentMapOptions::create_functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};

  // Get existing function of time names that are used for the maps and assign
  // their initial expiration time to infinity (i.e. not expiring)
  std::unordered_map<std::string, double> expiration_times{
      {expansion_name, std::numeric_limits<double>::infinity()},
      {rotation_name, std::numeric_limits<double>::infinity()},
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
  // domain::CoordinateMaps::TimeDependent::CubicScale map
  result[expansion_name] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time_,
          std::array<DataVector, 3>{
              {{gsl::at(expansion_map_options_.initial_values, 0)},
               {gsl::at(expansion_map_options_.initial_values, 1)},
               {0.0}}},
          expiration_times.at(expansion_name));

  // ExpansionMap FunctionOfTime for the function \f$b(t)\f$ in the
  // domain::CoordinateMaps::TimeDependent::CubicScale map
  result[expansion_outer_boundary_name] =
      std::make_unique<FunctionsOfTime::FixedSpeedCubic>(
          1.0, initial_time_, expansion_map_options_.outer_boundary_velocity,
          expansion_map_options_.outer_boundary_decay_time);

  // RotationMap FunctionOfTime for the rotation angles about each
  // axis.  The initial rotation angles don't matter as we never
  // actually use the angles themselves. We only use their derivatives
  // (omega) to determine map parameters. In theory we could determine
  // each initial angle from the input axis-angle representation, but
  // we don't need to.
  result[rotation_name] =
      std::make_unique<FunctionsOfTime::QuaternionFunctionOfTime<3>>(
          initial_time_,
          std::array<DataVector, 1>{DataVector{1.0, 0.0, 0.0, 0.0}},
          std::array<DataVector, 4>{{{3, 0.0},
                                     {gsl::at(initial_angular_velocity_, 0),
                                      gsl::at(initial_angular_velocity_, 1),
                                      gsl::at(initial_angular_velocity_, 2)},
                                     {3, 0.0},
                                     {3, 0.0}}},
          expiration_times.at(rotation_name));

  // CompressionMap FunctionOfTime for objects A and B
  for (size_t i = 0; i < size_names.size(); i++) {
    result[gsl::at(size_names, i)] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
            initial_time_,
            std::array<DataVector, 4>{
                {{gsl::at(gsl::at(initial_size_values_, i), 0)},
                 {gsl::at(gsl::at(initial_size_values_, i), 1)},
                 {gsl::at(gsl::at(initial_size_values_, i), 2)},
                 {0.0}}},
            expiration_times.at(gsl::at(size_names, i)));
  }

  // ShapeMap FunctionOfTime for objects A and B
  for (size_t i = 0; i < shape_names.size(); i++) {
    const DataVector shape_zeros{
        ylm::Spherepack::spectral_size(gsl::at(initial_l_max_, i),
                                       gsl::at(initial_l_max_, i)),
        0.0};
    result[gsl::at(shape_names, i)] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
            initial_time_,
            std::array<DataVector, 3>{shape_zeros, shape_zeros, shape_zeros},
            expiration_times.at(gsl::at(shape_names, i)));
  }

  return result;
}

void TimeDependentMapOptions::build_maps(
    const std::array<std::array<double, 3>, 2>& centers,
    const std::array<std::optional<double>, 2>& object_inner_radii,
    const std::array<std::optional<double>, 2>& object_outer_radii,
    const double domain_outer_radius) {
  expansion_map_ = CubicScaleMap{domain_outer_radius, expansion_name,
                                 expansion_outer_boundary_name};
  rotation_map_ = RotationMap3D{rotation_name};
  for (size_t i = 0; i < 2; i++) {
    if (gsl::at(object_inner_radii, i).has_value() and
        gsl::at(object_outer_radii, i).has_value()) {
      std::unique_ptr<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                          ShapeMapTransitionFunction>
          transition_func = std::make_unique<
              domain::CoordinateMaps::ShapeMapTransitionFunctions::
                  SphereTransition>(gsl::at(object_inner_radii, i).value(),
                                    gsl::at(object_outer_radii, i).value());

      gsl::at(shape_maps_, i) =
          ShapeMap{gsl::at(centers, i),        gsl::at(initial_l_max_, i),
                   gsl::at(initial_l_max_, i), std::move(transition_func),
                   gsl::at(shape_names, i),    gsl::at(size_names, i)};
    }
  }
}

TimeDependentMapOptions::MapType<Frame::Distorted, Frame::Inertial>
TimeDependentMapOptions::distorted_to_inertial_map(
    const bool include_distorted_map) const {
  if (include_distorted_map) {
    return std::make_unique<DistortedToInertialComposition>(expansion_map_,
                                                            rotation_map_);
  } else {
    return nullptr;
  }
}

template <domain::ObjectLabel Object>
TimeDependentMapOptions::MapType<Frame::Grid, Frame::Distorted>
TimeDependentMapOptions::grid_to_distorted_map(
    const bool include_distorted_map) const {
  if (include_distorted_map) {
    const size_t index = get_index(Object);
    return std::make_unique<GridToDistortedComposition>(
        gsl::at(shape_maps_, index));
  } else {
    return nullptr;
  }
}

template <domain::ObjectLabel Object>
TimeDependentMapOptions::MapType<Frame::Grid, Frame::Inertial>
TimeDependentMapOptions::grid_to_inertial_map(
    const bool include_distorted_map) const {
  if (include_distorted_map) {
    const size_t index = get_index(Object);
    return std::make_unique<GridToInertialComposition<true>>(
        gsl::at(shape_maps_, index), expansion_map_, rotation_map_);
  } else {
    return std::make_unique<GridToInertialComposition<false>>(expansion_map_,
                                                              rotation_map_);
  }
}

size_t TimeDependentMapOptions::get_index(const domain::ObjectLabel object) {
  ASSERT(object == domain::ObjectLabel::A or object == domain::ObjectLabel::B,
         "object label for TimeDependentMapOptions must be either A or B, not"
             << object);
  return object == domain::ObjectLabel::A ? 0_st : 1_st;
}

#define OBJECT(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                \
  template TimeDependentMapOptions::MapType<Frame::Grid, Frame::Distorted>  \
  TimeDependentMapOptions::grid_to_distorted_map<OBJECT(data)>(bool) const; \
  template TimeDependentMapOptions::MapType<Frame::Grid, Frame::Inertial>   \
  TimeDependentMapOptions::grid_to_inertial_map<OBJECT(data)>(bool) const;

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (domain::ObjectLabel::A, domain::ObjectLabel::B,
                         domain::ObjectLabel::None))

#undef OBJECT
#undef INSTANTIATE
}  // namespace domain::creators::bco
