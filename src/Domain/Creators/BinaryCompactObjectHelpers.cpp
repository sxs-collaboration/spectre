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
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/YlmSpherepack.hpp"
#include "Options/Options.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain::creators::bco {
TimeDependentMapOptions::TimeDependentMapOptions(
    double initial_time, ExpansionMapOptions expansion_map_options,
    std::array<double, 3> initial_angular_velocity,
    std::array<double, 3> initial_size_values_A,
    std::array<double, 3> initial_size_values_B)
    : initial_time_(initial_time),
      expansion_map_options_(expansion_map_options),
      initial_angular_velocity_(initial_angular_velocity),
      initial_size_values_(
          std::array{initial_size_values_A, initial_size_values_B}) {}

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
      {gsl::at(size_names, 1), std::numeric_limits<double>::infinity()}};

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
      gsl::at(size_maps_, i) = CompressionMap{
          gsl::at(size_names, i), gsl::at(object_inner_radii, i).value(),
          gsl::at(object_outer_radii, i).value(), gsl::at(centers, i)};
    }
  }
}

template <typename SourceFrame>
TimeDependentMapOptions::MapType<SourceFrame, Frame::Inertial>
TimeDependentMapOptions::frame_to_inertial_map() const {
  return std::make_unique<
      CubicScaleAndRotationMapForComposition<SourceFrame, Frame::Inertial>>(
      expansion_map_, rotation_map_);
}

template <domain::ObjectLabel Object>
TimeDependentMapOptions::MapType<Frame::Grid, Frame::Distorted>
TimeDependentMapOptions::grid_to_distorted_map(const bool use_identity) const {
  if (use_identity) {
    return std::make_unique<
        IdentityForComposition<Frame::Grid, Frame::Distorted>>(IdentityMap{});
  } else {
    const size_t index = get_index<Object>();
    return std::make_unique<
        CompressionMapForComposition<Frame::Grid, Frame::Distorted>>(
        gsl::at(size_maps_, index));
  }
}

template <domain::ObjectLabel Object>
TimeDependentMapOptions::MapType<Frame::Grid, Frame::Inertial>
TimeDependentMapOptions::everything_grid_to_inertial_map(
    const bool include_distorted_map) const {
  if (include_distorted_map) {
    const size_t index = get_index<Object>();
    return std::make_unique<EverythingMapForComposition>(
        gsl::at(size_maps_, index), expansion_map_, rotation_map_);
  } else {
    return std::make_unique<EverythingMapNoDistortedForComposition>(
        IdentityMap{}, expansion_map_, rotation_map_);
  }
}

template <domain::ObjectLabel Object>
size_t TimeDependentMapOptions::get_index() const {
  ASSERT(Object == domain::ObjectLabel::A or Object == domain::ObjectLabel::B,
         "Object label for TimeDependentMapOptions must be either A or B, not"
             << Object);
  return Object == domain::ObjectLabel::A ? 0_st : 1_st;
}

#define OBJECT(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template TimeDependentMapOptions::MapType<Frame::Grid, Frame::Distorted>     \
  TimeDependentMapOptions::grid_to_distorted_map<OBJECT(data)>(bool) const;    \
  template TimeDependentMapOptions::MapType<Frame::Grid, Frame::Inertial>      \
  TimeDependentMapOptions::everything_grid_to_inertial_map<OBJECT(data)>(bool) \
      const;                                                                   \
  template size_t TimeDependentMapOptions::get_index<OBJECT(data)>() const;

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (domain::ObjectLabel::A, domain::ObjectLabel::B))

#undef OBJECT
#undef INSTANTIATE

#define SOURCE_FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                    \
  template TimeDependentMapOptions::MapType<SOURCE_FRAME(data), \
                                            Frame::Inertial>    \
  TimeDependentMapOptions::frame_to_inertial_map<SOURCE_FRAME(data)>() const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (Frame::Grid, Frame::Distorted))

#undef SOURCE_FRAME
#undef INSTANTIATE

}  // namespace domain::creators::bco
