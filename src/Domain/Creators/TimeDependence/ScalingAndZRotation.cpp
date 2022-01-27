// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependence/ScalingAndZRotation.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/MapInstantiationMacros.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain::creators::time_dependence {

template <size_t MeshDim>
ScalingAndZRotation<MeshDim>::ScalingAndZRotation(
    const double initial_time, const double angular_velocity,
    const double outer_boundary, bool use_linear_scaling,
    const std::array<double, 2>& initial_expansion,
    const std::array<double, 2>& velocity,
    const std::array<double, 2>& acceleration)
    : initial_time_(initial_time),
      angular_velocity_(angular_velocity),
      outer_boundary_(outer_boundary),
      use_linear_scaling_(use_linear_scaling),
      initial_expansion_(initial_expansion),
      velocity_(velocity),
      acceleration_(acceleration) {
  // If we are using linear scaling, then CubicScale names must be the same
  if (use_linear_scaling_) {
    functions_of_time_names_[0] = "CubicScale";
    functions_of_time_names_[1] = "CubicScale";
  }
}

template <size_t MeshDim>
std::unique_ptr<TimeDependence<MeshDim>>
ScalingAndZRotation<MeshDim>::get_clone() const {
  return std::make_unique<ScalingAndZRotation>(
      initial_time_, angular_velocity_, outer_boundary_, use_linear_scaling_,
      initial_expansion_, velocity_, acceleration_);
}

template <size_t MeshDim>
std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, MeshDim>>>
ScalingAndZRotation<MeshDim>::block_maps_grid_to_inertial(
    const size_t number_of_blocks) const {
  ASSERT(number_of_blocks > 0, "Must have at least one block to create.");
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, MeshDim>>>
      result{number_of_blocks};
  result[0] = std::make_unique<GridToInertialMap>(grid_to_inertial_map());
  for (size_t i = 1; i < number_of_blocks; ++i) {
    result[i] = result[0]->get_clone();
  }
  return result;
}

template <size_t MeshDim>
std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, MeshDim>>>
ScalingAndZRotation<MeshDim>::block_maps_grid_to_distorted(
    const size_t number_of_blocks) const {
  ASSERT(number_of_blocks > 0, "Must have at least one block to create.");
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, MeshDim>>>
      result{number_of_blocks};
  result[0] = std::make_unique<GridToDistortedMap>(grid_to_distorted_map());
  for (size_t i = 1; i < number_of_blocks; ++i) {
    result[i] = result[0]->get_clone();
  }
  return result;
}

template <size_t MeshDim>
std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, MeshDim>>>
ScalingAndZRotation<MeshDim>::block_maps_distorted_to_inertial(
    const size_t number_of_blocks) const {
  ASSERT(number_of_blocks > 0, "Must have at least one block to create.");
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, MeshDim>>>
      result{number_of_blocks};
  result[0] =
      std::make_unique<DistortedToInertialMap>(distorted_to_inertial_map());
  for (size_t i = 1; i < number_of_blocks; ++i) {
    result[i] = result[0]->get_clone();
  }
  return result;
}

template <size_t MeshDim>
std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
ScalingAndZRotation<MeshDim>::functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};

  // Functions of time don't expire by default
  std::unordered_map<std::string, double> expiration_times{};
  for (auto& this_name : functions_of_time_names_) {
    expiration_times[this_name] = std::numeric_limits<double>::infinity();
  }

  // If we have control systems, overwrite these expiration times with the ones
  // supplied by the control system
  for (auto& [name, expr_time] : initial_expiration_times) {
    if (expiration_times.count(name) == 1) {
      expiration_times[name] = expr_time;
    }
  }

  result[functions_of_time_names_[2]] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time_,
          std::array<DataVector, 4>{{{0.0}, {angular_velocity_}, {0.0}, {0.0}}},
          expiration_times[functions_of_time_names_[2]]);
  // If we are using linear scaling, the scaling function of time
  // names will be the same so the first assignment will be
  // overwritten by the second assignment.  This is expected.  Use a
  // 3rd deriv function of time so that it can be used with a control
  // system.
  for (size_t i = 0; i < functions_of_time_names_.size() - 1; i++) {
    result[gsl::at(functions_of_time_names_, i)] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
            initial_time_,
            std::array<DataVector, 4>{{{gsl::at(initial_expansion_, i)},
                                       {gsl::at(velocity_, i)},
                                       {gsl::at(acceleration_, i)},
                                       {0.0}}},
            expiration_times.at(gsl::at(functions_of_time_names_, i)));
  }
  return result;
}

template <>
auto ScalingAndZRotation<2>::grid_to_inertial_map() const -> GridToInertialMap {
  return GridToInertialMap{
      CubicScaleMap{outer_boundary_, functions_of_time_names_[0],
                    functions_of_time_names_[1]},
      domain::CoordinateMaps::TimeDependent::Rotation<2>{
          functions_of_time_names_[2]}};
}

template <>
auto ScalingAndZRotation<3>::grid_to_inertial_map() const -> GridToInertialMap {
  using ProductRotationMap =
      domain::CoordinateMaps::TimeDependent::ProductOf2Maps<
          domain::CoordinateMaps::TimeDependent::Rotation<2>,
          domain::CoordinateMaps::Identity<1>>;
  return GridToInertialMap{
      CubicScaleMap{outer_boundary_, functions_of_time_names_[0],
                    functions_of_time_names_[1]},
      ProductRotationMap{domain::CoordinateMaps::TimeDependent::Rotation<2>{
                             functions_of_time_names_[2]},
                         domain::CoordinateMaps::Identity<1>{}}};
}

template <size_t MeshDim>
auto ScalingAndZRotation<MeshDim>::grid_to_distorted_map() const
    -> GridToDistortedMap {
  return GridToDistortedMap{CubicScaleMap{outer_boundary_,
                                          functions_of_time_names_[0],
                                          functions_of_time_names_[1]}};
}

template <>
auto ScalingAndZRotation<2>::distorted_to_inertial_map() const
    -> DistortedToInertialMap {
  return DistortedToInertialMap{
      domain::CoordinateMaps::TimeDependent::Rotation<2>{
          functions_of_time_names_[2]}};
}

template <>
auto ScalingAndZRotation<3>::distorted_to_inertial_map() const
    -> DistortedToInertialMap {
  using ProductRotationMap =
      domain::CoordinateMaps::TimeDependent::ProductOf2Maps<
          domain::CoordinateMaps::TimeDependent::Rotation<2>,
          domain::CoordinateMaps::Identity<1>>;
  return DistortedToInertialMap{
      ProductRotationMap{domain::CoordinateMaps::TimeDependent::Rotation<2>{
                             functions_of_time_names_[2]},
                         domain::CoordinateMaps::Identity<1>{}}};
}

template <size_t Dim>
bool operator==(const ScalingAndZRotation<Dim>& lhs,
                const ScalingAndZRotation<Dim>& rhs) {
  return lhs.initial_time_ == rhs.initial_time_ and
         lhs.angular_velocity_ == rhs.angular_velocity_ and
         lhs.outer_boundary_ == rhs.outer_boundary_ and
         lhs.use_linear_scaling_ == rhs.use_linear_scaling_ and
         lhs.initial_expansion_ == rhs.initial_expansion_ and
         lhs.velocity_ == rhs.velocity_ and
         lhs.acceleration_ == rhs.acceleration_;
}

template <size_t Dim>
bool operator!=(const ScalingAndZRotation<Dim>& lhs,
                const ScalingAndZRotation<Dim>& rhs) {
  return not(lhs == rhs);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                    \
  template class ScalingAndZRotation<GET_DIM(data)>;              \
  template bool operator==                                        \
      <GET_DIM(data)>(const ScalingAndZRotation<GET_DIM(data)>&,  \
                      const ScalingAndZRotation<GET_DIM(data)>&); \
  template bool operator!=                                        \
      <GET_DIM(data)>(const ScalingAndZRotation<GET_DIM(data)>&,  \
                      const ScalingAndZRotation<GET_DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATION, (2, 3))

#undef INSTANTIATION
#undef GET_DIM
}  // namespace domain::creators::time_dependence
