// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependence/UniformRotationAboutZAxis.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/MapInstantiationMacros.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain {
namespace creators {
namespace time_dependence {

template <size_t MeshDim>
UniformRotationAboutZAxis<MeshDim>::UniformRotationAboutZAxis(
    const double initial_time, const double angular_velocity,
    std::string function_of_time_name) noexcept
    : initial_time_(initial_time),
      angular_velocity_(angular_velocity),
      function_of_time_name_(std::move(function_of_time_name)) {}

template <size_t MeshDim>
std::unique_ptr<TimeDependence<MeshDim>>
UniformRotationAboutZAxis<MeshDim>::get_clone() const noexcept {
  return std::make_unique<UniformRotationAboutZAxis>(
      initial_time_, angular_velocity_, function_of_time_name_);
}

template <size_t MeshDim>
std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, MeshDim>>>
UniformRotationAboutZAxis<MeshDim>::block_maps(
    const size_t number_of_blocks) const noexcept {
  ASSERT(number_of_blocks > 0, "Must have at least one block to create.");
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, MeshDim>>>
      result{number_of_blocks};
  result[0] = std::make_unique<MapForComposition>(map_for_composition());
  for (size_t i = 1; i < number_of_blocks; ++i) {
    result[i] = result[0]->get_clone();
  }
  return result;
}

template <size_t MeshDim>
std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
UniformRotationAboutZAxis<MeshDim>::functions_of_time() const noexcept {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};
  // We use a `PiecewisePolynomial` with 2 derivs since some transformations
  // between different frames for moving meshes can require Hessians.
  result[function_of_time_name_] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time_,
          std::array<DataVector, 3>{{{0.0}, {angular_velocity_}, {0.0}}});
  return result;
}

/// \cond
template <>
auto UniformRotationAboutZAxis<2>::map_for_composition() const noexcept
    -> MapForComposition {
  return MapForComposition{domain::CoordinateMaps::TimeDependent::Rotation<2>{
      function_of_time_name_}};
}

template <>
auto UniformRotationAboutZAxis<3>::map_for_composition() const noexcept
    -> MapForComposition {
  using ProductMap = domain::CoordinateMaps::TimeDependent::ProductOf2Maps<
      domain::CoordinateMaps::TimeDependent::Rotation<2>,
      domain::CoordinateMaps::Identity<1>>;
  return MapForComposition{
      ProductMap{domain::CoordinateMaps::TimeDependent::Rotation<2>{
                     function_of_time_name_},
                 domain::CoordinateMaps::Identity<1>{}}};
}

template <size_t Dim>
bool operator==(const UniformRotationAboutZAxis<Dim>& lhs,
                const UniformRotationAboutZAxis<Dim>& rhs) noexcept {
  return lhs.initial_time_ == rhs.initial_time_ and
         lhs.angular_velocity_ == rhs.angular_velocity_ and
         lhs.function_of_time_name_ == rhs.function_of_time_name_;
}

template <size_t Dim>
bool operator!=(const UniformRotationAboutZAxis<Dim>& lhs,
                const UniformRotationAboutZAxis<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                   \
  template class UniformRotationAboutZAxis<GET_DIM(data)>;       \
  template bool operator==<GET_DIM(data)>(                       \
      const UniformRotationAboutZAxis<GET_DIM(data)>&,           \
      const UniformRotationAboutZAxis<GET_DIM(data)>&) noexcept; \
  template bool operator!=<GET_DIM(data)>(                       \
      const UniformRotationAboutZAxis<GET_DIM(data)>&,           \
      const UniformRotationAboutZAxis<GET_DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (2, 3))

#undef GET_DIM
#undef INSTANTIATION
/// \endcond
}  // namespace time_dependence
}  // namespace creators

using Identity = CoordinateMaps::Identity<1>;
using Rotation2d = CoordinateMaps::TimeDependent::Rotation<2>;
using Rotation3d =
    CoordinateMaps::TimeDependent::ProductOf2Maps<Rotation2d, Identity>;

template class CoordinateMaps::TimeDependent::ProductOf2Maps<Rotation2d,
                                                             Identity>;

INSTANTIATE_MAPS_FUNCTIONS(((Rotation2d), (Rotation3d)), (Frame::Grid),
                           (Frame::Inertial), (double, DataVector))

}  // namespace domain
