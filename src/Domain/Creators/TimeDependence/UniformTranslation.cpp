// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/MapInstantiationMacros.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain {
namespace creators::time_dependence {
template <size_t MeshDim>
UniformTranslation<MeshDim>::UniformTranslation(
    const double initial_time,
    const std::optional<double> initial_expiration_delta_t,
    const std::array<double, MeshDim>& velocity,
    std::string function_of_time_name) noexcept
    : initial_time_(initial_time),
      initial_expiration_delta_t_(initial_expiration_delta_t),
      velocity_(velocity),
      function_of_time_name_(std::move(function_of_time_name)) {}

template <size_t MeshDim>
std::unique_ptr<TimeDependence<MeshDim>>
UniformTranslation<MeshDim>::get_clone() const noexcept {
  return std::make_unique<UniformTranslation>(
      initial_time_, initial_expiration_delta_t_, velocity_,
      function_of_time_name_);
}

template <size_t MeshDim>
std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, MeshDim>>>
UniformTranslation<MeshDim>::block_maps(
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
UniformTranslation<MeshDim>::functions_of_time() const noexcept {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};

  const double initial_expiration_time =
      initial_expiration_delta_t_ ? initial_time_ + *initial_expiration_delta_t_
                                  : std::numeric_limits<double>::max();

  // We use a `PiecewisePolynomial` with 2 derivs since some transformations
  // between different frames for moving meshes can require Hessians.
  DataVector velocity{MeshDim, 0.0};
  for (size_t i = 0; i < MeshDim; i++) {
    velocity[i] = gsl::at(velocity_, i);
  }
  result[function_of_time_name_] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time_,
          std::array<DataVector, 3>{{{MeshDim, 0.0}, velocity, {MeshDim, 0.0}}},
          initial_expiration_time);
  return result;
}

template <size_t MeshDim>
auto UniformTranslation<MeshDim>::map_for_composition() const noexcept
    -> MapForComposition {
  return MapForComposition{
      domain::CoordinateMaps::TimeDependent::Translation<MeshDim>{
          function_of_time_name_}};
}

template <size_t MeshDim>
std::string UniformTranslation<MeshDim>::default_function_name() noexcept {
  return "Translation";
}

template <size_t Dim>
bool operator==(const UniformTranslation<Dim>& lhs,
                const UniformTranslation<Dim>& rhs) noexcept {
  return lhs.initial_time_ == rhs.initial_time_ and
         lhs.initial_expiration_delta_t_ == rhs.initial_expiration_delta_t_ and
         lhs.velocity_ == rhs.velocity_ and
         lhs.function_of_time_name_ == rhs.function_of_time_name_;
}

template <size_t Dim>
bool operator!=(const UniformTranslation<Dim>& lhs,
                const UniformTranslation<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                            \
  template class UniformTranslation<GET_DIM(data)>;                       \
  template bool operator==                                                \
      <GET_DIM(data)>(const UniformTranslation<GET_DIM(data)>&,           \
                      const UniformTranslation<GET_DIM(data)>&) noexcept; \
  template bool operator!=                                                \
      <GET_DIM(data)>(const UniformTranslation<GET_DIM(data)>&,           \
                      const UniformTranslation<GET_DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace creators::time_dependence

template <size_t MeshDim>
using Translation = CoordinateMaps::TimeDependent::Translation<MeshDim>;

INSTANTIATE_MAPS_FUNCTIONS(((Translation<1>), (Translation<2>),
                            (Translation<3>)),
                           (Frame::Grid), (Frame::Inertial),
                           (double, DataVector))

}  // namespace domain
