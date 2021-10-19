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
template <size_t MeshDim, size_t Index>
UniformTranslation<MeshDim, Index>::UniformTranslation(
    const double initial_time, const std::array<double, MeshDim>& velocity)
    : initial_time_(initial_time), velocity_(velocity) {}

template <size_t MeshDim, size_t Index>
std::unique_ptr<TimeDependence<MeshDim>>
UniformTranslation<MeshDim, Index>::get_clone() const {
  return std::make_unique<UniformTranslation>(initial_time_, velocity_);
}

template <size_t MeshDim, size_t Index>
std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, MeshDim>>>
UniformTranslation<MeshDim, Index>::block_maps(
    const size_t number_of_blocks) const {
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

template <size_t MeshDim, size_t Index>
std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
UniformTranslation<MeshDim, Index>::functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};

  // We use a `PiecewisePolynomial` with 2 derivs since some transformations
  // between different frames for moving meshes can require Hessians.
  DataVector velocity{MeshDim, 0.0};
  for (size_t i = 0; i < MeshDim; i++) {
    velocity[i] = gsl::at(velocity_, i);
  }

  // Functions of time don't expire by default
  double expiration_time = std::numeric_limits<double>::infinity();

  // If we have control systems, overwrite the expiration time with the one
  // supplied by the control system
  if (initial_expiration_times.count(function_of_time_name_) == 1) {
    expiration_time = initial_expiration_times.at(function_of_time_name_);
  }

  result[function_of_time_name_] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time_,
          std::array<DataVector, 3>{{{MeshDim, 0.0}, velocity, {MeshDim, 0.0}}},
          expiration_time);
  return result;
}

template <size_t MeshDim, size_t Index>
auto UniformTranslation<MeshDim, Index>::map_for_composition() const
    -> MapForComposition {
  return MapForComposition{
      domain::CoordinateMaps::TimeDependent::Translation<MeshDim>{
          function_of_time_name_}};
}

template <size_t Dim, size_t Index>
bool operator==(const UniformTranslation<Dim, Index>& lhs,
                const UniformTranslation<Dim, Index>& rhs) {
  return lhs.initial_time_ == rhs.initial_time_ and
         lhs.velocity_ == rhs.velocity_;
}

template <size_t Dim, size_t Index>
bool operator!=(const UniformTranslation<Dim, Index>& lhs,
                const UniformTranslation<Dim, Index>& rhs) {
  return not(lhs == rhs);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INDEX(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(r, data)                                   \
  template class UniformTranslation<GET_DIM(data), INDEX(data)>; \
  template bool operator==<GET_DIM(data), INDEX(data)>(          \
      const UniformTranslation<GET_DIM(data), INDEX(data)>&,     \
      const UniformTranslation<GET_DIM(data), INDEX(data)>&);    \
  template bool operator!=<GET_DIM(data), INDEX(data)>(          \
      const UniformTranslation<GET_DIM(data), INDEX(data)>&,     \
      const UniformTranslation<GET_DIM(data), INDEX(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (0, 1, 2))

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
