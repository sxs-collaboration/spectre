// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"

#include <array>
#include <cstddef>
#include <memory>
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
#include "ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain {
namespace creators {
namespace time_dependence {
namespace {
std::array<std::string, 3> default_function_names_impl() noexcept {
  return {{"TranslationX", "TranslationY", "TranslationZ"}};
}
}  // namespace

template <size_t MeshDim>
UniformTranslation<MeshDim>::UniformTranslation(
    const double initial_time, const std::array<double, MeshDim>& velocity,
    std::array<std::string, MeshDim> functions_of_time_names) noexcept
    : initial_time_(initial_time),
      velocity_(velocity),
      functions_of_time_names_(std::move(functions_of_time_names)) {}

template <size_t MeshDim>
std::unique_ptr<TimeDependence<MeshDim>>
UniformTranslation<MeshDim>::get_clone() const noexcept {
  return std::make_unique<UniformTranslation>(initial_time_, velocity_,
                                              functions_of_time_names_);
}

template <size_t MeshDim>
std::vector<std::unique_ptr<
    domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, MeshDim>>>
UniformTranslation<MeshDim>::block_maps(const size_t number_of_blocks) const
    noexcept {
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
  // We use a `PiecewisePolynomial` with 2 derivs since some transformations
  // between different frames for moving meshes can require Hessians.
  result[functions_of_time_names_[0]] =
      std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time_,
          std::array<DataVector, 3>{{{0.0}, {velocity_[0]}, {0.0}}});
  if (MeshDim > 1) {
    result[gsl::at(functions_of_time_names_, 1)] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
            initial_time_,
            std::array<DataVector, 3>{{{0.0}, {gsl::at(velocity_, 1)}, {0.0}}});
  }
  if (MeshDim > 2) {
    result[gsl::at(functions_of_time_names_, 2)] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
            initial_time_,
            std::array<DataVector, 3>{{{0.0}, {gsl::at(velocity_, 2)}, {0.0}}});
  }
  return result;
}

/// \cond
template <>
auto UniformTranslation<1>::map_for_composition() const noexcept
    -> MapForComposition {
  return MapForComposition{
      domain::CoordMapsTimeDependent::Translation{functions_of_time_names_[0]}};
}

template <>
auto UniformTranslation<2>::map_for_composition() const noexcept
    -> MapForComposition {
  using ProductMap =
      domain::CoordMapsTimeDependent::ProductOf2Maps<Translation, Translation>;
  return MapForComposition{
      ProductMap{Translation{functions_of_time_names_[0]},
                 Translation{functions_of_time_names_[1]}}};
}

template <>
auto UniformTranslation<3>::map_for_composition() const noexcept
    -> MapForComposition {
  using ProductMap =
      domain::CoordMapsTimeDependent::ProductOf3Maps<Translation, Translation,
                                                     Translation>;
  return MapForComposition{
      ProductMap{Translation{functions_of_time_names_[0]},
                 Translation{functions_of_time_names_[1]},
                 Translation{functions_of_time_names_[2]}}};
}

template <>
std::array<std::string, 1>
UniformTranslation<1>::default_function_names() noexcept {
  return {{default_function_names_impl()[0]}};
}

template <>
std::array<std::string, 2>
UniformTranslation<2>::default_function_names() noexcept {
  return {{default_function_names_impl()[0], default_function_names_impl()[1]}};
}

template <>
std::array<std::string, 3>
UniformTranslation<3>::default_function_names() noexcept {
  return default_function_names_impl();
}

template <size_t Dim>
bool operator==(const UniformTranslation<Dim>& lhs,
                const UniformTranslation<Dim>& rhs) noexcept {
  return lhs.initial_time_ == rhs.initial_time_ and
         lhs.velocity_ == rhs.velocity_ and
         lhs.functions_of_time_names_ == rhs.functions_of_time_names_;
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
/// \endcond
}  // namespace time_dependence
}  // namespace creators

using Translation = CoordMapsTimeDependent::Translation;
using Translation2d =
    CoordMapsTimeDependent::ProductOf2Maps<Translation, Translation>;
using Translation3d =
    CoordMapsTimeDependent::ProductOf3Maps<Translation, Translation,
                                           Translation>;

template class CoordMapsTimeDependent::ProductOf2Maps<Translation, Translation>;
template class CoordMapsTimeDependent::ProductOf3Maps<Translation, Translation,
                                                      Translation>;

INSTANTIATE_MAPS_FUNCTIONS(((Translation), (Translation2d), (Translation3d)),
                           (Frame::Grid), (Frame::Inertial),
                           (double, DataVector))

}  // namespace domain
