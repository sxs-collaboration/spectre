// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
template <typename SourceFrame, typename TargetFrame, size_t Dim>
class CoordinateMapBase;
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
}  // namespace domain
namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame

namespace domain::creators::time_dependence {
template <size_t MeshDim>
class CubicScale;
template <size_t MeshDim>
class None;
class SphericalCompression;
template <size_t MeshDim>
class UniformRotationAboutZAxis;
template <size_t MeshDim>
class UniformTranslation;
template <typename TimeDependenceCompTag0, typename... TimeDependenceCompTags>
class Composition;
template <typename TimeDep, size_t Suffix>
struct TimeDependenceCompositionTag;
}  // namespace domain::creators::time_dependence
/// \endcond

namespace domain::creators {
/// \ingroup ComputationalDomainGroup
/// \brief Classes and functions for adding time dependence to a domain.
namespace time_dependence {
/// \brief The abstract base class off of which specific classes for adding
/// time dependence into a domain creator must inherit off of.
///
/// The simplest examples of a `TimeDependence` are `None` and
/// `UniformTranslation`. The `None` class is treated in a special manner to
/// communicate to the code that the domain is time-independent.
template <size_t MeshDim>
struct TimeDependence {
 private:
  using creatable_classes_1d = tmpl::list<>;
  using creatable_classes_2d = tmpl::list<UniformRotationAboutZAxis<2>>;
  using creatable_classes_3d = tmpl::list<
      SphericalCompression, UniformRotationAboutZAxis<3>,
      Composition<
          TimeDependenceCompositionTag<CubicScale<3>,
                                       std::numeric_limits<size_t>::max()>,
          TimeDependenceCompositionTag<UniformRotationAboutZAxis<3>,
                                       std::numeric_limits<size_t>::max()>>>;
  using creatable_classes_any_dim =
      tmpl::list<CubicScale<MeshDim>, None<MeshDim>,
                 UniformTranslation<MeshDim>>;

 public:
  using creatable_classes =
      tmpl::append<creatable_classes_any_dim,
                   tmpl::conditional_t<
                       MeshDim == 1, creatable_classes_1d,
                       tmpl::conditional_t<MeshDim == 2, creatable_classes_2d,
                                           creatable_classes_3d>>>;

  TimeDependence() = default;
  virtual ~TimeDependence() = 0;
  TimeDependence(const TimeDependence&) = default;
  TimeDependence& operator=(const TimeDependence&) = default;
  TimeDependence(TimeDependence&&) = default;
  TimeDependence& operator=(TimeDependence&&) = default;

  /// Returns a `std::unique_ptr` pointing to a copy of the `TimeDependence`.
  virtual auto get_clone() const -> std::unique_ptr<TimeDependence> = 0;

  /// Returns the coordinate maps from the `Frame::Grid` to the
  /// `Frame::Inertial` frame for each block.
  virtual auto
  block_maps(size_t number_of_blocks) const -> std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, MeshDim>>> = 0;

  /// Returns the functions of time for the domain.
  virtual auto functions_of_time() const -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> = 0;

  /// Returns `true` if the instance is `None`, meaning no time dependence.
  bool is_none() const {
    return dynamic_cast<const None<MeshDim>*>(this) != nullptr;
  }
};

template <size_t MeshDim>
TimeDependence<MeshDim>::~TimeDependence() = default;
}  // namespace time_dependence
}  // namespace creators::namespace domain

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/Creators/TimeDependence/Composition.hpp"
#include "Domain/Creators/TimeDependence/CubicScale.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/SphericalCompression.hpp"
#include "Domain/Creators/TimeDependence/UniformRotationAboutZAxis.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
