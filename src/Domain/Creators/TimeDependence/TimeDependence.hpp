// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
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

namespace domain {
namespace creators {
namespace time_dependence {
template <size_t MeshDim>
class CubicScale;
template <size_t MeshDim>
class None;
template <size_t MeshDim>
class UniformTranslation;
}  // namespace time_dependence
}  // namespace creators
}  // namespace domain
/// \endcond

namespace domain {
namespace creators {
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
  using creatable_classes = tmpl::list<CubicScale<MeshDim>, None<MeshDim>,
                                       UniformTranslation<MeshDim>>;

  TimeDependence() = default;
  virtual ~TimeDependence() = 0;
  TimeDependence(const TimeDependence&) = default;
  TimeDependence& operator=(const TimeDependence&) = default;
  TimeDependence(TimeDependence&&) = default;
  TimeDependence& operator=(TimeDependence&&) = default;

  /// Returns a `std::unique_ptr` pointing to a copy of the `TimeDependence`.
  virtual auto get_clone() const noexcept
      -> std::unique_ptr<TimeDependence> = 0;

  /// Returns the coordinate maps from the `Frame::Grid` to the
  /// `Frame::Inertial` frame for each block.
  virtual auto block_maps(size_t number_of_blocks) const noexcept
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Inertial, MeshDim>>> = 0;

  /// Returns the functions of time for the domain.
  virtual auto functions_of_time() const noexcept -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> = 0;

  /// Returns `true` if the instance is `None`, meaning no time dependence.
  bool is_none() const noexcept {
    return dynamic_cast<const None<MeshDim>*>(this) != nullptr;
  }
};

template <size_t MeshDim>
TimeDependence<MeshDim>::~TimeDependence() = default;
}  // namespace time_dependence
}  // namespace creators
}  // namespace domain

#include "Domain/Creators/TimeDependence/CubicScale.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
