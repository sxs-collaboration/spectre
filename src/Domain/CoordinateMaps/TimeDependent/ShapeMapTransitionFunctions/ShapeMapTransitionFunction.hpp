// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <optional>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "Parallel/CharmPupable.hpp"

namespace domain::CoordinateMaps::ShapeMapTransitionFunctions {

/*!
 * \brief Abstract base class for the transition functions used by the
 * domain::CoordinateMaps::TimeDependent::Shape map.
 *
 * \details This base class defines the required methods of a transition
 * function used by the shape map. Different domains require the shape map to
 * fall off towards the boundary in different ways. This behavior is controlled
 * by the transition function. It is also needed to find the inverse of the
 * shape map. Since the shape map preserves angles, the problem of finding its
 * inverse reduces to the 1-dimensional problem of finding the original radius
 * from the mapped radius. The mapped radius \f$\tilde{r}\f$ is related to the
 * original \f$r\f$ radius by:
 * \f{equation}{\label{eq:shape_map_radius}
 *  \tilde{r} = r (1 - f(r,\theta,\phi) \sum_{lm} \lambda_{lm}(t)Y_{lm}(\theta,
 *  \phi)),
 *  \f}
 * where \f$f(r,\theta,\phi)\f$ is the transition function. Depending
 * on the format of the transition function, it should be possible to
 * analytically derive this map's inverse because it preserves angles and shifts
 * only the radius of each point. Otherwise the inverse has to be computed
 * numerically.
 *
 * The transition function must also be able to compute the gradient and the
 * value of the function divided by the radius. Care must be taken that this
 * does not divide by zero.
 *
 * All member functions with the exception of `original_radius_over_radius`
 * exist as overloads for types `double` and `DataVector` so that they work with
 * the templated shape map methods calling them. To avoid code duplication these
 * can be forwarded to templated implementation methods held by the derived
 * classes only.
 *
 * For an example, see SphereTransition.
 *
 * #### Design Decisions:
 *
 * It was decided to make the ShapeMapTransitionFunction an abstract base class
 * with overloads for types `double` and `DataVector` corresponding to the
 * template parameter `T` of the shape map's methods. The shape map holds a
 * `unique_ptr` to this abstract class using a common dynamic dispatch design
 * pattern. This approach avoids templating the shape map all together.
 *
 * An alternative approach would be to directly template the transition
 * functions onto the shape map so that no abstract base class is necessary.
 * These approaches can also be combined by making the transition function an
 * abstract base class but also templating it onto the shape map. In this way
 * the shape map does not need to hold a `unique_ptr` but can hold the
 * transition function directly as a member.
 */
class ShapeMapTransitionFunction : public PUP::able {
 public:
  ShapeMapTransitionFunction() = default;

  /// @{
  /*!
   * Evaluate the transition function at the Cartesian coordinates
   *`source_coords`.
   */
  virtual double operator()(
      const std::array<double, 3>& source_coords) const = 0;
  virtual DataVector operator()(
      const std::array<DataVector, 3>& source_coords) const = 0;
  /// @}

  /*!
   * Given the mapped coordinates `target_coords` and the corresponding
   * spherical harmonic expansion \f$\sum_{lm} \lambda_{lm}(t)Y_{lm}\f$,
   * `distorted_radius`, this method evaluates the original radius from the
   * mapped radius by inverting the domain::CoordinateMaps::TimeDependent::Shape
   * map. It also divides by the mapped radius to simplify calculations in the
   * shape map.
   */
  virtual std::optional<double> original_radius_over_radius(
      const std::array<double, 3>& target_coords,
      double distorted_radius) const = 0;
  /*!
   * Evaluate the transition function at the Cartesian coordinates divided by
   * the radius. All divisions over the radius which could potentially be zero
   * divisions are left to the transition function which uses its knowledge
   * of the domain to do this safely or throw an appropriate error.
   */
  /// @{
  virtual double map_over_radius(
      const std::array<double, 3>& source_coords) const = 0;
  virtual DataVector map_over_radius(
      const std::array<DataVector, 3>& source_coords) const = 0;
  /// @}

  /*!
   * Evaluate the gradient of the transition function with respect to the
   * Cartesian coordinates x, y and z at the Cartesian coordinates
   * `source_coords`.
   */
  /// @{
  virtual std::array<double, 3> gradient(
      const std::array<double, 3>& source_coords) const = 0;
  virtual std::array<DataVector, 3> gradient(
      const std::array<DataVector, 3>& source_coords) const = 0;
  /// @}
  virtual bool operator==(const ShapeMapTransitionFunction& other) const = 0;
  virtual bool operator!=(const ShapeMapTransitionFunction& other) const = 0;

  WRAPPED_PUPable_abstract(ShapeMapTransitionFunction);
  explicit ShapeMapTransitionFunction(CkMigrateMessage* m) : PUP::able(m) {}
};
}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
