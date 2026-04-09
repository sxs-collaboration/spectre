// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace ray_tracing {

/*!
 * \brief Abstract base class for background spacetimes in the ray tracer.
 *
 * Derived classes have to provide spacetime quantities at a given point on
 * request, e.g. by evaluating an analytic spacetime or by interpolating numeric
 * data from a file. The `initialize` function can be used to set up the
 * background spacetime, e.g. by reading data from a file. Then, the background
 * spacetime should be valid within the bounds returned by the `time_bounds`
 * function.
 */
class BackgroundSpacetime : public PUP::able {
 protected:
  static constexpr size_t Dim = 3;
  using DataType = double;
  using Frame = ::Frame::Inertial;
  using DerivLapse =
      ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<Dim>, Frame>;
  using DerivShift =
      ::Tags::deriv<gr::Tags::Shift<DataType, Dim>, tmpl::size_t<Dim>, Frame>;
  using DerivInvSpatialMetric =
      ::Tags::deriv<gr::Tags::InverseSpatialMetric<DataType, Dim, Frame>,
                    tmpl::size_t<Dim>, Frame>;
  using DerivSpatialMetric =
      ::Tags::deriv<gr::Tags::SpatialMetric<DataType, Dim, Frame>,
                    tmpl::size_t<Dim>, Frame>;

  BackgroundSpacetime() = default;

 public:
  ~BackgroundSpacetime() override = default;

  /// \cond
  explicit BackgroundSpacetime(CkMigrateMessage* msg) : PUP::able(msg) {}
  WRAPPED_PUPable_abstract(BackgroundSpacetime);
  /// \endcond

  /// Copies the background spacetime. Must call `initialize` on the clone
  /// before using it.
  virtual auto get_clone() const -> std::unique_ptr<BackgroundSpacetime> = 0;

  /*!
   * \brief Initialize the background spacetime, e.g. by reading data from a
   * file.
   *
   * This function is called before the first call to `variables()`. It is
   * valid to call `initialize` again with new time bounds. Derived classes
   * must guarantee that the `variables` function can be called from other
   * threads while `initialize` is running (e.g. loading new data from files),
   * but only with times within the overlap of the previous and the new time
   * bounds.
   *
   * \param time_bounds The time bounds for which to initialize the
   * background spacetime. The spacetime should be valid for all times in this
   * range.
   */
  virtual void initialize(
      [[maybe_unused]] const std::array<double, 2> time_bounds) {}

  /// Time bounds for which the background spacetime is valid. The `variables`
  /// function can be called for any time in this range (inclusive).
  virtual std::array<double, 2> time_bounds() const {
    return {-std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()};
  }

  /// These tags can be retrieved from the background spacetime. They are
  /// required to evaluate the `gr::geodesic_equation`.
  using tags = tmpl::list<gr::Tags::Lapse<DataType>, DerivLapse,
                          gr::Tags::Shift<DataType, Dim, Frame>, DerivShift,
                          gr::Tags::InverseSpatialMetric<DataType, Dim, Frame>,
                          DerivInvSpatialMetric,
                          gr::Tags::ExtrinsicCurvature<DataType, Dim, Frame>>;

  /*!
   * \brief Returns all spacetime variables at a given point in space and time.
   *
   * This function must be thread-safe.
   *
   * \param x Spatial coordinates
   * \param t Time
   * \param block_order Optional priority order for processing blocks during
   * interpolation. If specified, it will be updated to push the block in which
   * the point was found to the front. Can be empty, in which case it will be
   * initially set to the default order. See `block_logical_coordinates` for
   * more details.
   */
  virtual tuples::tagged_tuple_from_typelist<tags> variables(
      const tnsr::I<DataType, Dim, Frame>& x, double t,
      std::optional<gsl::not_null<std::vector<size_t>*>> block_order =
          std::nullopt) const = 0;
};

}  // namespace ray_tracing
