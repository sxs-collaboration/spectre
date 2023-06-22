// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
template <typename Frame>
class Strahlkorper;
namespace domain {
enum class ObjectLabel;
}  // namespace domain
/// \endcond

/// \ingroup ControlSystemGroup
/// All tags that will be used in the LinkedMessageQueue's within control
/// systems.
///
/// These tags will be used to retrieve the results of the measurements that
/// were sent to the control system which have been placed inside a
/// LinkedMessageQueue.
namespace control_system::QueueTags {
/// \ingroup ControlSystemGroup
/// Holds the centers of each horizon from measurements as DataVectors
template <::domain::ObjectLabel Horizon>
struct Center {
  using type = DataVector;
};

/// \ingroup ControlSystemGroup
/// Holds a full strahlkorper from measurements that represents a horizon
template <typename Frame>
struct Horizon {
  using type = ::Strahlkorper<Frame>;
};

/// \ingroup ControlSystemGroup
/// Holds a full strahlkorper from measurements for the excision surface
template <typename Frame>
struct ExcisionSurface {
  using type = ::Strahlkorper<Frame>;
};

/// \ingroup ControlSystemGroup
/// Holds the lapse on the `ExcisionSurface`
struct LapseOnExcisionSurface {
  using type = Scalar<DataVector>;
};

/*!
 * \ingroup ControlSystemGroup
 * \brief Holds a quantity that's similar to the shift, but isn't the shift, on
 * the `ExcisionSurface`.
 *
 * \details This holds
 *
 * \f{equation}{
 * \beta^i \frac{\partial x^\hat{i}}{\partial x^i} =
 * \hat{beta}^\hat{i} + \frac{\partial x^\hat{i}}{\partial t}
 * \f}
 *
 * where hatted quantities are in the distorted frame and non-hatted quantities
 * are in the grid frame.
 */
template <typename Frame>
struct ShiftyQuantity {
  using type = tnsr::I<DataVector, 3, Frame>;
};

/// \ingroup ControlSystemGroup
/// Holds the spatial metric on the `ExcisionSurface`
template <typename Frame>
struct SpatialMetricOnExcisionSurface {
  using type = tnsr::ii<DataVector, 3, Frame>;
};

/// \ingroup ControlSystemGroup
/// Holds the inverse spatial metric on the `ExcisionSurface`
template <typename Frame>
struct InverseSpatialMetricOnExcisionSurface {
  using type = tnsr::II<DataVector, 3, Frame>;
};

/*!
 * \ingroup ControlSystemGroup
 * \brief A queue tag that holds a TaggedTuple of all quantities needed for the
 * excision measurement of size control.
 *
 * \details Holds the following queue tags in a TaggedTuple in order:
 *
 * - `control_system::QueueTags::ExcisionSurface`
 * - `control_system::QueueTags::LapseOnExcisionSurface`
 * - `control_system::QueueTags::ShiftyQuantity`
 * - `control_system::QueueTags::SpatialMetricOnExcisionSurface`
 * - `control_system::QueueTags::InverseSpatialMetricOnExcisionSurface`
 */
template <typename Frame>
struct SizeExcisionQuantities {
  using type =
      tuples::TaggedTuple<ExcisionSurface<Frame>, LapseOnExcisionSurface,
                          ShiftyQuantity<Frame>,
                          SpatialMetricOnExcisionSurface<Frame>,
                          InverseSpatialMetricOnExcisionSurface<Frame>>;
};

/*!
 * \ingroup ControlSystemGroup
 * \brief A queue tag that holds a TaggedTuple of all quantities needed for the
 * horizon measurement of size control.
 *
 * \details Holds the following queue tags in a TaggedTuple in order:
 *
 * - `StrahlkorperTags::Strahlkorper`
 * - `::Tags::dt<StrahlkorperTags::Strahlkorper>`
 */
template <typename Frame>
struct SizeHorizonQuantities {
  using type =
      tuples::TaggedTuple<StrahlkorperTags::Strahlkorper<Frame>,
                          ::Tags::dt<StrahlkorperTags::Strahlkorper<Frame>>>;
};
}  // namespace control_system::QueueTags
