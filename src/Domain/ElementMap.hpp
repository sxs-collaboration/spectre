// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace PUP {
class er;
}  // namespace PUP

/*!
 * \ingroup ComputationalDomainGroup
 * \brief The CoordinateMap for the Element from the Logical frame to the
 * `TargetFrame`
 *
 * An ElementMap takes a CoordinateMap for a Block and an ElementId as input,
 * and then "prepends" the correct affine map to the CoordinateMap so that the
 * map corresponds to the coordinate map for the Element rather than the Block.
 * This allows DomainCreators to only specify the maps for the Blocks without
 * worrying about how the domain may be decomposed beyond that.
 */
template <size_t Dim, typename TargetFrame>
class ElementMap {
 public:
  static constexpr size_t dim = Dim;
  using source_frame = Frame::ElementLogical;
  using target_frame = TargetFrame;

  /// \cond HIDDEN_SYMBOLS
  ElementMap() = default;
  /// \endcond

  ElementMap(ElementId<Dim> element_id,
             std::unique_ptr<domain::CoordinateMapBase<Frame::BlockLogical,
                                                       TargetFrame, Dim>>
                 block_map);

  const domain::CoordinateMapBase<Frame::BlockLogical, TargetFrame, Dim>&
  block_map() const {
    return *block_map_;
  }

  const ElementId<Dim>& element_id() const { return element_id_; }

  template <typename T>
  tnsr::I<T, Dim, TargetFrame> operator()(
      const tnsr::I<T, Dim, Frame::ElementLogical>& source_point) const {
    auto block_source_point =
        apply_affine_transformation_to_point(source_point);
    return block_map_->operator()(std::move(block_source_point));
  }

  template <typename T>
  tnsr::I<T, Dim, Frame::ElementLogical> inverse(
      tnsr::I<T, Dim, TargetFrame> target_point) const {
    auto block_source_point{
        block_map_->inverse(std::move(target_point)).value()};
    // Apply the affine map to the points
    tnsr::I<T, Dim, Frame::ElementLogical> source_point;
    for (size_t d = 0; d < Dim; ++d) {
      source_point.get(d) =
          block_source_point.get(d) * gsl::at(map_inverse_slope_, d) +
          gsl::at(map_inverse_offset_, d);
    }
    return source_point;
  }

  template <typename T>
  InverseJacobian<T, Dim, Frame::ElementLogical, TargetFrame> inv_jacobian(
      const tnsr::I<T, Dim, Frame::ElementLogical>& source_point) const {
    auto block_source_point =
        apply_affine_transformation_to_point(source_point);
    auto block_inv_jac =
        block_map_->inv_jacobian(std::move(block_source_point));
    InverseJacobian<T, Dim, Frame::ElementLogical, TargetFrame> inv_jac;
    for (size_t d = 0; d < Dim; ++d) {
      for (size_t i = 0; i < Dim; ++i) {
        inv_jac.get(d, i) =
            block_inv_jac.get(d, i) * gsl::at(inverse_jacobian_, d);
      }
    }
    return inv_jac;
  }

  template <typename T>
  Jacobian<T, Dim, Frame::ElementLogical, TargetFrame> jacobian(
      const tnsr::I<T, Dim, Frame::ElementLogical>& source_point) const {
    auto block_source_point =
        apply_affine_transformation_to_point(source_point);
    auto block_jac = block_map_->jacobian(std::move(block_source_point));
    Jacobian<T, Dim, Frame::ElementLogical, TargetFrame> jac;
    for (size_t d = 0; d < Dim; ++d) {
      for (size_t i = 0; i < Dim; ++i) {
        jac.get(i, d) = block_jac.get(i, d) * gsl::at(jacobian_, d);
      }
    }
    return jac;
  }

  // clang-tidy: do not use references
  void pup(PUP::er& p);  // NOLINT

 private:
  template <typename T>
  tnsr::I<T, Dim, Frame::BlockLogical> apply_affine_transformation_to_point(
      const tnsr::I<T, Dim, Frame::ElementLogical>& source_point) const {
    tnsr::I<T, Dim, Frame::BlockLogical> block_source_point;
    for (size_t d = 0; d < Dim; ++d) {
      block_source_point.get(d) = source_point.get(d) * gsl::at(map_slope_, d) +
                                  gsl::at(map_offset_, d);
    }
    return block_source_point;
  }

  std::unique_ptr<
      domain::CoordinateMapBase<Frame::BlockLogical, TargetFrame, Dim>>
      block_map_{nullptr};
  ElementId<Dim> element_id_{};
  // map_slope_[i] = 0.5 * (segment_ids[i].endpoint(Side::Upper) -
  //                        segment_ids[i].endpoint(Side::Lower))
  std::array<double, Dim> map_slope_{
      make_array<Dim>(std::numeric_limits<double>::signaling_NaN())};
  // map_offset_[i] = 0.5 * (segment_ids[i].endpoint(Side::Upper) +
  //                         segment_ids[i].endpoint(Side::Lower))
  std::array<double, Dim> map_offset_{
      make_array<Dim>(std::numeric_limits<double>::signaling_NaN())};
  // map_inverse_slope_[i] = 1.0 / map_slope_[i]
  std::array<double, Dim> map_inverse_slope_{
      make_array<Dim>(std::numeric_limits<double>::signaling_NaN())};
  // map_inverse_offset_[i] = -map_offset_[i] / map_slope_[i]
  std::array<double, Dim> map_inverse_offset_{
      make_array<Dim>(std::numeric_limits<double>::signaling_NaN())};
  // Note: The Jacobian is diagonal
  std::array<double, Dim> jacobian_{map_slope_};
  std::array<double, Dim> inverse_jacobian_{map_inverse_slope_};
};
