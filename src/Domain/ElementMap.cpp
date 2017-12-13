// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/ElementMap.hpp"

/// \cond
template <size_t Dim, typename TargetFrame>
ElementMap<Dim, TargetFrame>::ElementMap(
    ElementId<Dim> element_id,
    std::unique_ptr<CoordinateMapBase<Frame::Logical, TargetFrame, Dim>>
        block_map) noexcept
    : block_map_(std::move(block_map)),
      element_id_(std::move(element_id)),
      map_slope_{[](const ElementId<Dim>& id) {
        std::array<double, Dim> result{};
        for (size_t d = 0; d < Dim; ++d) {
          gsl::at(result, d) =
              0.5 * (gsl::at(id.segment_ids(), d).endpoint(Side::Upper) -
                     gsl::at(id.segment_ids(), d).endpoint(Side::Lower));
        }
        return result;
      }(element_id_)},
      map_offset_{[](const ElementId<Dim>& id) {
        std::array<double, Dim> result{};
        for (size_t d = 0; d < Dim; ++d) {
          gsl::at(result, d) =
              0.5 * (gsl::at(id.segment_ids(), d).endpoint(Side::Upper) +
                     gsl::at(id.segment_ids(), d).endpoint(Side::Lower));
        }
        return result;
      }(element_id_)},
      map_inverse_slope_{[this]() {
        std::array<double, Dim> result{};
        for (size_t d = 0; d < Dim; ++d) {
          gsl::at(result, d) = 1.0 / gsl::at(this->map_slope_, d);
        }
        return result;
      }()},
      map_inverse_offset_{[this]() {
        std::array<double, Dim> result{};
        for (size_t d = 0; d < Dim; ++d) {
          gsl::at(result, d) =
              -gsl::at(this->map_offset_, d) / gsl::at(this->map_slope_, d);
        }
        return result;
      }()},
      jacobian_{map_slope_},
      inverse_jacobian_{map_inverse_slope_} {}

template <size_t Dim, typename TargetFrame>
void ElementMap<Dim, TargetFrame>::pup(PUP::er& p) noexcept {
  p | block_map_;
  p | element_id_;
  p | map_slope_;
  p | map_offset_;
  p | map_inverse_slope_;
  p | map_inverse_offset_;
  p | jacobian_;
  p | inverse_jacobian_;
}

template class ElementMap<1, Frame::Inertial>;
template class ElementMap<2, Frame::Inertial>;
template class ElementMap<3, Frame::Inertial>;
// For dual frame evolutions the ElementMap only goes to the grid frame
template class ElementMap<1, Frame::Grid>;
template class ElementMap<2, Frame::Grid>;
template class ElementMap<3, Frame::Grid>;
/// \endcond
