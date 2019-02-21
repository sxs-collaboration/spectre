// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/ElementMap.hpp"

#include "Domain/CoordinateMaps/CoordinateMap.hpp"  // IWYU pragma: keep
#include "Domain/Side.hpp"
#include "Parallel/PupStlCpp11.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"

/// \cond
template <size_t Dim, typename TargetFrame>
ElementMap<Dim, TargetFrame>::ElementMap(
    ElementId<Dim> element_id,
    std::unique_ptr<domain::CoordinateMapBase<Frame::Logical, TargetFrame, Dim>>
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

// For dual frame evolutions the ElementMap only goes to the grid frame
#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(r, data) \
  template class ElementMap<GET_DIM(data), GET_FRAME(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3),
                        (Frame::Inertial, Frame::Grid))

#undef GET_DIM
#undef GET_FRAME
#undef INSTANTIATION
/// \endcond
