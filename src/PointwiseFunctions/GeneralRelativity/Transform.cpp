// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Transform.hpp"

#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace transform {

template <size_t VolumeDim, typename SrcFrame, typename DestFrame>
void to_different_frame(
    const gsl::not_null<tnsr::ii<DataVector, VolumeDim, DestFrame>*> dest,
    const tnsr::ii<DataVector, VolumeDim, SrcFrame>& src,
    const Jacobian<DataVector, VolumeDim, DestFrame, SrcFrame>&
        jacobian) noexcept {
  destructive_resize_components(dest, src.begin()->size());
  for (size_t i = 0; i < VolumeDim; ++i) {
    for (size_t j = i; j < VolumeDim; ++j) {  // symmetry
      // To avoid initializing to zero, split out k=0 here, and start
      // k loop below at k=1.
      dest->get(i, j) = jacobian.get(0, i) * jacobian.get(0, j) * src.get(0, 0);
      for (size_t p = 1; p < VolumeDim; ++p) {
        dest->get(i, j) +=
            jacobian.get(0, i) * jacobian.get(p, j) * src.get(0, p);
      }
      for (size_t k = 1; k < VolumeDim; ++k) {
        for (size_t p = 0; p < VolumeDim; ++p) {
          dest->get(i, j) +=
              jacobian.get(k, i) * jacobian.get(p, j) * src.get(k, p);
        }
      }
    }
  }
}

template <size_t VolumeDim, typename SrcFrame, typename DestFrame>
auto to_different_frame(const tnsr::ii<DataVector, VolumeDim, SrcFrame>& src,
                        const Jacobian<DataVector, VolumeDim, DestFrame,
                                       SrcFrame>& jacobian) noexcept
    -> tnsr::ii<DataVector, VolumeDim, DestFrame> {
  auto dest = make_with_value<tnsr::ii<DataVector, VolumeDim, DestFrame>>(
      src, std::numeric_limits<double>::signaling_NaN());
  to_different_frame(make_not_null(&dest), src, jacobian);
  return dest;
}

template <size_t VolumeDim, typename SrcFrame, typename DestFrame>
void first_index_to_different_frame(
    const gsl::not_null<tnsr::ijj<DataVector, VolumeDim, DestFrame>*> dest,
    const Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                 index_list<SpatialIndex<VolumeDim, UpLo::Lo, SrcFrame>,
                            SpatialIndex<VolumeDim, UpLo::Lo, DestFrame>,
                            SpatialIndex<VolumeDim, UpLo::Lo, DestFrame>>>& src,
    const Jacobian<DataVector, VolumeDim, DestFrame, SrcFrame>&
        jacobian) noexcept {
  destructive_resize_components(dest, src.begin()->size());
  for (size_t i = 0; i < VolumeDim; ++i) {
    for (size_t j = i; j < VolumeDim; ++j) {  // symmetry
      for (size_t k = 0; k < VolumeDim; ++k) {
        dest->get(k, i, j) = jacobian.get(0, k) * src.get(0, i, j);
        for (size_t p = 1; p < VolumeDim; ++p) {
          dest->get(k, i, j) += jacobian.get(p, k) * src.get(p, i, j);
        }
      }
    }
  }
}

template <size_t VolumeDim, typename SrcFrame, typename DestFrame>
auto first_index_to_different_frame(
    const Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                 index_list<SpatialIndex<VolumeDim, UpLo::Lo, SrcFrame>,
                            SpatialIndex<VolumeDim, UpLo::Lo, DestFrame>,
                            SpatialIndex<VolumeDim, UpLo::Lo, DestFrame>>>& src,
    const Jacobian<DataVector, VolumeDim, DestFrame, SrcFrame>&
        jacobian) noexcept -> tnsr::ijj<DataVector, VolumeDim, DestFrame> {
  auto dest = make_with_value<tnsr::ijj<DataVector, VolumeDim, DestFrame>>(
      src, std::numeric_limits<double>::signaling_NaN());
  first_index_to_different_frame(make_not_null(&dest), src, jacobian);
  return dest;
}

}  // namespace transform

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define SRCFRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DESTFRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                  \
  template void transform::to_different_frame(                                \
      const gsl::not_null<tnsr::ii<DataVector, DIM(data), DESTFRAME(data)>*>  \
          dest,                                                               \
      const tnsr::ii<DataVector, DIM(data), SRCFRAME(data)>& src,             \
      const Jacobian<DataVector, DIM(data), DESTFRAME(data), SRCFRAME(data)>& \
          jacobian) noexcept;                                                 \
  template auto transform::to_different_frame(                                \
      const tnsr::ii<DataVector, DIM(data), SRCFRAME(data)>& src,             \
      const Jacobian<DataVector, DIM(data), DESTFRAME(data), SRCFRAME(data)>& \
          jacobian) noexcept                                                  \
      ->tnsr::ii<DataVector, DIM(data), DESTFRAME(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Grid),
                        (Frame::Inertial))
GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Inertial),
                        (Frame::Grid))
#undef INSTANTIATE

#define INSTANTIATE(_, data)                                                  \
  template void transform::first_index_to_different_frame(                    \
      const gsl::not_null<tnsr::ijj<DataVector, DIM(data), DESTFRAME(data)>*> \
          dest,                                                               \
      const Tensor<                                                           \
          DataVector, tmpl::integral_list<std::int32_t, 2, 1, 1>,             \
          index_list<SpatialIndex<DIM(data), UpLo::Lo, SRCFRAME(data)>,       \
                     SpatialIndex<DIM(data), UpLo::Lo, DESTFRAME(data)>,      \
                     SpatialIndex<DIM(data), UpLo::Lo, DESTFRAME(data)>>>&    \
          src,                                                                \
      const Jacobian<DataVector, DIM(data), DESTFRAME(data), SRCFRAME(data)>& \
          jacobian) noexcept;                                                 \
  template auto transform::first_index_to_different_frame(                    \
      const Tensor<                                                           \
          DataVector, tmpl::integral_list<std::int32_t, 2, 1, 1>,             \
          index_list<SpatialIndex<DIM(data), UpLo::Lo, SRCFRAME(data)>,       \
                     SpatialIndex<DIM(data), UpLo::Lo, DESTFRAME(data)>,      \
                     SpatialIndex<DIM(data), UpLo::Lo, DESTFRAME(data)>>>&    \
          src,                                                                \
      const Jacobian<DataVector, DIM(data), DESTFRAME(data), SRCFRAME(data)>& \
          jacobian) noexcept                                                  \
      ->tnsr::ijj<DataVector, DIM(data), DESTFRAME(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::ElementLogical),
                        (Frame::Grid))

#undef DIM
#undef SRCFRAME
#undef DESTFRAME
#undef INSTANTIATE
