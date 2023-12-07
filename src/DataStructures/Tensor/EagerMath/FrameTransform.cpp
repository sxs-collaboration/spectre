// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Tensor/EagerMath/FrameTransform.hpp"

#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace transform {

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
void to_different_frame(
    const gsl::not_null<tnsr::ii<DataType, VolumeDim, DestFrame>*> dest,
    const tnsr::ii<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian) {
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

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
auto to_different_frame(
    const tnsr::ii<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian)
    -> tnsr::ii<DataType, VolumeDim, DestFrame> {
  auto dest = make_with_value<tnsr::ii<DataType, VolumeDim, DestFrame>>(
      src, std::numeric_limits<double>::signaling_NaN());
  to_different_frame(make_not_null(&dest), src, jacobian);
  return dest;
}

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
void to_different_frame(
    const gsl::not_null<Scalar<DataType>*> dest, const Scalar<DataType>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& /*jacobian*/,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
    /*inv_jacobian*/) {
  *dest = src;
}

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
auto to_different_frame(
    Scalar<DataType> src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& /*jacobian*/,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
    /*inv_jacobian*/) -> Scalar<DataType> {
  return src;
}

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
void to_different_frame(
    const gsl::not_null<tnsr::I<DataType, VolumeDim, DestFrame>*> dest,
    const tnsr::I<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& /*jacobian*/,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian) {
  for (size_t i = 0; i < VolumeDim; ++i) {
    dest->get(i) = inv_jacobian.get(i, 0) * src.get(0);
    for (size_t p = 1; p < VolumeDim; ++p) {
      dest->get(i) += inv_jacobian.get(i, p) * src.get(p);
    }
  }
}

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
auto to_different_frame(
    const tnsr::I<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian) -> tnsr::I<DataType, VolumeDim, DestFrame> {
  auto dest = make_with_value<tnsr::I<DataType, VolumeDim, DestFrame>>(
      src, std::numeric_limits<double>::signaling_NaN());
  to_different_frame(make_not_null(&dest), src, jacobian, inv_jacobian);
  return dest;
}

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
void to_different_frame(
    const gsl::not_null<tnsr::i<DataType, VolumeDim, DestFrame>*> dest,
    const tnsr::i<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
    /*inv_jacobian*/) {
  for (size_t i = 0; i < VolumeDim; ++i) {
    dest->get(i) = jacobian.get(0, i) * src.get(0);
    for (size_t p = 1; p < VolumeDim; ++p) {
      dest->get(i) += jacobian.get(p, i) * src.get(p);
    }
  }
}

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
auto to_different_frame(
    const tnsr::i<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian) -> tnsr::i<DataType, VolumeDim, DestFrame> {
  auto dest = make_with_value<tnsr::i<DataType, VolumeDim, DestFrame>>(
      src, std::numeric_limits<double>::signaling_NaN());
  to_different_frame(make_not_null(&dest), src, jacobian, inv_jacobian);
  return dest;
}

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
void to_different_frame(
    const gsl::not_null<tnsr::iJ<DataType, VolumeDim, DestFrame>*> dest,
    const tnsr::iJ<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian) {
  for (size_t i = 0; i < VolumeDim; ++i) {
    for (size_t j = 0; j < VolumeDim; ++j) {
      // To avoid initializing to zero, split out k=0 here, and start
      // k loop below at k=1.
      dest->get(i, j) =
          jacobian.get(0, i) * inv_jacobian.get(j, 0) * src.get(0, 0);
      for (size_t p = 1; p < VolumeDim; ++p) {
        dest->get(i, j) +=
            jacobian.get(0, i) * inv_jacobian.get(j, p) * src.get(0, p);
      }
      for (size_t k = 1; k < VolumeDim; ++k) {
        for (size_t p = 0; p < VolumeDim; ++p) {
          dest->get(i, j) +=
              jacobian.get(k, i) * inv_jacobian.get(j, p) * src.get(k, p);
        }
      }
    }
  }
}

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
auto to_different_frame(
    const tnsr::iJ<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian) -> tnsr::iJ<DataType, VolumeDim, DestFrame> {
  auto dest = make_with_value<tnsr::iJ<DataType, VolumeDim, DestFrame>>(
      src, std::numeric_limits<double>::signaling_NaN());
  to_different_frame(make_not_null(&dest), src, jacobian, inv_jacobian);
  return dest;
}

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
void to_different_frame(
    const gsl::not_null<tnsr::ii<DataType, VolumeDim, DestFrame>*> dest,
    const tnsr::ii<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
    /*inv_jacobian*/) {
  to_different_frame(dest, src, jacobian);
}

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
auto to_different_frame(
    const tnsr::ii<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian) -> tnsr::ii<DataType, VolumeDim, DestFrame> {
  auto dest = make_with_value<tnsr::ii<DataType, VolumeDim, DestFrame>>(
      src, std::numeric_limits<double>::signaling_NaN());
  to_different_frame(make_not_null(&dest), src, jacobian, inv_jacobian);
  return dest;
}

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
void to_different_frame(
    const gsl::not_null<tnsr::II<DataType, VolumeDim, DestFrame>*> dest,
    const tnsr::II<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& /*jacobian*/,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian) {
  for (size_t i = 0; i < VolumeDim; ++i) {
    for (size_t j = i; j < VolumeDim; ++j) {  // symmetry
      // To avoid initializing to zero, split out k=0 here, and start
      // k loop below at k=1.
      dest->get(i, j) =
          inv_jacobian.get(i, 0) * inv_jacobian.get(j, 0) * src.get(0, 0);
      for (size_t p = 1; p < VolumeDim; ++p) {
        dest->get(i, j) +=
            inv_jacobian.get(i, 0) * inv_jacobian.get(j, p) * src.get(0, p);
      }
      for (size_t k = 1; k < VolumeDim; ++k) {
        for (size_t p = 0; p < VolumeDim; ++p) {
          dest->get(i, j) +=
              inv_jacobian.get(i, k) * inv_jacobian.get(j, p) * src.get(k, p);
        }
      }
    }
  }
}

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
auto to_different_frame(
    const tnsr::II<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian) -> tnsr::II<DataType, VolumeDim, DestFrame> {
  auto dest = make_with_value<tnsr::II<DataType, VolumeDim, DestFrame>>(
      src, std::numeric_limits<double>::signaling_NaN());
  to_different_frame(make_not_null(&dest), src, jacobian, inv_jacobian);
  return dest;
}

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
void to_different_frame(
    const gsl::not_null<tnsr::ijj<DataType, VolumeDim, DestFrame>*> dest,
    const tnsr::ijj<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
    /*inv_jacobian*/) {
  for (size_t q = 0; q < VolumeDim; ++q) {
    for (size_t i = 0; i < VolumeDim; ++i) {
      for (size_t j = i; j < VolumeDim; ++j) {  // symmetry
        dest->get(q, i, j) = jacobian.get(0, q) * jacobian.get(0, i) *
                             jacobian.get(0, j) * src.get(0, 0, 0);
        for (size_t r = 0; r < VolumeDim; ++r) {
          for (size_t k = 0; k < VolumeDim; ++k) {
            for (size_t p = 0; p < VolumeDim; ++p) {
              if (r == 0 and k == 0 and p == 0) {
                continue;
              }
              dest->get(q, i, j) += jacobian.get(r, q) * jacobian.get(k, i) *
                                    jacobian.get(p, j) * src.get(r, k, p);
            }
          }
        }
      }
    }
  }
}

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
auto to_different_frame(
    const tnsr::ijj<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian) -> tnsr::ijj<DataType, VolumeDim, DestFrame> {
  auto dest = make_with_value<tnsr::ijj<DataType, VolumeDim, DestFrame>>(
      src, std::numeric_limits<double>::signaling_NaN());
  to_different_frame(make_not_null(&dest), src, jacobian, inv_jacobian);
  return dest;
}

template <typename ResultTensor, typename InputTensor, typename DataType,
          size_t Dim, typename SourceFrame, typename TargetFrame>
void first_index_to_different_frame(
    const gsl::not_null<ResultTensor*> result, const InputTensor& input,
    const InverseJacobian<DataType, Dim, SourceFrame, TargetFrame>&
        inv_jacobian) {
  static_assert(InputTensor::rank() > 0,
                "Cannot transform scalar to different frame");
  static_assert(
      std::is_same_v<TensorMetafunctions::remove_first_index<ResultTensor>,
                     TensorMetafunctions::remove_first_index<InputTensor>>,
      "The input and result tensors must be the same except for the first "
      "index");
  using first_index = tmpl::front<typename InputTensor::index_list>;
  static_assert(
      std::is_same_v<first_index, SpatialIndex<Dim, UpLo::Up, TargetFrame>>,
      "This function is currently only tested for transforming an upper "
      "spatial index but can be generalized.");
  for (size_t storage_index = 0; storage_index < ResultTensor::size();
       ++storage_index) {
    const auto result_index = ResultTensor::get_tensor_index(storage_index);
    auto input_index = result_index;
    input_index[0] = 0;
    result->get(result_index) =
        input.get(input_index) * inv_jacobian.get(result_index[0], 0);
    for (size_t d = 1; d < Dim; ++d) {
      input_index[0] = d;
      result->get(result_index) +=
          input.get(input_index) * inv_jacobian.get(result_index[0], d);
    }
  }
}

template <typename InputTensor, typename DataType, size_t Dim,
          typename SourceFrame, typename TargetFrame, typename ResultTensor>
ResultTensor first_index_to_different_frame(
    const InputTensor& input,
    const InverseJacobian<DataType, Dim, SourceFrame, TargetFrame>&
        inv_jacobian) {
  ResultTensor result{};
  first_index_to_different_frame(make_not_null(&result), input, inv_jacobian);
  return result;
}

}  // namespace transform

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define SRCFRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define DESTFRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(3, data)

#define INSTANTIATE(_, data)                                                   \
  template void transform::to_different_frame(                                 \
      const gsl::not_null<tnsr::ii<DTYPE(data), DIM(data), DESTFRAME(data)>*>  \
          dest,                                                                \
      const tnsr::ii<DTYPE(data), DIM(data), SRCFRAME(data)>& src,             \
      const Jacobian<DTYPE(data), DIM(data), DESTFRAME(data), SRCFRAME(data)>& \
          jacobian);                                                           \
  template auto transform::to_different_frame(                                 \
      const tnsr::ii<DTYPE(data), DIM(data), SRCFRAME(data)>& src,             \
      const Jacobian<DTYPE(data), DIM(data), DESTFRAME(data), SRCFRAME(data)>& \
          jacobian)                                                            \
      ->tnsr::ii<DTYPE(data), DIM(data), DESTFRAME(data)>;                     \
  template void transform::to_different_frame(                                 \
      const gsl::not_null<Scalar<DTYPE(data)>*> dest,                          \
      const Scalar<DTYPE(data)>& src,                                          \
      const Jacobian<DTYPE(data), DIM(data), DESTFRAME(data), SRCFRAME(data)>& \
          jacobian,                                                            \
      const InverseJacobian<DTYPE(data), DIM(data), DESTFRAME(data),           \
                            SRCFRAME(data)>& inv_jacobian);                    \
  template auto transform::to_different_frame(                                 \
      Scalar<DTYPE(data)> src,                                                 \
      const Jacobian<DTYPE(data), DIM(data), DESTFRAME(data), SRCFRAME(data)>& \
          jacobian,                                                            \
      const InverseJacobian<DTYPE(data), DIM(data), DESTFRAME(data),           \
                            SRCFRAME(data)>& inv_jacobian)                     \
      ->Scalar<DTYPE(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3),
                        (Frame::BlockLogical, Frame::ElementLogical,
                         Frame::Grid, Frame::Distorted),
                        (Frame::Inertial), (double, DataVector))
GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Inertial),
                        (Frame::BlockLogical, Frame::ElementLogical,
                         Frame::Grid, Frame::Distorted),
                        (double, DataVector))
#undef INSTANTIATE

#define TENSOR(data) BOOST_PP_TUPLE_ELEM(4, data)

#define INSTANTIATE(_, data)                                                   \
  template void transform::to_different_frame(                                 \
      const gsl::not_null<tnsr::TENSOR(data) < DTYPE(data), DIM(data),         \
                          DESTFRAME(data)>* > dest,                            \
      const tnsr::TENSOR(data) < DTYPE(data), DIM(data),                       \
      SRCFRAME(data) > &src,                                                   \
      const Jacobian<DTYPE(data), DIM(data), DESTFRAME(data), SRCFRAME(data)>& \
          jacobian,                                                            \
      const InverseJacobian<DTYPE(data), DIM(data), DESTFRAME(data),           \
                            SRCFRAME(data)>& inv_jacobian);                    \
  template auto transform::to_different_frame(                                 \
      const tnsr::TENSOR(data) < DTYPE(data), DIM(data),                       \
      SRCFRAME(data) > &src,                                                   \
      const Jacobian<DTYPE(data), DIM(data), DESTFRAME(data), SRCFRAME(data)>& \
          jacobian,                                                            \
      const InverseJacobian<DTYPE(data), DIM(data), DESTFRAME(data),           \
                            SRCFRAME(data)>& inv_jacobian)                     \
      ->tnsr::TENSOR(data)<DTYPE(data), DIM(data), DESTFRAME(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3),
                        (Frame::BlockLogical, Frame::ElementLogical,
                         Frame::Grid, Frame::Distorted),
                        (Frame::Inertial), (double, DataVector),
                        (I, i, iJ, ii, II, ijj))
GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Inertial),
                        (Frame::BlockLogical, Frame::ElementLogical,
                         Frame::Grid, Frame::Distorted),
                        (double, DataVector), (I, i, iJ, ii, II, ijj))

#undef INSTANTIATE

#define INSTANTIATE(_, data)                                                   \
  template void transform::first_index_to_different_frame(                     \
      const gsl::not_null<TensorMetafunctions::prepend_spatial_index<          \
                              TensorMetafunctions::remove_first_index<         \
                                  tnsr::TENSOR(data) < DTYPE(data), DIM(data), \
                                  DESTFRAME(data)>>,                           \
                          DIM(data), UpLo::Up, SRCFRAME(data)>* > result,      \
      const tnsr::TENSOR(data) < DTYPE(data), DIM(data),                       \
      DESTFRAME(data) > &input,                                                \
      const InverseJacobian<DTYPE(data), DIM(data), SRCFRAME(data),            \
                            DESTFRAME(data)>& inv_jacobian);                   \
  template auto transform::first_index_to_different_frame(                     \
      const tnsr::TENSOR(data) < DTYPE(data), DIM(data),                       \
      DESTFRAME(data) > &input,                                                \
      const InverseJacobian<DTYPE(data), DIM(data), SRCFRAME(data),            \
                            DESTFRAME(data)>& inv_jacobian)                    \
      ->TensorMetafunctions::prepend_spatial_index<                            \
          TensorMetafunctions::remove_first_index<                             \
              tnsr::TENSOR(data) < DTYPE(data), DIM(data), DESTFRAME(data)>>,  \
      DIM(data), UpLo::Up, SRCFRAME(data) > ;
GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::ElementLogical),
                        (Frame::Inertial), (double, DataVector), (I, II, Ij))

#undef DIM
#undef SRCFRAME
#undef DESTFRAME
#undef TENSOR
#undef DTYPE
#undef INSTANTIATE
