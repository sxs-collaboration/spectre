// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/SliceIterator.hpp"

#include <functional>
#include <numeric>

#include "DataStructures/Index.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Numeric.hpp"

template <size_t Dim>
SliceIterator::SliceIterator(const Index<Dim>& extents, const size_t fixed_dim,
                             const size_t fixed_index)
    : size_(extents.product()),
      stride_(std::accumulate(extents.begin(), extents.begin() + fixed_dim,
                              1_st, std::multiplies<size_t>())),
      stride_count_(0),
      jump_((extents[fixed_dim] - 1) * stride_),
      initial_offset_(fixed_index * stride_),
      volume_offset_(initial_offset_),
      slice_offset_(0) {}

SliceIterator& SliceIterator::operator++() {
  ++volume_offset_;
  ++slice_offset_;
  ++stride_count_;
  if (stride_count_ == stride_) {
    volume_offset_ += jump_;
    stride_count_ = 0;
  }
  return *this;
}

void SliceIterator::reset() {
  volume_offset_ = initial_offset_;
  slice_offset_ = 0;
}

template <size_t VolumeDim>
std::pair<std::unique_ptr<std::pair<size_t, size_t>[], decltype(&free)>,
          std::array<std::pair<gsl::span<std::pair<size_t, size_t>>,
                               gsl::span<std::pair<size_t, size_t>>>,
                     VolumeDim>>
volume_and_slice_indices(const Index<VolumeDim>& extents) noexcept {
  std::unique_ptr<std::pair<size_t, size_t>[], decltype(&free)> indices_buffer(
      nullptr, &free);
  // array over dim, pair over lower/upper, span<pair> over volume/boundary
  std::array<std::pair<gsl::span<std::pair<size_t, size_t>>,
                       gsl::span<std::pair<size_t, size_t>>>,
             VolumeDim>
      volume_and_slice_indices{};

  const size_t half_number_boundary_points = alg::accumulate(
      alg::iota(std::array<size_t, VolumeDim>{{}}, 0_st),
      0_st, [&extents](const size_t state, const size_t d) noexcept {
        return state + extents.slice_away(d).product();
      });
  ASSERT(half_number_boundary_points != 0,
         "If you encounter this assert you've found a bug in the "
         "'volume_and_slice_indices' function. Please file an issue describing "
         "the necessary steps to reproduce this error. Thank you!");
  // clang-tidy thinks we make size zero allocations. The ASSERT prevents that.
  // NOLINTNEXTLINE(clang-analyzer-unix.API)
  indices_buffer.reset(static_cast<std::pair<size_t, size_t>*>(malloc(
      sizeof(std::pair<size_t, size_t>) * half_number_boundary_points * 2)));
  size_t alloc_offset = 0;
  for (size_t d = 0; d < VolumeDim; ++d) {
    const auto boundary_extents = extents.slice_away(d);
    const auto num_points = boundary_extents.product();
    gsl::at(volume_and_slice_indices, d).first =
        gsl::make_span(indices_buffer.get() + alloc_offset, num_points);
    alloc_offset += num_points;
    gsl::at(volume_and_slice_indices, d).second =
        gsl::make_span(indices_buffer.get() + alloc_offset, num_points);
    alloc_offset += num_points;

    size_t index = 0;
    // si_lower is a macro in some standard libs, so name sli_
    for (SliceIterator sli_lower(extents, d, 0),
         sli_upper(extents, d, extents[d] - 1);
         sli_lower and sli_upper;
         (void)++sli_lower, (void)++sli_upper, (void)++index) {
      gsl::at(gsl::at(volume_and_slice_indices, d).first, index).first =
          sli_lower.volume_offset();
      gsl::at(gsl::at(volume_and_slice_indices, d).first, index).second =
          sli_lower.slice_offset();

      gsl::at(gsl::at(volume_and_slice_indices, d).second, index).first =
          sli_upper.volume_offset();
      gsl::at(gsl::at(volume_and_slice_indices, d).second, index).second =
          sli_upper.slice_offset();
    }
  }

  return {std::move(indices_buffer), std::move(volume_and_slice_indices)};
}

/// \cond HIDDEN_SYMBOLS
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data)                                                 \
  template SliceIterator::SliceIterator(const Index<DIM(data)>&, const size_t, \
                                        const size_t);                         \
  template std::pair<                                                          \
      std::unique_ptr<std::pair<size_t, size_t>[], decltype(&free)>,           \
      std::array<std::pair<gsl::span<std::pair<size_t, size_t>>,               \
                           gsl::span<std::pair<size_t, size_t>>>,              \
                 DIM(data)>>                                                   \
  volume_and_slice_indices<DIM(data)>(                                         \
      const Index<DIM(data)>& extents) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond
