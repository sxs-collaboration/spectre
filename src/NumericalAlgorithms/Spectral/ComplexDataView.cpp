// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Math.hpp"  // IWYU pragma: keep

namespace Spectral {
namespace Swsh {
namespace detail {

template <typename T>
void sizes_assert(const T& vector, const size_t size) {
  ASSERT(vector.size() == size,
         "Assignment must be to the same size,"
         " not "
             << vector.size() << " assigned to " << size);
}

// The constructor of `ComplexDataView<ComplexRepresentation::Interleaved>`
// makes no copies or allocations

// Due to the nature of this class performing manual memory management, indexing
// into complex data with perscribed strides, it is necessary to violate type
// system conventions to index into the complex data as though it is double
// data, thus using a `reinterpret_cast`. The C++ standard guarantees that
// complex numbers are represented in memory in such a way that this is correct,
// despite the type subversion. We also cannot take the address of the
// components of the complex data, as those are returned by value, not lvalue
// reference.
template <>
ComplexDataView<ComplexRepresentation::Interleaved>::ComplexDataView(
    const gsl::not_null<ComplexDataVector*> vector, const size_t size,
    const size_t offset) noexcept
    : size_{size},
      real_slices_up_to_date_{false},
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
      data_real_{reinterpret_cast<double*>(vector->data() + offset)},
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      data_imag_{data_real_ + 1} {
  complex_view_.set_data_ref(vector->data() + offset, size);
}

// Due to the nature of this class performing manual memory management, indexing
// into complex data with perscribed strides, it is necessary to violate type
// system conventions to index into the complex data as though it is double
// data, thus using a `reinterpret_cast`. The C++ standard guarantees that
// complex numbers are represented in memory in such a way that this is correct,
// despite the type subversion. We also cannot take the address of the
// components of the complex data, as those are returned by value, not lvalue
// reference.
template <>
ComplexDataView<ComplexRepresentation::Interleaved>::ComplexDataView(
    std::complex<double>* const start, const size_t size) noexcept
    : size_{size},
      real_slices_up_to_date_{false},
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
      data_real_{reinterpret_cast<double*>(start)},
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      data_imag_{data_real_ + 1} {
  complex_view_.set_data_ref(start, size);
}

// RealsThenImags constructors cause an allocation and a copy for each of the
// real and imaginary components of the vector. If it becomes a performance
// concern, the `real_slice_` and `imag_slice_` could be made as non-owning
// DataVectors which reference a larger block of memory allocated only once.
template <>
ComplexDataView<ComplexRepresentation::RealsThenImags>::ComplexDataView(
    const gsl::not_null<ComplexDataVector*> vector, const size_t size,
    const size_t offset) noexcept
    : size_{size}, real_slices_up_to_date_{true}, complex_view_{} {
  complex_view_.set_data_ref(vector->data() + offset, size_);
  real_slice_ = real(complex_view_);
  imag_slice_ = imag(complex_view_);
  data_real_ = real_slice_.data();
  data_imag_ = imag_slice_.data();
}

template <>
ComplexDataView<ComplexRepresentation::RealsThenImags>::ComplexDataView(
    std::complex<double>* const start, const size_t size) noexcept
    : size_{size}, real_slices_up_to_date_{true}, complex_view_{} {
  complex_view_.set_data_ref(start, size_);
  real_slice_ = real(complex_view_);
  imag_slice_ = imag(complex_view_);
  data_real_ = real_slice_.data();
  data_imag_ = imag_slice_.data();
}

// For `ComplexDataView<ComplexRepresentation::Interleaved>`, assigning
// individual components is comparatively slow, and is currently implemented as
// loop with the appropriate stride in the contiguous view.

template <>
ComplexDataView<ComplexRepresentation::Interleaved>&
ComplexDataView<ComplexRepresentation::Interleaved>::assign_real(
    const DataVector& vector) noexcept {
  if (real_slice_.data() != vector.data() or not real_slices_up_to_date_) {
    sizes_assert(vector, size_);
    for (size_t i = 0; i < size_; i++) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      data_real_[stride_ * i] = vector[i];
    }
    real_slices_up_to_date_ = false;
  }
  return *this;
}

template <>
ComplexDataView<ComplexRepresentation::Interleaved>&
ComplexDataView<ComplexRepresentation::Interleaved>::assign_imag(
    const DataVector& vector) noexcept {
  if (imag_slice_.data() != vector.data() or not real_slices_up_to_date_) {
    sizes_assert(vector, size_);
    for (size_t i = 0; i < size_; i++) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      data_imag_[stride_ * i] = vector[i];
    }
    real_slices_up_to_date_ = false;
  }
  return *this;
}

// For `ComplexDataView<ComplexRepresentation::RealsThenImags>` views, the
// assignment to individual components assigns directly to the internal
// DataVectors of the separate components, which is comparatively fast.
template <>
ComplexDataView<ComplexRepresentation::RealsThenImags>&
ComplexDataView<ComplexRepresentation::RealsThenImags>::assign_real(
    const DataVector& vector) noexcept {
  if (real_slice_.data() != vector.data()) {
    sizes_assert(vector, size_);
    real_slice_ = vector;
  }
  return *this;
}

template <>
ComplexDataView<ComplexRepresentation::RealsThenImags>&
ComplexDataView<ComplexRepresentation::RealsThenImags>::assign_imag(
    const DataVector& vector) noexcept {
  if (imag_slice_.data() != vector.data()) {
    sizes_assert(vector, size_);
    imag_slice_ = vector;
  }
  return *this;
}

// `ComplexDataView<ComplexRepresentation::Interleaved>` views are simply
// references back to the source data, therefore the source is perpetually in
// sync with the view. This function is a no-op.
template <>
void ComplexDataView<
    ComplexRepresentation::Interleaved>::copy_back_to_source() noexcept {}

// `ComplexDataView<ComplexRepresentation::RealsThenImags>` views perform a true
// copy back to the source vector when this function is called
template <>
void ComplexDataView<
    ComplexRepresentation::RealsThenImags>::copy_back_to_source() noexcept {
  for (size_t i = 0; i < size_; i++) {
    complex_view_[i] = std::complex<double>{real_slice_[i], imag_slice_[i]};
  }
}

// `ComplexDataView<ComplexRepresentation::Interleaved>` views perform
// assignments from complex vectors comparatively quickly, via the assignments
// using the underlying complex vector types.
template <>
ComplexDataView<ComplexRepresentation::Interleaved>&
ComplexDataView<ComplexRepresentation::Interleaved>::operator=(
    const ComplexDataVector& vector) noexcept {
  if (complex_view_.data() != vector.data()) {
    sizes_assert(vector, size_);
    complex_view_ = vector;
    real_slices_up_to_date_ = false;
  }
  return *this;
}

template <>
ComplexDataView<ComplexRepresentation::Interleaved>&
ComplexDataView<ComplexRepresentation::Interleaved>::operator=(
    const ComplexDataView<ComplexRepresentation::Interleaved>& view) noexcept {
  if (this != &view) {
    sizes_assert(view, size_);
    complex_view_ = view.complex_view_;
    real_slices_up_to_date_ = false;
  }
  return *this;
}

template <>
void ComplexDataView<ComplexRepresentation::Interleaved>::conjugate() noexcept {
  complex_view_ = conj(complex_view_);
  real_slices_up_to_date_ = false;
}

// `ComplexDataView<ComplexRepresentation::RealsThenImags>` views perform
// assignments from complex vectors comparatively slowly. The complex data must
// be separately converted to the real and imaginary components in the form of
// DataVectors, then stored in the appropriate internal components.
template <>
ComplexDataView<ComplexRepresentation::RealsThenImags>&
ComplexDataView<ComplexRepresentation::RealsThenImags>::operator=(
    const ComplexDataVector& vector) noexcept {
  sizes_assert(vector, size_);
  real_slice_ = real(vector);
  imag_slice_ = imag(vector);
  return *this;
}

template <>
ComplexDataView<ComplexRepresentation::RealsThenImags>&
ComplexDataView<ComplexRepresentation::RealsThenImags>::operator=(
    const ComplexDataView<ComplexRepresentation::RealsThenImags>&
        view) noexcept {
  sizes_assert(view, size_);
  if (real_slice_.data() != view.real_slice_.data()) {
    real_slice_ = view.real_slice_;
  }
  if (imag_slice_.data() != view.imag_slice_.data()) {
    imag_slice_ = view.imag_slice_;
  }
  return *this;
}

template <>
void ComplexDataView<
    ComplexRepresentation::RealsThenImags>::conjugate() noexcept {
  imag_slice_ = -imag_slice_;
}

// `ComplexDataView<ComplexRepresentation::Interleaved>` views have the
// individual pointer positions only one double length apart, with a memory
// stride of 2, as the real and imaginary parts alternate in memory.
template <>
double*
ComplexDataView<ComplexRepresentation::Interleaved>::real_data() noexcept {
  return data_real_;
}

template <>
const double* ComplexDataView<ComplexRepresentation::Interleaved>::real_data()
    const noexcept {
  return data_real_;
}

template <>
double*
ComplexDataView<ComplexRepresentation::Interleaved>::imag_data() noexcept {
  return data_imag_;
}

template <>
const double* ComplexDataView<ComplexRepresentation::Interleaved>::imag_data()
    const noexcept {
  return data_imag_;
}

// `ComplexDataView<ComplexRepresentation::RealsThenImags>` views have
// potentially very different pointer positions for the start of the individual
// vectors, but have a memory stride of 1, as each of the two blocks is a
// contiguous representation.
template <>
double*
ComplexDataView<ComplexRepresentation::RealsThenImags>::real_data() noexcept {
  return real_slice_.data();
}

template <>
const double*
ComplexDataView<ComplexRepresentation::RealsThenImags>::real_data() const
    noexcept {
  return real_slice_.data();
}

template <>
double*
ComplexDataView<ComplexRepresentation::RealsThenImags>::imag_data() noexcept {
  return imag_slice_.data();
}

template <>
const double*
ComplexDataView<ComplexRepresentation::RealsThenImags>::imag_data() const
    noexcept {
  return imag_slice_.data();
}
}  // namespace detail
}  // namespace Swsh
}  // namespace Spectral
