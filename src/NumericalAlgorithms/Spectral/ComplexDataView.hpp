// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "Utilities/Gsl.hpp"

namespace Spectral {
/// \ingroup SpectralGroup
/// Namespace for spin-weighted spherical harmonic utilities.
namespace Swsh {

/// \brief A set of labels for the possible representations of complex numbers
/// for the `ComplexDataView` and the computations performed in the
/// spin-weighted spherical harmonic transform library.
///
/// \details The representation describes one of two behaviors:
///  - `Interleaved`: The vectors of complex numbers will be represented by
///  alternating doubles in memory. This causes both the real and imaginary part
///  at a given gridpoint to be near one another, but successive real values
///  farther. This is the native representation of complex data in the C++
///  standard, and is the representation needed for Blaze math
///  operations. Therefore, using this representation type in libsharp
///  computations will cause operations which access only the real or imaginary
///  parts individually to trace over larger memory regions. However, this
///  representation will give rise to fewer copying operations to perform the
///  libsharp operations.
///
///  - `RealsThenImags`: The vectors of complex numbers will primarily be
///  represented by a pair of vectors of doubles, one for the real values and
///  one for the imaginary values (the full computation cannot be performed
///  exclusively in this representation, as it must return to a vector of
///  `std::complex<double>` for Blaze math operations). This causes the
///  successive real values for different gridpoints to be closer in memory, but
///  the real and imaginary parts for a given gridpoint to be farther in
///  memory. This is not the native representation for complex data in C++, so
///  the data must be transformed between operations which use Blaze and the
///  transform operations which use `RealsThenImags`. Therefore, using this
///  representation in libsharp computations will cause operations which act on
///  real or imaginary parts individually to have better memory locality (so
///  likely improved cache performance, but such statements are highly
///  hardware-dependent). However, this representation will give rise to more
///  copying operations to perform the libsharp operations.
///
/// \note The pair of representations is provided as a means to 'turn a dial' in
/// optimizations. It is unclear which of these representations will be
/// preferable, and it may well be the case that different representations are
/// better for different calculation types or different hardware. Therefore,
/// when optimizing code which uses libsharp, it is encouraged to profile the
/// cost of each representation for a computation and choose the one which
/// performs best.
enum class ComplexRepresentation { Interleaved, RealsThenImags };

namespace detail {
// A storage container for storing sub-vector references ("views") of a
// ComplexDataVector.
//
// This class takes as a template argument a ComplexRepresentation to use when
// returning pointers to the components of the complex data. The representation
// is either:
// - `ComplexRepresentation::Interleaved`, which indicates that the complex data
// is represented as a vector of `std::complex<double>`, and then
// ComplexDataView acts as a view (a pure reference to a subvector).
// - `ComplexRepresentation::RealsThenImags`, which indicates that the complex
// data is represented as a pair of vectors of `double`. This is no longer a
// strict view in the typical sense, as the memory layout is different from
// the data which it references. In order to minimize copies, the user must
// specify when edits to the ComplexDataView are complete by calling the
// `copy_back_to_source()` member function.
//
// Warning: For optimization reasons, mutations applied via member functions or
// manipulations of data pointed to by pointers returned by member functions
// *may or may not* be immediately applied to the source vector used to
// construct the ComplexDataView. Correct use of this class will first perform
// (potentially several) manipulations of the data via the ComplexDataView, then
// as a final step flush the data to the source vector via the member function
// `copy_back_to_source()`. At that point, the vector is guaranteed to be
// placed in the same state as the ComplexDataView. **The process of
// constructing a ComplexDataView, then mutating data in both the
// ComplexDataView and the source vector, then calling `copy_back_to_source()`
// is considered undefined behavior, and depends on the details of the memory
// layout chosen.**
template <ComplexRepresentation Representation>
class ComplexDataView {
 public:
  // The Representation determines the internal data representation
  static const ComplexRepresentation complex_representation = Representation;

  // The internal storage type
  using value_type = std::complex<double>;

  // Construct a view which starts at index `offset` of the supplied
  // vector, and extends for `size` elements
  ComplexDataView(gsl::not_null<ComplexDataVector*> vector, size_t size,
                  size_t offset = 0) noexcept;

  // Construct a view which starts at pointer `start` and extends for
  // `size` elements. Need not be a part of a ComplexDataVector.
  ComplexDataView(std::complex<double>* start, size_t size) noexcept;

  // For the lifetime of the data view, it points to the same portion of a
  // single vector. We disallow default move assignment, as it would exhibit
  // behavior that would contradict that. All assignment operations permitted by
  // the ComplexDataView are copying operations which act only on the data, not
  // the reference.
  ComplexDataView() = delete;
  ComplexDataView(const ComplexDataView&) = default;
  ComplexDataView(ComplexDataView&&) = default;
  ComplexDataView operator=(ComplexDataView&&) = delete;
  ~ComplexDataView() = default;

  // assign into the view the values of a same-sized ComplexDataVector
  ComplexDataView<Representation>& operator=(
      const ComplexDataVector& vector) noexcept;

  // Assign into the view the values from a same-sized view
  ComplexDataView<Representation>& operator=(
      const ComplexDataView<Representation>& view) noexcept;

  // Conjugate the data.
  void conjugate() noexcept;

  // Assign into the real components of a view the values from a
  // provided `DataVector`
  ComplexDataView<Representation>& assign_real(
      const DataVector& vector) noexcept;

  // Assign into the imaginary components of a view the values from a
  // provided `DataVector`
  ComplexDataView<Representation>& assign_imag(
      const DataVector& vector) noexcept;

  // Gets the size of the view (the number of complex entries).
  size_t size() const noexcept { return size_; }

  // Gets the stride between successive real or successive imaginary components
  // in memory.
  static constexpr size_t stride() noexcept { return stride_; }

  // Gets the raw pointer to the start of the real data, which are separated
  // from one another by `stride()`.
  double* real_data() noexcept;
  const double* real_data() const noexcept;

  // Gets the raw pointer to the start of the imaginary data, which are
  // separated from one another by `stride()`.
  double* imag_data() noexcept;
  const double* imag_data() const noexcept;

  // A function for manually flushing the data back to the source. This class
  // minimizes copies wherever possible, so the manual control of bringing the
  // data back to the source vector only when necessary is important for
  // optimization.
  // Warning: Until this function is called, mutations applied via other member
  // functions or mutations applied via the pointers obtained from other member
  // functions are not guaranteed to be applied to the source vector.
  void copy_back_to_source() noexcept;

 private:
  size_t size_;

  static const size_t stride_ =
      Representation == ComplexRepresentation::RealsThenImags ? 1 : 2;

  // In either case, we want to avoid unnecessary copies, so we track whether
  // the data has been copied from one representation to the other.
  bool real_slices_up_to_date_;
  // These two DataVectors are unused in the case of `Interleaved`
  // representation
  DataVector real_slice_;
  DataVector imag_slice_;

  double* data_real_;
  double* data_imag_;

  ComplexDataVector complex_view_;
};

}  // namespace detail
}  // namespace Swsh
}  // namespace Spectral
