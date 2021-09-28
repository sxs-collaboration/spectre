// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Utilities/TypeTraits/GetFundamentalType.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/**
 * \ingroup DataStructuresGroup
 *
 * \brief A dynamically sized vector of `DataVector`s. For convenience it can
 * also be instantiated for fundamental types.
 *
 * \details This class is useful when one wants to create a `std::vector<T>`
 * with a size that is unknown at compile time. It allocates all `DataVector`s
 * in a single memory chunk rather than allocating each individually. If the
 * size of the vector is known at compile time, a `TempBuffer` object should be
 * used instead.
 *
 * If needed it should be fairly straightforward to generalize to
 * `ComplexDataVector`.
 */
template <typename T>
class DynamicBuffer {
 public:
  static constexpr bool is_data_vector_type =
      std::is_base_of_v<VectorImpl<tt::get_fundamental_type_t<T>, T>, T>;

  DynamicBuffer() = default;

  /*!
   * Constructs a `DynamicBuffer`. The `number_of_vectors` corresponds to the
   * number of `DataVector`s which are saved inside, each of which has size
   * `number_of_grid_points`. `number_of_grid_points` has to be 1 if T is a
   * fundamental type.
   */
  DynamicBuffer(size_t number_of_vectors, size_t number_of_grid_points);
  ~DynamicBuffer() = default;
  DynamicBuffer(DynamicBuffer&& other) = default;
  DynamicBuffer& operator=(DynamicBuffer&& other) = default;

  DynamicBuffer(const DynamicBuffer& other);

  DynamicBuffer& operator=(const DynamicBuffer& other);

  T& operator[](size_t index) { return data_[index]; }
  T& at(size_t index) { return data_.at(index); }
  const T& operator[](size_t index) const { return data_[index]; }
  const T& at(size_t index) const { return data_.at(index); }

  auto begin() { return data_.begin(); }
  auto end() { return data_.end(); }
  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end(); }

  size_t size() const { return data_.size(); }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  // sets data references for all `data_` into `buffer_`
  void set_references();

  template <typename LocalT>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const DynamicBuffer<LocalT>& lhs,
                         const DynamicBuffer<LocalT>& rhs);

  size_t number_of_grid_points_;
  // vector of non-owning DataVectors pointing into `buffer_`. In case of
  // fundamental type T the data is saved in `data_` directly.
  std::vector<T> data_;
  // memory buffer for all DataVectors. Unused in case of fundamental type T.
  std::vector<double> buffer_;
};

template <typename T>
bool operator!=(const DynamicBuffer<T>& lhs, const DynamicBuffer<T>& rhs);
