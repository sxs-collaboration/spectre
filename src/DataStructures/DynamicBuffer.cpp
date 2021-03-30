// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DynamicBuffer.hpp"

#include <pup_stl.h>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template <typename T>
DynamicBuffer<T>::DynamicBuffer(const size_t number_of_vectors,
                                const size_t number_of_grid_points) noexcept
    : number_of_grid_points_(number_of_grid_points), data_(number_of_vectors) {
  if constexpr (is_data_vector_type) {
    buffer_.resize(number_of_vectors * number_of_grid_points);
    set_references();
  } else {
    static_assert(
        std::is_fundamental_v<T>,
        "T was found to be neither a DataVector nor a fundamental type. "
        "`DynamicBuffer` is not implemented yet for complex numbers.");
    if (number_of_grid_points != 1) {
      ERROR(
          "DynamicBuffer must have number_of_grid_points == 1 when T is a "
          "fundamental type but has number_of_grid_points = "
          << number_of_grid_points_);
    }
  }
}

template <typename T>
DynamicBuffer<T>::DynamicBuffer(const DynamicBuffer<T>& other) noexcept
    : number_of_grid_points_(other.number_of_grid_points_) {
  if constexpr (is_data_vector_type) {
    data_.resize(other.size());
    buffer_ = other.buffer_;
    set_references();
  } else {
    data_ = other.data_;
  }
}

template <typename T>
DynamicBuffer<T>& DynamicBuffer<T>::operator=(
    const DynamicBuffer& other) noexcept {
  if (this == &other) {
    return *this;
  }
  number_of_grid_points_ = other.number_of_grid_points_;
  if constexpr (is_data_vector_type) {
    data_.resize(other.size());
    buffer_ = other.buffer_;
    set_references();
  } else {
    data_ = other.data_;
  }
  return *this;
}

template <typename T>
void DynamicBuffer<T>::pup(PUP::er& p) noexcept {
  p | number_of_grid_points_;
  p | buffer_;
  if constexpr (is_data_vector_type) {
    if (p.isUnpacking()) {
      data_.resize(buffer_.size() / number_of_grid_points_);
      set_references();
    }
  } else {
    p | data_;
  }
}

template <typename T>
void DynamicBuffer<T>::set_references() noexcept {
  if constexpr (is_data_vector_type) {
    for (size_t i = 0; i < size(); ++i) {
      data_[i].set_data_ref(&buffer_[number_of_grid_points_ * i],
                            number_of_grid_points_);
    }
  }
}

template <typename T>
bool operator==(const DynamicBuffer<T>& lhs,
                const DynamicBuffer<T>& rhs) noexcept {
  return lhs.data_ == rhs.data_ and
         lhs.number_of_grid_points_ == rhs.number_of_grid_points_;
}

template <typename T>
bool operator!=(const DynamicBuffer<T>& lhs,
                const DynamicBuffer<T>& rhs) noexcept {
  return not(lhs == rhs);
}

/// \cond HIDDEN_SYMBOLS
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GEN_OP(op, type)                                    \
  template bool operator op(const DynamicBuffer<type>& lhs, \
                            const DynamicBuffer<type>& rhs) noexcept;
#define INSTANTIATE(_, data)                 \
  template class DynamicBuffer<DTYPE(data)>; \
  GEN_OP(==, DTYPE(data))                    \
  GEN_OP(!=, DTYPE(data))

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef GEN_OP
#undef INSTANTIATE
/// \endcond
