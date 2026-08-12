// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <utility>

#include "DataStructures/Variables.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

/*!
 * \ingroup DataStructuresGroup
 * \brief A data structure holding Variables associated with directions.
 *
 * This class holds Variables of sizes `points_per_direction`, all
 * referencing a single memory allocation.  The individual Variables
 * can be accessed through the `variables()` member.  Access to the
 * underlying buffer is provided so that the class can, for example,
 * be used as the evolved variables for an evolution system.
 */
template <size_t Dim, typename Tags>
class BoundaryVariables {
  static_assert(
      DirectionMap<Dim, size_t>::hash_is_perfect,
      "Map must use a perfect hash to ensure consistent data ordering.");

 public:
  using tags_list = Tags;
  using VariablesType = Variables<tags_list>;
  using Vector = typename VariablesType::vector_type;

  BoundaryVariables() = default;
  BoundaryVariables(BoundaryVariables&& other)
      : points_per_direction_(std::move(other.points_per_direction_)),
        buffer_(std::move(other.buffer_)) {
    other.clear();
    setup_variables();
  }
  BoundaryVariables(const BoundaryVariables& other)
      : points_per_direction_(other.points_per_direction_),
        buffer_(other.buffer_) {
    setup_variables();
  }

  BoundaryVariables& operator=(BoundaryVariables&& other) {
    if (this == &other) {
      return *this;
    }
    points_per_direction_ = std::move(other.points_per_direction_);
    buffer_ = std::move(other.buffer_);
    other.clear();
    setup_variables();
    return *this;
  }
  BoundaryVariables& operator=(const BoundaryVariables& other) {
    points_per_direction_ = other.points_per_direction_;
    buffer_ = other.buffer_;
    setup_variables();
    return *this;
  }

  ~BoundaryVariables() = default;

  explicit BoundaryVariables(DirectionMap<Dim, size_t> points_per_direction) {
    initialize(std::move(points_per_direction));
  }

  BoundaryVariables(DirectionMap<Dim, size_t> points_per_direction,
                    const typename Vector::value_type& initial_value) {
    initialize(std::move(points_per_direction), initial_value);
  }

  void initialize(DirectionMap<Dim, size_t> points_per_direction) {
    points_per_direction_ = std::move(points_per_direction);
    buffer_ = Vector(buffer_size());
    setup_variables();
  }

  void initialize(DirectionMap<Dim, size_t> points_per_direction,
                  const typename Vector::value_type& initial_value) {
    points_per_direction_ = std::move(points_per_direction);
    buffer_ = Vector(buffer_size(), initial_value);
    setup_variables();
  }

  size_t size() const { return buffer_.size(); }
  bool empty() const { return buffer_.size() == 0; }

  void clear() {
    points_per_direction_.clear();
    buffer_.clear();
    variables_.clear();
  }

  const DirectionMap<Dim, size_t>& points_per_direction() const {
    return points_per_direction_;
  }

  const Vector& buffer() const {
    check_pointers();
    return buffer_;
  }
  Vector& buffer() {
    check_pointers();
    return buffer_;
  }

  DirectionMap<Dim, VariablesType>& variables() {
    check_pointers();
    return variables_;
  }
  const DirectionMap<Dim, VariablesType>& variables() const {
    check_pointers();
    return variables_;
  }

  void pup(PUP::er& p) {
    p | points_per_direction_;
    p | buffer_;
    if (p.isUnpacking()) {
      setup_variables();
    }
  }

 private:
  size_t buffer_size() const {
    size_t num_points = 0;
    for (const auto& [direction, points] : points_per_direction_) {
      num_points += points;
    }
    return num_points * VariablesType::number_of_independent_components;
  }

  void setup_variables() {
    variables_.clear();
    size_t offset = 0;
    for (const auto& [direction, points] : points_per_direction_) {
      const size_t vars_size =
          points * VariablesType::number_of_independent_components;
      variables_.emplace(direction,
                         VariablesType(buffer_.data() + offset, vars_size));
      offset += vars_size;
    }
  }

  void check_pointers() const {
#ifdef SPECTRE_DEBUG
    if (points_per_direction_.empty()) {
      return;
    }
    ASSERT(buffer_.size() == buffer_size(),
           "Buffer has been manually resized.  Have "
               << buffer_.size() << " entries but should be " << buffer_size());
    const auto first_direction = points_per_direction_.begin()->first;
    ASSERT(variables_.at(first_direction).data() == buffer_.data(),
           "Pointer mismatch.  Data buffers have been replaced.");
#endif
  }

  DirectionMap<Dim, size_t> points_per_direction_{};
  Vector buffer_{};
  DirectionMap<Dim, VariablesType> variables_{};
};

template <size_t Dim, typename Tags>
bool operator==(const BoundaryVariables<Dim, Tags>& a,
                const BoundaryVariables<Dim, Tags>& b) {
  return a.points_per_direction() == b.points_per_direction() and
         a.buffer() == b.buffer();
}

template <size_t Dim, typename Tags>
bool operator!=(const BoundaryVariables<Dim, Tags>& a,
                const BoundaryVariables<Dim, Tags>& b) {
  return not(a == b);
}

template <size_t Dim, typename Tags>
auto make_math_wrapper(
    const gsl::not_null<BoundaryVariables<Dim, Tags>*> data) {
  return make_math_wrapper(&data->buffer());
}

template <size_t Dim, typename Tags>
auto make_math_wrapper(const BoundaryVariables<Dim, Tags>& data) {
  return make_math_wrapper(data.buffer());
}

template <size_t Dim, typename Tags>
auto into_math_wrapper_type(BoundaryVariables<Dim, Tags>&& data) {
  auto result = into_math_wrapper_type(std::move(data.buffer()));
  data.clear();
  return result;
}

// We can't set the size from an arbitrary object, but we can copy
// sizes from one BoundaryVariables to another.
template <size_t Dim, typename Tags1, typename Tags2>
struct MakeWithValueImpls::MakeWithValueImpl<BoundaryVariables<Dim, Tags1>,
                                             BoundaryVariables<Dim, Tags2>> {
  template <typename ValueType>
  static BoundaryVariables<Dim, Tags1> apply(
      const BoundaryVariables<Dim, Tags2>& input, const ValueType value) {
    return {input.points_per_direction(), value};
  }
};

template <size_t Dim, typename Tags1, typename Tags2>
void set_number_of_grid_points(
    const gsl::not_null<BoundaryVariables<Dim, Tags1>*> result,
    const BoundaryVariables<Dim, Tags2>& pattern) {
  if (UNLIKELY(result->points_per_direction() !=
               pattern.points_per_direction())) {
    *result = BoundaryVariables<Dim, Tags1>(pattern.points_per_direction());
  }
}

template <size_t Dim, typename Tags>
bool contains_allocations(const BoundaryVariables<Dim, Tags>& value) {
  return contains_allocations(value.buffer());
}
