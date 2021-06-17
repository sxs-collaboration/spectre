// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <ostream>
#include <pup.h>
#include <string>
#include <tuple>

#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver::Schwarz {

/// \cond
template <bool Const, size_t Dim, typename TagsList>
struct ElementCenteredSubdomainDataIterator;
/// \endcond

/*!
 * \brief Data on an element-centered subdomain
 *
 * An element-centered subdomain consists of a central element and overlap
 * regions with all neighboring elements. This class holds data on such a
 * subdomain. It supports vector space operations (addition and scalar
 * multiplication) and an inner product, which allows the use of this data type
 * with linear solvers (see e.g. `LinearSolver::Serial::Gmres`).
 */
template <size_t Dim, typename TagsList>
struct ElementCenteredSubdomainData {
  static constexpr size_t volume_dim = Dim;
  using ElementData = Variables<TagsList>;
  using OverlapData = ElementData;
  using iterator = ElementCenteredSubdomainDataIterator<false, Dim, TagsList>;
  using const_iterator =
      ElementCenteredSubdomainDataIterator<true, Dim, TagsList>;

  ElementCenteredSubdomainData() = default;
  ElementCenteredSubdomainData(const ElementCenteredSubdomainData&) = default;
  ElementCenteredSubdomainData& operator=(
      const ElementCenteredSubdomainData&) noexcept = default;
  ElementCenteredSubdomainData(ElementCenteredSubdomainData&&) noexcept =
      default;
  ElementCenteredSubdomainData& operator=(
      ElementCenteredSubdomainData&&) noexcept = default;
  ~ElementCenteredSubdomainData() noexcept = default;

  explicit ElementCenteredSubdomainData(
      const size_t element_num_points) noexcept
      : element_data{element_num_points} {}

  template <typename UsedForSizeTagsList>
  void destructive_resize(
      const ElementCenteredSubdomainData<Dim, UsedForSizeTagsList>&
          used_for_size) noexcept {
    if (UNLIKELY(element_data.number_of_grid_points() !=
                 used_for_size.element_data.number_of_grid_points())) {
      element_data.initialize(
          used_for_size.element_data.number_of_grid_points());
    }
    for (const auto& [overlap_id, used_for_overlap_size] :
         used_for_size.overlap_data) {
      if (UNLIKELY(overlap_data[overlap_id].number_of_grid_points() !=
                   used_for_overlap_size.number_of_grid_points())) {
        overlap_data[overlap_id].initialize(
            used_for_overlap_size.number_of_grid_points());
      }
    }
  }

  ElementCenteredSubdomainData(
      Variables<TagsList> local_element_data,
      OverlapMap<Dim, Variables<TagsList>> local_overlap_data) noexcept
      : element_data{std::move(local_element_data)},
        overlap_data{std::move(local_overlap_data)} {}

  size_t size() const noexcept {
    return std::accumulate(
        overlap_data.begin(), overlap_data.end(), element_data.size(),
        [](const size_t size, const auto& overlap_id_and_data) noexcept {
          return size + overlap_id_and_data.second.size();
        });
  }
  iterator begin() noexcept { return {this}; }
  iterator end() noexcept { return {}; }
  const_iterator begin() const noexcept { return {this}; }
  const_iterator end() const noexcept { return {}; }
  const_iterator cbegin() const noexcept { return begin(); }
  const_iterator cend() const noexcept { return end(); }

  void pup(PUP::er& p) noexcept {  // NOLINT
    p | element_data;
    p | overlap_data;
  }

  template <typename RhsTagsList>
  ElementCenteredSubdomainData& operator+=(
      const ElementCenteredSubdomainData<Dim, RhsTagsList>& rhs) noexcept {
    element_data += rhs.element_data;
    for (auto& [overlap_id, data] : overlap_data) {
      data += rhs.overlap_data.at(overlap_id);
    }
    return *this;
  }

  template <typename RhsTagsList>
  ElementCenteredSubdomainData& operator-=(
      const ElementCenteredSubdomainData<Dim, RhsTagsList>& rhs) noexcept {
    element_data -= rhs.element_data;
    for (auto& [overlap_id, data] : overlap_data) {
      data -= rhs.overlap_data.at(overlap_id);
    }
    return *this;
  }

  ElementCenteredSubdomainData& operator*=(const double scalar) noexcept {
    element_data *= scalar;
    for (auto& [overlap_id, data] : overlap_data) {
      data *= scalar;
      // Silence unused-variable warning on GCC 7
      (void)overlap_id;
    }
    return *this;
  }

  ElementCenteredSubdomainData& operator/=(const double scalar) noexcept {
    element_data /= scalar;
    for (auto& [overlap_id, data] : overlap_data) {
      data /= scalar;
      // Silence unused-variable warning on GCC 7
      (void)overlap_id;
    }
    return *this;
  }

  ElementData element_data{};
  OverlapMap<Dim, OverlapData> overlap_data{};
};

template <size_t Dim, typename LhsTagsList, typename RhsTagsList>
decltype(auto) operator+(
    ElementCenteredSubdomainData<Dim, LhsTagsList> lhs,
    const ElementCenteredSubdomainData<Dim, RhsTagsList>& rhs) noexcept {
  lhs += rhs;
  return lhs;
}

template <size_t Dim, typename LhsTagsList, typename RhsTagsList>
decltype(auto) operator-(
    ElementCenteredSubdomainData<Dim, LhsTagsList> lhs,
    const ElementCenteredSubdomainData<Dim, RhsTagsList>& rhs) noexcept {
  lhs -= rhs;
  return lhs;
}

template <size_t Dim, typename TagsList>
decltype(auto) operator*(
    const double scalar,
    ElementCenteredSubdomainData<Dim, TagsList> data) noexcept {
  data *= scalar;
  return data;
}

template <size_t Dim, typename TagsList>
decltype(auto) operator*(ElementCenteredSubdomainData<Dim, TagsList> data,
                         const double scalar) noexcept {
  data *= scalar;
  return data;
}

template <size_t Dim, typename TagsList>
decltype(auto) operator/(ElementCenteredSubdomainData<Dim, TagsList> data,
                         const double scalar) noexcept {
  data /= scalar;
  return data;
}

template <size_t Dim, typename TagsList>
std::ostream& operator<<(std::ostream& os,
                         const ElementCenteredSubdomainData<Dim, TagsList>&
                             subdomain_data) noexcept {
  os << "Element data:\n"
     << subdomain_data.element_data << "\nOverlap data:\n"
     << subdomain_data.overlap_data;
  return os;
}

template <size_t Dim, typename TagsList>
bool operator==(
    const ElementCenteredSubdomainData<Dim, TagsList>& lhs,
    const ElementCenteredSubdomainData<Dim, TagsList>& rhs) noexcept {
  return lhs.element_data == rhs.element_data and
         lhs.overlap_data == rhs.overlap_data;
}

template <size_t Dim, typename TagsList>
bool operator!=(
    const ElementCenteredSubdomainData<Dim, TagsList>& lhs,
    const ElementCenteredSubdomainData<Dim, TagsList>& rhs) noexcept {
  return not(lhs == rhs);
}

namespace detail {
// Defines a consistent ordering of overlap IDs
template <size_t Dim, typename ValueType>
std::vector<OverlapId<Dim>> ordered_overlap_ids(
    const OverlapMap<Dim, ValueType>& overlap_map) noexcept {
  std::vector<OverlapId<Dim>> overlap_ids{};
  overlap_ids.reserve(overlap_map.size());
  std::transform(overlap_map.begin(), overlap_map.end(),
                 std::back_inserter(overlap_ids),
                 [](const auto& overlap_id_and_value) noexcept {
                   return overlap_id_and_value.first;
                 });
  std::sort(overlap_ids.begin(), overlap_ids.end(),
            [](const OverlapId<Dim>& lhs, const OverlapId<Dim>& rhs) noexcept {
              if (lhs.first.axis() != rhs.first.axis()) {
                return lhs.first.axis() < rhs.first.axis();
              }
              if (lhs.first.side() != rhs.first.side()) {
                return lhs.first.side() < rhs.first.side();
              }
              if (lhs.second.block_id() != rhs.second.block_id()) {
                return lhs.second.block_id() < rhs.second.block_id();
              }
              for (size_t d = 0; d < Dim; ++d) {
                const auto lhs_segment_id = lhs.second.segment_id(d);
                const auto rhs_segment_id = rhs.second.segment_id(d);
                if (lhs_segment_id.refinement_level() !=
                    rhs_segment_id.refinement_level()) {
                  return lhs_segment_id.refinement_level() <
                         rhs_segment_id.refinement_level();
                }
                if (lhs_segment_id.index() != rhs_segment_id.index()) {
                  return lhs_segment_id.index() < rhs_segment_id.index();
                }
              }
              return false;
            });
  return overlap_ids;
}
}  // namespace detail

/*!
 * \brief Iterate over `LinearSolver::Schwarz::ElementCenteredSubdomainData`
 *
 * This iterator guarantees that it steps through the data in the same order as
 * long as these conditions are satisfied:
 *
 * - The set of overlap IDs in the `overlap_data` doesn't change
 * - The extents of the `element_data` and the `overlap_data doesn't change
 *
 * Iterating requires sorting the overlap IDs. If you find this impacts
 * performance, be advised to implement the internal data storage in
 * `ElementCenteredSubdomainData` so it stores its data contiguously, e.g. by
 * implementing non-owning variables.
 */
template <bool Const, size_t Dim, typename TagsList>
struct ElementCenteredSubdomainDataIterator {
 private:
  using PtrType =
      tmpl::conditional_t<Const,
                          const ElementCenteredSubdomainData<Dim, TagsList>*,
                          ElementCenteredSubdomainData<Dim, TagsList>*>;

 public:
  using difference_type = ptrdiff_t;
  using value_type = double;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category = std::forward_iterator_tag;

  /// Construct begin state
  ElementCenteredSubdomainDataIterator(PtrType data) noexcept : data_(data) {
    overlap_ids_ = detail::ordered_overlap_ids(data_->overlap_data);
    reset();
  }

  void reset() noexcept {
    overlap_index_ = (data_->element_data.size() == 0 and overlap_ids_.empty())
                         ? std::numeric_limits<size_t>::max()
                         : 0;
    data_index_ = 0;
  }

  /// Construct end state
  ElementCenteredSubdomainDataIterator() noexcept {
    overlap_index_ = std::numeric_limits<size_t>::max();
    data_index_ = 0;
  }

  ElementCenteredSubdomainDataIterator& operator++() noexcept {
    ++data_index_;
    if (data_index_ ==
        (overlap_index_ == 0
             ? data_->element_data
             : data_->overlap_data.at(overlap_ids_[overlap_index_ - 1]))
            .size()) {
      ++overlap_index_;
      data_index_ = 0;
    }
    if (overlap_index_ == overlap_ids_.size() + 1) {
      overlap_index_ = std::numeric_limits<size_t>::max();
    }
    return *this;
  }

  tmpl::conditional_t<Const, double, double&> operator*() const noexcept {
    if (overlap_index_ == 0) {
      return data_->element_data.data()[data_index_];
    } else {
      return data_->overlap_data.at(overlap_ids_[overlap_index_ - 1])
          .data()[data_index_];
    }
  }

 private:
  friend bool operator==(
      const ElementCenteredSubdomainDataIterator& lhs,
      const ElementCenteredSubdomainDataIterator& rhs) noexcept {
    return lhs.overlap_index_ == rhs.overlap_index_ and
           lhs.data_index_ == rhs.data_index_;
  }

  friend bool operator!=(
      const ElementCenteredSubdomainDataIterator& lhs,
      const ElementCenteredSubdomainDataIterator& rhs) noexcept {
    return not(lhs == rhs);
  }

  PtrType data_;
  std::vector<OverlapId<Dim>> overlap_ids_;
  size_t overlap_index_;
  size_t data_index_;
};

}  // namespace LinearSolver::Schwarz

namespace LinearSolver::InnerProductImpls {

template <size_t Dim, typename LhsTagsList, typename RhsTagsList>
struct InnerProductImpl<
    Schwarz::ElementCenteredSubdomainData<Dim, LhsTagsList>,
    Schwarz::ElementCenteredSubdomainData<Dim, RhsTagsList>> {
  static double apply(
      const Schwarz::ElementCenteredSubdomainData<Dim, LhsTagsList>& lhs,
      const Schwarz::ElementCenteredSubdomainData<Dim, RhsTagsList>&
          rhs) noexcept {
    double result = inner_product(lhs.element_data, rhs.element_data);
    for (const auto& [overlap_id, lhs_data] : lhs.overlap_data) {
      result += inner_product(lhs_data, rhs.overlap_data.at(overlap_id));
    }
    return result;
  }
};

}  // namespace LinearSolver::InnerProductImpls

namespace MakeWithValueImpls {

template <size_t Dim, typename TagsListOut, typename TagsListIn>
struct MakeWithValueImpl<
    LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, TagsListOut>,
    LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, TagsListIn>> {
  using SubdomainDataIn =
      LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, TagsListIn>;
  using SubdomainDataOut =
      LinearSolver::Schwarz::ElementCenteredSubdomainData<Dim, TagsListOut>;
  static SPECTRE_ALWAYS_INLINE SubdomainDataOut
  apply(const SubdomainDataIn& input, const double value) noexcept {
    SubdomainDataOut output{};
    output.element_data =
        make_with_value<typename SubdomainDataOut::ElementData>(
            input.element_data, value);
    for (const auto& [overlap_id, input_data] : input.overlap_data) {
      output.overlap_data.emplace(
          overlap_id, make_with_value<typename SubdomainDataOut::OverlapData>(
                          input_data, value));
    }
    return output;
  }
};

}  // namespace MakeWithValueImpls
