// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <blaze/math/Column.h>
#include <cstddef>

#include "DataStructures/DenseMatrix.hpp"
#include "DataStructures/DenseVector.hpp"
#include "Options/Options.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver::Schwarz::subdomain_preconditioners {

template <size_t Dim>
struct ExplicitInverse {
  using options = tmpl::list<>;
  static constexpr Options::String help =
      "Builds a matrix representation of the subdomain operator and inverts it "
      "directly. This means that each element has a large initialization cost, "
      "but all subdomain solves converge immediately.";

  ExplicitInverse() = default;
  ExplicitInverse(const ExplicitInverse& /*rhs*/) = default;
  ExplicitInverse& operator=(const ExplicitInverse& /*rhs*/) = default;
  ExplicitInverse(ExplicitInverse&& /*rhs*/) = default;
  ExplicitInverse& operator=(ExplicitInverse&& /*rhs*/) = default;
  ~ExplicitInverse() = default;

  template <typename SubdomainOperator, typename TagsList>
  ExplicitInverse(const SubdomainOperator& subdomain_operator,
                  const ElementCenteredSubdomainData<Dim, TagsList>&
                      used_for_size) noexcept {
    ordered_overlap_ids_.reserve(used_for_size.overlap_data.size());
    std::transform(used_for_size.overlap_data.begin(),
                   used_for_size.overlap_data.end(),
                   std::back_inserter(ordered_overlap_ids_),
                   [](const auto& overlap_id_and_data) noexcept {
                     return overlap_id_and_data.first;
                   });
    size_ = used_for_size.element_data.size() +
            std::accumulate(used_for_size.overlap_data.begin(),
                            used_for_size.overlap_data.end(), size_t{0},
                            [](const size_t state,
                               const auto& overlap_id_and_data) noexcept {
                              return state + overlap_id_and_data.second.size();
                            });
    arg_workspace_.resize(size_);
    result_workspace_.resize(size_);
    // Construct explicit matrix representation
    auto unit_vector =
        make_with_value<ElementCenteredSubdomainData<Dim, TagsList>>(
            used_for_size, 0.);
    inverse_.resize(size_, size_);
    auto& operator_matrix = inverse_;
    double* data_index;
    const auto set_column =
        [&operator_matrix, this](
            const size_t col, const ElementCenteredSubdomainData<Dim, TagsList>&
                                  column) noexcept {
          for (size_t i = 0; i < column.element_data.size(); ++i) {
            operator_matrix(i, col) = column.element_data.data()[i];
          }
          size_t i_cont = column.element_data.size();
          for (const auto& overlap_id : ordered_overlap_ids_) {
            for (size_t i = 0; i < column.overlap_data.at(overlap_id).size();
                 ++i) {
              operator_matrix(i_cont, col) =
                  column.overlap_data.at(overlap_id).data()[i];
              ++i_cont;
            }
          }
        };
    for (size_t i = 0; i < unit_vector.element_data.size(); ++i) {
      if (LIKELY(i > 0)) {
        *data_index = 0.;
      }
      data_index = &(unit_vector.element_data.data()[i]);
      *data_index = 1.;
      set_column(i, subdomain_operator(unit_vector));
    }
    size_t i_cont = unit_vector.element_data.size();
    for (const auto& overlap_id : ordered_overlap_ids_) {
      for (size_t i = 0; i < unit_vector.overlap_data.at(overlap_id).size();
           ++i) {
        *data_index = 0.;
        data_index = &(unit_vector.overlap_data.at(overlap_id).data()[i]);
        *data_index = 1.;
        set_column(i_cont, subdomain_operator(unit_vector));
        ++i_cont;
      }
    }
    // Directly invert the matrix
    invert_matrix();
  }

  template <typename TagsList>
  ElementCenteredSubdomainData<Dim, TagsList> operator()(
      const ElementCenteredSubdomainData<Dim, TagsList>& arg) const noexcept {
    // Copy subdomain data into contiguous workspace
    for (size_t i = 0; i < arg.element_data.size(); ++i) {
      arg_workspace_[i] = arg.element_data.data()[i];
    }
    size_t i_cont = arg.element_data.size();
    for (const auto& overlap_id : ordered_overlap_ids_) {
      const auto& overlap_data = arg.overlap_data.at(overlap_id);
      for (size_t i = 0; i < overlap_data.size(); ++i) {
        arg_workspace_[i_cont] = overlap_data.data()[i];
        ++i_cont;
      }
    }
    // Apply inverse
    apply_inverse();
    // Reconstruct subdomain data from contiguous workspace
    auto result =
        make_with_value<ElementCenteredSubdomainData<Dim, TagsList>>(arg, 0.);
    for (size_t i = 0; i < result.element_data.size(); ++i) {
      result.element_data.data()[i] = result_workspace_[i];
    }
    i_cont = result.element_data.size();
    for (const auto& overlap_id : ordered_overlap_ids_) {
      auto& result_overlap_data = result.overlap_data.at(overlap_id);
      for (size_t i = 0; i < result_overlap_data.size(); ++i) {
        result_overlap_data.data()[i] = result_workspace_[i_cont];
        ++i_cont;
      }
    }
    return result;
  }

  size_t size() const noexcept;

  DenseMatrix<double> matrix_representation() const noexcept;

  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  void invert_matrix() noexcept;
  void apply_inverse() const noexcept;

  std::vector<OverlapId<Dim>> ordered_overlap_ids_{};
  DenseMatrix<double, blaze::columnMajor> inverse_{};
  size_t size_ = std::numeric_limits<size_t>::max();

  // Buffers to avoid re-allocating memory for applying the operator
  mutable DenseVector<double> arg_workspace_{};
  mutable DenseVector<double> result_workspace_{};
};

}  // namespace LinearSolver::Schwarz::subdomain_preconditioners
