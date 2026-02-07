// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/ValidateSelection.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Type.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t>
class ElementId;
/// \endcond

namespace amr::Criteria {

namespace TruncationError_detail {
/*!
 * \brief Apply the truncation error criterion to a single tensor component
 *
 * The `result` is the current decision in each dimension based on the previous
 * tensor components. This function will update the flags if necessary. It takes
 * the "max" of the current and new flags, where the "highest" flag is
 * `Flag::IncreaseResolution`, followed by `Flag::DoNothing`, and then
 * `Flag::DecreaseResolution`.
 */
template <typename ValueType, size_t Dim>
void max_over_components(
    gsl::not_null<std::array<Flag, Dim>*> result,
    const gsl::not_null<std::array<DataVector, Dim>*> power_monitors_buffer,
    const ValueType& tensor_component, const Mesh<Dim>& mesh,
    double target_abs_truncation_error, double target_rel_truncation_error);
}  // namespace TruncationError_detail

/*!
 * \brief Refine the grid towards the target truncation error
 *
 * - If any tensor component has a truncation error above the target value, the
 *   element will be p-refined.
 * - If all tensor components still satisfy the target even with one mode
 *   removed, the element will be p-coarsened.
 *
 * For details on how the truncation error is computed see
 * `PowerMonitors::truncation_error`.
 *
 * \tparam Dim Spatial dimension of the grid
 * \tparam TensorTags List of tags of the tensors to be monitored
 */
template <size_t Dim, typename TensorTags>
class TruncationError : public Criterion {
 public:
  struct VariablesToMonitor {
    using type = std::vector<std::string>;
    static constexpr Options::String help = {
        "The tensors to monitor the truncation error of."};
    static size_t lower_bound_on_size() { return 1; }
  };
  struct AbsoluteTargetTruncationError {
    static std::string name() { return "AbsoluteTarget"; }
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The absolute target truncation error."};
  };
  struct RelativeTargetTruncationError {
    static std::string name() { return "RelativeTarget"; }
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The relative target truncation error."};
  };

  using options = tmpl::list<VariablesToMonitor, AbsoluteTargetTruncationError,
                             RelativeTargetTruncationError>;

  static constexpr Options::String help = {
      "Refine the grid towards the target truncation error"};

  TruncationError() = default;

  TruncationError(std::vector<std::string> vars_to_monitor,
                  double target_abs_truncation_error,
                  double target_rel_truncation_error,
                  const Options::Context& context = {});

  /// \cond
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(TruncationError);  // NOLINT
  /// \endcond

  Type type() override { return Type::p; }

  std::string observation_name() override { return "TruncationError"; }

  using compute_tags_for_observation_box = tmpl::list<>;

  using argument_tags = tmpl::list<::Tags::DataBox>;

  template <typename DbTagsList, typename Metavariables>
  std::array<Flag, Dim> operator()(const db::DataBox<DbTagsList>& box,
                                   Parallel::GlobalCache<Metavariables>& cache,
                                   const ElementId<Dim>& element_id) const;

  void pup(PUP::er& p) override;

 private:
  std::vector<std::string> vars_to_monitor_{};
  double target_abs_truncation_error_{};
  double target_rel_truncation_error_{};
};

// Out-of-line definitions
/// \cond

template <size_t Dim, typename TensorTags>
TruncationError<Dim, TensorTags>::TruncationError(
    std::vector<std::string> vars_to_monitor,
    const double target_abs_truncation_error,
    const double target_rel_truncation_error, const Options::Context& context)
    : vars_to_monitor_(std::move(vars_to_monitor)),
      target_abs_truncation_error_(target_abs_truncation_error),
      target_rel_truncation_error_(target_rel_truncation_error) {
  db::validate_selection<TensorTags>(vars_to_monitor_, context);
}

template <size_t Dim, typename TensorTags>
template <typename DbTagsList, typename Metavariables>
std::array<Flag, Dim> TruncationError<Dim, TensorTags>::operator()(
    const db::DataBox<DbTagsList>& box,
    Parallel::GlobalCache<Metavariables>& /*cache*/,
    const ElementId<Dim>& /*element_id*/) const {
  auto result = make_array<Dim>(Flag::Undefined);
  const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
  std::array<DataVector, Dim> power_monitors_buffer{};
  // Check all tensors and all tensor components in turn
  tmpl::for_each<TensorTags>(
      [&result, &box, &mesh, &power_monitors_buffer, this](const auto tag_v) {
        // Stop if we have already decided to refine every dimension
        if (result == make_array<Dim>(Flag::IncreaseResolution)) {
          return;
        }
        using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
        const std::string tag_name = db::tag_name<tag>();
        // Skip if this tensor is not being monitored
        if (not alg::found(vars_to_monitor_, tag_name)) {
          return;
        }
        const auto& tensor = db::get<tag>(box);
        for (const auto& tensor_component : tensor) {
          TruncationError_detail::max_over_components(
              make_not_null(&result), make_not_null(&power_monitors_buffer),
              tensor_component, mesh, target_abs_truncation_error_,
              target_rel_truncation_error_);
        }
      });
  return result;
}

template <size_t Dim, typename TensorTags>
void TruncationError<Dim, TensorTags>::pup(PUP::er& p) {
  p | vars_to_monitor_;
  p | target_abs_truncation_error_;
  p | target_rel_truncation_error_;
}

#if defined(SPECTRE_USE_CHARM)
template <size_t Dim, typename TensorTags>
PUP::able::PUP_ID TruncationError<Dim, TensorTags>::my_PUP_ID = 0;  // NOLINT
#endif  // SPECTRE_USE_CHARM
/// \endcond

}  // namespace amr::Criteria
