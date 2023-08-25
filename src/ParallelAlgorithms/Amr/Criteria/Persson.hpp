// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
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
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t>
class ElementId;
/// \endcond

namespace amr::Criteria {

/// @{
/*!
 * \brief Computes an anisotropic smoothness indicator based on the power in the
 * highest modes
 *
 * This smoothness indicator is the L2 norm of the tensor component with the
 * lowest N - M modes filtered out, where N is the number of grid points in the
 * given dimension and M is `num_highest_modes`.
 *
 * This strategy is similar to the Persson troubled-cell indicator (see
 * `evolution::dg::subcell::persson_tci`) with a few modifications:
 *
 * - The indicator is computed in each dimension separately for an anisotropic
 *   measure.
 * - The number of highest modes to keep can be chosen as a parameter.
 * - We don't normalize by the L2 norm of the unfiltered data $u$ here. This
 *   function just returns the L2 norm of the filtered data.
 */
template <size_t Dim>
double persson_smoothness_indicator(
    gsl::not_null<DataVector*> filtered_component_buffer,
    const DataVector& tensor_component, const Mesh<Dim>& mesh, size_t dimension,
    size_t num_highest_modes);
template <size_t Dim>
std::array<double, Dim> persson_smoothness_indicator(
    const DataVector& tensor_component, const Mesh<Dim>& mesh,
    size_t num_highest_modes);
/// @}

namespace Persson_detail {
template <size_t Dim>
void max_over_components(gsl::not_null<std::array<Flag, Dim>*> result,
                         gsl::not_null<DataVector*> buffer,
                         const DataVector& tensor_component,
                         const Mesh<Dim>& mesh, size_t num_highest_modes,
                         double alpha, double absolute_tolerance,
                         double coarsening_factor);
}

/*!
 * \brief h-refine the grid based on power in the highest modes
 *
 * \see persson_smoothness_indicator
 */
template <size_t Dim, typename TensorTags>
class Persson : public Criterion {
 public:
  struct VariablesToMonitor {
    using type = std::vector<std::string>;
    static constexpr Options::String help = {
        "The tensors to monitor for h-refinement."};
    static size_t lower_bound_on_size() { return 1; }
  };
  struct NumHighestModes {
    using type = size_t;
    static constexpr Options::String help = {
        "Number of highest modes to monitor the power of."};
    static size_t lower_bound() { return 1; }
  };
  struct Exponent {
    using type = double;
    static constexpr Options::String help = {
        "The exponent at which the modes should decrease. "
        "Corresponds to a \"relative tolerance\" of N^(-alpha), where N is the "
        "number of grid points minus 'NumHighestModes'. "
        "If any tensor component has power in the highest modes above this "
        "value times the max of the absolute tensor component over the "
        "element, the element will be h-refined in that direction."};
    static double lower_bound() { return 0.; }
  };
  struct AbsoluteTolerance {
    using type = double;
    static constexpr Options::String help = {
        "If any tensor component has a power in the highest modes above this "
        "value, the element will be h-refined in that direction. "
        "Set to 0 to disable."};
    static double lower_bound() { return 0.; }
  };
  struct CoarseningFactor {
    using type = double;
    static constexpr Options::String help = {
        "Factor applied to both relative and absolute tolerance to trigger "
        "h-coarsening. Set to 0 to disable h-coarsening altogether. "
        "Set closer to 1 to trigger h-coarsening more aggressively. "
        "Values too close to 1 risk that coarsened elements will immediately "
        "trigger h-refinement again. A reasonable value is 0.1."};
    static double lower_bound() { return 0.; }
    static double upper_bound() { return 1.; }
  };

  using options = tmpl::list<VariablesToMonitor, NumHighestModes, Exponent,
                             AbsoluteTolerance, CoarseningFactor>;

  static constexpr Options::String help = {
      "Refine the grid so the power in the highest modes stays below the "
      "tolerance"};

  Persson() = default;

  Persson(std::vector<std::string> vars_to_monitor,
          const size_t num_highest_modes, double alpha,
          double absolute_tolerance, double coarsening_factor,
          const Options::Context& context = {});

  /// \cond
  explicit Persson(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Persson);  // NOLINT
  /// \endcond

  using compute_tags_for_observation_box = tmpl::list<>;

  using argument_tags = tmpl::list<::Tags::DataBox>;

  template <typename DbTagsList, typename Metavariables>
  std::array<Flag, Dim> operator()(const db::DataBox<DbTagsList>& box,
                                   Parallel::GlobalCache<Metavariables>& cache,
                                   const ElementId<Dim>& element_id) const;

  void pup(PUP::er& p) override;

 private:
  std::vector<std::string> vars_to_monitor_{};
  size_t num_highest_modes_{};
  double alpha_ = std::numeric_limits<double>::signaling_NaN();
  double absolute_tolerance_ = std::numeric_limits<double>::signaling_NaN();
  double coarsening_factor_ = std::numeric_limits<double>::signaling_NaN();
};

// Out-of-line definitions
/// \cond

template <size_t Dim, typename TensorTags>
Persson<Dim, TensorTags>::Persson(std::vector<std::string> vars_to_monitor,
                                  const size_t num_highest_modes,
                                  const double alpha,
                                  const double absolute_tolerance,
                                  const double coarsening_factor,
                                  const Options::Context& context)
    : vars_to_monitor_(std::move(vars_to_monitor)),
      num_highest_modes_(num_highest_modes),
      alpha_(alpha),
      absolute_tolerance_(absolute_tolerance),
      coarsening_factor_(coarsening_factor) {
  db::validate_selection<TensorTags>(vars_to_monitor_, context);
}

template <size_t Dim, typename TensorTags>
Persson<Dim, TensorTags>::Persson(CkMigrateMessage* msg) : Criterion(msg) {}

template <size_t Dim, typename TensorTags>
template <typename DbTagsList, typename Metavariables>
std::array<Flag, Dim> Persson<Dim, TensorTags>::operator()(
    const db::DataBox<DbTagsList>& box,
    Parallel::GlobalCache<Metavariables>& /*cache*/,
    const ElementId<Dim>& /*element_id*/) const {
  auto result = make_array<Dim>(Flag::Undefined);
  const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
  // Check all tensors and all tensor components in turn. We take the
  // highest-priority refinement flag in each dimension, so if any tensor
  // component is non-smooth, the element will split in that dimension. And only
  // if all tensor components are smooth enough will elements join in that
  // dimension.
  DataVector buffer(mesh.number_of_grid_points());
  tmpl::for_each<TensorTags>(
      [&result, &box, &mesh, &buffer, this](const auto tag_v) {
        // Stop if we have already decided to refine every dimension
        if (result == make_array<Dim>(Flag::Split)) {
          return;
        }
        using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
        const std::string tag_name = db::tag_name<tag>();
        // Skip if this tensor is not being monitored
        if (not alg::found(vars_to_monitor_, tag_name)) {
          return;
        }
        const auto& tensor = db::get<tag>(box);
        for (const DataVector& tensor_component : tensor) {
          Persson_detail::max_over_components(
              make_not_null(&result), make_not_null(&buffer), tensor_component,
              mesh, num_highest_modes_, alpha_, absolute_tolerance_,
              coarsening_factor_);
        }
      });
  return result;
}

template <size_t Dim, typename TensorTags>
void Persson<Dim, TensorTags>::pup(PUP::er& p) {
  p | vars_to_monitor_;
  p | num_highest_modes_;
  p | alpha_;
  p | absolute_tolerance_;
  p | coarsening_factor_;
}

template <size_t Dim, typename TensorTags>
PUP::able::PUP_ID Persson<Dim, TensorTags>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace amr::Criteria
