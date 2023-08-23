// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/ValidateSelection.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Amr/Flag.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t>
class ElementId;
/// \endcond

namespace amr::Criteria {

namespace Constraints_detail {

/*!
 * \brief Computes the (squared) normalization factor $N_\hat{k}^2$ (see
 * `logical_constraints` below)
 */
template <size_t Dim>
void normalization_factor_square(
    gsl::not_null<tnsr::i<DataVector, Dim, Frame::ElementLogical>*> result,
    const Jacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>&
        jacobian);

/*!
 * \brief Computes a measure of the constraints in each logical direction of the
 * grid
 *
 * \requires `constraints_tensor` to be a tensor of rank 2 or higher. The first
 * index must be a lower spatial index that originates from a derivative.
 *
 * We follow \cite Szilagyi2014fna, Eq. (62)-(64) to compute:
 *
 * \begin{align}
 * \Epsilon_\hat{k} &= \frac{1}{N_\hat{k}} \sqrt{\sum_{a\ldots} \left(\sum_{i}
 *   \frac{\partial x^i}{\partial x^\hat{k}} C_{ia\ldots}\right)^2} \\
 * N_\hat{k} &= \sqrt{\sum_{i} \left(\frac{\partial x^i}{\partial x^\hat{k}}
 *   \right)^2}
 * \end{align}
 *
 * This transform the first lower spatial index of the tensor to the
 * element-logical frame, then takes an L2 norm over all remaining indices.
 * The (squared) normalization factor $N_\hat{k}^2$ is computed by
 * `normalization_factor` and passed in as an argument.
 */
template <size_t Dim, typename TensorType>
void logical_constraints(
    gsl::not_null<tnsr::i<DataVector, Dim, Frame::ElementLogical>*> result,
    gsl::not_null<DataVector*> buffer, const TensorType& constraints_tensor,
    const Jacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>&
        jacobian,
    const tnsr::i<DataVector, Dim, Frame::ElementLogical>&
        normalization_factor_square);

/*!
 * \brief Apply the AMR criterion to one of the constraints
 *
 * The `result` is the current decision in each dimension based on the previous
 * constraints. This function will update the flags if necessary. It takes
 * the "max" of the current and new flags, where the "highest" flag is
 * `Flag::IncreaseResolution`, followed by `Flag::DoNothing`, and then
 * `Flag::DecreaseResolution`.
 */
template <size_t Dim>
void max_over_components(
    gsl::not_null<std::array<Flag, Dim>*> result,
    const tnsr::i<DataVector, Dim, Frame::ElementLogical>& logical_constraints,
    double abs_target, double coarsening_factor);

}  // namespace Constraints_detail

/*!
 * \brief Refine the grid towards the target constraint violation
 *
 * - If any constraint is above the target value, the element will be p-refined.
 * - If all constraints are below the target times the "coarsening factor" the
 *   element will be p-coarsened.
 *
 * This criterion is based on Sec. 6.1.4 in \cite Szilagyi2014fna .
 *
 * If the coarsening factor turns out to be hard to choose, then we can try to
 * eliminate it by projecting the variables to a lower polynomial order before
 * computing constraints, or something like that.
 *
 * \tparam Dim Spatial dimension of the grid
 * \tparam TensorTags List of tags of the constraints to be monitored. These
 * must be tensors of rank 2 or higher. The first index must be a lower spatial
 * index that originates from a derivative.
 */
template <size_t Dim, typename TensorTags>
class Constraints : public Criterion {
 public:
  struct ConstraintsToMonitor {
    using type = std::vector<std::string>;
    static constexpr Options::String help = {"The constraints to monitor."};
    static size_t lower_bound_on_size() { return 1; }
  };
  struct AbsoluteTarget {
    using type = double;
    static constexpr Options::String help = {
        "The absolute target constraint violation. If any constraint is above "
        "this value, the element will be p-refined."};
    static double lower_bound() { return 0.; }
  };
  struct CoarseningFactor {
    using type = double;
    static constexpr Options::String help = {
        "If all constraints are below the 'AbsoluteTarget' times this factor, "
        "the element will be p-coarsened. "
        "A reasonable value is 0.1."};
    static double lower_bound() { return 0.; }
    static double upper_bound() { return 1.; }
  };

  using options =
      tmpl::list<ConstraintsToMonitor, AbsoluteTarget, CoarseningFactor>;

  static constexpr Options::String help = {
      "Refine the grid towards the target constraint violation"};

  Constraints() = default;

  Constraints(std::vector<std::string> vars_to_monitor, double abs_target,
              double coarsening_factor, const Options::Context& context = {});

  /// \cond
  explicit Constraints(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Constraints);  // NOLINT
  /// \endcond

  using compute_tags_for_observation_box = TensorTags;

  using argument_tags = tmpl::list<::Tags::ObservationBox>;

  template <typename ComputeTagsList, typename DataBoxType,
            typename Metavariables>
  std::array<Flag, Dim> operator()(
      const ObservationBox<ComputeTagsList, DataBoxType>& box,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id) const;

  void pup(PUP::er& p) override;

 private:
  std::vector<std::string> vars_to_monitor_{};
  double abs_target_ = std::numeric_limits<double>::signaling_NaN();
  double coarsening_factor_ = std::numeric_limits<double>::signaling_NaN();
};

// Out-of-line definitions
/// \cond

template <size_t Dim, typename TensorTags>
Constraints<Dim, TensorTags>::Constraints(
    std::vector<std::string> vars_to_monitor, const double abs_target,
    const double coarsening_factor, const Options::Context& context)
    : vars_to_monitor_(std::move(vars_to_monitor)),
      abs_target_(abs_target),
      coarsening_factor_(coarsening_factor) {
  db::validate_selection<TensorTags>(vars_to_monitor_, context);
}

template <size_t Dim, typename TensorTags>
Constraints<Dim, TensorTags>::Constraints(CkMigrateMessage* msg)
    : Criterion(msg) {}

template <size_t Dim, typename TensorTags>
template <typename ComputeTagsList, typename DataBoxType,
          typename Metavariables>
std::array<Flag, Dim> Constraints<Dim, TensorTags>::operator()(
    const ObservationBox<ComputeTagsList, DataBoxType>& box,
    Parallel::GlobalCache<Metavariables>& /*cache*/,
    const ElementId<Dim>& /*element_id*/) const {
  auto result = make_array<Dim>(Flag::Undefined);
  const auto& jacobian =
      get<Events::Tags::ObserverJacobian<Dim, Frame::ElementLogical,
                                         Frame::Inertial>>(box);
  // Set up memory buffers
  const size_t num_points = jacobian.begin()->size();
  TempBuffer<tmpl::list<::Tags::Tempi<0, Dim, Frame::ElementLogical>,
                        ::Tags::Tempi<1, Dim, Frame::ElementLogical>,
                        ::Tags::TempScalar<2>>>
      buffer{num_points};
  auto& normalization_factor_square =
      get<::Tags::Tempi<0, Dim, Frame::ElementLogical>>(buffer);
  auto& logical_constraints =
      get<::Tags::Tempi<1, Dim, Frame::ElementLogical>>(buffer);
  auto& scalar_buffer = get(get<::Tags::TempScalar<2>>(buffer));
  Constraints_detail::normalization_factor_square(
      make_not_null(&normalization_factor_square), jacobian);
  // Check all constraints in turn
  tmpl::for_each<TensorTags>([&result, &box, &jacobian,
                              &normalization_factor_square,
                              &logical_constraints, &scalar_buffer,
                              this](const auto tag_v) {
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
    Constraints_detail::logical_constraints(
        make_not_null(&logical_constraints), make_not_null(&scalar_buffer),
        get<tag>(box), jacobian, normalization_factor_square);
    Constraints_detail::max_over_components(make_not_null(&result),
                                            logical_constraints, abs_target_,
                                            coarsening_factor_);
  });
  return result;
}

template <size_t Dim, typename TensorTags>
void Constraints<Dim, TensorTags>::pup(PUP::er& p) {
  p | vars_to_monitor_;
  p | abs_target_;
  p | coarsening_factor_;
}

template <size_t Dim, typename TensorTags>
PUP::able::PUP_ID Constraints<Dim, TensorTags>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace amr::Criteria
