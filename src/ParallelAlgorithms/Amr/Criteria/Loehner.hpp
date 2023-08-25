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
 * \brief Computes an anisotropic smoothness indicator based on the magnitude of
 * second derivatives
 *
 * This smoothness indicator is simply the L2 norm of the logical second
 * derivative of the tensor component in the given `dimension`:
 *
 * \begin{equation}
 * \epsilon_k =
 *   \sqrt{\frac{1}{N_\mathrm{points}} \sum_{p=1}^N_\mathrm{points}
 *     \left(\partial^2 u / \partial \xi_k^2\right)^2}
 * \end{equation}
 *
 * If the smoothness indicator is large in a direction, meaning the tensor
 * component has a large second derivative in that direction, the element should
 * be h-refined. If the smoothness indicator is small, the element should be
 * h-coarsened. A coarsing threshold of about a third of the refinement
 * threshold seems to work well, but this will need more testing.
 *
 * Note that it is not at all clear that a smoothness indicator based on the
 * magnitude of second derivatives is useful in a DG context. Smooth functions
 * with higher-order derivatives can be approximated just fine by higher-order
 * DG elements without the need for h-refinement. The use of second derivatives
 * to indicate the need for refinement originated in the finite element context
 * with linear elements. Other smoothness indicators might prove more useful for
 * DG elements, e.g. based on jumps or oscillations of the solution. We can also
 * explore applying the troubled-cell indicators (TCIs) used in hydrodynamic
 * simulations as h-refinement indicators.
 *
 * Specifically, this smoothness indicator is based on \cite Loehner1987 (hence
 * the name of the function), which is popular in the finite element community
 * and also used in a DG context by \cite Dumbser2013, Eq. (34), and by
 * \cite Renkhoff2023, Eq. (15). We make several modifications:
 *
 * - The original smoothness indicator is isotropic, i.e. it computes the norm
 *   over all (mixed) second derivatives. Here we compute an anisotropic
 *   indicator by computing second derivatives in each dimension separately
 *   and ignoring mixed derivatives.
 * - The original smoothness indicator is normalized by measures of the first
 *   derivative which don't generalize well to spectral elements. Therefore, we
 *   simplify the normalization to a standard relative and absolute tolerance.
 *   An alternative approach is proposed in \cite Renkhoff2023, Eq.(15), where
 *   the authors take the absolute value of the differentiation matrix and apply
 *   the resulting matrix to the absolute value of the data on the grid to
 *   compute the normalization. However, this quantity can produce quite large
 *   numbers and hence overestimates the smoothness by suppressing the second
 *   derivative.
 * - We compute the second derivative in logical coordinates. This seems
 *   easiest for spectral elements, but note that \cite Renkhoff2023 seem to
 *   use inertial coordinates.
 *
 * In addition to the above modifications, we can consider approximating the
 * second derivative using finite differences, as explored in the prototype
 * https://github.com/sxs-collaboration/dg-charm/blob/HpAmr/Evolution/HpAmr/LohnerRefiner.hpp.
 * This would allow falling back to the normalization used by Löhner and might
 * be cheaper to compute, but it requires an interpolation to the center and
 * maybe also to the faces, depending on the desired stencil.
 */
template <size_t Dim>
double loehner_smoothness_indicator(
    gsl::not_null<DataVector*> first_deriv_buffer,
    gsl::not_null<DataVector*> second_deriv_buffer,
    const DataVector& tensor_component, const Mesh<Dim>& mesh,
    size_t dimension);
template <size_t Dim>
std::array<double, Dim> loehner_smoothness_indicator(
    const DataVector& tensor_component, const Mesh<Dim>& mesh);
/// @}

namespace Loehner_detail {
template <size_t Dim>
void max_over_components(
    gsl::not_null<std::array<Flag, Dim>*> result,
    gsl::not_null<std::array<DataVector, 2>*> deriv_buffers,
    const DataVector& tensor_component, const Mesh<Dim>& mesh,
    double relative_tolerance, double absolute_tolerance,
    double coarsening_factor);
}

/*!
 * \brief h-refine the grid based on a smoothness indicator
 *
 * The smoothness indicator used here is based on the magnitude of second
 * derivatives. See `amr::Criteria::loehner_smoothness_indicator` for details
 * and caveats.
 *
 * \see amr::Criteria::loehner_smoothness_indicator
 */
template <size_t Dim, typename TensorTags>
class Loehner : public Criterion {
 public:
  struct VariablesToMonitor {
    using type = std::vector<std::string>;
    static constexpr Options::String help = {
        "The tensors to monitor for h-refinement."};
    static size_t lower_bound_on_size() { return 1; }
  };
  struct RelativeTolerance {
    using type = double;
    static constexpr Options::String help = {
        "If any tensor component has a second derivative magnitude above this "
        "value times the max of the absolute tensor component over the "
        "element, the element will be h-refined in that direction. "
        "Set to 0 to disable."};
    static double lower_bound() { return 0.; }
  };
  struct AbsoluteTolerance {
    using type = double;
    static constexpr Options::String help = {
        "If any tensor component has a second derivative magnitude above this "
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
        "trigger h-refinement again. A reasonable value is 1/3."};
    static double lower_bound() { return 0.; }
    static double upper_bound() { return 1.; }
  };

  using options = tmpl::list<VariablesToMonitor, RelativeTolerance,
                             AbsoluteTolerance, CoarseningFactor>;

  static constexpr Options::String help = {
      "Refine the grid towards resolving an estimated error in the second "
      "derivative"};

  Loehner() = default;

  Loehner(std::vector<std::string> vars_to_monitor, double relative_tolerance,
          double absolute_tolerance, double coarsening_factor,
          const Options::Context& context = {});

  /// \cond
  explicit Loehner(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Loehner);  // NOLINT
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
  double relative_tolerance_ = std::numeric_limits<double>::signaling_NaN();
  double absolute_tolerance_ = std::numeric_limits<double>::signaling_NaN();
  double coarsening_factor_ = std::numeric_limits<double>::signaling_NaN();
};

// Out-of-line definitions
/// \cond

template <size_t Dim, typename TensorTags>
Loehner<Dim, TensorTags>::Loehner(std::vector<std::string> vars_to_monitor,
                                  const double relative_tolerance,
                                  const double absolute_tolerance,
                                  const double coarsening_factor,
                                  const Options::Context& context)
    : vars_to_monitor_(std::move(vars_to_monitor)),
      relative_tolerance_(relative_tolerance),
      absolute_tolerance_(absolute_tolerance),
      coarsening_factor_(coarsening_factor) {
  db::validate_selection<TensorTags>(vars_to_monitor_, context);
  if (relative_tolerance == 0. and absolute_tolerance == 0.) {
    PARSE_ERROR(
        context,
        "Must specify non-zero RelativeTolerance, AbsoluteTolerance, or both.");
  }
}

template <size_t Dim, typename TensorTags>
Loehner<Dim, TensorTags>::Loehner(CkMigrateMessage* msg) : Criterion(msg) {}

template <size_t Dim, typename TensorTags>
template <typename DbTagsList, typename Metavariables>
std::array<Flag, Dim> Loehner<Dim, TensorTags>::operator()(
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
  std::array<DataVector, 2> deriv_buffers{};
  tmpl::for_each<TensorTags>(
      [&result, &box, &mesh, &deriv_buffers, this](const auto tag_v) {
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
          Loehner_detail::max_over_components(
              make_not_null(&result), make_not_null(&deriv_buffers),
              tensor_component, mesh, relative_tolerance_, absolute_tolerance_,
              coarsening_factor_);
        }
      });
  return result;
}

template <size_t Dim, typename TensorTags>
void Loehner<Dim, TensorTags>::pup(PUP::er& p) {
  p | vars_to_monitor_;
  p | relative_tolerance_;
  p | absolute_tolerance_;
  p | coarsening_factor_;
}

template <size_t Dim, typename TensorTags>
PUP::able::PUP_ID Loehner<Dim, TensorTags>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace amr::Criteria
