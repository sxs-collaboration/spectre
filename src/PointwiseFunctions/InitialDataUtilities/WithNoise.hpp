// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace evolution::initial_data {

/*!
 * \brief Add uniform random noise to all stored components of \p tensor.
 *
 * Each independent stored component receives noise drawn independently from
 * \f$\text{Uniform}[-A,\,A]\f$, where \f$A\f$ is \p amplitude. The RNG seed
 * for component \f$i\f$ is `hash(element_seed, component_offset + i)`, so
 * calling this function with different \p component_offset values keeps
 * separate tensor fields independent even when \p element_seed is the same.
 *
 * \param tensor Tensor to perturb in place.
 * \param amplitude Half-width \f$A\f$ of the noise interval.
 * \param element_seed Per-element seed, typically constructed by hashing a
 *   base seed with the inertial coordinates of the element's first grid point.
 * \param component_offset Offset added to the component index before hashing,
 *   used to keep different tensor fields independent.
 */
template <typename TensorType>
void add_noise_to_tensor(gsl::not_null<TensorType*> tensor, double amplitude,
                         size_t element_seed, size_t component_offset) {
  if (amplitude == 0.0) {
    return;
  }
  std::uniform_real_distribution<double> dist{-amplitude, amplitude};
  for (size_t i = 0; i < TensorType::size(); ++i) {
    size_t comp_seed = element_seed;
    boost::hash_combine(comp_seed, component_offset + i);
    std::mt19937_64 gen{comp_seed};
    for (double& val : (*tensor)[i]) {
      val += dist(gen);
    }
  }
}

/*!
 * \brief Combine \p base_seed with the first grid point of \p inertial_coords
 * to produce a per-element RNG seed.
 *
 * Because most domains have elements have distinct first grid points, this
 * gives independent noise on almost every element even when `base_seed` is the
 * same.
 */
template <size_t Dim>
size_t make_element_seed(
    size_t base_seed,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords) {
  size_t element_seed = base_seed;
  for (size_t d = 0; d < Dim; ++d) {
    boost::hash_combine(element_seed, inertial_coords.get(d)[0]);
  }
  return element_seed;
}

/*!
 * \brief Wraps any analytic `InitialData` and adds uniform random noise to
 * selected evolution variables after the inner solution has initialized them.
 *
 * \details The inner solution first sets all evolution variables in the normal
 * way. Then each variable whose `db::tag_name` appears in `Variables` (or
 * every variable if `All` is listed) has independent noise from
 * \f$\text{Uniform}[-A,\,A]\f$ added to every grid point and every independent
 * tensor component, where \f$A\f$ is `Amplitude`.
 *
 * Valid variable names are the `db::tag_name`s of the system's evolution
 * variables. Examples:
 * - **ScalarWave**: `Psi`, `Pi`, `Phi`
 * - **GeneralizedHarmonic**: `SpacetimeMetric`, `Pi`, `Phi`
 *
 * Different elements receive independent noise because the RNG seed is mixed
 * with the inertial coordinates of each element's first grid point. Set
 * `Seed` to a fixed integer for reproducible results.
 *
 * \note Only analytic initial data (analytic solutions or analytic data) are
 * supported as the inner `Solution`. Numeric initial data uses a two-phase
 * asynchronous loading process: the `SetInitialData` action dispatches a file
 * read request to `ElementDataReader` and returns immediately; variables are
 * only set later when `ReceiveNumericInitialData` processes the inbox data.
 * Because `WithNoise` applies noise immediately after the inner solution
 * initializes variables, it has no hook into that second action and therefore
 * cannot wrap numeric initial data.
 *
 * \note Nesting `WithNoise` inside `WithNoise` is not supported.
 *
 * \note **GeneralizedHarmonic systems**: the initialization phase
 * `InitializeInitialDataDependentQuantities` runs
 * `gh::gauges::SetPiAndPhiFromConstraints` *after* `WithNoise` applies noise.
 * That action unconditionally overwrites `Phi` with the numerical spatial
 * derivative of `SpacetimeMetric` and recomputes `Pi` from the gauge source
 * function and the current geometry. As a result, noise added to `Pi` or `Phi`
 * is effectively erased. Only noise added to `SpacetimeMetric` persists and
 * propagates into the other variables: it appears in `Phi` amplified by
 * roughly one over the grid spacing (due to differentiation), and in `Pi`
 * through the geometric quantities (lapse, shift) extracted from the metric.
 *
 * Example input-file snippet for ScalarWave:
 * \code{.yaml}
 * InitialData:
 *   WithNoise:
 *     Amplitude: 1.0e-6
 *     Seed: 42
 *     Variables: [Psi, Pi]
 *     Solution:
 *       PlaneWave:
 *         ...
 * \endcode
 */
class WithNoise : public evolution::initial_data::InitialData {
 public:
  /// The analytic initial data to wrap.
  struct Solution {
    using type = std::unique_ptr<evolution::initial_data::InitialData>;
    static constexpr Options::String help =
        "The analytic initial data to wrap. Must be an analytic solution or "
        "analytic data (not numeric initial data).";
  };

  /// Half-amplitude \f$A\f$ of the noise interval \f$[-A,\,A]\f$.
  struct Amplitude {
    using type = double;
    static constexpr Options::String help =
        "Half-amplitude A of the noise interval [-A, A] applied "
        "independently to each grid point and tensor component.";
  };

  /// Seed for the random number generator.
  struct Seed {
    using type = Options::Auto<size_t, Options::AutoLabel::None>;
    static constexpr Options::String help =
        "Seed for the noise RNG. Set to 'None' for a random seed chosen at "
        "runtime (runs will not be reproducible).";
  };

  /// Names of evolution variables to perturb, or `[All]` for all.
  struct Variables {
    using type = std::vector<std::string>;
    static constexpr Options::String help =
        "List of variable names to add noise to, or [All] to noise every "
        "evolution variable. Valid names depend on the system; see class "
        "documentation.";
  };

  using options = tmpl::list<Solution, Amplitude, Seed, Variables>;
  static constexpr Options::String help =
      "Wraps any analytic initial data and adds uniform random noise to "
      "selected evolution variables after the inner solution sets them.";

  WithNoise() = default;
  WithNoise(const WithNoise& rhs);
  WithNoise& operator=(const WithNoise& rhs);
  WithNoise(WithNoise&&) = default;
  WithNoise& operator=(WithNoise&&) = default;
  ~WithNoise() override = default;

  WithNoise(std::unique_ptr<evolution::initial_data::InitialData> solution,
            double amplitude, std::optional<size_t> seed,
            std::vector<std::string> variables);

  /// \cond
  explicit WithNoise(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(WithNoise);
  /// \endcond

  std::unique_ptr<evolution::initial_data::InitialData> get_clone()
      const override;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  const evolution::initial_data::InitialData& solution() const {
    return *solution_;
  }
  /// Unwrap to the inner analytic solution, recursively resolving nested
  /// `WithNoise` wrappers.
  const evolution::initial_data::InitialData& unwrap() const override {
    return solution_->unwrap();
  }
  double amplitude() const { return amplitude_; }
  size_t seed() const { return seed_; }
  const std::vector<std::string>& variables() const { return variables_; }

  friend bool operator==(const WithNoise& lhs, const WithNoise& rhs);

  friend bool operator!=(const WithNoise& lhs, const WithNoise& rhs);

 private:
  std::unique_ptr<evolution::initial_data::InitialData> solution_{};
  double amplitude_{0.};
  size_t seed_{0};
  std::vector<std::string> variables_{};
};

}  // namespace evolution::initial_data
