// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlm.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t Dim>
class Mesh;
/// \endcond

namespace Filters {
/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief A modal filter for spherical-shell elements: a top-$\ell$ Heaviside
 * cutoff in the angular direction plus optional smooth exponential roll-offs
 * in both the angular $\ell$ direction and the radial direction.
 *
 * Concrete implementation of `Filters::Filter` for spherical-shell elements,
 * driven by the DG filtering action. See `Filters::Filter` for the framing of
 * volume vs. boundary application, the substep / every-N-steps cadence
 * controls, and the `blocks_to_filter` semantics.
 *
 * For each component of the tensors in `TagList`, the filter rescales the
 * Spherepack-normalized angular modal coefficients $c_{\ell'}$ as
 *
 * \f{align*}{
 *  c_{\ell'} \to c_{\ell'} \exp\!\left[-36 \left(\frac{\ell'}
 *  {\ell^+_{\mathrm{cut}}+1}\right)^{2\sigma_a}\right],
 * \f}
 *
 * where $\ell^+_{\mathrm{cut}} = \ell_{\mathrm{max}} -$ `NumModesToKill` is
 * the largest angular mode that is retained and $\sigma_a$ is the
 * `AngularHalfPower` option. With the fixed coefficient 36 and $\sigma_a$ in
 * the typical range 28-32 the angular filter is smooth below
 * $\ell^+_{\mathrm{cut}}$ and reduces to a sharp Heaviside cutoff as
 * $\sigma_a \to \infty$. When `AngularHalfPower` is `None`, only the
 * Heaviside cutoff is applied. See `ylm::TensorYlm` for the derivation of
 * the underlying angular filter.
 *
 * When `RadialHalfPower` has a value $\sigma_r$, an additional 1D
 * exponential filter is applied independently along each radial column. The
 * radial nodal coefficients $c_i$ are rescaled in modal space as
 *
 * \f{align*}{
 *  c_i \to c_i \exp\!\left[-36 \left(\frac{i}{N_r}\right)^{2\sigma_r}\right],
 * \f}
 *
 * where $N_r$ is the radial basis degree (radial extent minus one). When
 * `RadialHalfPower` is `None`, the radial direction is left untouched.
 *
 * #### Design decision:
 *
 * - The exponential coefficient is hardcoded to 36, matching the choice in
 * `Hypercube` in `Cube.hpp` and `Filters::Exponential`. `SphericalShell` is
 * the `Filters::Filter`-based implementation that plugs into the filtering
 * action and supports per-block selection together with independent volume-
 * and boundary-filtering cadences. It is intended for spherical-shell
 * blocks, which store Spherepack-normalized spherical-harmonic modes.
 */
template <typename TagList>
class SphericalShell : public Filter<3, TagList> {
 public:
  /// \brief The number of top $\ell$ modes to set to zero.
  struct NumModesToKill {
    using type = size_t;
    static constexpr Options::String help =
        "The number of top ell modes to set to zero.";
  };

  /*!
   * \brief Half of the exponent $\sigma_a$ in the smooth exponential roll-off
   * applied to the angular $\ell$ modes below the top-$\ell$ cutoff.
   *
   * \f{align*}{
   *  c_{\ell'} \to c_{\ell'} \exp\left[-36 \left(\frac{\ell'}
   *  {\ell^+_{\mathrm{cut}}+1}\right)^{2\sigma_a}\right]
   * \f}
   *
   * If `None`, only the Heaviside top-$\ell$ cutoff is applied to the
   * angular modes.
   */
  struct AngularHalfPower {
    using type = Options::Auto<size_t, Options::AutoLabel::None>;
    static constexpr Options::String help =
        "The half-power sigma for the angular ell-mode exponential roll-off. "
        "If None, only the top-ell Heaviside cutoff is applied.";
  };

  /*!
   * \brief Half of the exponent $\sigma_r$ in the smooth exponential
   * roll-off applied to the radial modal coefficients.
   *
   * \f{align*}{
   *  c_i \to c_i \exp\left[-36 \left(\frac{i}{N_r}\right)^{2\sigma_r}\right]
   * \f}
   *
   * where $N_r$ is the radial basis degree. If `None`, the radial direction
   * is not filtered.
   */
  struct RadialHalfPower {
    using type = Options::Auto<size_t, Options::AutoLabel::None>;
    static constexpr Options::String help =
        "The half-power sigma for the radial exponential filter. "
        "If None, no radial filtering is applied.";
  };

  /// \brief Enable (true) or disable (false) the filter
  struct Enable {
    using type = bool;
    static constexpr Options::String help = {"Enable the filter"};
  };

  /// \brief Which blocks the filter should be applied to.
  struct BlocksToFilter {
    using type =
        Options::Auto<std::vector<std::string>, Options::AutoLabel::All>;
    static constexpr Options::String help = {
        "List of blocks or block groups to apply filtering to. All other "
        "blocks will have no filtering. You can also specify 'All' to do "
        "filtering in all blocks of the domain that are spherical shells."};
  };

  /// \brief Apply the volume filter inside every Runge-Kutta substep
  /// instead of only at whole-step boundaries.
  struct VolumeFilterOnSubstep {
    using type = bool;
    static constexpr Options::String help = {
        "Enable the volume filter on every substep."};
  };

  /// \brief Apply the boundary correction filter inside every Runge-Kutta
  /// substep instead of only at whole-step boundaries.
  struct BoundaryCorrectionFilterOnSubstep {
    using type = bool;
    static constexpr Options::String help = {
        "Enable the boundary filter on every substep."};
  };

  /// \brief Apply the volume filter once every `N` steps. `None`
  /// (`std::nullopt`) disables the every-N-steps trigger.
  struct VolumeFilterEveryNSteps {
    using type = Options::Auto<size_t, Options::AutoLabel::None>;
    static constexpr Options::String help = {
        "Enable the volume filter on every N steps. 'None' to disable."};
  };

  /// \brief Apply the boundary correction filter once every `N` steps. `None`
  /// (`std::nullopt`) disables the every-N-steps trigger.
  struct BoundaryCorrectionFilterEveryNSteps {
    using type = Options::Auto<size_t, Options::AutoLabel::None>;
    static constexpr Options::String help = {
        "Enable the boundary filter on every N steps. 'None' to disable."};
  };

  using options =
      tmpl::list<NumModesToKill, AngularHalfPower, RadialHalfPower, Enable,
                 BlocksToFilter, VolumeFilterOnSubstep,
                 BoundaryCorrectionFilterOnSubstep, VolumeFilterEveryNSteps,
                 BoundaryCorrectionFilterEveryNSteps>;

  static constexpr Options::String help = {
      "A spherical-shell filter applying a top-ell Heaviside cutoff in the "
      "angular direction with optional smooth exponential roll-offs in both "
      "the angular ell direction and the radial direction."};

  SphericalShell() = default;

  SphericalShell(
      size_t num_modes_to_kill, std::optional<size_t> angular_half_power,
      std::optional<size_t> radial_half_power, bool enable,
      const std::optional<std::vector<std::string>>& blocks_to_filter,
      bool volume_filter_on_substep, bool boundary_filter_on_substep,
      std::optional<size_t> volume_filter_every_n_steps,
      std::optional<size_t> boundary_filter_every_n_steps,
      const Options::Context& context = {});

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(Filter<3, TagList>), SphericalShell);
  explicit SphericalShell(CkMigrateMessage* msg) : Filter<3, TagList>(msg) {}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  std::unique_ptr<Filter<3, TagList>> get_clone() const override;

  bool apply_volume_filter_on_substep() const override;
  bool apply_volume_filter_on_this_step(size_t step_number) const override;

  bool apply_boundary_filter_on_substep() const override;
  bool apply_boundary_filter_on_this_step(size_t step_number) const override;

  bool need_jacobians() const override { return true; }

  bool supports_mesh(const Mesh<3>& mesh) const override;

  std::string name() const override { return "SphericalShell"; }

  const std::optional<std::vector<size_t>>& blocks_to_filter() const override;

  void set_blocks_to_filter(
      const std::vector<std::string>& all_block_names,
      const std::unordered_map<std::string, std::unordered_set<std::string>>&
          block_groups) override;

  void apply_in_volume(
      gsl::not_null<Variables<TagList>*> vars, const Mesh<3>& mesh,
      const std::optional<
          InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>>&
          inv_jac_grid_to_inertial,
      const std::optional<
          Jacobian<DataVector, 3, Frame::Grid, Frame::Inertial>>&
          jac_grid_to_inertial) const override;

  void apply_on_boundary(
      gsl::not_null<Variables<TagList>*> vars, const Mesh<2>& mesh,
      const std::optional<
          InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>>&
          inv_jac_grid_to_inertial,
      const std::optional<
          Jacobian<DataVector, 3, Frame::Grid, Frame::Inertial>>&
          jac_grid_to_inertial) const override;

  bool is_equal(const Filter<3, TagList>& other) const override;

 private:
  template <typename LocalTagList>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const SphericalShell<LocalTagList>& lhs,
                         const SphericalShell<LocalTagList>& rhs);

  size_t num_modes_to_kill_{0};
  std::optional<size_t> angular_half_power_{std::nullopt};
  std::optional<size_t> radial_half_power_{std::nullopt};
  bool enable_{true};
  std::optional<std::vector<std::string>> blocks_and_groups_to_filter_{};
  std::optional<std::vector<size_t>> blocks_to_filter_{};
  bool volume_filter_on_substep_{false};
  bool boundary_filter_on_substep_{false};
  std::optional<size_t> volume_filter_every_n_steps_{std::nullopt};
  std::optional<size_t> boundary_filter_every_n_steps_{std::nullopt};

  // Use Spherepack normalization because the variables are stored as Spherepack
  // modes
  static constexpr ylm::TensorYlm::CoefficientNormalization normalization_ =
      ylm::TensorYlm::CoefficientNormalization::Spherepack;
  // Caches and memory buffers
  // NOLINTNEXTLINE(spectre-mutable)
  mutable size_t cached_l_max_{0};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable ylm::TensorYlm::FilterMatrixHolder filter_matrices_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable size_t cached_radial_extents_{0};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable Matrix cached_radial_filter_matrix_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable Variables<TagList> temp_storage_{};
};

template <typename TagList>
bool operator==(const SphericalShell<TagList>& lhs,
                const SphericalShell<TagList>& rhs);

template <typename TagList>
bool operator!=(const SphericalShell<TagList>& lhs,
                const SphericalShell<TagList>& rhs);
}  // namespace Filters
