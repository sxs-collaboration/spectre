// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesDeclaration.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
namespace PUP {
class er;
}  // namespace PUP
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace Filters {
/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief A modal filter for hollow-cylinder elements: independent exponential
 * roll-offs in the radial, angular (Fourier), and axial directions plus an
 * optional top-mode Heaviside cutoff in the angular direction.
 *
 * Concrete implementation of `Filters::Filter` for the hollow-cylinder shells
 * created by `domain::creators::AngularCylinder`, driven by the DG filtering
 * action. See `Filters::Filter` for the framing of volume vs. boundary
 * application, the substep / every-N-steps cadence controls, and the
 * `blocks_to_filter` semantics.
 *
 * A hollow-cylinder shell element uses a Legendre/Chebyshev basis in the
 * radial direction (logical dimension 0), a Fourier basis in the angular
 * direction (logical dimension 1), and a Legendre/Chebyshev basis in the axial
 * \f$z\f$ direction (logical dimension 2).
 *
 * For each component of the tensors in `TagList`, the filter rescales the 1-D
 * modal coefficients in a given logical direction by a smooth exponential
 * roll-off. For the radial and axial (Legendre/Chebyshev) directions the
 * coefficient \f$c_i\f$ of the \f$i\f$-th modal basis function is rescaled as
 *
 * \f{align*}{
 *  c_i \to c_i \exp\!\left[-36 \left(\frac{i}{N}\right)^{2m}\right],
 * \f}
 *
 * where \f$N\f$ is the basis degree (number of grid points per element per
 * dimension minus one) and \f$m\f$ is the half-power option for that direction
 * (`RadialHalfPower` or `ZHalfPower`). For the angular (Fourier) direction the
 * roll-off is applied in the angular mode number \f$k\f$ instead, with the
 * maximum resolved mode \f$K = (N_\mathrm{pts} - 1) / 2\f$ taking the role of
 * \f$N\f$:
 *
 * \f{align*}{
 *  c_k \to c_k \exp\!\left[-36 \left(\frac{k}{K}\right)^{2m}\right],
 * \f}
 *
 * where \f$m\f$ is `AngularHalfPower` and both the \f$\cos\f$ and \f$\sin\f$
 * contributions to a given mode \f$k\f$ are weighted identically. Any direction
 * whose half-power is `None` is left untouched. When `NumModesToKill` is
 * nonzero, the top `NumModesToKill` angular modes are additionally set to zero
 * by a Heaviside cutoff.
 *
 * \note The angular (Fourier) direction is periodic and therefore has no
 * boundary faces: a hollow-cylinder element only has radial and axial
 * boundaries. Consequently `apply_on_boundary` is only ever valid on a face
 * obtained by slicing away the radial direction (a `(angular, z)` face) or the
 * axial direction (a `(radial, angular)` face); both of these retain the
 * angular direction. A face that sliced away the angular direction would have
 * no Fourier direction and cannot occur, so `apply_on_boundary` errors if
 * handed one.
 */
template <typename TagList>
class HollowCylinder : public Filter<3, TagList> {
 public:
  /// \brief The number of top angular (Fourier) \f$m\f$-modes to set to zero.
  struct NumModesToKill {
    using type = size_t;
    static constexpr Options::String help =
        "The number of top angular (Fourier) m-modes to set to zero.";
  };

  /*!
   * \brief Half of the exponent \f$m\f$ in the smooth exponential roll-off
   * applied to the angular (Fourier) modal coefficients.
   *
   * \f{align*}{
   *  c_k \to c_k \exp\left[-36 \left(\frac{k}{K}\right)^{2m}\right]
   * \f}
   *
   * Here \f$k\f$ is the angular mode number and \f$K = (N_\mathrm{pts} -
   * 1)/2\f$ is the maximum resolved mode; the \f$\cos\f$ and \f$\sin\f$ parts
   * of each mode \f$k\f$ are weighted identically. If `None`, only the top-mode
   * Heaviside cutoff (if any) is applied to the angular modes.
   */
  struct AngularHalfPower {
    using type = Options::Auto<size_t, Options::AutoLabel::None>;
    static constexpr Options::String help =
        "The half-power for the angular (Fourier) exponential roll-off. If "
        "None, only the top-mode cutoff is applied to the angular modes.";
  };

  /*!
   * \brief Half of the exponent \f$m\f$ in the smooth exponential roll-off
   * applied to the radial modal coefficients (logical dimension 0). If `None`,
   * the radial direction is not filtered.
   */
  struct RadialHalfPower {
    using type = Options::Auto<size_t, Options::AutoLabel::None>;
    static constexpr Options::String help =
        "The half-power for the radial exponential filter. If None, no radial "
        "filtering is applied.";
  };

  /*!
   * \brief Half of the exponent \f$m\f$ in the smooth exponential roll-off
   * applied to the axial \f$z\f$ modal coefficients (logical dimension 2). If
   * `None`, the axial direction is not filtered.
   */
  struct ZHalfPower {
    using type = Options::Auto<size_t, Options::AutoLabel::None>;
    static constexpr Options::String help =
        "The half-power for the axial (z) exponential filter. If None, no "
        "axial filtering is applied.";
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
        "filtering in all blocks of the domain that are hollow cylinders."};
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
  ///
  /// \note Currently the check for whether to filter on every `N` steps is done
  /// relative to the start of the current Slab. This means that for GTS,
  /// independent of the value of `N` for every `N` steps, every step has a
  /// filter applied since GTS has one step per slab.
  struct VolumeFilterEveryNSteps {
    using type = Options::Auto<size_t, Options::AutoLabel::None>;
    static constexpr Options::String help = {
        "Enable the volume filter on every N steps. 'None' to disable."};
  };

  /// \brief Apply the boundary correction filter once every `N` steps. `None`
  /// (`std::nullopt`) disables the every-N-steps trigger.
  ///
  /// \note Currently the check for whether to filter on every `N` steps is done
  /// relative to the start of the current Slab. This means that for GTS,
  /// independent of the value of `N` for every `N` steps, every step has a
  /// filter applied since GTS has one step per slab.
  struct BoundaryCorrectionFilterEveryNSteps {
    using type = Options::Auto<size_t, Options::AutoLabel::None>;
    static constexpr Options::String help = {
        "Enable the boundary filter on every N steps. 'None' to disable."};
  };

  using options =
      tmpl::list<NumModesToKill, AngularHalfPower, RadialHalfPower, ZHalfPower,
                 Enable, BlocksToFilter, VolumeFilterOnSubstep,
                 BoundaryCorrectionFilterOnSubstep, VolumeFilterEveryNSteps,
                 BoundaryCorrectionFilterEveryNSteps>;

  static constexpr Options::String help = {
      "A hollow-cylinder filter applying independent exponential roll-offs in "
      "the radial, angular (Fourier), and axial directions with an optional "
      "top-mode Heaviside cutoff in the angular direction."};

  HollowCylinder() = default;

  HollowCylinder(
      size_t num_modes_to_kill, std::optional<size_t> angular_half_power,
      std::optional<size_t> radial_half_power,
      std::optional<size_t> z_half_power, bool enable,
      const std::optional<std::vector<std::string>>& blocks_to_filter,
      bool volume_filter_on_substep, bool boundary_filter_on_substep,
      std::optional<size_t> volume_filter_every_n_steps,
      std::optional<size_t> boundary_filter_every_n_steps,
      const Options::Context& context = {});

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(Filter<3, TagList>), HollowCylinder);
  explicit HollowCylinder(CkMigrateMessage* msg) : Filter<3, TagList>(msg) {}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  std::unique_ptr<Filter<3, TagList>> get_clone() const override;

  bool apply_volume_filter_on_substep() const override;
  bool apply_volume_filter_on_this_step(size_t step_number) const override;

  bool apply_boundary_filter_on_substep() const override;
  bool apply_boundary_filter_on_this_step(size_t step_number) const override;

  bool need_jacobians() const override { return false; }

  bool supports_mesh(const Mesh<3>& mesh) const override;

  std::string name() const override { return "HollowCylinder"; }

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
  friend bool operator==(const HollowCylinder<LocalTagList>& lhs,
                         const HollowCylinder<LocalTagList>& rhs);

  // Caches a single filter matrix together with the parameters it was built
  // from: the 1-D extent, the half-power, and (for the angular direction) the
  // number of top modes killed. Rebuilt (replacing the previous entry) whenever
  // queried with any different parameter, keeping the memory footprint bounded
  // instead of growing with the number of distinct extents encountered.
  struct SingleExtentCache {
    std::optional<size_t> extent{std::nullopt};
    std::optional<size_t> half_power{std::nullopt};
    size_t num_modes_to_kill{0};
    Matrix matrix{};
  };

  // Returns the cached radial/axial exponential filter matrix for a Legendre or
  // Chebyshev direction with the given half-power and 1-D mesh. Returns an
  // empty (identity) matrix when the half-power is None.
  const Matrix& exponential_filter_matrix(std::optional<size_t> half_power,
                                          const Mesh<1>& mesh_1d,
                                          SingleExtentCache& cache) const;

  // Returns the cached angular (Fourier) filter matrix combining the optional
  // exponential roll-off and the optional top-mode cutoff. Returns an empty
  // (identity) matrix when neither is requested.
  const Matrix& angular_filter_matrix(const Mesh<1>& mesh_1d) const;

  size_t num_modes_to_kill_{0};
  std::optional<size_t> angular_half_power_{std::nullopt};
  std::optional<size_t> radial_half_power_{std::nullopt};
  std::optional<size_t> z_half_power_{std::nullopt};
  bool enable_{true};
  std::optional<std::vector<std::string>> blocks_and_groups_to_filter_{};
  std::optional<std::vector<size_t>> blocks_to_filter_{};
  bool volume_filter_on_substep_{false};
  bool boundary_filter_on_substep_{false};
  std::optional<size_t> volume_filter_every_n_steps_{std::nullopt};
  std::optional<size_t> boundary_filter_every_n_steps_{std::nullopt};

  // One filter matrix per direction, each bound to the single extent it was
  // built for. These caches serve both the volume filter (`apply_in_volume`,
  // which filters all three directions) and the boundary filter
  // (`apply_on_boundary`). The empty matrix is returned (and not cached) when a
  // direction is not filtered.
  // NOLINTNEXTLINE(spectre-mutable)
  mutable SingleExtentCache cached_radial_filter_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable SingleExtentCache cached_angular_filter_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable SingleExtentCache cached_z_filter_{};
  // Returned by const reference to represent the identity for unfiltered
  // directions; never mutated, so it needs no `mutable`.
  Matrix empty_matrix_{};
};

template <typename TagList>
bool operator==(const HollowCylinder<TagList>& lhs,
                const HollowCylinder<TagList>& rhs);

template <typename TagList>
bool operator!=(const HollowCylinder<TagList>& lhs,
                const HollowCylinder<TagList>& rhs);
}  // namespace Filters
