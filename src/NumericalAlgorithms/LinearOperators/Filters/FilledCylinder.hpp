// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;
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
 * \brief A modal filter for filled-cylinder elements: a smooth exponential
 * roll-off of the coupled ZernikeB2 radial-angular (disk) modes and an
 * independent exponential roll-off in the axial direction, plus an optional
 * top-mode Heaviside cutoff in the angular direction.
 *
 * Concrete implementation of `Filters::Filter` for the central filled-cylinder
 * block created by `domain::creators::AngularCylinder`, driven by the DG
 * filtering action. See `Filters::Filter` for the framing of volume vs.
 * boundary application, the substep / every-N-steps cadence controls, and the
 * `blocks_to_filter` semantics.
 *
 * A filled-cylinder element uses a ZernikeB2 basis in the radial direction
 * (logical dimension 0), a ZernikeB2 basis in the angular direction (logical
 * dimension 1), and a Legendre/Chebyshev basis in the axial \f$z\f$ direction
 * (logical dimension 2). Unlike the Fourier-angular `Filters::HollowCylinder`
 * shells, the ZernikeB2 radial and angular spectral spaces are intertwined (a
 * two-dimensional disk basis), so the radial and angular roll-offs cannot be
 * specified independently: a single `RadialAngularHalfPower` drives the
 * combined disk exponential filter, while `ZHalfPower` drives the axial filter.
 *
 * In the volume the disk and axial filters are applied by
 * `Spectral::filtering::zernike_b2_cylinder_filter`. The coefficient of the
 * exponential roll-off is hardcoded to 36, matching `Filters::Hypercube`,
 * `Filters::SphericalShell`, and `Filters::HollowCylinder`. When
 * `NumModesToKill` is nonzero, the top `NumModesToKill` angular (Fourier
 * \f$m\f$) modes are additionally set to zero by a Heaviside cutoff (the
 * \f$m=0\f$ mode is always retained).
 *
 * For boundary-correction filtering the face mesh is classified by basis and
 * quadrature:
 * - an axial face (both directions ZernikeB2) is a full disk and is filtered by
 *   `Spectral::filtering::zernike_b2_disk_filter`;
 * - a mantle/radial face (ZernikeB2 with `Equiangular` quadrature in the
 *   angular direction, Legendre/Chebyshev in \f$z\f$) is filtered by treating
 *   the equiangular angular direction as a Fourier direction (an exponential
 *   roll-off using `RadialAngularHalfPower` plus the `NumModesToKill` cutoff),
 *   together with the axial filter;
 */
template <typename TagList>
class FilledCylinder : public Filter<3, TagList> {
 public:
  /// \brief The number of top angular (Fourier) \f$m\f$-modes to set to zero.
  struct NumModesToKill {
    using type = size_t;
    static constexpr Options::String help =
        "The number of top angular (Fourier) m-modes to set to zero.";
  };

  /*!
   * \brief Half of the exponent \f$m\f$ in the smooth exponential roll-off
   * applied to the coupled ZernikeB2 radial-angular (disk) modal coefficients
   * (logical dimensions 0 and 1).
   *
   * \f{align*}{
   *  c \to c \exp\left[-36 \left(\frac{n}{N_r}\right)^{2m}\right]
   *           \exp\left[-36 \left(\frac{k}{K}\right)^{2m}\right]
   * \f}
   *
   * where \f$n\f$ and \f$k\f$ are the radial and angular mode numbers and
   * \f$N_r\f$, \f$K\f$ are the corresponding maximum resolved modes. If `None`,
   * only the top-mode Heaviside cutoff (if any) is applied to the disk modes.
   */
  struct RadialAngularHalfPower {
    using type = Options::Auto<size_t, Options::AutoLabel::None>;
    static constexpr Options::String help =
        "The half-power for the coupled ZernikeB2 radial-angular (disk) "
        "exponential roll-off. If None, only the top-mode cutoff is applied to "
        "the disk modes.";
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
        "filtering in all blocks of the domain that are filled cylinders."};
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
      tmpl::list<NumModesToKill, RadialAngularHalfPower, ZHalfPower, Enable,
                 BlocksToFilter, VolumeFilterOnSubstep,
                 BoundaryCorrectionFilterOnSubstep, VolumeFilterEveryNSteps,
                 BoundaryCorrectionFilterEveryNSteps>;

  static constexpr Options::String help = {
      "A filled-cylinder filter applying a smooth exponential roll-off of the "
      "coupled ZernikeB2 radial-angular (disk) modes and an independent "
      "exponential roll-off in the axial direction, with an optional top-mode "
      "Heaviside cutoff in the angular direction."};

  FilledCylinder() = default;

  FilledCylinder(
      size_t num_modes_to_kill, std::optional<size_t> radial_angular_half_power,
      std::optional<size_t> z_half_power, bool enable,
      const std::optional<std::vector<std::string>>& blocks_to_filter,
      bool volume_filter_on_substep, bool boundary_filter_on_substep,
      std::optional<size_t> volume_filter_every_n_steps,
      std::optional<size_t> boundary_filter_every_n_steps,
      const Options::Context& context = {});

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(Filter<3, TagList>), FilledCylinder);
  explicit FilledCylinder(CkMigrateMessage* msg) : Filter<3, TagList>(msg) {}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  std::unique_ptr<Filter<3, TagList>> get_clone() const override;

  bool apply_volume_filter_on_substep() const override;
  bool apply_volume_filter_on_this_step(size_t step_number) const override;

  bool apply_boundary_filter_on_substep() const override;
  bool apply_boundary_filter_on_this_step(size_t step_number) const override;

  bool need_jacobians() const override { return false; }

  bool supports_mesh(const Mesh<3>& mesh) const override;

  std::string name() const override { return "FilledCylinder"; }

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
  friend bool operator==(const FilledCylinder<LocalTagList>& lhs,
                         const FilledCylinder<LocalTagList>& rhs);

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

  // Returns the cached axial exponential filter matrix for a Legendre or
  // Chebyshev direction with the given half-power and 1-D mesh, stored in the
  // passed-in cache. Returns an empty (identity) matrix when the half-power is
  // None.
  const Matrix& exponential_filter_matrix(std::optional<size_t> half_power,
                                          const Mesh<1>& mesh_1d,
                                          SingleExtentCache& cache) const;

  // Returns the cached angular (Fourier) filter matrix combining the optional
  // exponential roll-off and the optional top-mode cutoff. On the mantle face
  // the equiangular ZernikeB2-angular direction is treated as Fourier, so the
  // caller passes a Fourier/Equiangular `Mesh<1>` with the matching extent.
  // Returns an empty (identity) matrix when neither is requested.
  const Matrix& angular_filter_matrix(const Mesh<1>& mesh_1d) const;

  size_t num_modes_to_kill_{0};
  std::optional<size_t> radial_angular_half_power_{std::nullopt};
  std::optional<size_t> z_half_power_{std::nullopt};
  bool enable_{true};
  std::optional<std::vector<std::string>> blocks_and_groups_to_filter_{};
  std::optional<std::vector<size_t>> blocks_to_filter_{};
  bool volume_filter_on_substep_{false};
  bool boundary_filter_on_substep_{false};
  std::optional<size_t> volume_filter_every_n_steps_{std::nullopt};
  std::optional<size_t> boundary_filter_every_n_steps_{std::nullopt};

  // Boundary-filter matrix caches, each bound to the single extent it was built
  // for and rebuilt when queried with a different extent. The angular filter
  // is used only by `apply_on_boundary` right now: the radial and angular
  // filter is applied uncached by
  // `Spectral::filtering::zernike_b2_cylinder_filter`, which intertwines the
  // radial and angular disk modes and so cannot reuse a per-direction 1-D
  // matrix. `cached_z_filter_` holds the axial (Legendre/Chebyshev) filter and
  // `cached_angular_filter_` the Fourier-treated mantle-angular filter. The
  // empty matrix is returned (and not cached) when a direction is not filtered.
  // NOLINTNEXTLINE(spectre-mutable)
  mutable SingleExtentCache cached_z_filter_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable SingleExtentCache cached_angular_filter_{};
  // Returned by const reference to represent the identity for unfiltered
  // directions; never mutated, so it needs no `mutable`.
  Matrix empty_matrix_{};
  // Buffer used internally by `Spectral::filtering::zernike_b2_cylinder_filter`
  // NOLINTNEXTLINE(spectre-mutable)
  mutable DataVector temp_storage_{};
};

template <typename TagList>
bool operator==(const FilledCylinder<TagList>& lhs,
                const FilledCylinder<TagList>& rhs);

template <typename TagList>
bool operator!=(const FilledCylinder<TagList>& lhs,
                const FilledCylinder<TagList>& rhs);
}  // namespace Filters
