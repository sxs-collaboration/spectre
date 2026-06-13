// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;
/// \endcond

namespace Filters {
/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Abstract base class for spectral filters.
 *
 * Derived classes implement spectral filters that suppress high-frequency
 * modes in the modal expansion of a discontinuous Galerkin solution. Writing
 * the 1-D modal coefficients of a tensor component as \f$c_i\f$, a spectral
 * filter rescales them as
 *
 * \f{align*}{
 *  c_i \to \sigma\!\left(\frac{i}{N}\right) c_i,
 * \f}
 *
 * where \f$N\f$ is the basis degree (number of grid points per element per
 * dimension minus one) and \f$\sigma\f$ is a derived-class-specific damping
 * factor (for example, an exponential in `Filters::Hypercube`). For a
 * discussion of filtering see section 5.3 of \cite HesthavenWarburton.
 *
 * The interface is organized along two axes:
 *
 * - **Volume vs. boundary.** `apply_in_volume` filters the evolved
 *   `Variables` on the element's `Dim`-dimensional mesh, while
 *   `apply_on_boundary` filters data on a `Dim - 1` face mesh. The boundary
 *   hook is intended for filtering the lifted boundary corrections produced
 *   by the DG numerical flux before they enter the volume right-hand side.
 *
 * - **Substep vs. every-N-steps.** The driving action queries
 *   `apply_volume_filter_on_substep()` (and the boundary analogue) to decide
 *   whether to filter inside every Runge-Kutta substep, and
 *   `apply_volume_filter_on_this_step(step_number)` (and the boundary
 *   analogue) to gate filtering at coarser cadences such as every `N` full
 *   steps. The two return values are independent so a derived class may pick
 *   any combination.
 *
 * Derived classes additionally declare via `need_jacobians()` whether they
 * require the grid-to-inertial Jacobian and its inverse in `apply_in_volume`
 * / `apply_on_boundary`. When `need_jacobians()` returns `false`, the
 * driving action passes `std::nullopt` for both Jacobian arguments and skips
 * computing them.
 *
 * The set of domain blocks to which the filter applies is reported by
 * `blocks_to_filter()`; a `std::nullopt` return value means the filter
 * applies to every block in the domain.
 */
template <size_t Dim, typename TagList>
class Filter : public PUP::able {
 public:
  Filter() = default;
  ~Filter() noexcept override = default;

  WRAPPED_PUPable_abstract(Filter);  // NOLINT
  explicit Filter(CkMigrateMessage* m) : PUP::able(m) {}

  /// Return a heap-allocated deep copy of this filter.
  virtual std::unique_ptr<Filter<Dim, TagList>> get_clone() const = 0;

  /// Whether the volume filter should be applied inside every Runge-Kutta
  /// substep.
  virtual bool apply_volume_filter_on_substep() const = 0;

  /// Whether the volume filter should be applied at the given `step_number`.
  virtual bool apply_volume_filter_on_this_step(size_t step_number) const = 0;

  /// Whether the boundary filter should be applied inside every Runge-Kutta
  /// substep.
  virtual bool apply_boundary_filter_on_substep() const = 0;

  /// Whether the boundary filter should be applied at the given
  /// `step_number`.
  virtual bool apply_boundary_filter_on_this_step(size_t step_number) const = 0;

  /// \brief Whether the filter needs the grid-to-inertial Jacobian and its
  /// inverse.
  ///
  /// When `false`, the driving action passes `std::nullopt` for the Jacobian
  /// arguments of `apply_in_volume` and `apply_on_boundary` and avoids
  /// constructing them.
  virtual bool need_jacobians() const = 0;

  /// \brief Returns `true` if this filter can filter the `mesh`.
  virtual bool supports_mesh(const Mesh<Dim>& mesh) const = 0;

  /// \brief A human-readable name for the concrete filter type, used in
  /// diagnostics.
  virtual std::string name() const = 0;

  /// \brief Returns `true` if `other` is the same concrete filter type and is
  /// equivalent to `*this`.
  virtual bool is_equal(const Filter& other) const = 0;

  /// \brief The set of block (or block group) names the filter applies to.
  ///
  /// `std::nullopt` means the filter applies to every block in the domain.
  virtual const std::optional<std::vector<size_t>>& blocks_to_filter()
      const = 0;

  /// \brief Used after construction to change blocks and groups labeled by
  /// strings to `size_t`s, which is what is used throughout the code.
  virtual void set_blocks_to_filter(
      const std::vector<std::string>& all_block_names,
      const std::unordered_map<std::string, std::unordered_set<std::string>>&
          block_groups) = 0;

  /*!
   * \brief Apply the filter in place to the volume `vars` on the
   * `Dim`-dimensional `mesh`.
   *
   * The grid-to-inertial Jacobian `jac_grid_to_inertial` and its inverse
   * `inv_jac_grid_to_inertial` are populated only when `need_jacobians()`
   * returns `true`; otherwise both arguments are `std::nullopt`.
   */
  virtual void apply_in_volume(
      gsl::not_null<Variables<TagList>*> vars, const Mesh<Dim>& mesh,
      const std::optional<
          InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
          inv_jac_grid_to_inertial,
      const std::optional<
          Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
          jac_grid_to_inertial) const = 0;

  /*!
   * \brief Apply the filter in place to face `vars` on the `Dim - 1` face
   * `mesh`.
   *
   * Used to filter the lifted boundary corrections produced by the DG numerical
   * flux before they enter the volume right-hand side. The grid-to-inertial
   * Jacobian `jac_grid_to_inertial` and its inverse `inv_jac_grid_to_inertial`
   * are populated only when `need_jacobians()` returns `true`; otherwise both
   * arguments are `std::nullopt`.
   */
  virtual void apply_on_boundary(
      gsl::not_null<Variables<TagList>*> vars, const Mesh<Dim - 1>& mesh,
      const std::optional<
          InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
          inv_jac_grid_to_inertial,
      const std::optional<
          Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
          jac_grid_to_inertial) const = 0;
};
}  // namespace Filters
