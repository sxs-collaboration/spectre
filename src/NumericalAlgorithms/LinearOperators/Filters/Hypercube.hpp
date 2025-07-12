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

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesDeclaration.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
class Matrix;
/// \endcond

namespace Filters {
/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief An exponential spectral filter applied in each logical direction of
 * a tensor-product (line, square, cube, ...) element.
 *
 * Concrete implementation of `Filters::Filter` for tensor-product
 * (hypercube) elements, driven by the DG filtering action. See
 * `Filters::Filter` for the framing of volume vs. boundary application, the
 * substep / every-N-steps cadence controls, and the `blocks_to_filter`
 * semantics.
 *
 * For each component of the tensors in `TagList`, the filter rescales the
 * 1-D modal coefficients \f$c_i\f$ in each logical direction as
 *
 * \f{align*}{
 *  c_i \to c_i \exp\!\left[-36 \left(\frac{i}{N}\right)^{2m}\right],
 * \f}
 *
 * where \f$N\f$ is the basis degree (number of grid points per element per
 * dimension minus one) and \f$m\f$ is the `HalfPower` option. The same
 * coefficient and `HalfPower` are used in every logical direction. With the
 * fixed coefficient 36 the highest mode is rescaled by approximately machine
 * epsilon, i.e. effectively zeroed. For a discussion of filtering see
 * section 5.3 of \cite HesthavenWarburton.
 *
 * #### Design decision:
 *
 * The exponential coefficient is hardcoded to 36 since this is what has
 * worked well in practice for several decades in SpEC. If we ever use
 * quad or double-double types, we may want to try 72, but that is unlikely to
 * be necessary since 36 decreases the highest coefficient by 1e-16. I.e.,
 * this is not relative to the largest coefficient.
 */
template <size_t Dim, typename TagList>
class Hypercube : public Filter<Dim, TagList> {
 public:
  /*!
   * \brief Half of the exponent in the exponential.
   *
   * I.e., this is \f$m\f$ in
   *
   * \f{align*}{
   *  c_i\to c_i \exp\left[-\alpha \left(\frac{i}{N}\right)^{2m}\right]
   * \f}
   */
  struct HalfPower {
    using type = unsigned;
    static constexpr Options::String help =
        "Half of the exponent in the generalized Gaussian";
    static type lower_bound() { return 1; }
  };

  /// \brief Enable the filter
  struct Enable {
    using type = bool;
    static constexpr Options::String help = {"Enable the filter"};
  };

  /// \brief Which blocks and block groups the filter should be applied to.
  struct BlocksToFilter {
    using type =
        Options::Auto<std::vector<std::string>, Options::AutoLabel::All>;
    static constexpr Options::String help = {
        "List of blocks or block groups to apply filtering to. All other "
        "blocks will have no filtering. You can also specify 'All' to do "
        "filtering in all blocks of the domain that are hypercubes."};
  };

  /// \brief Apply the volume filter inside every substep instead of only at
  /// step boundaries.
  struct VolumeFilterOnSubstep {
    using type = bool;
    static constexpr Options::String help = {
        "Enable the volume filter on every substep."};
  };

  /// \brief Apply the boundary correction filter inside every substep instead
  /// of only at step boundaries.
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
      tmpl::list<HalfPower, Enable, BlocksToFilter, VolumeFilterOnSubstep,
                 BoundaryCorrectionFilterOnSubstep, VolumeFilterEveryNSteps,
                 BoundaryCorrectionFilterEveryNSteps>;

  static constexpr Options::String help = {
      "An exponential filter applied in each direction of a line, square, or "
      "cube (hypercube)."};

  Hypercube();

  Hypercube(unsigned half_power, bool enable,
            const std::optional<std::vector<std::string>>& blocks_to_filter,
            bool volume_filter_on_substep, bool boundary_filter_on_substep,
            std::optional<size_t> volume_filter_every_n_steps,
            std::optional<size_t> boundary_filter_every_n_steps,
            const Options::Context& context = {});

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(Filter<Dim, TagList>), Hypercube);
  explicit Hypercube(CkMigrateMessage* msg) : Filter<Dim, TagList>(msg) {}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  std::unique_ptr<Filter<Dim, TagList>> get_clone() const override;

  bool apply_volume_filter_on_substep() const override;
  bool apply_volume_filter_on_this_step(size_t step_number) const override;

  bool apply_boundary_filter_on_substep() const override;
  bool apply_boundary_filter_on_this_step(size_t step_number) const override;

  bool need_jacobians() const override { return false; }

  bool supports_mesh(const Mesh<Dim>& mesh) const override;

  std::string name() const override { return "Hypercube"; }

  const std::optional<std::vector<size_t>>& blocks_to_filter() const override;

  void set_blocks_to_filter(
      const std::vector<std::string>& all_block_names,
      const std::unordered_map<std::string, std::unordered_set<std::string>>&
          block_groups) override;

  void apply_in_volume(
      gsl::not_null<Variables<TagList>*> vars, const Mesh<Dim>& mesh,
      const std::optional<
          InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
          inv_jac_grid_to_inertial,
      const std::optional<
          Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
          jac_grid_to_inertial) const override;

  void apply_on_boundary(
      gsl::not_null<Variables<TagList>*> vars, const Mesh<Dim - 1>& mesh,
      const std::optional<
          InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
          inv_jac_grid_to_inertial,
      const std::optional<
          Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>>&
          jac_grid_to_inertial) const override;

  bool is_equal(const Filter<Dim, TagList>& other) const override;

 private:
  const Matrix& filter_matrix(
      const Mesh<1>& mesh,
      Spectral::Parity parity = Spectral::Parity::Uninitialized) const;

  // Apply the parity-aware ZernikeB1 filter to a LocalDim-dimensional mesh
  // where direction 0 uses ZernikeB1. Each tensor component is filtered with
  // the Even or Odd direction-0 matrix according to its radial parity;
  // directions 1..LocalDim-1 use the ordinary parity-independent filter matrix.
  template <size_t LocalDim>
  void apply_zernikeb1_filter(gsl::not_null<Variables<TagList>*> vars,
                              const Mesh<LocalDim>& mesh) const;

  template <size_t LocalDim, typename LocalTagList>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const Hypercube<LocalDim, LocalTagList>& lhs,
                         const Hypercube<LocalDim, LocalTagList>& rhs);

  unsigned half_power_{0};
  bool enable_{true};
  std::optional<std::vector<std::string>> blocks_and_groups_to_filter_{};
  std::optional<std::vector<size_t>> blocks_to_filter_{};
  bool volume_filter_on_substep_{false};
  bool boundary_filter_on_substep_{false};
  std::optional<size_t> volume_filter_every_n_steps_{std::nullopt};
  std::optional<size_t> boundary_filter_every_n_steps_{std::nullopt};
};

template <size_t Dim, typename TagList>
bool operator==(const Hypercube<Dim, TagList>& lhs,
                const Hypercube<Dim, TagList>& rhs);

template <size_t Dim, typename TagList>
bool operator!=(const Hypercube<Dim, TagList>& lhs,
                const Hypercube<Dim, TagList>& rhs);
}  // namespace Filters
