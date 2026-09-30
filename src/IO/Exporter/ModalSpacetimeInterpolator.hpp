// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/ElementSearchTree.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Interpolation/UniformCardinalBSpline.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace spectre::Exporter {

/*!
 * \brief Interpolate tensor components in both space and time using modal
 * data. This is much more efficient and accurate than interpolating nodal
 * data.
 *
 * \details This class builds a time interpolant for every modal coefficient of
 * every tensor component on each element. It reads volume data one element at
 * a time, converts nodal values to modal coefficients, and constructs a
 * cardinal cubic B-spline in time for each mode. At evaluation time it locates
 * the element containing the target point, evaluates the modal interpolants at
 * the requested time, and evaluates the Legendre series at the element-logical
 * point.
 *
 * ### How to use
 *
 * In a simulation, create a few events that write out volume data at different
 * time intervals using ObserveFields. The observation times need to be uniform
 * within each subfile, so either observe at a fixed slab interval if the slab
 * size is constant, or at fixed times using a dense trigger. The idea is that
 * there is a very coarse grid written very frequently and a finer grid written
 * less frequently. This way, the lowest modes are resolved very accurately and
 * the higher modes are approximated with a larger error. A useful choice in
 * practice is:
 *
 * - Observe the lowest 3-4 modes very frequently, e.g. every 0.5M or even more
 *   often.
 * - Observe the lowest 6-7 modes 10 times less frequently.
 * - Observe all modes 10 times less frequently again.
 *
 * Important: use the `ProjectToMesh` option of ObserveFields to ensure that
 * the data is truncated cleanly in modal space. Do not use `InterpolateToMesh`
 * as this creates a catastrophic aliasing error. Also, do not use the
 * `CleanFunctionsOfTime` action, as this removes the required global history of
 * the functions of time.
 *
 * Then construct the ModalSpacetimeInterpolator from these subfiles in
 * priority order, from the coarse mesh observed most frequently to the fine
 * mesh that defines the final spatial resolution. The first subfile containing
 * a mode owns it, and later subfiles only supply the modes that are missing on
 * the earlier meshes. The subfile meshes must therefore be nested, i.e. have
 * component-wise nondecreasing extents, and all subfiles must contain the same
 * elements and the same domain. The subfiles may have different time steps and
 * nonaligned observation times; the available time interval is their
 * intersection.
 *
 * ### Error tolerance and compression
 *
 * The time series of every owned mode is compressed into an
 * `intrp::UniformCardinalBSpline` on the coarsest uniform time grid that
 * reproduces the samples to within an absolute error tolerance, and modes
 * whose amplitude never exceeds the tolerance are dropped entirely since they
 * are not important for reconstructing the field (this can be up to 80% of the
 * modes in practice). This way, the interpolator only keeps the data that is
 * necessary.
 *
 * By default, the tolerance for each element and tensor component is estimated
 * from the lowest mode in the first subfile: it is the interpolation error of
 * that mode when using all of its observed data, so all other modes are
 * interpolated with a conservatively similar error. Note that this infers the
 * tolerance from the temporal variability of the data rather than from an
 * accuracy requirement, so specify the tolerance explicitly when you know the
 * accuracy you need. The estimate requires at least 9 observations in the
 * first subfile; an explicit tolerance does not. Either way this is a per-mode
 * compression heuristic. It does not bound the total reconstructed field error
 * and does not detect temporal undersampling in the original volume data.
 *
 * ### Limitations
 *
 * Only the Legendre basis is supported for now. Changes to the element
 * topology or per-element mesh within one time series (e.g. AMR), restarts
 * that produce non-uniform observation times, and element migration between
 * files within one time series are not supported. Files from different
 * simulation segments must be joined first, with overlapping observations
 * removed. Elements may reside in different files in different subfiles.
 * Moving domains use the global functions of time stored with the final
 * subfile, and all functions required by the domain must cover the full
 * available time interval.
 *
 * Since the time grid of each mode is uniform, it is a good idea to use a
 * separate interpolator for parts of the evolution that are dynamically
 * different (e.g. junk radiation vs. inspiral vs. merger vs. ringdown). A
 * wrapper that patches together several interpolators may be added in the
 * future.
 *
 * \note Due to a bug in the boost cardinal cubic B-spline implementation that
 * was fixed only recently, see
 * https://github.com/boostorg/math/commit/4809e714d4806c07da3a3def0c4550daa0529b8d,
 * the interpolator can only be built with boost version 1.81 or later. See
 * `intrp::UniformCardinalBSpline` for details.
 */
template <size_t Dim, typename Frame = ::Frame::Inertial>
class ModalSpacetimeInterpolator {
 public:
  /*!
   * \brief Construct from one or more volume files and nested volume
   * subfiles.
   *
   * \param volume_files_or_glob A list of volume H5 files, or a glob string
   *     that resolves to volume files. The files may distribute the elements
   *     of each observation across nodes.
   * \param subfiles_in_priority_order Volume subfile names ordered from the
   *     preferred source of low modes to the source of the final spatial
   *     mesh. The meshes must have component-wise nondecreasing extents.
   * \param tensor_components Tensor component names to interpolate. The
   *     returned values have this order.
   * \param start_time Optional lower bound used to select observations from
   *     every subfile.
   * \param end_time Optional upper bound used to select observations from
   *     every subfile.
   * \param error_tolerance Optional absolute tolerance used to drop and
   *     compress modes. By default the tolerance is estimated from the
   *     lowest mode in the first subfile, which then needs at least 9
   *     observations.
   * \param verbosity Controls diagnostic output during construction.
   */
  ModalSpacetimeInterpolator(
      const std::variant<std::vector<std::string>, std::string>&
          volume_files_or_glob,
      std::vector<std::string> subfiles_in_priority_order,
      std::vector<std::string> tensor_components,
      std::optional<double> start_time = std::nullopt,
      std::optional<double> end_time = std::nullopt,
      std::optional<double> error_tolerance = std::nullopt,
      Verbosity verbosity = Verbosity::Quiet);

  ModalSpacetimeInterpolator(ModalSpacetimeInterpolator&&) = default;
  ModalSpacetimeInterpolator& operator=(ModalSpacetimeInterpolator&&) = default;
  ModalSpacetimeInterpolator(const ModalSpacetimeInterpolator&) = delete;
  ModalSpacetimeInterpolator& operator=(const ModalSpacetimeInterpolator&) =
      delete;
  ~ModalSpacetimeInterpolator() = default;

  /*!
   * \brief Interpolate the tensor components at a spacetime point.
   *
   * \param result Output buffer, resized to the number of tensor components.
   * \param target_point Coordinates of the target point in `Frame`.
   * \param time Time at which to evaluate the interpolant.
   * \param block_order Optional block-ordering hint that speeds up repeated
   *     block searches.
   */
  void interpolate_to_point(gsl::not_null<std::vector<double>*> result,
                            const tnsr::I<double, Dim, Frame>& target_point,
                            double time,
                            std::optional<gsl::not_null<std::vector<size_t>*>>
                                block_order = std::nullopt) const;

  /// Tensor components in the order returned by `interpolate_to_point()`
  const std::vector<std::string>& tensor_components() const {
    return tensor_components_;
  }

  /// Inclusive time interval shared by all input subfiles
  const std::array<double, 2>& time_bounds() const { return time_bounds_; }

 private:
  using ModeInterpolant = std::optional<intrp::UniformCardinalBSpline>;

  struct ElementData {
    Mesh<Dim> mesh{};
    // Indexed by [component][mode on the final mesh]
    std::vector<std::vector<ModeInterpolant>> interpolants{};
  };

  std::vector<std::string> tensor_components_{};
  std::array<double, 2> time_bounds_{};
  Domain<Dim> domain_{};
  domain::FunctionsOfTimeMap functions_of_time_{};
  std::map<size_t, domain::ElementSearchTree<Dim>> element_search_trees_{};
  std::unordered_map<ElementId<Dim>, ElementData> element_data_{};
};

}  // namespace spectre::Exporter
