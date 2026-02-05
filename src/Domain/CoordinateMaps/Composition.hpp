// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/Autodiff/Autodiff.hpp"
#include "Utilities/Simd/Simd.hpp"

namespace domain::CoordinateMaps {

/*!
 * \brief A composition of coordinate maps at runtime
 *
 * Composes a sequence of `domain::CoordinateMapBase` that step through the
 * `Frames`. The result is another `domain::CoordinateMapBase`. This is
 * different to `domain::CoordinateMap`, which does the composition at compile
 * time. The use cases are different:
 *
 * - Use `domain::CoordinateMap` to compose maps at compile time to go from one
 *   (named) frame to another using any number of coordinate transformation. The
 *   coordinate transformations are concatenated to effectively form a single
 *   map, and intermediate frames have no meaning. This has the performance
 *   benefit that looking up pointers and calling into virtual member functions
 *   of intermediate maps is avoided, and it has the semantic benefit that
 *   intermediate frames without meaning are not named or even accessible.
 *   **Example:** A static BlockLogical -> Grid map that deforms the logical
 *   cube to a wedge, applies a radial redistribution of grid points, and
 *   translates + rotates the wedge.
 * - Use `domain::CoordinateMaps::Composition` (this class) to compose maps at
 *   runtime to step through a sequence of (named) frames.
 *   **Example:** A time-dependent ElementLogical -> BlockLogical -> Grid ->
 *   Inertial map that applies an affine transformation between the
 *   ElementLogical and BlockLogical frames (see
 *   domain::element_to_block_logical_map), then deforms the logical cube to a
 *   wedge using the map described above (Grid frame), and then rotates the grid
 *   with a time-dependent rotation map (Inertial frame).
 *
 * \warning Think about performance implications before using this Composition
 * class. In an evolution it's usually advantageous to keep at least some of the
 * maps separate, e.g. to avoid reevaluating the static maps in every time step.
 * You can also access individual components of the composition in this class.
 *
 * \tparam Frames The list of frames in the composition, as a tmpl::list<>.
 * The first entry in the list is the source frame, and the last entry is the
 * target frame of the composition. Maps in the composition step through the
 * `Frames`. For example, if `Frames = tmpl::list<Frame::ElementLogical,
 * Frame::BlockLogical, Frame::Inertial>`, then the composition has two maps:
 * ElementLogical -> BlockLogical and BlockLogical -> Inertial.
 *
 * ## Note on inverse Hessian computation
 * Below we work out the algebra for computing the inverse Hessian
 * used in this struct (if SPECTRE_AUTODIFF=ON). Suppose we have
 * a composition of maps \f$ \xi^i \longrightarrow \cdots \longrightarrow x^i
 * \longrightarrow y^i \f$, where \f$ \xi^i \f$ are the source coordinates, \f$
 * x^i \f$ are some intermediate coordinates, and \f$ y^i \f$ are the final
 * target coordinates,  We compute inverse Hessian by first
 * propagating autodiff dual types through the composed `call_impl` function to
 * automatically get the Hessian \f$ \frac{\partial^2 y^i}{\partial\xi^j
 * \partial\xi^k} \f$ Then the inverse Hessian is computed by the identity
 * \f[ \frac{\partial^2\xi^i}{\partial y^m \partial y^n} =
 * -\frac{\partial\xi^i}{\partial y^j}\frac{\partial\xi^k}{\partial y^m}
 * \frac{\partial\xi^l}{\partial y^n}\frac{\partial^2 y^j}{\partial\xi^k
 * \partial\xi^l}, \f] where the inverse Jacobian is passed in as a function
 * argument.
 *
 * See Test_Composition.cpp for a different implementation. The current
 * implementation is chosen in production as it is faster when we have
 * the inverse Jacobian already, and in most cases we do.
 *
 * ## Note on auto differentiation
 * We use forward mode autodiff here as it is simpler to implement and
 * has better optimization than the reverse mode. Reverse mode in
 * the [Autodiff](https://github.com/autodiff/autodiff) library
 * has higher cost per propagation and does not support taping.
 * Also see this
 * [github issue](https://github.com/autodiff/autodiff/issues/332).
 */
template <typename Frames, size_t Dim,
          typename Is = std::make_index_sequence<tmpl::size<Frames>::value - 1>>
struct Composition;

template <typename Frames, size_t Dim, size_t... Is>
struct Composition<Frames, Dim, std::index_sequence<Is...>>
    : public CoordinateMapBase<tmpl::front<Frames>, tmpl::back<Frames>, Dim> {
  using frames = Frames;
  using SourceFrame = tmpl::front<Frames>;
  using TargetFrame = tmpl::back<Frames>;
  static constexpr size_t num_frames = tmpl::size<Frames>::value;
  using Base = CoordinateMapBase<SourceFrame, TargetFrame, Dim>;
  using FuncOfTimeMap = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;
  using Base::operator();

  Composition() = default;
  Composition(const Composition& rhs) : PUP::able(rhs) { *this = rhs; }
  Composition& operator=(const Composition& rhs);
  Composition(Composition&& /*rhs*/) = default;
  Composition& operator=(Composition&& /*rhs*/) = default;
  ~Composition() override = default;

  std::unique_ptr<Base> get_clone() const override {
    return std::make_unique<Composition>(*this);
  }

  /// \cond
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Composition);  // NOLINT
  /// \endcond

  Composition(std::unique_ptr<CoordinateMapBase<
                  tmpl::at<frames, tmpl::size_t<Is>>,
                  tmpl::at<frames, tmpl::size_t<Is + 1>>, Dim>>... maps);

  const std::tuple<std::unique_ptr<
      CoordinateMapBase<tmpl::at<frames, tmpl::size_t<Is>>,
                        tmpl::at<frames, tmpl::size_t<Is + 1>>, Dim>>...>&
  maps() const {
    return maps_;
  }

  template <typename LocalSourceFrame, typename LocalTargetFrame>
  const CoordinateMapBase<LocalSourceFrame, LocalTargetFrame, Dim>&
  get_component() const {
    return *get<std::unique_ptr<
        CoordinateMapBase<LocalSourceFrame, LocalTargetFrame, Dim>>>(maps_);
  }

  bool is_identity() const override;

  bool inv_jacobian_is_time_dependent() const override;

  bool jacobian_is_time_dependent() const override;

  const std::unordered_set<std::string>& function_of_time_names()
      const override {
    return function_of_time_names_;
  }

  bool supports_hessian() const override;

  tnsr::I<double, Dim, TargetFrame> operator()(
      tnsr::I<double, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FuncOfTimeMap& functions_of_time = {}) const override;

  tnsr::I<DataVector, Dim, TargetFrame> operator()(
      tnsr::I<DataVector, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FuncOfTimeMap& functions_of_time = {}) const override;

  std::optional<tnsr::I<double, Dim, SourceFrame>> inverse(
      tnsr::I<double, Dim, TargetFrame> target_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FuncOfTimeMap& functions_of_time = {}) const override;

  InverseJacobian<double, Dim, SourceFrame, TargetFrame> inv_jacobian(
      tnsr::I<double, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FuncOfTimeMap& functions_of_time = {}) const override;

  InverseJacobian<DataVector, Dim, SourceFrame, TargetFrame> inv_jacobian(
      tnsr::I<DataVector, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FuncOfTimeMap& functions_of_time = {}) const override;

#ifdef SPECTRE_AUTODIFF
  InverseJacobian<autodiff::HigherOrderDual<2, double>, Dim, SourceFrame,
                  TargetFrame>
  inv_jacobian(tnsr::I<autodiff::HigherOrderDual<2, double>, Dim, SourceFrame>
                   source_point,
               double time = std::numeric_limits<double>::signaling_NaN(),
               const FuncOfTimeMap& functions_of_time = {}) const override;

  // Compute inverse hessian by calling operator()
  InverseHessian<double, Dim, SourceFrame, TargetFrame> inv_hessian(
      tnsr::I<double, Dim, SourceFrame> source_point,
      const InverseJacobian<double, Dim, SourceFrame, TargetFrame>& inverse_jac,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FuncOfTimeMap& functions_of_time = {}) const override;

  // Compute inverse hessian by calling operator()
  InverseHessian<DataVector, Dim, SourceFrame, TargetFrame> inv_hessian(
      tnsr::I<DataVector, Dim, SourceFrame> source_point,
      const InverseJacobian<DataVector, Dim, SourceFrame, TargetFrame>&
          inverse_jac,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FuncOfTimeMap& functions_of_time = {}) const override;
#endif  // SPECTRE_AUTODIFF

  Jacobian<double, Dim, SourceFrame, TargetFrame> jacobian(
      tnsr::I<double, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FuncOfTimeMap& functions_of_time = {}) const override;

  Jacobian<DataVector, Dim, SourceFrame, TargetFrame> jacobian(
      tnsr::I<DataVector, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FuncOfTimeMap& functions_of_time = {}) const override;

  [[noreturn]] std::tuple<
      tnsr::I<double, Dim, TargetFrame>,
      InverseJacobian<double, Dim, SourceFrame, TargetFrame>,
      Jacobian<double, Dim, SourceFrame, TargetFrame>,
      tnsr::I<double, Dim, TargetFrame>>
  coords_frame_velocity_jacobians(
      tnsr::I<double, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FuncOfTimeMap& functions_of_time = {}) const override;

  [[noreturn]] std::tuple<
      tnsr::I<DataVector, Dim, TargetFrame>,
      InverseJacobian<DataVector, Dim, SourceFrame, TargetFrame>,
      Jacobian<DataVector, Dim, SourceFrame, TargetFrame>,
      tnsr::I<DataVector, Dim, TargetFrame>>
  coords_frame_velocity_jacobians(
      tnsr::I<DataVector, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FuncOfTimeMap& functions_of_time = {}) const override;

  [[noreturn]] std::unique_ptr<CoordinateMapBase<SourceFrame, Frame::Grid, Dim>>
  get_to_grid_frame() const override;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  template <typename DataType>
  tnsr::I<DataType, Dim, TargetFrame> call_impl(
      tnsr::I<DataType, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FuncOfTimeMap& functions_of_time = {}) const;

  template <typename DataType>
  std::optional<tnsr::I<DataType, Dim, SourceFrame>> inverse_impl(
      tnsr::I<DataType, Dim, TargetFrame> target_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FuncOfTimeMap& functions_of_time = {}) const;

  template <typename DataType>
  InverseJacobian<DataType, Dim, SourceFrame, TargetFrame> inv_jacobian_impl(
      tnsr::I<DataType, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FuncOfTimeMap& functions_of_time = {}) const;

  template <typename DataType>
  InverseHessian<DataType, Dim, SourceFrame, TargetFrame> inv_hessian_impl(
      tnsr::I<DataType, Dim, SourceFrame> source_point,
      const ::InverseJacobian<DataType, Dim, SourceFrame, TargetFrame>&
          inverse_jac,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FuncOfTimeMap& functions_of_time = {}) const;

  template <typename DataType>
  Jacobian<DataType, Dim, SourceFrame, TargetFrame> jacobian_impl(
      tnsr::I<DataType, Dim, SourceFrame> source_point,
      double time = std::numeric_limits<double>::signaling_NaN(),
      const FuncOfTimeMap& functions_of_time = {}) const;

  bool is_equal_to(const CoordinateMapBase<SourceFrame, TargetFrame, Dim>&
                       other) const override;

  std::tuple<std::unique_ptr<
      CoordinateMapBase<tmpl::at<frames, tmpl::size_t<Is>>,
                        tmpl::at<frames, tmpl::size_t<Is + 1>>, Dim>>...>
      maps_;
  std::unordered_set<std::string> function_of_time_names_;
};

// Template deduction guide
template <typename FirstMap, typename... Maps>
Composition(std::unique_ptr<FirstMap>, std::unique_ptr<Maps>... maps)
    -> Composition<tmpl::list<typename FirstMap::source_frame,
                              typename FirstMap::target_frame,
                              typename Maps::target_frame...>,
                   FirstMap::dim>;

/// \cond
template <typename Frames, size_t Dim, size_t... Is>
PUP::able::PUP_ID
    Composition<Frames, Dim, std::index_sequence<Is...>>::my_PUP_ID = 0;
/// \endcond

}  // namespace domain::CoordinateMaps
