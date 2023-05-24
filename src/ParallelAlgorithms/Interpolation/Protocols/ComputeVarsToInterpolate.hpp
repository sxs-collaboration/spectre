// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct ElementLogical;
struct Grid;
struct Inertial;
}  // namespace Frame
template <size_t Dim>
struct Mesh;
struct DataVector;
/// \endcond

namespace intrp::protocols {
/*!
 * \brief A protocol for the type alias `compute_vars_to_interpolate` that can
 * potentially be found in an InterpolationTargetTag (potentially because an
 * InterpolationTargetTag does not require this type alias).
 *
 * \details A struct conforming to the `ComputeVarsToInterpolate` protocol must
 * have
 *
 * - a type alias `allowed_src_tags` which is a `tmpl::list` of tags from the
 *   volume that can be used to compute quantities that will be interpolated
 *   onto the target.
 *
 * - a type alias `required_src_tags` which is a `tmpl::list` of tags from the
 *   volume that are necessary to compute quantities that will be interpolated
 *   onto the target. This list must be a subset of `allowed_src_tags`.
 *
 * - a type alias `allowed_dest_tags<Frame>` which is a `tmpl::list` of tags on
 *   the target that can be computed from the source tags.
 *
 * - a type alias `required_dest_tags<Frame>` which is a `tmpl::list` of tags on
 *   the target that must be computed from the source tags. This list must be a
 *   subset of `allowed_dest_tags<Frame>`
 *
 * - an `apply` function with at least one of the signatures in the example.
 *   This apply function fills the `vars_to_interpolate_to_target` type alias
 *   from the InterpolationTargetTag. Here `Dim` is `Metavariables::volume_dim`,
 *   SrcFrame is the frame of `Metavariables::interpolator_source_vars` and
 *   TargetFrame is the frame of `vars_to_interpolate_to_target` from the
 *   InterpolationTargetTag. The overload without Jacobians treats the case in
 *   which TargetFrame is the same as SrcFrame.
 *
 * Here is an example of a class that conforms to this protocols:
 *
 * \snippet Helpers/ParallelAlgorithms/Interpolation/Examples.hpp ComputeVarsToInterpolate
 */
struct ComputeVarsToInterpolate {
  template <typename ConformingType>
  struct test {
    template <size_t Dim>
    struct DummyTag : db::SimpleTag {
      using type = tnsr::a<DataVector, Dim, Frame::Grid>;
    };
    template <size_t Dim>
    using example_list = tmpl::list<DummyTag<Dim>>;

    template <typename T, size_t Dim, typename = std::void_t<>>
    struct has_signature_1 : std::false_type {};

    template <typename T, size_t Dim>
    struct has_signature_1<
        T, Dim,
        std::void_t<decltype(T::apply(
            std::declval<const gsl::not_null<Variables<example_list<Dim>>*>>(),
            std::declval<const Variables<example_list<Dim>>&>(),
            std::declval<const Mesh<Dim>&>()))>> : std::true_type {};

    template <typename T, size_t Dim, typename = std::void_t<>>
    struct has_signature_2 : std::false_type {};

    template <typename T, size_t Dim>
    struct has_signature_2<
        T, Dim,
        std::void_t<decltype(T::apply(
            std::declval<const gsl::not_null<Variables<example_list<Dim>>*>>(),
            std::declval<const Variables<example_list<Dim>>&>(),
            std::declval<const Mesh<Dim>&>(),
            std::declval<const Jacobian<DataVector, Dim, Frame::Grid,
                                        Frame::Inertial>&>(),
            std::declval<const InverseJacobian<DataVector, Dim, Frame::Grid,
                                               Frame::Inertial>&>(),
            std::declval<const Jacobian<DataVector, Dim, Frame::ElementLogical,
                                        Frame::Grid>&>(),
            std::declval<const InverseJacobian<
                DataVector, Dim, Frame::ElementLogical, Frame::Grid>&>(),
            std::declval<const tnsr::I<DataVector, Dim, Frame::Inertial>&>(),
            std::declval<const tnsr::I<DataVector, Dim, Frame::Grid>&>()))>>
        : std::true_type {};

    static_assert(has_signature_1<ConformingType, 1>::value or
                  has_signature_2<ConformingType, 1>::value or
                  has_signature_1<ConformingType, 2>::value or
                  has_signature_2<ConformingType, 2>::value or
                  has_signature_1<ConformingType, 3>::value or
                  has_signature_2<ConformingType, 3>::value);

    using allowed_src_tags = typename ConformingType::allowed_src_tags;
    using required_src_tags = typename ConformingType::required_src_tags;
    static_assert(
        tmpl::size<tmpl::list_difference<required_src_tags,
                                         allowed_src_tags>>::value == 0,
        "Some of the required source tags are not in the allowed source tags.");

    // Checking this in the grid frame is just a choice, but it suffices to
    // check conformance.
    using allowed_dest_tags =
        typename ConformingType::template allowed_dest_tags<Frame::Grid>;
    using required_dest_tags =
        typename ConformingType::template required_dest_tags<Frame::Grid>;
    static_assert(tmpl::size<tmpl::list_difference<required_dest_tags,
                                                   allowed_dest_tags>>::value ==
                      0,
                  "Some of the required destination tags are not in the "
                  "allowed destination tags.");
  };
};
}  // namespace intrp::protocols
