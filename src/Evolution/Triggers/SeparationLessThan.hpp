// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Grid;
struct Inertial;
struct Distorted;
}  // namespace Frame
namespace domain::Tags {
template <size_t VolumeDim>
struct Domain;
struct FunctionsOfTime;
template <size_t VolumeDim>
struct Element;
template <ObjectLabel Label>
struct ObjectCenter;
}  // namespace domain::Tags
template <size_t VolumeDim>
struct Domain;
template <size_t VolumeDim>
struct Element;
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace Triggers {
/*!
 * \brief A standard trigger that monitors the separation between
 * `domain::Tags::ObjectCenter<A>` and `domain::Tags::ObjectCenter<B>` in the
 * inertial frame. Once the separation is smaller than (or equal to) the input
 * separation, then the `operator()` returns true.
 *
 * \note This trigger requires that
 * `domain::Tags::ObjectCenter<domain::ObjectLabel::A>` and
 * `domain::Tags::ObjectCenter<domain::ObjectLabel::B>` are in the DataBox. It
 * also requires that there are two ExcisionSphere%s in the Domain named
 * `ExcisionSphereA/B` and that these ExcisionSphere%s have had time dependent
 * maps injected into them. The coordinate maps from these ExcisionSphere%s will
 * be used to calculate the separation in the inertial frame.
 */
class SeparationLessThan : public Trigger {
 public:
  /// \cond
  SeparationLessThan() = default;
  explicit SeparationLessThan(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(SeparationLessThan);  // NOLINT
  /// \endcond

  struct Value {
    using type = double;
    static constexpr Options::String help = {
        "Separation of the two horizons to compare against."};
  };

  using options = tmpl::list<Value>;
  static constexpr Options::String help{
      "Trigger when the separation between the two horizons is less than a "
      "certain distance."};

  explicit SeparationLessThan(double separation);

  using argument_tags =
      tmpl::list<Tags::Time, domain::Tags::Domain<3>,
                 domain::Tags::FunctionsOfTime,
                 domain::Tags::ObjectCenter<domain::ObjectLabel::A>,
                 domain::Tags::ObjectCenter<domain::ObjectLabel::B>>;

  bool operator()(
      const double time, const ::Domain<3>& domain,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const tnsr::I<double, 3, Frame::Grid>& grid_object_center_a,
      const tnsr::I<double, 3, Frame::Grid>& grid_object_center_b) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  double separation_{};
};
}  // namespace Triggers
