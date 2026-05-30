// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Mesh;
/// \endcond

namespace domain::BoundaryConditions {
/// Mark a boundary condition as being used as an internal Cartoon boundary.
///
/// The cartoon method requires a ZernikeB1 basis to be stable at small
/// \f$x\f$, which should not have a boundary condition applied. However, FD
/// does require ghost zones to be filled on this boundary, so `is_cartoon()`
/// can be used to determine whether to treat this boundary condition as
/// something to implement or skip.
class MarkAsCartoon {
 public:
  MarkAsCartoon() = default;
  MarkAsCartoon(MarkAsCartoon&&) = default;
  MarkAsCartoon& operator=(MarkAsCartoon&&) = default;
  MarkAsCartoon(const MarkAsCartoon&) = default;
  MarkAsCartoon& operator=(const MarkAsCartoon&) = default;
  virtual ~MarkAsCartoon() = 0;
};

/*!
 * \brief Cartoon boundary conditions, to be used as the default placeholder in
 * systems without Subcell.
 *
 * To use with a specific system, add:
 *
 * \code
 *  domain::BoundaryConditions::Cartoon<your::system::BoundaryConditionBase>
 * \endcode
 *
 * to the list of creatable classes.
 *
 * Note: Cartoon boundary conditions should only be specified with systems
 * set-up to use the cartoon method. It should not be used as an external
 * boundary.
 */
template <typename SystemBoundaryConditionBaseClass>
struct Cartoon final : public SystemBoundaryConditionBaseClass,
                       public MarkAsCartoon {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Cartoon boundary condition, to be used in systems that do not implement "
      "Subcell.\n\nNote: This should never be used as an external boundary, it "
      "is only used on specific cartoon-system boundaries that are "
      "automatically handled in the domain creators."};
  static std::string name() { return "Cartoon"; }

  Cartoon() = default;
  Cartoon(Cartoon&&) = default;
  Cartoon& operator=(Cartoon&&) = default;
  Cartoon(const Cartoon&) = default;
  Cartoon& operator=(const Cartoon&) = default;
  ~Cartoon() override = default;

  explicit Cartoon(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(
      domain::BoundaryConditions::BoundaryCondition, Cartoon);

  auto get_clone() const -> std::unique_ptr<
      domain::BoundaryConditions::BoundaryCondition> override;

  void pup(PUP::er& p) override;
};

template <typename SystemBoundaryConditionBaseClass>
Cartoon<SystemBoundaryConditionBaseClass>::Cartoon(CkMigrateMessage* const msg)
    : SystemBoundaryConditionBaseClass(msg) {}

template <typename SystemBoundaryConditionBaseClass>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Cartoon<SystemBoundaryConditionBaseClass>::get_clone() const {
  return std::make_unique<Cartoon>(*this);
}

template <typename SystemBoundaryConditionBaseClass>
void Cartoon<SystemBoundaryConditionBaseClass>::pup(PUP::er& p) {
  BoundaryCondition::pup(p);
}

/// \cond
template <typename SystemBoundaryConditionBaseClass>
// NOLINTNEXTLINE
PUP::able::PUP_ID Cartoon<SystemBoundaryConditionBaseClass>::my_PUP_ID = 0;
/// \endcond

/// Check if a boundary condition inherits from `MarkAsCartoon`, which
/// constitutes as it being marked as an internal Cartoon boundary condition.
bool is_cartoon(const std::unique_ptr<BoundaryCondition>& boundary_condition);

/// Check if a mesh is compatible with a Cartoon boundary condition, i.e. it is
/// using cartoon bases in a proper way.
template <size_t Dim>
bool dg_mesh_is_cartoon_compatible(const Mesh<Dim>& dg_mesh);

namespace detail {
template <typename T>
struct inherits_from_mark_as_cartoon : std::is_base_of<MarkAsCartoon, T> {};

template <typename List>
struct find_cartoon_bc_impl {
  using filtered_list =
      tmpl::filter<List, inherits_from_mark_as_cartoon<tmpl::_1>>;

  // Ensure there's exactly one cartoon BC, not zero or multiple
  static_assert(tmpl::size<filtered_list>::value <= 1,
                "Multiple cartoon boundary conditions found in factory list. "
                "Only one cartoon boundary condition is allowed per system.");

  // Need lazy evaluation in case list is empty
  template <typename L>
  using get_maybe_first = tmpl::apply<tmpl::apply<
      tmpl::if_<std::bool_constant<(tmpl::size<L>::value != 0)>,
                tmpl::defer<tmpl::bind<tmpl::front, tmpl::pin<L>>>, void>>>;

  using type = get_maybe_first<filtered_list>;
};

/// Find the unique type in a tmpl::list that inherits from MarkAsCartoon
template <typename List>
using find_cartoon_bc = typename find_cartoon_bc_impl<List>::type;

/// Check if a tmpl::list contains any types that inherit from MarkAsCartoon
template <typename List>
constexpr bool has_cartoon_bc_v = not std::is_void_v<find_cartoon_bc<List>>;

/// Filter out cartoon boundary conditions from a list, leaving only external
/// BCs
template <typename List>
using filter_out_cartoon_bcs =
    tmpl::remove_if<List, inherits_from_mark_as_cartoon<tmpl::_1>>;
}  // namespace detail

/// Extract the cartoon boundary condition type from a system's boundary
/// condition list. Returns void if no cartoon boundary condition is found.
template <typename Metavariables>
using get_cartoon_boundary_condition_from_system = detail::find_cartoon_bc<
    tmpl::at<typename Metavariables::factory_creation::factory_classes,
             typename Metavariables::system::boundary_conditions_base>>;

/// Extract only the external (non-cartoon) boundary conditions from a system's
/// boundary condition list. This should be used for user-selectable boundary
/// condition options to prevent cartoon BCs from being specified as external
/// BCs.
template <typename Metavariables>
using get_external_boundary_conditions_from_system =
    detail::filter_out_cartoon_bcs<
        tmpl::at<typename Metavariables::factory_creation::factory_classes,
                 typename Metavariables::system::boundary_conditions_base>>;

/// Check if a system has a cartoon boundary condition available
template <typename Metavariables>
constexpr bool system_has_cartoon_bc_v = not std::is_void_v<
    get_cartoon_boundary_condition_from_system<Metavariables>>;

/// Create a cartoon boundary condition for systems that support it.
/// Returns nullptr if the system doesn't have a cartoon boundary condition.
template <typename Metavariables>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
make_cartoon_boundary_condition() {
  if constexpr (system_has_cartoon_bc_v<Metavariables>) {
    using cartoon_bc_type =
        get_cartoon_boundary_condition_from_system<Metavariables>;
    return std::make_unique<cartoon_bc_type>();
  } else {
    return nullptr;
  }
}
}  // namespace domain::BoundaryConditions
