\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

# Adaptive mesh refinement (AMR) {#dev_guide_amr}

\tableofcontents

### Introduction

SpECTRE implements a block-based anisotropic h-p adaptive mesh
refinement (AMR) algorithm.  Currently AMR can only occur at global
synchronization points, but it is planned to remove this restriction
in the future.

\warning AMR is still under active development. The elliptic
executables fully support AMR.  The evolution executables currently
only support p-adaptivity, and only when using a global time stepper,
and only when using a DG method.  The executable will abort with an
ERROR if local time stepping or h-refinement is done.

Here is an overview of what is described in detail in the sections below:

- \ref dev_guide_amr_algorithm "AMR Algorithm": An overview of the
  parallel phases, actions, and components that are used to do
  adaptive mesh refinement.
- \ref dev_guide_amr_criteria "AMR Criteria": Describes the
  expectations and restrictions on concrete classes implementing
  adaptive mesh refinement criteria.
- \ref dev_guide_amr_policies "AMR Policies": Describes restrictions
  and assumptions of the current algorithm, and what would be
  required to relax them.
- \ref dev_guide_amr_projectors "AMR Projectors": Describes the
  requirements on concrete classes which project data to the new or
  modified elements after mesh refinement.
- \ref dev_guide_amr_metavariables "AMR Metavariables": Describes the
  code that needs to be added to the metavariables of an executable in
  order to enable adaptive mesh refinement.
- \ref dev_guide_amr_input_options "Controlling AMR": Describes the
  options in the input file that control the behavior of adaptive mesh
  refinement.

### AMR Algorithm {#dev_guide_amr_algorithm}

The AMR algorithm requires communication via simple actions between
the elements of the array component that are responsible for the
Element%s of the Domain and a singleton amr::Component.  See \ref
dev_guide_parallelization_foundations
"Parallelization and Core Concepts" for an overview of phases,
actions, metavariables, and other SpECTRE parallelization concepts.

Currently the AMR algorithm is run over several phases when a phase
change is triggered by an executable.  See \ref
dev_guide_amr_input_options "Controlling AMR" for how to control when
the AMR phases are triggered.

#### Evaluating refinement criteria

When AMR is triggered, the executable should enter
Parallel::Phase::EvaluateAmrCriteria.  In this phase, the
amr::Component triggers the simple action
amr::Actions::EvaluateRefinementCriteria on each element of the array
component (specified by `metavariables::amr::element_array`).  Each
element evaluates (in order) the user-specified refinement criteria,
each of which returns a refinement decision in each logical dimension
of the element represented as an amr::Flag.  The overall refinement
decision in each logical dimension is obtained by taking the highest
priority decision in the dimension over all criteria, where
amr::Flag::Split has the highest priority and amr::Flag::Join has the
lowest.  See \ref dev_guide_amr_criteria "AMR Criteria" for more
details on implementing criteria.  Once the overall refinement
decision based on the refinement criteria is made, the decision may be
modified to satisfy the \ref dev_guide_amr_policies "AMR Policies".
Finally, the element sends its decision (and other information in
amr::Info) to its neighbors by calling the simple action
amr::Actions::UpdateAmrDecision.

In amr::Actions::UpdateAmrDecision, an element decides whether or not it
needs to update its own refinement decision (or other information in
amr::Info) based on decisions (or other information in amr::Info) sent
by its neighbor.  Possible reasons to update the refinement decision
are, e.g., if a sibling neighbor does not want to join an element that
had initially decided to join, or if any of the amr::Policies would
be violated.  If the refinement decision (or other information in
amr::Info) of the element changes, the element will send its
updated amr::Info to its neighbors.

Note that simple actions are executed asynchronusly, so both
amr::Actions::EvaluateRefinementCriteria and
amr::Actions::UpdateAmrDecision have to handle the possibility that
they are executed out of order.  For example,
amr::Actions::UpdateAmrDecision could be executed before
amr::Actions::EvaluateRefinementCriteria.  Eventually all elements
will reach a state where they do not need to send new information to
their neighbors, thus reaching a state of quiescence.  When this
happens Parallel::Phase::EvaluateAmrCriteria ends. At this point all
elements have reached a consensus on how the domain should be adapted,
but no changes have been made to the domain.  In order to adjust the
domain, the executable should next start
Parallel::Phase::AdjustDomain.

#### Adjusting the domain

At the beginning of Parallel::Phase::AdjustDomain, the amr::Component
calls amr::Actions::AdjustDomain on each element of the array
component (specified by `metavariables::amr::element_array`).  If the
element wants to split in any dimension, it will call the simple
action amr::Actions::CreateChild on the amr::Component in order to
create new refined elements that will replace the existing element.
If the element wants to join in any dimension, it will either call
amr::Actions::CreateParent on the amr::Component (if it is the joining
sibling with the lowest SegmentId in each dimension being joined) or
do nothing in order to create a single new coarsened element that will
replace multiple existing elements.  If the element neither wants to
split nor join in any dimension, then the element mutates items in its
db::DataBox.  First the Element, Mesh and neighbor Mesh are updated.
Then the element calls the
\ref dev_guide_amr_projectors "AMR Projectors" that will update all
the mutable items in the db::DataBox.
Finally the amr::Info of the element and its neighbors are reset and
cleared so they are ready for a future invocation of AMR.

When amr::Actions::CreateChild is executed by the amr::Component, it
creates a new element in the `metavariables::amr::element_array`
passing a Parallel::SimpleActionCallback that will be executed after
the new element is created.  The callback will be either to call
amr::Actions::CreateChild again for the next child to be created, or
amr::Actions::SendDataToChildren which will be invoked on the element
that is being split.  amr::Actions::SendDataToChildren will then
invoke amr::Actions::InitializeChild on each new child element,
passing along all of the mutable items in its
db::DataBox. amr::Actions::SendDataToChildren will then deregister and
delete the element being split. amr::Actions::InitializeChild will
create its Element, Mesh, and neighbor Mesh, and then call the \ref
dev_guide_amr_projectors "AMR Projectors" that will update all the
mutable items in its db::DataBox.

When amr::Actions::CreateParent is executed by the amr::Component, it
creates a new element in the `metavariables::amr::element_array`
passing a Parallel::SimpleActionCallback that will be executed after
the new element is created.  The callback will be
amr::Actions::CollectDataFromChildren which will be invoked on one
of the elements that are joining.
amr::Actions::CollectDataFromChildren will then invoke either itself
on another element that is joining or amr::Actions::InitializeParent
on the new parent element, collecting all of the mutable items in each
joining element's db::DataBox. amr::Actions::CollectDataFromChildren
will then deregister and delete the element that is
joining. amr::Actions::InitializeParent will create its Element, Mesh,
and neighbor Mesh, and then call the \ref dev_guide_amr_projectors
"AMR Projectors" that will update all the mutable items in its
db::DataBox.

When all of the actions finish, a state of quiescence is reached.
When this happens Parallel::Phase::AdjustDomain ends.  The executable
should now either return to the phase which triggered AMR, or enter
Parallel::Phase::CheckDomain.

### AMR Criteria {#dev_guide_amr_criteria}

When amr::Actions::EvaluateRefinementCriteria is executed, for each
element, it loops over a list of criteria that is specified in the
input file options
(See \ref dev_guide_amr_input_options "Controling AMR").
Each criterion must be an option-creatable class derived from
amr::Criterion (see its documentation for a list of requirements for
the derived classes).  In particular, each criterion should compute an
array of amr::Flag%s that represent the recommended refinemenent
choice in each logical dimension.  These choices can be computed from
any items in the db::DataBox of the element or additional compute
items specified by the criterion.  It is expected that a criterion
will compute something and then either choose to recommend refinement,
no change, or coarsening the element either in size (h-refinement) or
polynomial order (p-refinement).  A criterion does not need to handle
anything that is enforced by the amr::Policies such as worry about
bounds on the refinement level or number of grid points.

### AMR Policies {#dev_guide_amr_policies}

The current AMR algorithm has a set of rules that are enforced after
the overall refinement choice is determined from the AMR criteria.

The following rules simplify the code:

- An Element cannot join if it is splitting in any dimension.  Instead
  any amr::Flag::Join is changed to amr::Flag::DoNothing. Relaxing
  this restriction would be a non-trivial change that would require
  more complicated logic in determining if neighboring elements can
  join, as well as an additional code path to handle the creation and
  initialization of the new elements. This should not be a significant
  restriction as in most cases when this would occur it is unlikely
  that the siblings of the Element would agree to a valid join.

- In two or three spatial dimensions, there must be a 2:1 balance
  (i.e. within one refinement level) between neighbors in the
  dimension(s) parallel to the face of the element.  Relaxing this
  restriction would be a significant change to the code as the
  code for communicating boundary corrections assumes 2:1 balance.  In
  order to maintain 2:1 balance with neighboring elements, a choice of
  amr::Flag::Join may be changed to amr::Flag::DoNothing, and
  amr::Flag::DoNothing may be changed to amr::Flag::Split.

The following policies are among those controlled by input file
options (see \ref dev_guide_amr_input_options "Controlling AMR" and
the documentation for amr::Policies):

- Whether to do isotropic or anisotropic refinement.  If isotropic
  refinement is done, the AMR decision in each dimension are changed
  to the decision of the dimension with the highest priority amr::Flag
  (i.e. amr::Flag::Split has the highest priority).

- Whether or not to enforce 2:1 balance for the refinement level in
  the direction perpendicular to the face.

- The minimum and maximum of the refinement level and the number of
  grid points.  The actual bounds on these will be the stricter of
  those specified by the input file and those set by the code.  For
  the number of grid points, the code bounds are set by
  Spectral::minimum_number_of_points and
  Spectral::maximum_number_of_points for the Spectral::Basis and
  Spectral::Quadrature of the Mesh.  For the refinement level, the
  minimum is zero, and the maximum is set by
  `ElementId::max_refinement_level`.

### AMR Projectors {#dev_guide_amr_projectors}

After the grid adapts, the mutable items in the DataBox of new or
existing elements must be updated.  The Element, Mesh, neighbor Mesh,
amr::Info, and neighbor amr::Info along with the items in
`Parallel::Tags::distributed_object_tags` are automatically updated by
the AMR actions, but all other items must be explicitly updated via
projectors that conform to amr::protocols::Projector.  If any mutable
item has not been explicitly handled by any of the projectors, a
`static_assert` will be triggered listing the tags for the items that
have not been projected.

When a new element is created, the items in its DataBox are default
constructed.  Their Element, Mesh, and neighbor Mesh are mutated to
their desired state.  Then the AMR projectors are called, passing
along the items in the DataBox(es) from their splitting parent element
or joining children elements.  Existing elements first mutate their
Element, Mesh, and neighbor Mesh, and then call the AMR projectors,
passing along the old Element and Mesh.

The `return_tags` of each projector lists the tags for the items that
the projector mutates.  From the pre-AMR state, the projector must
compute the post-AMR state of these items.  Typically during
Parallel::Phase::Initialization, a group of items will be initialized
by a specific action or mutator in the action
`Initialization::Actions::InitializeItems`.  Therefore it makes sense
to create a projector that will handle the same group of items.
However some items are initialized from input file options, while
others are left default initialized and not mutated by any
initialization action.  These items will still need to be handled by
some projector.  Two convenience projectors are provided:
amr::projectors::DefaultInitialize which value initializes the items
corresponding to the listed tags; and
amr::projectors::CopyFromCreatorOrLeaveAsIs that will either copy the
item from the parent (or children, asserting that all children agree)
of a newly created element, or leave the item unmodified for an
existing element for the items corresponding to the listed tags.

The list of projectors is specified in
`metavariables::amr::projectors` in the executable, and the projectors
are evaluated in the order in which they are specified.

### Enabling AMR for an executable {#dev_guide_amr_metavariables}

In order to enable AMR for an executable, the following changes need
to be made in the metavariables:

- The addition of a struct named `amr` such as the following example:
```
struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using element_array = dg_element_array;
    using projectors = tmpl::list<::amr::projectors::DefaultInitialize<
        domain::Tags::InitialExtents<Dim>,
        domain::Tags::InitialRefinementLevels<Dim>,
        evolution::dg::Tags::Quadrature>>;
  };
```

  where `element_array` specifies the array component on which AMR
  will operate and `projectors` is a type list of amr::projectors
  that govern how the items in the DataBox for the `element_array` are
  updated after refinement.

- The addition of amr::Component to the `component_list` of the metavariables

- The addition (or modification) of the following `tmpl::pair`s in
  `factory_creation::factory_classes`:
```
        tmpl::pair<amr::Criterion,
                   tmpl::list<LIST_OF_CRITERIA>,
        tmpl::pair<
            PhaseChange,
            tmpl::list<
                PhaseControl::VisitAndReturn<
                    Parallel::Phase::EvaluateAmrCriteria>,
                PhaseControl::VisitAndReturn<Parallel::Phase::AdjustDomain>,
                PhaseControl::VisitAndReturn<Parallel::Phase::CheckDomain>>,
```
where `LIST_OF_CRITERIA` should be a list of amr::Criteria.

- The addition of the following item in the list of PhaseActions for
  the component specified in `amr::element_array`:

```
          Parallel::PhaseActions<Parallel::Phase::CheckDomain,
                                 tmpl::list<::amr::Actions::SendAmrDiagnostics,
                                            Parallel::Actions::TerminatePhase>>,
```

- The possible addition of `PhaseControl::Actions::ExecutePhaseChange`
  to the action list for the appropriate Phase in the PhaseAction of
  the `amr::element_array`

- The addition of the mutator `amr::Initialization::Initialize` to the
  list of mutators in `Initialization::Actions::InitializeItems` in
  the Parallel::PhaseActions list for Parallel::Phase::Initialization.

- The appropriate includes for all of the above.

### Controlling AMR {#dev_guide_amr_input_options}

There are two places in the input file that control how and when AMR happens.

The "how" is controlled by the option group `Amr`.  Here you list the
amr::Criteria being used, the available options for the amr::Policies, and
the `Verbosity` of diagnostic messages that are printed.  Here is an
example:
```
Amr:
  Criteria:
    - TruncationError:
        VariablesToMonitor: [Psi]
        AbsoluteTarget: 1.e-6
        RelativeTarget: 1.0
  Policies:
    EnforceTwoToOneBalanceInNormalDirection: true
    Isotropy: Anisotropic
    Limits:
      RefinementLevel: Auto
      NumGridPoints: Auto
      ErrorBeyondLimits: False
  Verbosity: Verbose
```

Note that the values `Auto` for the Limits options choose the default
limits set by the code. Also note that if a criterion tries to refine beyond the
limits, whether or not the code should error is controlled by
`ErrorBeyondLimits`.

"When" AMR happens is controlled by specifying a Trigger and a list of
`PhaseChanges` in the top-level option `PhaseChangesAndTriggers`.  For
example:
```
PhaseChangeAndTriggers:
  - Trigger:
      Slabs:
        EvenlySpaced:
          Interval: 10
          Offset: 0
    PhaseChanges:
      - VisitAndReturn(EvaluateAmrCriteria)
      - VisitAndReturn(AdjustDomain)
      - VisitAndReturn(CheckDomain)
```

Both `EvaluateAmrCriteria` and `AdjustDomain` are required in order
for AMR to work.  `VisitAndReturn(CheckDomain)` performs diagnostics
and can be omitted.  To turn off AMR, omit the three phase changes above.
