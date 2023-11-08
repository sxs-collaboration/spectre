\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# IMEX {#tutorial_imex}

\tableofcontents

### Introduction to IMEX

IMEX (implicit-explicit) integration is a method for stabilizing the numerical
evolution of stiff differential equations.  While "stiff" is not a precisely
defined term, it generally refers to equations where the time step required to
achieve numerical stability is much shorter than the timescale of the evolution
of the solution, often by orders of magnitude.  An example of a stiff equation
is

\f{equation}
  \label{eq:example-equation}
  \frac{dy}{dt} = -k (y - f(t))
\f}

with $k$ very large.  After a brief initial transient, the solution is very
well-approximated as $y = f(t) + O(k^{-1})$, but will be unstable if evolved
using an explicit time stepper with a step size larger than approximately
$k^{-1}$, even if $f$ is slowly varying.

To see the source of the stability problem, consider evolving
$\eqref{eq:example-equation}$ using Euler's method.  For a step from $t_0$ to
$t_1 = t_0 + \Delta t$, we find

\f{equation}
  \label{eq:example-Euler}
  y_1 = y_0 + \Delta t \frac{dy}{dt}(y_0, t_0)
  = y_0 - k \Delta t (y_0 - f(t_0)),
\f}

giving the divergence from the known infinite-$k$ solution of

\f{equation}
  y_1 - f(t_1) = y_0 - f(t_1) - k \Delta t (y_0 - f(t_0)).
\f}

If $k \Delta t$, is large, the last term will dominate and the deviation from
the approximate solution will increase by roughly that factor every step.

To improve this, we can evolve using an implicit time stepper.  Taking the same
step as above with the backwards Euler method gives

\f{equation}
  \label{eq:example-backwards-Euler}
  y_1 = y_0 + \Delta t \frac{dy}{dt}(y_1, t_1)
  = y_0 - k \Delta t (y_1 - f(t_1)),
\f}

for a deviation of

\f{equation}
  y_1 - f(t_1) = \frac{y_0 - f(t_1)}{1 + k \Delta t}.
\f}

This goes to zero as $k \Delta t$ becomes large, showing convergence to the
infinite-$k$ solution.

The unconditional (linear) stability achievable by implicit integrators allows
them to solve problems not feasible with explicit methods, but it comes at a
significant price.  While the explicit update $\eqref{eq:example-Euler}$
directly gave $y_1$, the implicit update $\eqref{eq:example-backwards-Euler}$
required the equation to be solved for the updated quantity.  In this example
that was fairly easy, but, in a real application, finding an analytic solution
is likely to be difficult, if not impossible.  Implicit integrators are
therefore usually used with a numerical solve of the update equation.  Even for
simple systems, this root-finding procedure adds significantly to the
computational cost of the method, and for a system like a large PDE evolution
it is intractable.

To mitigate this problem, we turn to the hybrid IMEX methods.  The idea behind
IMEX is to split the derivative into two terms, one without stability problems
that will be treated explicitly, and another that will be treated implicitly.
Notationally, we will write

\f{equation}
  \frac{dy}{dt}(y, t) = E(y, t) + I(y, t).
\f}

The simplest IMEX integrator combines the two forms of Euler's method from
above:

\f{equation}
  y_1 = y_0 + \Delta t (E(y_0, t_0) + I(y_1, t_1)).
\f}

(This integrator is not implemented in SpECTRE.)  In our example above, we can,
as an example, split $\eqref{eq:example-equation}$ as

\f{align}
  E(y, t) &= k f(t) &
  I(y, t) &= -k y,
\f}

which gives

\f{equation}
  y_1 = y_0 + k \Delta t (f(t_0) - y_1)
  = \frac{y_0 + k \Delta t\, f(t_0)}{1 + k \Delta t}.
\f}

In the limit where $k \Delta t$ goes to infinity, this gives $y_1 = f(t_0)$,
which is as small a deviation from the infinite-$k$ solution as possible when
$f$ is treated explicitly.

Again, in this simple case there is no downside to using an IMEX integrator
over an explicit one, or an implicit method over IMEX.  The gain appears in
real cases where the implicit-step equation cannot be solved exactly.

#### Properties of IMEX time steppers

Any IMEX time stepper can be used as either an implicit or an explicit time
stepper by choosing $E$ or $I$ to be zero.  As such, many significant
properties of an IMEX time stepper are actually properties of one or the other
part.  We won't discuss the standard properties of explicit time steppers here,
but most of them, such as error estimation, are used in SpECTRE ignoring the
implicit part.

As the entire point of using implicit methods is to increase stability, the
most important properties of the implicit part of a time stepper are stability
properties.  There are many stability classifications, but we will only discuss
the most important here, and those only qualitatively.  More information can be
found in \cite Hairer1996 and \cite Kennedy2016.

The basic desire for an implicit method is that as the equation being evolved
becomes increasingly stiff the evolution remains stable.  (Linear) stability
for an infinitely large step size (or, equivalently, an infinitely stiff
equation) is known as *A-stability*.  Adams methods above second order cannot
be A-stable, so IMEX work generally focuses on Runge-Kutta schemes. All IMEX
time steppers implemented in SpECTRE are A-stable.

A stronger stability condition is known as *L-stability*.  L-stability is
similar to A-stability, but instead of merely requiring that an analytically
decaying equation does not numerically diverge in the large-step-size limit, we
require that the solution reaches zero in a single step.  Since IMEX
applications can take steps orders of magnitude larger than the decay timescale
of stiff terms, this property is often desirable.

In addition to the stability properties, there is one interesting property of
the IMEX stepper as a whole.  A wide class of time steppers, including all
explicit (global time-stepping) methods implementable in the current SpECTRE
interface, preserve (linear) conserved quantities.  A linear conserved quantity
is a linear combination of the evolved variables that is analytically constant
under evolution, independent of the initial conditions.  In physics, a common
example is the integral of the density, i.e., the total mass.  Such quantities
will be numerically preserved during evolution to the level of roundoff error,
even if that is much smaller than the truncation error in the solution.

Both the explicit and implicit portions of all IMEX time steppers that we will
consider are conservative in this manner.  However, the combination of the two
parts into an IMEX scheme does not necessarily preserve this property.  In
general, both the explicit and implicit parts of the split derivative must
individually conserve something for the full IMEX method to conserve it as
well.  Some IMEX schemes, however, do preserve conserved quantities,
independent of how the derivative is split between the explicit and implicit
parts.  Such methods are termed *stiffly accurate*.  (This property is also
useful in deriving and analyzing implicit schemes, but that is not of great
importance to users of the methods.)

### IMEX in SpECTRE

SpECTRE supports IMEX integration, with restrictions to reduce the cost of the
solve of the implicit equation as much as possible.  Additionally, not all
integration schemes extend to IMEX integration, so the list of available time
steppers is limited for IMEX evolutions.

#### Mathematical restrictions

For problems of interest to us, the evolved variables consist of several fields
coupled by the system derivative.  The implicit solve is more expensive when
performed on more variables, so there is a method to restrict the solve to a
subset of the system if only some variables are affected by stiff terms.  The
system can define one or more *implicit sectors*, which are subsets of the
evolved variables with implicit terms that can be solved independently from one
another.  (If sectors depend on variables from other sectors, the evolution
system defines the order in which the implicit updates are applied.)

As an example, for a fluid coupled to neutrinos, the neutrinos obey stiff
equations under some conditions.  If we consider neutrino flavors to be
non-interacting, they can each be placed in their own implicit sector and
solved independently.  The fluid equations will usually be non-stiff, so the
fluid variables will not be in any sector, even though they will still be used
as arguments for computing the stiff sources.

Within each sector, restrictions are placed on the form of the implicit
derivative (the $I$ above).  When evolving a hyperbolic PDE, in general, the
equation for an implicit step becomes an elliptic PDE.  This is far too complex
and expensive to solve for every integration substep.  We therefore restrict
the implicit derivative to only contain source terms, i.e., terms that depend
on the sector variables only in a pointwise manner, rather than through
derivatives or similar.  (The terms may still depend on the derivatives of
non-sector variables.)  This results in each point in the domain having an
independent set of algebraic equations to solve.

There is a small complexity in handling non-autonomous (time-dependent) systems
of equations, such as those for evolutions with control systems.  Some methods
are defined in a way such that the values of time for the explicit and implicit
parts of a substep appear to be inconsistent.  SpECTRE always uses the time
derived from the explicit portion of the method, which is equivalent to
treating time as an additional evolved variable with a constant explicit
derivative of 1 and implicit derivative of 0.

SpECTRE supports using an analytic solution of the implicit-step equation as
well as two modes for numerical root-finding: implicit and semi-implicit.  The
analytic mode can be chosen in the definition of the implicit sector if the
form of the implicit source allows an analytic solution.  The numerical methods
can be toggled at runtime.  The fully implicit method does a numerical
root-find, while the semi-implicit one linearizes the equation and solves that.
Semi-implicit solves are faster, and, experimentally, they still stabilize the
evolution fairly well.  Using a semi-implicit solve instead of a fully implicit
one does not affect conservation properties of the method.  Both numerical
solvers require the jacobian of the implicit source to be coded, but an
analytic solution does not.

#### Creation of SpECTRE IMEX executables

The SpECTRE IMEX interface is defined by two protocols: \link
imex::protocols::ImexSystem \endlink and \link imex::protocols::ImplicitSector
\endlink.  The normal evolution-system interface now only contains the explicit
portion of the derivative, and the implicit portion is defined by the contents
of the `implicit_sectors` typelist.  See those protocols for details.

In order to use an IMEX system, the implicit solver must be explicitly
instantiated for each sector.  This is done in a separate source file in the
system directory.  As an example, the instantiation file for the sectors in one
of the tests is

\include tests/Unit/Helpers/Evolution/Imex/DoImplicitStepInstantiate.cpp

In the executable itself, four sets of changes are necessary.  First, the
time-stepper type, usually defined as a type alias near the start of the
metavariables, must be changed from `TimeStepper` to `ImexTimeStepper`
(IMEX-LTS is not currently supported), and an entry must be added to the
`factory_creation` list:

```
tmpl::pair<ImexTimeStepper, TimeSteppers::imex_time_steppers>
```

(The `TimeStepper` line can be removed, although doing so is not necessary.)

Second, the IMEX actions must be added after the corresponding explicit
actions:

* In the initialization phase, the `imex::Initialize<system>` mutator must be
  applied by adding it to the arguments of
  `Initialization::Actions::InitializeItems`.  It uses objects initialized by
  `Initialization::TimeStepperHistory`, so must appear after that.

* In the step actions,
```
imex::Actions::RecordTimeStepperData<system>
```
  must be added after each occurrence of
  `Actions::RecordTimeStepperData<system>`.  (There may be only one.)

* Again in the step actions,
```
imex::Actions::DoImplicitStep<system>
```
  must be added after each occurrence of `Actions::UpdateU<system>`.  (Again,
  there may be only one.)

Third, the `imex::ImplicitDenseOutput<system>` dense output postprocessor must
be added to the argument of the `evolution::Actions::RunEventsAndDenseTriggers`
action.  It must appear before any other preprocessors that use the evolved
variables.

Finally, header includes must be added for all these things.  The required
headers are
```
#include "Evolution/Imex/Actions/DoImplicitStep.hpp"
#include "Evolution/Imex/Actions/RecordTimeStepperData.hpp"
#include "Evolution/Imex/ImplicitDenseOutput.hpp"
#include "Evolution/Imex/Initialize.hpp"
#include "Time/TimeSteppers/ImexTimeStepper.hpp"
```
