\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

# Notes on SpECTRE load-balancing using Charm++'s built-in load balancers {#load_balancing_notes}

The goal of load-balancing (LB) is to ensure that HPC resources are well-used
while performing large inhomogeneous simulations. In 2020-2021, Jordan Moxon
and Francois Hebert performed a number of tests using Charm++'s built-in load
balancers with SpECTRE.

These notes highlight the key points and give (at the bottom) some general
recommendations.

### Overview of how LBs work with SpECTRE

In late 2020 FH tested Charm's LBs on simple, homogeneous, SpECTRE test cases.
These tests reveal the following broad behavior patterns:
- Without using any LBs (no LB command-line args, or `+balancer NullLB`),
  SpECTRE's performance depends sensitively on how the DG elements are
  distributed over the HPC system. This indicates that communication costs are
  very important in SpECTRE runs. This statement remains true for more expensive
  evolution systems like Generalized Harmonic.
- Charm's LBs that are not communications aware (e.g., `GreedyLB`, `RefineLB`,
  ...) all result in low parallel efficiency (20-30%). This is consistent with
  the understanding that communication costs are large. A good initial
  distribution of elements over processors will be degraded by these LBs,
  leading to more complicated communication graph and loss of performance.
- Some of Charm's communication-aware LBs perform well: they approach the
  efficiency of a "manually tuned" initial distribution of elements onto
  processors. This suggests these LBs do a good job of partitioning the
  communications graph. In FH's simple tests, the best results were from
  `RecBipartLB`, which came within 10-20% of a manual initial distribution.
  However, it is a slow algorithm and is best used infrequently or only a few
  times near the start of the simulation.

Note that at the 2020 Charm++ Workshop, the Charm team recommended that we use
`MetisLB`, or that we combined `MetisLB` with `RefineLB` (syntax:
`+balancer MetisLB +balancer RefineLB`, which applies the first balancer on the
first invocation and the second balancer on all subsequent invocations, to
'polish' the results of the first LB). In practice, this failed for two reasons:
- `MetisLB` tends to error with FPEs
- When falling back to the pairing of `RecBipartLB` followed by `RefineLB`, the
  run starts with good performance. However, within a few applications of
  `RefineLB`, the performance is heavily degraded (down to 20-30% efficiency).
  It appears that we should stick to the comm-aware LB strategies.

### Contaminated LB measurements on first invocation

There is reason to suspect that the Charm load balancing may incorrectly balance
the load when applied near the start of some simulations. This is because the
'one-time' setup of the system may involve nontrivial computation, and (e.g.
for numeric initial data) communication patterns between components that differ
significantly from the patterns during evolution. Then, the first balancer
invocation, based partially on the measurements taken during the
non-generalizable initialization phase, can give rise to a poorly-chosen balance
and injure performance.
This is the suspected cause of poor performance that has been noticed
in cases of homogeneous load and numeric initial data in Generalized Harmonic
tests performed by Geoffrey Lovelace. JM confirmed the problem.
It appears that the balance is not similarly degraded when using cases that do
not involve numeric initial data -- some basic 2-node re-tests with Generalized
Harmonic by JM seem to produce useful balance (improved performance) when not
using numeric initial data.

This issue has not been investigated to a completely satisfactory conclusion,
though the above explanation seems most plausible.

In cases for which it appears that the LB data is problematically impacted by
the set up of the evolution system, we can try two main strategies to mitigate
the problem:
- Apply the load balancer at least two times near the start of the simulation,
  with sufficient gaps to collect useful balancing information. The LB database
  in Charm is cleared every time a balance is applied, so the later balances
  during the evolution should be uncontaminated. This strategy has not yet
  been carefully tested. To do this, use an input file similar to
```
PhaseChangeAndTriggers:
  - - Slabs:
        Specified:
          Values: [5, 10, 15]
    - - VisitAndReturn(LoadBalancing)
```
- Use `LBTurnInstrumentOff` and `LBTurnInstrumentOn` to specifically exclude
  setup procedures from the LB instrumentation. First attempts indicate that
  this process might be challenging to accomplish correctly, and may require
  correspondence with the Charm developers to clarify at what points in the
  code execution those commands may be used, and precisely how they affect
  the load-balancing database. A first attempt by JM was to turn instrumentation
  off during array element construction, then turn instrumentation on for each
  element during the start of the `Evolve` phase, but that attempt led to a
  hang of the system, so the utility must have more constraints than were
  initially apparent.

### Scotch load balancer

JM tested `ScotchLB`, and found better performance than with `RecBipartLB`. The
margin varied a great deal among the number of nodes used, but at multiple
points tested, the runtime was less than 65% of the `RecBipartLB` runtime.
The tests were performed with homogeneous load, but starting from the
round-robin element distribution. The indication is therefore that `ScotchLB`
is very effective at minimizing communication costs.

However, in practical applications, JM found that the `ScotchLB` often generates
FPEs during the graph partition step and causes the simulation to crash.
The issue [charm++ issue #3401](https://github.com/UIUC-PPL/charm/issues/3401)
tracks the progress to determine the cause of the problem and fix it in Charm.
The source of the problem has largely been identified, but the fix is still
pending.

`ScotchLB` will likely replace `RecBipartLB` as the most-frequently recommended
centralized communication based balancer for SpECTRE once the FPE bugs have
been fixed.

### General recommendations

#### Homogeneous loads

For homogeneous loads, it is likely best to omit load-balancing and just use
the z-curve distribution (default) to give a good initial distribution and use
that for the entire evolution. This means calling the SpECTRE executable with
no LB-related command-line args, or with `+balancer NullLB`.

You may find modest gains from using a communication-based load balancer, but
likely only from the 'extra' parallel components of the system that cause the
load to be not completely homogeneous (e.g., components like the interpolator
or horizon finder).
If you need a very long evolution or intend to submit a large number of
evolutions, it may be worth experimenting to see whether 1-3 applications of
`RecBipartLB` (or `ScotchLB` once its bugs are fixed, see above) improve
performance for the system, for instance by using the input file:
```
PhaseChangeAndTriggers:
  - - Slabs:
        Specified:
          Values: [5, 10, 15]
    - - VisitAndReturn(LoadBalancing)
```
and command-line args `+balancer RecBipartLB` (or `ScotchLB` when its
bugs are fixed). This may be particularly relevant for cases with numeric
initial data or other complicated set-up procedures.

#### Inhomogeneous loads

Based on our experiments, we anticipate that using a load-balancer may
significantly improve runtimes with inhomogeneous loads. Our testing on this
case is far more sparse, but for SpECTRE executables, it is probably remains
true that managing communication costs will be an important goal for the
balancer. It is likely worth attempting the evolution with a
periodically-applied centralized communication-aware balancer, e.g.:
```
PhaseChangeAndTriggers:
  - - Slabs:
        EvenlySpaced:
          Interval: 1000
          Offset: 5
    - - VisitAndReturn(LoadBalancing)
```
paired with command-line args `+balancer RecBipartLB` (or `ScotchLB` when its
bugs are fixed).

Important considerations when choosing the interval with which to balance are:
- you will want to ensure that the balancer is applied frequently enough to
  prioritize expensive parts of the simulation before any relevent features
  'move' to other elements. For example, if a shock is moving across the
  simulation domain causing certain elements to be more expensive to compute,
  you want to balance often enough that the LB 'keeps up' with the movement of
  the shock.
- you will want to avoid balancing so frequently that the synchronization
  and balancer calculation itself becomes a significant portion of runtime.

We have not yet taken much detailed data on using the load-balancers for
inhomogeneous loads, so more detailed tests determining their efficacy would be
valuable.
