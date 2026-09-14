\subsection{Case Study}
\label{app:case_study}

\paragraph{Case 1: Prompt-level reuse may fail to exploit available positive experience.}
To understand why explicit distribution-level guidance is more effective than prompt-level experience reuse, we analyze a representative example from HMMT24 (Q10) with Qwen3-4B-Thinking-2507. 
In Iter0, the search process already discovers both a successful reasoning pattern and a misleading shortcut. 
The successful trajectory identifies that the number of tilings can be reduced to the integer solutions of
$4x+5y=24$, yielding two possible column compositions:
$(x,y)=(6,0)$ and $(1,4)$.
The first case corresponds to a single composition
$[4,4,4,4,4,4]$, while the second produces five permutations of
$[4,5,5,5,5]$, resulting in the correct answer
$1+5=6$.
In contrast, unsuccessful trajectories incorrectly assume that since $24$ is not divisible by $5$, no width-$5$ blocks can exist, leading to the erroneous conclusion that only the uniform composition is possible and the answer is $1$.

Although RSE stores the successful reasoning pattern into the experience prompt, it fails to consistently utilize this information during subsequent generation.
In Iter1, all 64 RSE trajectories retrieve the experience containing the correct lemma
$4x+5y=24$ and explicitly mention the corresponding mixed compositions, but all trajectories still return the incorrect answer $1$.
The model first acknowledges the successful experience but then overrides it with the original erroneous assumption:
``the width is not divisible by 5, therefore width-5 strips are impossible''.
This illustrates the limitation of prompt-level reuse: useful experience is introduced only as additional context and must compete with existing model priors and conflicting failure patterns.

In contrast, CTO and RG-CTO convert the same trajectory-derived experience into explicit decoding-time guidance.
The positive branch increases the probability of continuations following the valid composition-based reasoning path, rather than relying on the model to interpret the retrieved experience correctly.
Consequently, CTO obtains the correct answer in 21/32 Iter1 rollouts, while RG-CTO further improves robustness and reaches 24/32 correct rollouts.
This example demonstrates that the benefit of experience reuse does not merely come from exposing the model to previous reasoning traces, but from directly shaping the generation distribution toward useful continuations.

\paragraph{Case 2: Negative experience as an explicit anti-expert improves reasoning recovery.}
We further examine a representative example from HMMT24 (Q26) to illustrate how negative experience can guide subsequent reasoning. 
The problem asks for the perimeter of the orthic triangle $DEF$ in an acute triangle $ABC$, where
$AQ=20$, $BC=15$, and $AD=24$.
The correct answer is $8\sqrt{11}$.
In Iter0, the search process produces both successful and failed reasoning trajectories.
A common failure mode incorrectly applies a projection identity to the orthic triangle:
\[
P_{\mathrm{wrong}}
=
a(1+\cos A)
=
15\left(1+\frac{AQ}{AD}\right)
=
15\left(1+\frac{5}{6}\right)
=
\frac{55}{2}.
\]
Although this derivation appears mathematically plausible, it relies on an invalid use of the projection relation without verifying whether the required conditions hold for the orthic perimeter.

RSE stores this failure pattern as negative experience and provides warnings such as avoiding the unverified projection identity in the subsequent prompt. 
However, the negative information remains only textual context and does not directly influence the generation distribution.
In Iter1, 64/64 RSE trajectories still mention the projection formula or the intermediate expression $a(1+\cos A)$, and 55/64 trajectories eventually reproduce the same incorrect answer $\frac{55}{2}$.
The model can explicitly recognize the warning but still follows the high-probability erroneous continuation:
``the projection formula gives $P=a(1+\cos A)$''.
This demonstrates that prompt-level negative experience reuse only exposes failure patterns, while leaving the model to decide whether these patterns should affect future token generation.

In contrast, CTO converts the failure trajectory into an explicit negative decoding branch.
The negative branch acts as an online anti-expert, reducing the probability of continuations associated with the previously observed failure pattern.
Meanwhile, the positive branch preserves alternative valid reasoning directions.
Consequently, successful CTO trajectories avoid committing to the invalid projection identity and recover a correct derivation based on geometric relations:
\[
\sin A=\frac{\sqrt{11}}{6},\qquad
R=\frac{BC}{2\sin A},
\qquad
P=\frac{2\Delta}{R}=8\sqrt{11}.
\]
CTO improves the number of correct rollouts from 10/32 in Iter0 to 21/32 in Iter1, while RG-CTO further reduces the failure mode and achieves 31/32 correct rollouts.
This example highlights the key distinction between prompt-level reuse and contrastive trajectory optimization: RSE can describe what reasoning patterns should be avoided, whereas CTO directly incorporates negative experience into next-token distribution control.

\paragraph{Case 3: Reliability gating prevents harmful suppression from unreliable negative experience.}
Finally, we study HMMT25 Q28 to demonstrate why negative experience requires reliability-aware control. 
This example highlights a failure mode where a seemingly reasonable negative pattern extracted from unsuccessful trajectories becomes harmful when directly used for suppression.
The task asks for the length of $AB$ in a rectangle configuration with
$BC=24$, $\angle AXB=90^\circ$, and the circumradii of $\triangle AXD$ and $\triangle BXC$ given as $13$ and $15$, respectively.
The correct answer is
$14+4\sqrt{37}$.

In Iter0, successful trajectories already discover the correct reasoning framework.
They introduce coordinates
$A(0,0), B(a,0), C(a,24), D(0,24)$,
derive the circumcenter constraint, and combine it with the right-angle condition
\[
p^2+q^2=ap .
\]
By substituting the two circumradius equations, the solution can be obtained as
\[
p=\frac{a(a-18)}{2(a-14)},
\]
which leads to
\[
a=14+4\sqrt{37}.
\]
However, some failed trajectories follow the same general approach but contain algebraic mistakes during substitution, eventually producing incorrect candidates such as $38$.
When extracting negative experience from these failures, several incorrect failure patterns are generated, including treating the coordinate substitution procedure, perpendicular-bisector reasoning, or the valid solution candidate $14+4\sqrt{37}$ itself as dead ends.

Applying these negative experiences directly in CTO leads to harmful suppression.
Although the extracted pitfalls originate from failed attempts, they conflict with reliable positive evidence already discovered in successful trajectories.
The negative branch therefore suppresses useful continuations involving coordinate construction and substitution, preventing the model from revisiting the correct solution path.
As a result, CTO performance degrades over further iterations: only 9/32 rollouts are correct at the initial guided step, and the accuracy further drops to 7/32 in the subsequent step.
This illustrates that not every observed failure pattern is a safe anti-expert signal; suppressing an unreliable negative experience can remove essential reasoning trajectories.

RG-CTO addresses this issue by estimating the reliability of each negative item before applying suppression.
For this example, all extracted pitfalls fail the reliability threshold: they have limited cross-rollout support, weak problem locality, and strong conflict with positive evidence.
Consequently, all negative items are filtered out
($n_{\mathrm{pitfalls\_kept}}=0/108$),
leading to
\[
\alpha_r=0,
\]
and RG-CTO automatically falls back to positive-only guidance.
Without suppressing the coordinate-based reasoning path, RG-CTO recovers the valid continuation and achieves 30/32 correct rollouts at the first guided step and 31/32 at the following step.
This case demonstrates the necessity of reliability-aware negative experience utilization: a failure observed in one trajectory may represent an execution error rather than an undesirable reasoning strategy, and indiscriminate suppression can introduce negative transfer.