\subsection{Reliability-Gated Contrastive Trajectory Optimization}
\label{sec:rg_cto}

Although CTO enables explicit distribution-level experience reuse,
directly treating all negative experience as an anti-expert implicitly
assumes that every extracted failure pattern provides a reliable
suppression signal. This assumption is often violated. A negative item
may arise from an isolated rollout, be weakly related to the current
reasoning state, or conflict with otherwise useful positive evidence.
We therefore introduce \emph{Reliability-Gated Contrastive Trajectory
Optimization} (RG-CTO), which estimates the reliability of each negative
experience and uses it to control both whether and how strongly the
negative branch affects decoding.

\paragraph{Reliability-aware negative experience.}
For each negative item
\(e\in\mathcal{R}_{-}^{(r)}\), RG-CTO evaluates three complementary
signals: cross-rollout support \(u(e)\), reasoning locality \(l(e,q)\),
and conflict risk \(c(e)\). The support score measures whether the failure pattern is repeatedly
observed across previous trajectories. Specifically, a semantic matching
function determines whether \(e\) is supported by a previous rollout,
with matching threshold \(\tau_{\mathrm{match}}\). A larger
\(\tau_{\mathrm{match}}\) therefore requires stronger semantic agreement
before a trajectory contributes to the support estimate. The locality score measures whether the negative pattern is relevant to
the current problem and accumulated positive reasoning evidence, while
the conflict score estimates whether suppressing the negative item may
interfere with a reasoning direction that is strongly supported by
positive experience. Detailed definitions of these components are
provided in Appendix~\ref{app:reliability_estimation}.

We combine the three signals into an item-level reliability score:
\begin{equation}
    w(e)
    =
    \operatorname{clip}
    \left(
        u(e)^{\lambda_u}
        l(e,q)^{\lambda_l}
        (1-c(e)),
        0,
        1
    \right),
    \label{eq:rg_item_weight}
\end{equation}
where \(\lambda_u\) and \(\lambda_l\) control the relative sensitivity
of reliability estimation to cross-rollout support and reasoning
locality, respectively. Together,
\(\tau_{\mathrm{match}}, \lambda_u,\lambda_l\) determine how evidence
reliability is estimated before negative suppression is applied.
Intuitively, a negative item should receive a high reliability score only
when it is consistently supported by previous rollouts, locally relevant
to the current reasoning process, and unlikely to conflict with reliable
positive evidence.

RG-CTO then removes insufficiently reliable negative experience using a
reliability threshold \(\delta\):
\begin{equation}
    \widetilde{\mathcal{R}}_{-}^{(r)}
    =
    \left\{
        e\in\mathcal{R}_{-}^{(r)}
        \mid
        w(e)\geq\delta
    \right\}.
    \label{eq:rg_filter}
\end{equation}
Thus, \(\delta\) directly controls the selectivity of the negative
branch: a larger value retains fewer but more reliable failure signals,
whereas a smaller value allows broader negative guidance.

The retained negative items are used to construct the negative decoding
branch following CTO. To additionally control suppression strength, we
aggregate their item-level reliability into a round-level gate:
\begin{equation}
    g^{(r)}
    =
    \frac{1}
    {|\widetilde{\mathcal{R}}_{-}^{(r)}|}
    \sum_{e\in\widetilde{\mathcal{R}}_{-}^{(r)}} w(e),
    \label{eq:rg_round_gate}
\end{equation}
where \(g^{(r)}=0\) when no negative item survives filtering. The
contrastive coefficient is then adapted from the CTO base strength
\(\alpha_0\) as
\begin{equation}
    \alpha_r=\alpha_0 g^{(r)}.
    \label{eq:rg_alpha}
\end{equation}
Hence, the reliability mechanism operates at two levels:
\(\delta\) determines which negative experiences are allowed to
participate in decoding, while \(g^{(r)}\) determines how strongly the
retained evidence influences the token distribution.

At decoding step \(j\), RG-CTO replaces the fixed CTO coefficient with
the reliability-adaptive coefficient \(\alpha_r\):
\begin{equation}
    s_i^{(j)}
    =
    \begin{cases}
        \ell_{\mathrm{pos},i}^{(j)}
        -
        \alpha_r
        \ell_{\mathrm{neg},i}^{(j)},
        &
        i\in\mathcal{S}^{(j)},\\
        \ell_{\mathrm{pos},i}^{(j)},
        &
        i\notin\mathcal{S}^{(j)} .
    \end{cases}
    \label{eq:rg_cto_score}
\end{equation}
Here, \(\mathcal{S}^{(j)}\) is the \(K_{\mathrm{CTO}}\)-sized plausible
candidate set defined in CTO. Therefore, the complete intervention is
controlled by two groups of hyperparameters: \(\alpha_0\) and
\(K_{\mathrm{CTO}}\) determine the base strength and token-level scope
of contrastive decoding, while
\(\tau_{\mathrm{match}}, \lambda_u, \lambda_l,\) and \(\delta\)
determine the reliability and selectivity of negative experience.
Their default values are fixed before the main evaluation and reported
in Section~\ref{sec:exp_setup} and
Appendix~\ref{app:implementation_details}.

All other decoding operations remain identical to CTO. RG-CTO therefore
interpolates between full contrastive guidance and positive-only
guidance: when negative evidence is well supported, locally relevant,
and non-conflicting, its suppression effect approaches that of CTO;
when reliability is low, negative items are filtered or their aggregate
influence is automatically reduced.

\subsubsection{Reliability Estimation for Negative Experience}
\label{app:reliability_estimation}


Negative experience can provide useful anti-expert signals, but directly
applying all extracted failure patterns may introduce harmful suppression when
they originate from accidental mistakes, irrelevant reasoning states, or
conflicts with valid solution paths. RG-CTO therefore estimates the reliability
of each negative experience before incorporating it into contrastive decoding.

For a negative experience item $e$, we first measure its support across
previous rollouts. Intuitively, a failure pattern that repeatedly appears in
independent reasoning attempts is more likely to represent a generalizable
failure mode rather than an isolated execution error. Given an experience item
$e$ and a previous trajectory $y$, we define the matching function:

\begin{equation}
m(e,y)
=
\mathbb{I}
(
\operatorname{Match}(e,y)>\tau_{\rm match}
),
\end{equation}

where $\operatorname{Match}(\cdot)$ measures the semantic consistency between
the experience item and the trajectory, and $\tau_{\rm match}$ denotes the
matching threshold. The support score is then calculated as:

\begin{equation}
u(e)
=
\frac{1}{N}
\sum_{i=1}^{N}
m(e,y_i),
\end{equation}

where $N$ denotes the number of previous rollouts.

In addition to rollout-level support, RG-CTO considers whether a negative
experience is relevant to the current reasoning process. We define the
locality score as:

\begin{equation}
l(e,q)
=
\rho(e,q)
\cdot
\rho(e,E_+),
\end{equation}

where $\rho(\cdot,\cdot)$ denotes cosine similarity between encoded semantic
representations:

\begin{equation}
\rho(a,b)
=
\frac{
\psi(a)^T\psi(b)
}{
||\psi(a)||_2||\psi(b)||_2
}.
\end{equation}

The first term measures the relevance between the negative experience and the
current problem, while the second term evaluates its relationship with
accumulated positive reasoning evidence. This prevents negative experiences
from unrelated problems or distant reasoning states from affecting current
generation.

Furthermore, a negative experience should not suppress reasoning paths that
are strongly supported by reliable positive evidence. RG-CTO therefore
introduces a conflict score to measure potential interference between negative
and positive evidence. Given negative experience $e_-$ and positive evidence
set $T_{\rm conf}$, we define:

\begin{equation}
c(e_-)
=
p_\theta
(
\operatorname{conflict}
|
T_{\rm conf}
),
\end{equation}

where the frozen model estimates whether applying the negative experience may
conflict with useful reasoning directions. A higher conflict score indicates
that the negative pattern may correspond to an execution error rather than an
undesirable reasoning strategy.

The three factors are aggregated into an experience-level reliability weight:

\begin{equation}
w(e)
=
\operatorname{clip}
(
u(e)l(e,q)(1-c(e)),
0,1
).
\end{equation}

Based on the estimated reliability, RG-CTO filters unreliable negative
experiences before constructing the negative decoding branch:

\begin{equation}
R_-^{(r)}
=
\{
e\in \tilde R_-^{(r)}
|
w(e)\geq\delta
\}.
\end{equation}

Only retained negative experiences are used for contrastive suppression. The
remaining reliability scores are further aggregated to adapt the suppression
strength:

\begin{equation}
g^{(r)}
=
\frac{1}{|R_-^{(r)}|}
\sum_{e\in R_-^{(r)}}w(e),
\end{equation}

where $g^{(r)}=0$ when no negative experience survives filtering. The final
contrastive coefficient is therefore adjusted as:

\begin{equation}
\alpha_r=\alpha_0 g^{(r)},
\end{equation}

allowing RG-CTO to preserve strong negative guidance when failure evidence is
reliable while automatically reducing suppression when the extracted negative
signals are uncertain.