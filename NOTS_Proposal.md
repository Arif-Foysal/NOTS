# RESEARCH PROPOSAL

## **Nash-Optimized Topological Shields:**

## **A Game-Theoretic Persistent Homology Framework**

## **for Adversarially Robust Network Intrusion Detection**

*Target Venue: IEEE Transactions on Information Forensics and Security*

*Secondary Target: IEEE Transactions on Network and Service Management*

| **Domain** | **Estimated Duration** |
|---|---|
| Network Security / Topological Data Analysis / Game Theory | 18 months (Phase I: 6 mo, Phase II: 6 mo, Phase III: 6 mo) |

---

## **1. Abstract**

This proposal outlines a novel framework — Nash-Optimized Topological Shields (NOTS) — for building adversarially robust Network Intrusion Detection Systems (NIDS). The framework combines two powerful mathematical disciplines: Algebraic Topology, specifically Persistent Homology, to extract structural "shape" features from network traffic that are provably invariant under small adversarial perturbations; and Game Theory, specifically Minimax Nash Equilibrium analysis, to design a randomized feature-sampling strategy that guarantees a mathematically provable lower bound on detection performance regardless of attacker behavior.

The central insight is that while an attacker can manipulate the statistical content of their packets (byte distributions, packet sizes, timing), they cannot efficiently alter the high-dimensional topological structure of their attack's coordination pattern. The NOTS framework detects this topological signature using Wasserstein distance between Persistence Diagrams, and the game-theoretic layer makes the detection strategy a mathematically optimal moving target.

---

## **2. Problem Statement**

### **2.1 The Adversarial Evasion Problem**

Modern NIDS, including both signature-based systems (Snort, Suricata) and machine learning-based anomaly detectors, share a critical vulnerability: they are susceptible to adversarial evasion attacks. An attacker who understands the detection logic can deliberately craft traffic that evades the detector while executing a harmful payload.

The most common evasion technique is packet perturbation: padding malicious packets with random or structured bytes to shift their statistical fingerprint toward the distribution of benign traffic. Against a classifier trained on features like packet size distributions, byte-frequency histograms, or inter-arrival times, this is often sufficient for evasion.

| **Core Problem** |
|---|
| Given a trained NIDS classifier f(x) that detects attacks with accuracy A, an attacker can construct a perturbed sample x' = x + delta such that f(x') = benign, while x' still executes the original attack payload. Existing defenses are either computationally expensive, require retraining on adversarial examples (which requires knowing the attack a priori), or provide no formal guarantee on detection performance. |

### **2.2 Why Current Approaches Fall Short**

| **Approach** | **Strength** | **Critical Weakness** |
|---|---|---|
| Signature-based (Snort) | Fast, low false positives | Cannot detect zero-day or perturbed known attacks |
| ML anomaly detection | Can detect novel attacks | Vulnerable to adversarial perturbations (delta-attack) |
| Adversarial training | Hardens against seen attacks | No formal guarantee; fails on unseen attack types |
| Ensemble methods | Improved robustness | No mathematical lower bound on detection rate |
| Topological methods (prior) | Structural features | Not yet combined with adversarial game-theoretic framework |

### **2.3 The Research Gap**

No existing NIDS framework simultaneously achieves (a) a formally provable lower bound on detection rate, (b) robustness to adversarial packet perturbations without requiring prior knowledge of the attack, and (c) practical computational efficiency on real network traffic. The NOTS framework is specifically designed to close all three gaps.

---

## **3. Proposed Framework: NOTS**

The NOTS framework operates in three mathematically integrated layers, described below.

### **3.1 Layer 1 — Topological Feature Extraction via Persistent Homology**

#### **3.1.1 Network Traffic as a Point Cloud**

In a time window of length T, we extract N network flow records. Each flow record is represented as a feature vector in R^d, capturing properties such as flow duration, byte counts, packet counts, inter-arrival time statistics, TCP flag counts, and port numbers. This yields a point cloud P_t = {x_1, ..., x_N} in R^d.

Rather than treating these points as raw features for a classifier, we analyze the topological structure — the global geometric shape — of the point cloud. This is done via Vietoris-Rips filtration: for increasing values of a scale parameter epsilon, we connect pairs of points within distance epsilon, forming a simplicial complex K_epsilon.

K_epsilon = { σ ⊆ P_t | diam(σ) ≤ ε }

#### **3.1.2 The Persistence Diagram**

As epsilon increases from 0 to infinity, topological features (connected components, loops, voids) are born and die. A feature born at ε_birth and dying at ε_death has persistence p = ε_death - ε_birth. Long-lived features (high persistence) represent genuine structural patterns; short-lived features represent noise.

The output of this process is a Persistence Diagram D: a multiset of points (birth, death) in R^2, one per topological feature. The k-th Betti number β_k counts the number of k-dimensional holes currently alive. For network traffic data, β_0 = number of connected clusters, β_1 = number of loops.

| **Key Insight — Topological Invariance** |
|---|
| The Structural Stability Theorem of Persistent Homology (Cohen-Steiner et al., 2007) proves that for two point clouds P and Q with Hausdorff distance d_H(P, Q) ≤ δ, their persistence diagrams D_P and D_Q satisfy: W_∞(D_P, D_Q) ≤ δ. This means: if an attacker's perturbation has L-∞ norm at most δ, the persistence diagram shifts by at most δ. Large-scale topological features (high-persistence points) cannot be eliminated by small perturbations. |

#### **3.1.3 Wasserstein Distance as the Detection Signal**

Let D_norm be the persistence diagram computed from a baseline window of known-benign traffic, and D_live be the persistence diagram of the current live window. The detection signal is the p-Wasserstein distance:

W_p(D_norm, D_live) = [ inf_{γ ∈ Γ} Σ_{(x,y) ∈ γ} ||x - y||^p ]^{1/p}

where γ ranges over all perfect matchings between the two diagrams (augmented with the diagonal). This measures the minimum total work needed to transform D_norm into D_live by moving points on the birth-death plane. A large W_p means the live traffic has a fundamentally different topological structure from baseline.

### **3.2 Layer 2 — Minimax Game-Theoretic Hardening**

#### **3.2.1 Formalizing the Adversarial Game**

We model the interaction between the NIDS (defender) and the attacker as a zero-sum game G = (S, H, U) where:

- S is the defender's strategy space: all probability distributions over feature subsets ω ⊆ {1,...,d}
- H is the attacker's strategy space: all perturbations δ with ||δ||_∞ ≤ Δ_max
- U(S, H) is the payoff (detection rate) = W_p(D_norm, D_live(ω, x+δ))

The NIDS wants to maximize U (maximize detection); the attacker wants to minimize U (evade detection). This is the classic minimax problem:

S* = argmax_S min_H U(S, H)

#### **3.2.2 Existence and Form of Nash Equilibrium**

By Nash's theorem, since S is a compact convex set (it is a simplex over feature subsets) and U is continuous and concave-convex (linear in S, continuous in H for fixed S), a Nash Equilibrium (S*, H*) exists. At equilibrium:

max_S min_H U(S,H) = min_H max_S U(S,H) = ε_min

The value ε_min is the guaranteed minimum detection payoff. The NIDS playing S* (the minimax optimal randomized feature sampling strategy) guarantees W_p ≥ ε_min for all attacker strategies H, as long as ||δ||_∞ ≤ Δ_max.

#### **3.2.3 The Critical Lemma — Topological Bound**

| **Theorem (to be proven formally in paper)** |
|---|
| Let P = {x_1,...,x_N} be the attacker's traffic point cloud and P' = {x_1+δ_1,...,x_N+δ_N} the perturbed cloud with max_i ||δ_i||_∞ ≤ Δ_max. Let D and D' be their persistence diagrams. Then: W_∞(D, D') ≤ Δ_max. Consequently, if W_p(D_norm, D) ≥ τ + Δ_max for some threshold τ > 0, then W_p(D_norm, D') ≥ τ regardless of attacker perturbation. The NOTS system sets the alert threshold at τ to guarantee detection is maintained. |

This theorem is the mathematical core of the paper. It shows that if the attack's true topological signature is far enough from normal traffic (which is the case for coordinated attacks like DDoS, port scans, and botnets), no bounded perturbation can shrink the detection signal below the threshold.

### **3.3 Layer 3 — Online Adaptive Baseline**

To handle concept drift (gradual evolution of normal traffic patterns), NOTS maintains a sliding baseline using an Exponentially Weighted Moving Average of persistence diagrams. New baseline diagrams are incorporated with weight α, providing adaptation while being robust to adversarially injected baseline poisoning.

D_norm(t+1) = (1-α) * D_norm(t) + α * D_live(t) if W_p < τ/2

Baseline updates are only accepted when the live traffic is sufficiently close to current baseline, preventing an attacker from gradually shifting the baseline to accept their attack signature.

---

## **4. Research Objectives**

This research has five concrete objectives, each corresponding to a deliverable:

1. Formally prove the Topological Stability Lemma stated in Section 3.2.3, establishing the exact relationship between perturbation bound Δ_max, the persistence threshold, and the guaranteed detection floor ε_min.

2. Develop an efficient algorithm for online persistence diagram computation on streaming network traffic, targeting O(N log N) time complexity using approximate nearest-neighbor filtration methods.

3. Solve the minimax optimization problem to find S* analytically or via efficient convex optimization, characterizing the optimal feature sampling distribution.

4. Implement the full NOTS pipeline and evaluate it on CICIDS-2017, NSL-KDD, and UNSW-NB15 datasets under white-box and black-box adversarial attack scenarios.

5. Conduct a computational complexity analysis and compare runtime performance against state-of-the-art baselines including Kitsune, LUCID, and RF-based anomaly detectors.

---

## **5. Novelty and Contribution**

The NOTS framework makes the following distinct contributions, each of which we believe is publishable on its own merits:

| **Contribution** | **Why it is Novel** |
|---|---|
| Formal provable detection lower bound | No existing NIDS provides a mathematical guarantee on minimum detection rate under bounded adversarial perturbation. |
| Topological stability applied to NIDS | While TDA has been applied to anomaly detection in other domains, the game-theoretic integration for NIDS is new. |
| Nash equilibrium feature sampling | Using Nash Equilibrium to derive an optimal randomized feature selection policy for NIDS is a new approach. |
| Closed-form perturbation bound Δ_max | Provides practitioners with an explicit formula relating their security budget to the detectable attack class. |
| Adaptive baseline with poisoning resistance | The conditional update rule for the baseline is novel and provably resistant to slow-drift adversarial poisoning. |

---

## **6. Related Work and Positioning**

### **6.1 Topological Data Analysis in Security**

Gidea & Katz (2018) applied persistent homology to time-series anomaly detection in financial systems, demonstrating the stability advantage. Islambekov et al. (2019) used TDA for network traffic clustering. Our work extends this to the adversarial setting with a formal game-theoretic guarantee, which neither prior work addresses.

### **6.2 Adversarial Machine Learning in NIDS**

Apruzzese et al. (2022) provide a comprehensive survey of adversarial attacks on ML-based NIDS, identifying packet perturbation as the dominant threat. Debicha et al. (2023) propose adversarial training as a defense, but without formal guarantees. Lin et al. (2022) use generative models for adversarial sample generation. NOTS differs fundamentally by providing a formal bound rather than empirical hardening.

### **6.3 Game Theory in Network Security**

Nguyen & Alpcan (2010) pioneered game-theoretic frameworks for intrusion detection, but without topological features. Patcha & Park (2007) model IDS as a two-player game but rely on classical statistical features. NOTS combines both threads for the first time.

---

## **7. Detailed Methodology**

### **7.1 Phase I — Theoretical Foundations (Months 1–6)**

- Complete formal proof of the Topological Stability Lemma using results from Cohen-Steiner et al. (2007) and the Bottleneck Stability Theorem
- Derive closed-form expression for ε_min as a function of (Δ_max, d, N, τ)
- Prove existence and uniqueness (or characterize the equilibrium set) of the Nash Equilibrium S*
- Develop the computational algorithm for S* using Linear Programming relaxation of the minimax problem
- Deliverable: Theoretical paper submitted to IEEE Transactions on Information Theory or Journal of Computer Security

### **7.2 Phase II — Implementation (Months 7–12)**

- Implement Vietoris-Rips filtration using the Ripser library (C++ with Python bindings) for efficiency
- Implement online Wasserstein distance computation using the POT (Python Optimal Transport) library
- Implement the randomized feature sampler using S* derived in Phase I
- Integrate full pipeline into a prototype NIDS agent that processes PCAP files and live tcpdump streams
- Deliverable: Open-source codebase on GitHub

### **7.3 Phase III — Evaluation (Months 13–18)**

Evaluation will follow the methodology of the NIDS benchmarking literature, using three publicly available datasets:

| **Dataset** | **Traffic Type** | **Attack Classes** |
|---|---|---|
| CICIDS-2017 | Realistic synthetic network traffic | DDoS, PortScan, Botnet, Web attacks, Infiltration |
| NSL-KDD | Classic benchmark, widely cited | DoS, Probe, R2L, U2R |
| UNSW-NB15 | Modern hybrid real/synthetic | Fuzzers, Backdoors, Shellcode, Worms, Generic |

For each dataset, we will evaluate: (1) detection rate and false positive rate at baseline, (2) detection rate under white-box L-∞ bounded perturbation attacks at varying Δ_max, (3) detection rate under black-box transfer attacks, (4) computational throughput (flows per second), and (5) adaptation speed under concept drift simulation.

---

## **8. Expected Results and Claims**

| **Primary Claim** |
|---|
| NOTS will maintain a detection rate of at least ε_min (derived analytically) under all adversarial perturbations with ||δ||_∞ ≤ Δ_max, provably. Empirically, we expect ε_min to correspond to DR ≥ 85% across all tested attack classes at Δ_max = 0.1 (10% packet perturbation budget). |

| **Evaluation Metric** | **Expected Result** |
|---|---|
| Detection Rate (baseline, no attack) | > 95% on CICIDS-2017 |
| Detection Rate (under white-box attack, Δ=0.05) | > 90% (formal bound: > ε_min) |
| Detection Rate (under white-box attack, Δ=0.10) | > 85% (formal bound: > ε_min) |
| False Positive Rate | < 3% (comparable to SOTA) |
| Throughput (flows/sec) | > 10,000 flows/sec on commodity hardware |
| Computation vs. baseline NIDS | 2–5x overhead vs. Kitsune (acceptable for security gain) |

---

## **9. Anticipated Challenges and Mitigations**

| **Challenge** | **Risk Level** | **Mitigation** |
|---|---|---|
| Persistent homology computation is O(N³) in the worst case | High | Use approximate methods (Ripser, landmark sampling) to achieve near-linear time; provide theoretical approximation guarantees. |
| Nash Equilibrium may not have a closed-form solution for all feature spaces | Medium | Use LP relaxation and prove that the ε-approximate equilibrium suffices for the detection bound. |
| Wasserstein distance computation is O(N² log N) | High | Use sliced Wasserstein distance (O(N log N)) as approximation; prove that the slack introduced is bounded by a known constant. |
| Baseline persistence diagram may drift under non-adversarial traffic changes | Medium | The conditional update rule in Section 3.3 directly addresses this; tune α on held-out validation traffic. |
| Reviewers may challenge the realism of the bounded-perturbation model | Medium | Include experiments at varying Δ_max and discuss the real-world cost of large perturbations for the attacker (bandwidth overhead, protocol violations). |

---

## **10. Publication and Dissemination Plan**

| **Timeline** | **Output** |
|---|---|
| Month 6 | Theory paper: 'Topological Stability Bounds for Adversarial NIDS' — IEEE T-IFS or J. Computer Security |
| Month 12 | Systems paper: 'NOTS: Efficient Online Persistent Homology for Network Intrusion Detection' — IEEE INFOCOM or NDSS |
| Month 16 | Full paper: 'Nash-Optimized Topological Shields for Provably Robust NIDS' — IEEE T-IFS (primary target) |
| Month 18 | Open-source release + reproducibility package on GitHub and Papers With Code |
| Ongoing | Preprints on arXiv under cs.CR and cs.NI |

---

## **11. Required Background and Tooling**

### **11.1 Mathematical Prerequisites**

- Algebraic Topology: simplicial complexes, homology groups, filtrations
- Metric geometry: Hausdorff distance, Wasserstein distance, optimal transport
- Game Theory: zero-sum games, Nash Equilibrium, minimax theorem (von Neumann)
- Functional analysis: compact convex sets, fixed-point theorems (for Nash existence proof)

### **11.2 Software and Libraries**

| **Tool / Library** | **Purpose** |
|---|---|
| Ripser (C++/Python) | Fast persistent homology computation via Vietoris-Rips filtration |
| POT — Python Optimal Transport | Wasserstein distance computation between persistence diagrams |
| Scikit-TDA / Gudhi | Topological data analysis pipeline utilities |
| CICFlowMeter | Feature extraction from PCAP files for CICIDS-2017 |
| CVXPY | Convex optimization for Nash Equilibrium computation (LP formulation) |
| Scapy / dpkt | Live packet capture and stream processing |

---

## **12. Key References**

The following references are central to the proposed framework. A full bibliography will be included in the submitted paper.

- Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007). Stability of persistence diagrams. *Discrete & Computational Geometry*, 37(1), 103–120. [Foundation of topological stability — central to the main theorem]

- Nash, J. (1951). Non-cooperative games. *Annals of Mathematics*, 54(2), 286–295. [Existence of Nash Equilibrium]

- Islambekov, U., Gel, Y. R., & Perea, J. A. (2019). Unsupervised topological learning approach for examining network traffic anomalies. arXiv:1911.03664. [Closest prior TDA-NIDS work]

- Apruzzese, G., et al. (2022). The role of machine learning in cybersecurity. *Digital Threats*, 3(3). [Survey of adversarial ML in NIDS — motivates the threat model]

- Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An Introduction*. AMS. [Primary reference for persistent homology theory]

- Villani, C. (2008). *Optimal Transport: Old and New*. Springer. [Theoretical foundation of Wasserstein distance]

- Patcha, A., & Park, J.-M. (2007). An overview of anomaly detection techniques. *Computer Networks*, 51(12). [Game-theoretic IDS framing]

- Bauer, U. (2021). Ripser: Efficient computation of Vietoris-Rips persistence barcodes. *Journal of Applied and Computational Topology*. [Algorithm paper for the core computational tool]

- Sharafaldin, I., et al. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *ICISSP*. [CICIDS-2017 dataset reference]

- Manzoor, E., et al. (2016). Fast memory-efficient anomaly detection in streaming heterogeneous graphs. *KDD*. [Online streaming anomaly detection baseline for comparison]

---

## **Appendix A — Mathematical Notation Summary**

| **Symbol** | **Definition** | **Section** |
|---|---|---|
| P_t | Point cloud of network flows in window t, in R^d | 3.1.1 |
| K_ε | Vietoris-Rips simplicial complex at scale ε | 3.1.1 |
| D | Persistence diagram: multiset of (birth, death) pairs | 3.1.2 |
| β_k | k-th Betti number (count of k-dimensional holes) | 3.1.2 |
| W_p(D, D') | p-Wasserstein distance between persistence diagrams | 3.1.3 |
| D_norm | Baseline persistence diagram of benign traffic | 3.1.3 |
| D_live | Persistence diagram of current live traffic window | 3.1.3 |
| δ | Adversarial perturbation vector; \|\|δ\|\|_∞ ≤ Δ_max | 3.2.1 |
| S | NIDS strategy: probability distribution over feature subsets | 3.2.1 |
| H | Attacker strategy: choice of perturbation δ | 3.2.1 |
| U(S,H) | Payoff = W_p(D_norm, D_live(ω, x+δ)) | 3.2.1 |
| S* | Nash-optimal randomized feature sampling strategy | 3.2.2 |
| ε_min | Guaranteed minimum detection payoff at Nash Equilibrium | 3.2.2 |
| τ | Detection alert threshold; set as τ = ε_min | 3.2.3 |
| α | Baseline adaptation rate (EWM weight) | 3.3 |
