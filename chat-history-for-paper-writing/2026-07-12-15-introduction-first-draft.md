# Introduction First Draft — LoRA-NF Narrative

**日期**: 2026-07-12  
**状态**: 英文初稿，用于讨论；尚未写入 LaTeX 主稿  
**写作约束**: 引用键沿用现有草稿；实验数字和强结论保留占位符；正式提交前需核验相关工作覆盖、理论条件和 claim–evidence ledger。

---

## 1. Draft

Vision-language models (VLMs), exemplified by CLIP~\cite{radford2021learning}, acquire broad visual concepts and open-vocabulary recognition capabilities by aligning images and natural language at scale. Such models provide an appealing foundation for continual learning: rather than learning each incoming task from scratch, a learner can adapt a shared representation as new classes and domains arrive. Yet this advantage also changes what must be preserved. A continually adapted CLIP model should retain not only its performance on previously learned classes, but also the cross-modal alignment that supports zero-shot recognition and image--text retrieval. We therefore study CLIP class-incremental learning through a dual objective: acquiring strong discrimination for sequentially introduced classes while preserving the cross-modal capability inherited from pre-training.

These objectives are inherently in tension. Supervision from a new task encourages the model to reshape its representation toward the currently observed classes, whereas repeated adaptation can both interfere with directions that are important to earlier tasks and drift away from the image--text relations learned during pre-training~\cite{zheng2023zscl,ni2023modx}. A frozen CLIP model retains its pre-trained geometry but cannot fully exploit new supervision, while unconstrained adaptation improves new-task discrimination at the cost of historical knowledge and cross-modal transfer. These two forms of degradation call for complementary controls: continual adaptation should avoid excessive interference with historical task directions, while cross-modal relations should be explicitly anchored during training.

Low-rank adaptation (LoRA)~\cite{hu2022lora} offers a natural residual pathway for updating a frozen foundation model. For a linear transformation, LoRA parameterizes the residual update as \(\Delta W=AB\), where the rank of \(AB\) is much smaller than the ambient dimension. This factorization substantially reduces the number of trainable parameters, but low rank alone is not a knowledge-preservation constraint. The trainable factors remain free to amplify input directions on which historical tasks already exhibit high activation energy. Put differently, LoRA determines how many degrees of freedom are used for adaptation, but not which historical activation directions those degrees of freedom should follow. This distinction is central in continual learning, where interference is highly anisotropic rather than uniformly distributed across the representation space.

Null-space methods provide a complementary strategy by identifying directions that minimally affect previously acquired knowledge. Gradient-projection approaches modify an update after conventional forward and backward computation so that it lies in an approximate null space of historical features~\cite{saha2021gradient,kang2025dynamic}. Some null-space-inspired LoRA variants instead initialize or constrain the input-side factor \(A\) using a basis derived from low-energy activation directions. Such designs inject null-space information into the low-rank factors, but do not place a fixed history-induced filter persistently before both trainable factors throughout optimization. This motivates a different question: can historical activation geometry be embedded directly into the LoRA forward function, such that the low-rank factors are optimized in a filtered input space throughout adaptation?

We answer this question with \textbf{LoRA-NF}, a low-rank adaptation method with persistent null-space filtering. LoRA-NF replaces the conventional residual \(AB\) with
\[
    \Delta W=PAB,
\]
where \(P\) is constructed from the eigenspectrum of historical layer-input second moments. The resulting adapter computes \((XP)AB\): it first attenuates high-energy historical directions and then learns the low-rank transformation in the filtered space. Unlike an initialization-only null-space bias, this filtering remains active throughout optimization. We use a leaky rather than strict filter, allowing the adapter to retain flexibility across directions while biasing learning away from those most strongly occupied by historical tasks. The formal conditions governing its expressivity and optimization dynamics are developed in the method section.

LoRA-NF is designed to preserve historical task knowledge by reducing effective adaptation along high-energy directions of past activations; it is not, by itself, a mechanism for preserving cross-modal retrieval. We assign the latter role primarily to cross-modal knowledge distillation, which explicitly constrains the adapted model to maintain image--text relations on reference pairs. At inference, we further combine the CLIP text classifier with low-rank regularized Gaussian discriminant analysis (LR-RGDA), which captures class-conditional statistics in the adapted visual space. The two branches provide complementary language-derived priors and supervised visual evidence. This prediction module contributes to continual classification performance but does not affect cross-modal retrieval.

We evaluate the resulting framework under the X-TAIL protocol~\cite{xu2024rail} across ten heterogeneous image-classification datasets, and assess both class-incremental learning and image--text retrieval. The experiments examine whether LoRA-NF improves over rank- and budget-matched LoRA, projected-gradient, and null-space initialization variants; whether the resulting framework is competitive with strong CLIP continual-learning methods; whether LR-RGDA and the CLIP text classifier provide complementary decision signals; and whether classification gains coexist with preserved cross-modal retrieval. \textbf{[RESULT PLACEHOLDER: insert verified headline results for Transfer/Average/Last, classifier ablations, and retrieval R@K, including uncertainty.]}

Our contributions are summarized as follows:

1. We introduce \textbf{LoRA-NF}, which reparameterizes the LoRA update from \(AB\) to \(PAB\) and persistently filters the adapter input using a history-induced approximate null-space operator, reducing effective learning along directions strongly occupied by historical tasks.
2. We build a complete continual-learning framework around LoRA-NF, using cross-modal knowledge distillation to constrain image--text relations and an LR-RGDA/text ensemble to combine supervised visual statistics with CLIP's semantic prior.
3. We systematically evaluate continual classification, cross-modal retrieval, training and classifier ablations, and the resulting adaptation--preservation trade-off under a unified X-TAIL protocol.

## 2. Paragraph Functions

| Paragraph | Function | Must not overclaim |
|---|---|---|
| 1 | Establish CLIP continual learning and the dual classification/retrieval objective | Do not call CLIP a natural continual learner without explaining adaptation |
| 2 | Define directional interference and the adaptation--preservation tension | Do not imply all representation change is harmful |
| 3 | Explain why standard LoRA is insufficient | Low rank is not claimed useless; only insufficient for directional protection |
| 4 | Locate the gap relative to gradient projection and null-space-inspired constraints on (A) | Do not turn the introduction into a named-method comparison |
| 5 | Introduce \(PAB\) and the intuition of persistent leaky filtering | Leave expressivity conditions and optimization derivations to Method |
| 6 | Complete the framework with cross-modal distillation and concise LR-RGDA/text fusion | Keep LoRA-NF as the narrative center; the classifier does not affect retrieval |
| 7 | State evaluation questions and contributions | Insert only verified results |

## 3. Citation and Evidence TODOs

- [ ] Verify that the cited gradient-projection papers match the exact forward/update-stage contrast;
- [ ] Check whether any prior LoRA method persistently applies an activation-derived null-space filter before both trainable factors;
- [ ] Verify that the generic description of null-space-inspired initialization/constraints on (A) accurately covers the intended methods;
- [ ] Replace the result placeholder using the final multi-seed continual-learning and retrieval tables;
- [x] Retain LR-RGDA/text fusion as an important inference-side component without making it a second introduction storyline;
- [ ] Decide whether the text branch in the final paper is pure frozen zero-shot CLIP or the matched LADA-style hybrid of tuned seen-class and frozen unseen-class text features;
- [ ] Align `cross-modal knowledge distillation` terminology with the implemented FD/CD losses;
- [ ] Confirm whether the final protocol should be called class-incremental X-TAIL, cross-domain task-agnostic incremental learning, or both with a precise definition.

## 4. Discussion Notes

- The introduction retains only the intuition that persistent leaky filtering biases adaptation away from historical high-energy directions. Formal claims about expressivity, invertibility and optimization geometry belong to Method and Appendix.
- The term `parameter-efficient` is not used as LoRA-NF's main advantage because the current implementation stores dense non-trainable filters.
- The introduction does not name LoRA-Null. It describes the broader family of methods that initialize or constrain the input-side factor (A) using an approximate null-space basis; named comparisons remain in Related Work and Experiments.
- LoRA-NF is responsible for historical-task preservation in the central narrative. Cross-modal retrieval preservation is attributed primarily to cross-modal knowledge distillation and verified directly with retrieval experiments.
- LR-RGDA/text fusion is retained as an important part of the complete method, but the introduction does not motivate it through a detailed comparison with LADA. That comparison belongs in Related Work and Experiments.
- The current code defaults to a LADA-style hybrid text classifier: cached tuned text features for seen classes and frozen CLIP text features for unseen classes. Calling this branch simply `zero-shot classifier` would be imprecise unless the final paper switches to the fully frozen text classifier setting.
