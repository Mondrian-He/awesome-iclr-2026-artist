# ICLR 2026 Main Papers — Part 169 of 199

27 papers.

- [R-WoM: Retrieval-augmented World Model For Computer-use Agents](#paper-4540)
- [Bongard-RWR+: Real-World Representations of Fine-Grained Concepts in Bongard Problems](#paper-4541)
- [Iterative Distillation for Reward-Guided Fine-Tuning of Diffusion Models in Biomolecular Design](#paper-4542)
- [UNDERSTANDING TRANSFORMERS FOR TIME SERIES FORECASTING: A CASE STUDY ON MOIRAI](#paper-4543)
- [Evidence for Limited Metacognition in LLMs](#paper-4544)
- [Guided Flow Policy: Learning from High-Value Actions in Offline Reinforcement Learning](#paper-4545)
- [Inter-Agent Relative Representations for Multi-Agent Option Discovery](#paper-4546)
- [Characterizing Human Semantic Navigation in Concept Production as Trajectories in Embedding Space](#paper-4547)
- [Time-Gated Multi-Scale Flow Matching for Time-Series Imputation](#paper-4548)
- [Bayesian Influence Functions for Hessian-Free Data Attribution](#paper-4549)
- [Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs](#paper-4550)
- [Decoding Dynamic Visual Experience from Calcium Imaging via Cell-Pattern-Aware Pretraining](#paper-4551)
- [Critique-RL: Training Language Models For Critiquing Through Two-Stage Reinforcement Learning](#paper-4552)
- [Distilling to Hybrid Attention Models via KL-Guided Layer Selection](#paper-4553)
- [Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving](#paper-4554)
- [SPRIG: Improving Large Language Model Performance by System Prompt Optimization](#paper-4555)
- [Bridging Kolmogorov Complexity and Deep Learning: Asymptotically Optimal Description Length Objectives for Transformers](#paper-4556)
- [PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks](#paper-4557)
- [GraphPlanner: Graph Memory-Augmented Agentic Routing for Multi-Agent LLMs](#paper-4558)
- [One Patch Doesn’t Fit All: Adaptive Patching for Native-Resolution Multimodal Large Language Models](#paper-4559)
- [Keep the Best, Forget the Rest: Reliable Alignment with Order-Aware Preference Optimization](#paper-4560)
- [Influence Dynamics and Stagewise Data Attribution](#paper-4561)
- [Output Supervision Can Obfuscate the Chain of Thought](#paper-4562)
- [Laplacian Kernelized Bandit](#paper-4563)
- [Closed-form $\ell_r$ norm scaling with data for overparameterized linear regression and diagonal linear networks under $\ell_p$ bias](#paper-4564)
- [Disentangled representation learning through unsupervised symmetry group discovery](#paper-4565)
- [CircuitSense: A Hierarchical MLLM Benchmark Bridging Visual Comprehension and Symbolic Reasoning in Engineering Design Process](#paper-4566)

---

<a id="paper-4540"></a>
## R-WoM: Retrieval-augmented World Model For Computer-use Agents

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=5ZaoXB3MdP)

### Images

![images](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/images/38e0c3a7da4ce8997e73fdaeefe4b7241f7b688697f35aa514b493d18057aef6.jpg)

![images](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/images/3b75d2dc4b0c0693eb3c37d8aab43bf3b702985b01a57c1b0d04352daea0d850.jpg)

![images](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/images/3f36fc03ba7cbfb8e422210f49b48c9f9780043550e1e11313f2f0cef57f7bed.jpg)

![images](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/images/a90d54dc546edabcb31cee02b32813375e061df3fc63b3248c766bbd73b78f36.jpg)

![images](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/images/be04145f38c16015ed6f7dddbee1e9b4e9a9039126f65fc418bc6253bc0680af.jpg)

![images](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/images/c20c9319598f7ffd21a11cac6b9cdc3293beef8290560e209a14364fe88a4b7f.jpg)

![images](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/images/ca66b1262478c220e4965cead8cbe4ad415d5941ff88f9f7918f2e81d489cc4b.jpg)

![images](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/images/e2ec030c94d112f346f57231798e7192661e1d1eb0a3baf464ba8c015da8cb8f.jpg)

### Tables

![tables](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/tables/07ec5dcbaad04af92639dd552d1a3b8d7e724f10775f0585207a586b4aae307d.jpg)

![tables](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/tables/25b9487bf496cb95c313d609b8a86cdef167d9433a0a7de98286bff35890d341.jpg)

![tables](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/tables/47c082ed8391d6517ef3fd041c421585e567dddac3a4b2ccd0c5e31de4cd30d3.jpg)

![tables](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/tables/56242a1de5a1679325e99149af59158b4e7220d0dab848da68769d3bd2fdab44.jpg)

![tables](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/tables/618cc246db9d0ffa1484c5797590230431ccbeccca1beaf8448e4b3dd9741614.jpg)

![tables](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/tables/6296b4b0ce138c9dc305f0d059ba8c4f860e515a7469953a8faaae47f39e0a7d.jpg)

![tables](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/tables/6ec82e0725b9329debc3bb2330d665630cba6e42c87b0d38799bd4a087cce74f.jpg)

![tables](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/tables/7719546dc742b42d275cbb8e3bfbeb92900d289dfaa0b94826387fac718ba3ff.jpg)

![tables](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/tables/7743703d302fad7c389f872a2bca656c67c101d4dabad10d26ff89b7ca7bf1c4.jpg)

![tables](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/tables/ac4458edd23b6282d8d99e75a38cb148738121b7c9ba43bf6edd0e222278d6f5.jpg)

![tables](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/tables/b8bab17b9e5d28246ab7acb291a627787b05012f15670f57e5101018f8b34065.jpg)

![tables](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/tables/dace0d31bfd570632fb0e1fba55c21b6ab6ff7ce7dacce9fe9ff2a7eb41a858c.jpg)

![tables](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/tables/e5d334504b183b14b9b0f7428c5d246a9ce9fa83a55f7196f78275cba191648d.jpg)

![tables](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/tables/e658bde4f80d1e8d13c1d6ab707aa020534eb6354e6b9e313da2db4fc6bf3851.jpg)

![tables](../iclr_results/4540_R-WoM_Retrieval-augmented_World_Model_For_Computer-use_Agents/tables/eedcb6bfb6f5043b1d76a0c81655866ae9e2d3de55c385d3fc24b9f0095c3990.jpg)

<a id="paper-4541"></a>
## Bongard-RWR+: Real-World Representations of Fine-Grained Concepts in Bongard Problems

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=3w4kZegWG5)

### Images

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/13803d5fdb55b21bdf7040d04df9e3a6ba27d646f38f0371eb21ea791db13093.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/193bca501395cf3ce0f97bf02c2c26232444eb371285442df4e39d47b05cba8d.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/24a9d7c1793eebde8988297485b3145dd8a3548a2fbe61c2b684c20915d224f5.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/2d57649513f8b1a55a7575cf7153c71e8d48ffb7a6746865e4e80310c67003a9.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/3134f87027d8f8f3f4fa0ce5811fef031a45c56f20062b870158486c92a8a8b0.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/36d97008dd095fc00eebfb36e95fb1886ef37ed3f16d2573a2d152487d524a43.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/3b899321b9bb2df20e6a6704eafd459330fb9e078441c9e9b93b087ee2083be6.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/3c311a76d8bd9a6a9b6705fb935043605e479c24e921ca5444ac2e28fc410996.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/3e8ee1226a8fe2eb18f75e640b1876bf7442269f42ed9189af667cad88776d6d.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/51a7129b49a0ac5b6800846bd12a00aa7c286dbee948dca7868341aeac5ae2f5.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/545219e0c8a74dedcaaa275d8f6b28497c6234077a1a88cc9c5caf57475a1b48.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/654fd37e7e20840a8aa8a5bc1c5a1e43b0cf6456fd2983a5e133839863005edf.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/6ccff3f682259953fd044be8da5c18dcc36bf690143ed08dfb6053ad30c7fadb.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/7831a41a0e508198e3b536822d9b73611c81c413243d541642aa2bce33d3c15a.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/7cc458ef66e93f889cb370aa57a48e348663fa72f4e38c30851432f8a56c56c5.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/82b3e63d392156c6778d1e780091b59a79b299c993b0a1c44f8a822de8eae4de.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/8b7f024e0130628faef56c0764cd432cee3c40d8652e93cb1404e003313edaff.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/8d1e3e90de7ff881a22130e053523ba368e52cd62e780a881069332ac50b3bb8.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/8d2bf392213197478b2658d87f88aded986c1dfd00a5840736b07ccfc247d5ce.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/90fe36e51c8895346877293abbd3d39231dfb760eea868be78d082f350563b5e.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/a20a274f6c71f09cc454e7fe0a6c8b4621c467a80b5120d1a021f51e330b3d30.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/a28bac8f41c4c008d67f7922622e63a6c10266b7a1f65ad071569951ccf0b143.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/a34a7aaa17b3019f83433fa8b94aae561b0cc23d6ef984c14bb72ff23dc94297.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/a5ff2f64bf4655dcddb8f79142df2cc3f344640d4c85186acf4209908352ddd5.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/b4e7b39ddc52bc069f099288ffb7024656c12883ad39c998f800b5c83de90e7f.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/b6a11d227ea1ffb8d402188a8581a642644726afc9a0a962328b294ceb01f3a1.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/bae43db79c2e432c1f70f6a1494b155af07baf90753410deb70b69bdc27dfc51.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/bce627e19388576d9acd1420a75f72daa4bf9c3140938a29f9288cb09f799599.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/ccd578488e1e1c148157b4bae27813fcd9c1173434163883a9b4d4af49804153.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/d26547d78a5c97ffac98aca7160b455d8cf5599e89200b754d158c40fe03276f.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/e24d491cb32d777913ccb37cafffda9800e6ebb34c06c14ee08e0aceb8097689.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/ef2759ee274e82ed860264f2fbba266bb90f79a81b54fee1f0cbc9a682ab0406.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/f9a6539627f845d97827e13c4d99213241863530077baac3fbf4c657038f48d4.jpg)

![images](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/images/f9e8d22d2184d63ca9a173024e18dd1e659d139c4cd70446efc2ca18a16e0253.jpg)

### Tables

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/09dd9fa0619cb4ec073855cb33fb7312809fc30d8aa312f1567001cdd1d6a92b.jpg)

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/0f7ea228baee12b7c0c1e810e05eb904dbdf17cb8fb0723c94817bfb4e3883e1.jpg)

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/12304cca7f6262503997eeb66116d93dd817d185c91bd769120efce38016dc2d.jpg)

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/2b2de22fea07e7f815defb3c29020f7fa189d81731d60b40806748933698a30c.jpg)

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/490cae7c72a9971e75df592e82998d2477720612b11077c3efaae7bf0e4eda8d.jpg)

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/55f09511325b4004d1b74b5a2f823e0a2ab4b3f0a8a47aeb2d0581b705730589.jpg)

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/5819c54c679085ec3b0d9a38955f6b77affcf0072d111f3745ff3493c30ef054.jpg)

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/5b3f467f849db10d7db73df068f628e471f13ccaaf999e4ef6603e24dbcc3410.jpg)

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/71bd52650fa16b12c2fa36b057a5bdeb1148f7aeff9aac042ff3338f9ea7b7b5.jpg)

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/7ad4d12ae408c3e29dc8db71b7ee300bd961bba7686628a5a1e7d4cb135ebf10.jpg)

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/859e7b117acc2de65a6e0b5048b75aaba35f51909201a934e9d1b7cd92249944.jpg)

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/a13d9364f97b07030c380b82ff1d78647a9b3e0dfd650746c84cec0109d04236.jpg)

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/a8f3cf14cfb03f7f7f20f0a207bc392e189eebc5cbff4ffbeebcf3129d862838.jpg)

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/aa37f8659c3c760b853252fa47cce36f8de2556b9fad3b94e7b81e995c8f3402.jpg)

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/ab35a0ffd763a7d5416ef6be6ee6c0ac94d20be099e4daf8304f1827abb742b9.jpg)

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/badfea13f55d923b3740079a4365cdbfe10cf37805956a4f404c2c518905e97b.jpg)

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/cba16b939251d9f4b417b064c236dfc414310327369caa0bb5aa5be42d93acec.jpg)

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/cd1f5908f096797d0e84d887153bc4bf57a5c48ed574aeca5e8919a7a7709938.jpg)

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/d74fb98e490f6dc428e0db4f6305587eefba85207dfb3e94cdea3e9908facb48.jpg)

![tables](../iclr_results/4541_Bongard-RWR_Real-World_Representations_of_Fine-Grained_Concepts_in_Bongard_Problems/tables/ed377a6e315bdd38004e7585929f03e0d18549d1865a35ad34ec7578bebb66db.jpg)

<a id="paper-4542"></a>
## Iterative Distillation for Reward-Guided Fine-Tuning of Diffusion Models in Biomolecular Design

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=NFffW9tBmC)

### Images

![images](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/images/36b582d3be48cc1686c1b56258a00dae02363c3a79416830c91f23b4a5c69761.jpg)

![images](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/images/3b0f62725554276254946cd76b15bfc22a927b86ac51d5fba8c81998a2c25dfd.jpg)

![images](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/images/3f4dad058d83b265b75d8fbc57119adc4596576458059bb2422bc354323a7509.jpg)

![images](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/images/543e64bb004876df4f1e91436bf27e7e78a67d8ec36f2f78ab38d4359ac538f8.jpg)

![images](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/images/5f8b894345dd8a9649c491963989a9b8f14d45e6d95b3c950b7bddb19018a934.jpg)

![images](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/images/f9cd0c5b19c0bdf4913ffb2c637d834145d606a5b64fe0a9bf118ac419d75744.jpg)

![images](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/images/ff6cb2aba7ac7b120bcb247760e11220d23c1a3277cf32ddd949e1502d88a68b.jpg)

### Tables

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/0c0f1d79bb2de7545511ee96963390b4fe6af1123a5cbe7b6a170e3f7a1681a4.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/12e52a932522ed69237cfa47a378d55233bbd09557a453e7d5ede43302c38523.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/1ef5834f69c8ce8594e32d67340211064fed338249a8e809875e8dcfe7dd0356.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/24ef85ea70bfb9186d96b3b8063732fbce9d0e0dbb1b3ff4c227dc882e99cf4b.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/2963375c43fdd0fe1f4150d6e063f4a67057ddc82ecb1acc3d5bef2dd8ad843d.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/2af3b5b2982c308564de4c5500ef18d17b6b24d9384afdbbbace145114ae6973.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/408c74e3635a8f25508f81f8d3b0dbb6f1aacbe2a600b0fc55c8ef4773202de2.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/50fb5e372da3cc302853a025848372141b49d2400f2466330c0f57ecad1e1ceb.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/5a3091aeb1a5038723fd8fe45780975d2efc1840180fa6de38b2c7a3e3fa15da.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/6bf259ba4722b3114fc6050a7b62f5b39a102d718f27dd6a918d9e572c015021.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/7453d1be5e6e90ec68848c9f2b4b1b8d7a4d171d4dfe11f68475857840949ea0.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/7508c0ad90529555d8bc39221e46525116b6e030d561565625f8dc20d9da8398.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/7a11c3e37ba82edd32573caf771b7f92628c6a33c3e0eeded89539d462c2e6b4.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/8119480565b0858a91733ee795286c4d929a0e97844ca95cc2697b659712a652.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/97f513bd33983d6cfe18a34df3308918da9f1b93e61e62dcf6af7ab9efe122ec.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/ad2c679f1a8d5895007f96825ddca153610a35f8957cbbd5d9dc5503f68e2d0c.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/b39aab844463afab7f6c8e7f284a58867d01f61f6f1d4b3fc78195f16f3cf45b.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/c25ab5311aa4abe5245f25846d2dcbf4eec32d37b9a32e0dc10882f524bd79bb.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/dbcf61955ccf4a9d584e2bb74b3578706a13eb2f6f9f309791c3452fd1420bab.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/ebe79133a078cdb8236de5e7d4a9b341c1e5c78edb2dda60e220877c59d617dd.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/ee31466ab4e7ffc268d7b3d5577733abd4c79c0289ca248154faf163bc77ee47.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/f106e970f9198ab2acdb2b8b712d4e042754c95d9774fe8f68255584429938fa.jpg)

![tables](../iclr_results/4542_Iterative_Distillation_for_Reward-Guided_Fine-Tuning_of_Diffusion_Models_in_Biomolecular_Design/tables/f1e19b956504e36e12f3a840cc9c0234268169c896138cde516772b1b650e411.jpg)

<a id="paper-4543"></a>
## UNDERSTANDING TRANSFORMERS FOR TIME SERIES FORECASTING: A CASE STUDY ON MOIRAI

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=iAPSx90gwJ)

### Images

![images](../iclr_results/4543_UNDERSTANDING_TRANSFORMERS_FOR_TIME_SERIES_FORECASTING_A_CASE_STUDY_ON_MOIRAI/images/3d3fcaad3827515047256c461b18c085d950932b7d8290da0caf4ecd2b5688d0.jpg)

![images](../iclr_results/4543_UNDERSTANDING_TRANSFORMERS_FOR_TIME_SERIES_FORECASTING_A_CASE_STUDY_ON_MOIRAI/images/908b26fdf485736d0e5fd44474b810ea000f3c7ebb6d67c82dcc19f1ca623e15.jpg)

![images](../iclr_results/4543_UNDERSTANDING_TRANSFORMERS_FOR_TIME_SERIES_FORECASTING_A_CASE_STUDY_ON_MOIRAI/images/d9f231752e51663f603dd425569a2bfcb46de86ca92af2fd29828baa29fc3344.jpg)

![images](../iclr_results/4543_UNDERSTANDING_TRANSFORMERS_FOR_TIME_SERIES_FORECASTING_A_CASE_STUDY_ON_MOIRAI/images/dd47686aa9c7cbd71130ae8f266412a58ba8a27bfe6bddccb6ecd8c162767171.jpg)

### Tables

![tables](../iclr_results/4543_UNDERSTANDING_TRANSFORMERS_FOR_TIME_SERIES_FORECASTING_A_CASE_STUDY_ON_MOIRAI/tables/06bebced15aa401b0121e891991ac72946142dcae92b9769347c3435cd01195f.jpg)

![tables](../iclr_results/4543_UNDERSTANDING_TRANSFORMERS_FOR_TIME_SERIES_FORECASTING_A_CASE_STUDY_ON_MOIRAI/tables/23e3c8e8352ced3a3a057884e754c0e12c90730e6901a881796f9aba5308a236.jpg)

![tables](../iclr_results/4543_UNDERSTANDING_TRANSFORMERS_FOR_TIME_SERIES_FORECASTING_A_CASE_STUDY_ON_MOIRAI/tables/2f3e090afb9b66998d039b26af862888a72af80f1f5ddf4a5b87c35c54e27f66.jpg)

<a id="paper-4544"></a>
## Evidence for Limited Metacognition in LLMs

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=gb9HR8hxtU)

### Images

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/0114a0eb031c1a8f87b36c4af5d3ff8053095b8fcc586b4f31c65ff534a68d5f.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/030a1af2a77ecd4eacf0b09f6ec0a2914434a3b5bed5e99b5a0ce1d0dbe8dedb.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/0a5fa85bdea8a11b21f0786c5dde9bd976c641181e4a65747af0ef7f45648058.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/0f88e5719b7f966703a8e60425935fea3bf6b3d4b12147f043e41aaf9792a2fd.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/35e522dc018bd3c2dc5f38b5d59a3b8f47fa46d09c1e2a2e8223f83fcc42f8c0.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/51c8ebfcac3ba749008da4bddc64aeea1e280cf256bb1dd21843d025452eaa89.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/526fd905601afb3b80403e48e71cc0159099d04ab0c628650efc163bed3cb514.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/52fbbe16326168e682b243ac57a602dccacea8aee37185b7b6aae68367b24b21.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/5324d08973c0e8ab2a780d497f09f537febeaf5828cc4f41ae9247d8368269f0.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/5fd4230e00ac069eed7cc23ab08a08e2c2a21524292a9e4cf12d79a6b239e7b9.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/64773eb2ae4e4ce8043c869874db4d91490964946b300750c917a208d24e3ccf.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/664854e5f7d3762351fa314c1017b3f6655f66622dc31062089fde1390393890.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/6aab9b29658d0b43429fc9ddb4146620ff3571283f8f7e9d1040bea4951f464a.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/728f1e7079928c633c9660c0c5cb7e80a4d9ca2fea6eb13928ae3ccb9a20e8e1.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/75e619791b7ba60b505b61ce186d0593a71ff2d571bb511b7fed155f77b89708.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/99ef174de8066198019ed0d20efc6e933fba2137cab702d10981fa28da2eabc0.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/a402567a8d37a10376d36ab01c5ddcf55bcad18acad57406062284824898dea9.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/b04e174fbc8995e43d75e8b0818b9e5c8fcf460f5bed8cc814951860ede7c378.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/b5c5cdd3618b575bcfe91cad67f177f89d06b0f5ff43b3ce6dd7b66c18d4059f.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/c045b00b370e5a5e16a1553fd36ff2deee4ef7232828896ef009816577224c86.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/d6ac7663481c77c6945a5a7f2c803f5797af125f875ad264fbfa3d8e87f3ca06.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/d6cbb69459b2129ee5eb91e25dd2b48a6517de9a4a6313ac00a43f6445008cf8.jpg)

![images](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/images/f755b34ed232ee621b8c2dfd5a4b8ddaae19e107837082b0938afeeeed52cddb.jpg)

### Tables

![tables](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/tables/8e1bc61ae82128bdcc0f18240c5f07ac899aecffeea878d340b9ffaf6e5ab97f.jpg)

![tables](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/tables/b437c8f1d1dd1eaa335bb9f810ee103644119fe52a8c033c5ea7cc530a9d2bbc.jpg)

![tables](../iclr_results/4544_Evidence_for_Limited_Metacognition_in_LLMs/tables/f27e4ed3edffc74a67d54ab014a12143bd882ee4b9a2c7286e11826bd5bdd4cb.jpg)

<a id="paper-4545"></a>
## Guided Flow Policy: Learning from High-Value Actions in Offline Reinforcement Learning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=EBjy1rmpv0)

### Images

![images](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/images/02ade64c3d30488a36899a8907c7d366f4f7b39392483c656834797dca2fe995.jpg)

![images](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/images/49670d415ec75b16ef335321439e09b60d2debc67b5f08269ff2412d31cc0828.jpg)

![images](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/images/4b403501b4925e0afcffeaeb5ca51a0c0922c24ec102f78be10ef0effae260b4.jpg)

![images](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/images/53711a57a0059cc9a18fa6f03cb5578594d9ed7e6d58455f66a76e6d85bd0b82.jpg)

![images](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/images/a52786e8ff8ade7214a71cb4469f3145099c7fc158d1ec7b3fb21a409537172c.jpg)

![images](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/images/cce81545a363901d1c84960907e7239a62da5c459064afa93a1f1a2fef206d27.jpg)

![images](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/images/f10b58e941bdacd272755a69a1b95caf78d89c03bd6683585e62f407cb001926.jpg)

![images](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/images/f7d3abc6fb85a0abe4c4679a6187eeafd47e73fceb9dc583cb0a67674b502278.jpg)

### Tables

![tables](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/tables/045df6827a4807b0ba56be6c640f04a310b6a14410e1b63505ab6b98316b1b83.jpg)

![tables](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/tables/05a294e21d6d159cd7fe0d78106388353b35560863f8a219b88d113fa59974b1.jpg)

![tables](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/tables/1210cf208cab7fcc2c7bcb2b60bc7d7277cf8f72c6d654fe5c32e307ce479de2.jpg)

![tables](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/tables/360af408cbe3ccb08099bed37de2f2be7e4bb6de3b36918e2aa9984f450ca972.jpg)

![tables](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/tables/418be7ebc066fe034676b1c1401368c80ab766900e29b8a71c963122ca4c45fc.jpg)

![tables](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/tables/620666077dbbdbe791517edb03dbc8a0b21272ca8f8ca9312bb09d5d5589d7ce.jpg)

![tables](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/tables/6ea8b7a29bcdf750437620a53bc27966f4cb3b5cd33a2e0c5125e0df09e88101.jpg)

![tables](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/tables/735d8b392d06fff6dfb1b4eafbec0468060e67e22c6897df851b9a1f29f55238.jpg)

![tables](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/tables/9f59fe75e5606dec5a33906c2784e03ed8cd67a3aa3025551abca8e78be30992.jpg)

![tables](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/tables/a4d70ba481bd30ed16a37617e24e3220ab0adc6d773cebf1528195b8bb305acd.jpg)

![tables](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/tables/afff189a4c9a84cfca502aa32f19a4928764f8067821284232e642765587db71.jpg)

![tables](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/tables/b7ed67a189e65917767934b45794aec9e7eac4394cdece7a24397e44c96ee98d.jpg)

![tables](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/tables/cc23f310ca402a32e3d524f3c8480a71807d05b468a2e4ea1fde6e853b8a0c39.jpg)

![tables](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/tables/d08b497c158e214445156a54551ee34db89bd756cd22e8c9b3de812c32b8e198.jpg)

![tables](../iclr_results/4545_Guided_Flow_Policy_Learning_from_High-Value_Actions_in_Offline_Reinforcement_Learning/tables/d9b7c771509e96ad7c34562638d7b6151496ea0f5d5088642cf778d7a8a73483.jpg)

<a id="paper-4546"></a>
## Inter-Agent Relative Representations for Multi-Agent Option Discovery

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=Fte7TOqnQp)

### Images

![images](../iclr_results/4546_Inter-Agent_Relative_Representations_for_Multi-Agent_Option_Discovery/images/13f4be44c92b3268d94583499f6d8995d9fd45474eafca1f568006fd6e0af8cf.jpg)

![images](../iclr_results/4546_Inter-Agent_Relative_Representations_for_Multi-Agent_Option_Discovery/images/269889815f2994eb6ebda9392487fc8baba821c6920eb7e12c9df8e523f435c9.jpg)

![images](../iclr_results/4546_Inter-Agent_Relative_Representations_for_Multi-Agent_Option_Discovery/images/2dd4d6c70e4f8ba9f93e569d6ffd757cec29cf2bc6d32f069c150456c3db0818.jpg)

![images](../iclr_results/4546_Inter-Agent_Relative_Representations_for_Multi-Agent_Option_Discovery/images/511e5c1efd7508494e45117917c8ce21496a4c0f7aacea78e9ec6c3528032496.jpg)

![images](../iclr_results/4546_Inter-Agent_Relative_Representations_for_Multi-Agent_Option_Discovery/images/5b446cad23874b473cc415a9e0409333a0d506c2074bc00e40699483f21abbc2.jpg)

![images](../iclr_results/4546_Inter-Agent_Relative_Representations_for_Multi-Agent_Option_Discovery/images/60f0cef383ec1b768a147b94898e71b49fcb54eca0d19482406522ebc60181d7.jpg)

![images](../iclr_results/4546_Inter-Agent_Relative_Representations_for_Multi-Agent_Option_Discovery/images/ba21e12a7b83e609847dfc00ce49be5d25e49ac31850007ea6f3827ddee81a5e.jpg)

![images](../iclr_results/4546_Inter-Agent_Relative_Representations_for_Multi-Agent_Option_Discovery/images/bdcc56c1f2a543844f4b56ad4ae206e91a04d700d66b3cbedae4b600881f5dc2.jpg)

![images](../iclr_results/4546_Inter-Agent_Relative_Representations_for_Multi-Agent_Option_Discovery/images/c310f09ae26678664b792efe65f99320c84b3eabf8e44f73618a8f643cf780dc.jpg)

![images](../iclr_results/4546_Inter-Agent_Relative_Representations_for_Multi-Agent_Option_Discovery/images/d24b32269e46cbbb8aef5490033e6730054d4163b3773ef009ae3411d81b19ea.jpg)

![images](../iclr_results/4546_Inter-Agent_Relative_Representations_for_Multi-Agent_Option_Discovery/images/f2553d97212a79112bf22fd7d637713644e1b5afe0f408dc11bec6e5027b5b64.jpg)

![images](../iclr_results/4546_Inter-Agent_Relative_Representations_for_Multi-Agent_Option_Discovery/images/fcddd17efe8852338ab5f230869663698fedf3c7f702008687c222e78325d394.jpg)

![images](../iclr_results/4546_Inter-Agent_Relative_Representations_for_Multi-Agent_Option_Discovery/images/fcdec766f1e126ed1eef9072cd8f6135560c08352e60fe73f16b44430c1db72b.jpg)

![images](../iclr_results/4546_Inter-Agent_Relative_Representations_for_Multi-Agent_Option_Discovery/images/ffde656d814692deb5c7990ff04b6980232a8237163eba31dcdca4cdcdd5808c.jpg)

### Tables

![tables](../iclr_results/4546_Inter-Agent_Relative_Representations_for_Multi-Agent_Option_Discovery/tables/23df249ae1f0aa5f0a92705e7ada91f60fd09a218276689728de2843302ad51c.jpg)

![tables](../iclr_results/4546_Inter-Agent_Relative_Representations_for_Multi-Agent_Option_Discovery/tables/2791abbdc6f8010c79b00d19621277ed0017eecbd786141bb8afa7d3d5f80b8c.jpg)

![tables](../iclr_results/4546_Inter-Agent_Relative_Representations_for_Multi-Agent_Option_Discovery/tables/308f8bb3355cee1dd641155d815323254df5b3cbce833bb3db9b6fb799b7bf09.jpg)

<a id="paper-4547"></a>
## Characterizing Human Semantic Navigation in Concept Production as Trajectories in Embedding Space

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=QQVmIR97sf)

### Images

![images](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/images/2ce0761db5c41b9561d4a29f5f22898ca25a6578b79646356eeec90cdc4cd97d.jpg)

![images](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/images/441f0f14310d6632ad00da93f38854533cb0feded245ef3401d31c09f91d8266.jpg)

![images](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/images/456f3c7119640ea1a6d98a49ff7135e1d3e7dc653cd8679e6a7e35fb0783f831.jpg)

![images](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/images/4d6e2511796e7847df7e76063da16db22140a6be576ea81297d7adbaa708f8da.jpg)

![images](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/images/51a138daf1242908444470c055730766f6414d3a19f8af9ac3ddf901db85fd5c.jpg)

![images](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/images/63d0ce5b75cae03bb42b1140baa3c5c2928f560cbdd157424cb83d87e9571bff.jpg)

![images](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/images/69ece89e973e453325acc44384a8e5e1c08a78a97600de5937f9811eb5e80c61.jpg)

![images](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/images/7dbb7c3720afffcb32a4de46aebdefd1d396dcdf95ea91f0de12213abe251d7c.jpg)

![images](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/images/7ea91e629d7ac17200796d73d01f5e4c72d17b34bebb0a44a211189e5aae8edf.jpg)

![images](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/images/8275254d13486ead1ffc37fe793fb2c2cce59369779a3fa1931f2b3cd187670a.jpg)

![images](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/images/8c02b8d255e564b25462d213ac16603424aaa5554ea441b4bbb2336056b9d4f3.jpg)

![images](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/images/952d259d5a18709583eeb312df101e271cfbc48960bc90661457999ee49f1e3e.jpg)

![images](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/images/974fb16b765f70b626c6ebe31ba41c066dd78c1734bcedf8dbf443eaba13bc9f.jpg)

![images](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/images/aa3062809a52402d9cc544d776fe4ffce01d9a0a2855a69058811d6439580d11.jpg)

![images](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/images/b0e2d9ec3a20891744d0a5959ddca0d37a05a2f27d148b7f8723b4ea276ec8ee.jpg)

![images](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/images/ca5095e2a8fa3103ce94c22d41d18054fcfc30bd1412887a6df7e7d0e6d7d737.jpg)

![images](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/images/cf5d0e632840539dbdbb98f0fe01e0285a64bfa3bc8327a09ce7fe17ce90d614.jpg)

![images](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/images/d4b2858af91152ba4f6e8203a1f64876e96feb6e01f0e1534e21024d822ec2fa.jpg)

![images](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/images/dd8b758494633a349fd83515c61736ef53fb0cff8f577f670258b8c4259b1378.jpg)

### Tables

![tables](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/tables/298d8565a86578b71b1f26943af4fc837eec425f8e3f5f993522b27b85ad9a86.jpg)

![tables](../iclr_results/4547_Characterizing_Human_Semantic_Navigation_in_Concept_Production_as_Trajectories_in_Embedding_Space/tables/edaeb7f50f947b34c9636d125a336000bc5b7e24b22f61893086a95cd74fa5dc.jpg)

<a id="paper-4548"></a>
## Time-Gated Multi-Scale Flow Matching for Time-Series Imputation

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=txvc61ONbs)

### Images

![images](../iclr_results/4548_Time-Gated_Multi-Scale_Flow_Matching_for_Time-Series_Imputation/images/2b0d0f308f75327abbec9146bc5e4a0e5b9a4c3bc38533147b5dfc5597ec673d.jpg)

![images](../iclr_results/4548_Time-Gated_Multi-Scale_Flow_Matching_for_Time-Series_Imputation/images/42e6f36737d93daeaf3ef4d5137768f2897941ec9728e3945fdb72877afdee6d.jpg)

![images](../iclr_results/4548_Time-Gated_Multi-Scale_Flow_Matching_for_Time-Series_Imputation/images/5e0aa124a0a99ec4a393eadc9ecdbb64559cbb52e0cf0a602d3beed9a8c84119.jpg)

![images](../iclr_results/4548_Time-Gated_Multi-Scale_Flow_Matching_for_Time-Series_Imputation/images/9deb5b1ef51937903c9c6e3524789ecad283550805e341dd9024902986f0756b.jpg)

![images](../iclr_results/4548_Time-Gated_Multi-Scale_Flow_Matching_for_Time-Series_Imputation/images/ae46d56e1ff06fe3bf5951583a928c17b9caa4be32dec3dd394c8e9a0d2cad6e.jpg)

![images](../iclr_results/4548_Time-Gated_Multi-Scale_Flow_Matching_for_Time-Series_Imputation/images/dc0c47c23f06e9662af8114e871f1cf4ae3b28abedd117ca853d663f000031a8.jpg)

![images](../iclr_results/4548_Time-Gated_Multi-Scale_Flow_Matching_for_Time-Series_Imputation/images/dc7451bf78b3661952729135e67ce95dfb3b57537197e4702ae8e64d7254f013.jpg)

![images](../iclr_results/4548_Time-Gated_Multi-Scale_Flow_Matching_for_Time-Series_Imputation/images/f8a8f83b5f742cd895e417ec5aa17a5fcc05fb946a2682899f36589e07cc7268.jpg)

### Tables

![tables](../iclr_results/4548_Time-Gated_Multi-Scale_Flow_Matching_for_Time-Series_Imputation/tables/4eaab354b70b43e11208528aebc81176c121960cdff735ee908b20ada739ec6e.jpg)

![tables](../iclr_results/4548_Time-Gated_Multi-Scale_Flow_Matching_for_Time-Series_Imputation/tables/68dabc57345e09170d5c544c4a673d57a3022ef971c06a38f0a649c609a3a680.jpg)

![tables](../iclr_results/4548_Time-Gated_Multi-Scale_Flow_Matching_for_Time-Series_Imputation/tables/6abb06c7aaf0c2572e530a24fe4fd9cad90dc6063f29a6338dd2f48bc0b11bba.jpg)

![tables](../iclr_results/4548_Time-Gated_Multi-Scale_Flow_Matching_for_Time-Series_Imputation/tables/a27560bfc9315951b223606f3760f1c61050da00f21926ad881445c4f1193ead.jpg)

![tables](../iclr_results/4548_Time-Gated_Multi-Scale_Flow_Matching_for_Time-Series_Imputation/tables/ae6eee272ba8cb4e04cc3821e272d9b7e7eecfc6a6c48fea89da792f1a253026.jpg)

![tables](../iclr_results/4548_Time-Gated_Multi-Scale_Flow_Matching_for_Time-Series_Imputation/tables/e9004a29942331c2620e8d640f9372f580d100872e518ae2337e2e0fc0bc98d2.jpg)

<a id="paper-4549"></a>
## Bayesian Influence Functions for Hessian-Free Data Attribution

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=YEBpZVm70i)

### Images

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/08a80885437d323b74376ec32928402bfe25b73f8d6b6edf443d26881c9187d5.jpg)

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/0a2325d8a86e5150da3d5bfd9ee2dbcf73bbe271236a9f0eb2cd900195de105a.jpg)

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/0adad54586dd145e0e370c40249dadba97be22c03bb53543832c21d417c8807f.jpg)

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/3606bb52e999513d646a7b131bba2ef0f2e8a756bb712eb1c2c304e22da574a8.jpg)

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/3c2925cc87af57498a4df269240fff84d754224c460b961d4c2825201b7ee69f.jpg)

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/3d15fcc500e84d0e5a44283401324e725041bb986e0a15edf306cae7766d52ea.jpg)

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/529a95a727c796741f6cad45ecc1e89ddc57af555fa814b6c8ec30ee9bd999c5.jpg)

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/59aa852d771a50373d38f04040df86cdd84ff49634b68df42251a4363904b8a0.jpg)

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/618fe896e574b1c0c5d57d1bb75c8b18e61291824bf926d3433d9e172f435008.jpg)

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/663e69f1e292de89d4713243ba1211b3899c9d5d1778cb9e9ac925aa27af6feb.jpg)

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/7b7e535c6e08615d7a4952f0830694045e08091cdc8cd096a5416981165806d2.jpg)

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/958651c00c3f76a25445803fd264fd166244181e7e2adc975c22dd9477947aff.jpg)

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/95cec6bac3b547c37e083f31521028d041b95096004e7c4955a646447bd4f99f.jpg)

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/9aafa8145f97d42dee9554a00d6db71e87461fce76617a14379866ff58c244d9.jpg)

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/a9ed8c0286eb1f1ba8ac86c0eba5e29ba8dc11204b47155f86cfca550bbb1575.jpg)

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/be1a0915ecc9e287ae05fc9148c4f21890cea4a737c5134331aab4f74093cc33.jpg)

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/d1fb4460270ca826242a7697292eb40ec7ca019ec385f72756a0e4decc2951e0.jpg)

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/d58e8254d014e5dfcbc75a782b194ef9256736b964242177095cf7605b1ec632.jpg)

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/de66522743120bd422cc9383a59759c90505cdb469cc7c9ff0a9f6fe66619e26.jpg)

![images](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/images/f1c19a0ba786f3a5d243fd97b9b31d1ed8400b9bb0631c752d8f29693ff48cb8.jpg)

### Tables

![tables](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/tables/0ad6e54be9a84a14654902ace571235a6ffc72acc7a2f0b4d5420837c581b3dd.jpg)

![tables](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/tables/2073d22468e456fcd936defa3d0384f61a9d8b5626122ccda886fc1106d2e0ca.jpg)

![tables](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/tables/23ed8a0e8f87fd446abdf158873f0cbb187a30e3b86edf7e0c67e0e050f5e025.jpg)

![tables](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/tables/44f82b6d12036f47bb8d5b63481c65083ba9acd19e42ed933df47d6ad29f92d2.jpg)

![tables](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/tables/76299da3e82f9870b74acd85a6e82cf8b6a3b69cbedd41ff8bc34db4e651c00b.jpg)

![tables](../iclr_results/4549_Bayesian_Influence_Functions_for_Hessian-Free_Data_Attribution/tables/ada084dbc3e37e0dea48eb89b2cf245972836496d53b9f27bf4192b095a0415e.jpg)

<a id="paper-4550"></a>
## Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=qCaq3jGb0S)

### Images

![images](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/images/01da845e7cae555401362e82767c069f98cfe24e25695e6337f5b31f606a883a.jpg)

![images](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/images/0e7a195889eee7f1d03f26938f408cb9ee83a52f727e6381a681c300ecaff5b1.jpg)

![images](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/images/1e9cc49bff2aabda18c966de1cb18b16714c0e689655052f92eb487af9e1ca11.jpg)

![images](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/images/286017654c22eea668e5f9180af4f816b40a490d3df351db42a02592ee573793.jpg)

![images](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/images/3b4ef1d75774984eba51c8170464ed27835da04d9f52f00e722dca2cba952c35.jpg)

![images](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/images/3e9a94817668c3f7ac03d93dd48e66f37a55990afa0b5fc6c4df8bf3b50c26a9.jpg)

![images](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/images/54ab9213585c925f0c3fec35e24a9e5d5a7eeb5f22fdb3de131a429b37f5e79f.jpg)

![images](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/images/61bac9d7a13838b076832c239053708a1caf9d6891b415fa87e08c84a8a7e448.jpg)

![images](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/images/70ffb9492fa97486c3a6ce694e7d8e80eb2c4e7bfd10a5fe023339f8b009693a.jpg)

![images](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/images/7228d24e7fdb2fa570183fc1b1d71254fe2fea99b1d805507ce00eb7f3b0ebc7.jpg)

![images](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/images/72b96fc9be1319789092cb9102cc3707cfb2a81ee1f2d81581abd9dc27d40320.jpg)

![images](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/images/7f5b86e3d3a26eb932ffbb6f8347d32d350b4ed4cadaf218deb02dc5342c35ec.jpg)

![images](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/images/8f453f3b25f9a83916340257c4ee97258216e396abeec65d320269a3027b1346.jpg)

![images](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/images/99061c0db47f8489aab7bad703c40f2728244b349fe45345287f68aa73b0e7cf.jpg)

![images](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/images/b74a48052a3e39874ab665ee2964c51dd9d0f3b6689ea27e0b3b53610ab3b9cc.jpg)

![images](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/images/beabec770d24c8846db6838eee4c88d318edfd6bc873ca5a5b81308ad28ce09d.jpg)

![images](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/images/c3ce2444e12f092094adb24b593a1b76e13da833fd8b67c1a8f2c47ea1117a9d.jpg)

![images](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/images/cd656503702c80d0d2cc690dcd29387796ae503af91e9db1825471d7fb31d20c.jpg)

![images](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/images/fc57d2ae87a021d83d2d8038221315f4e49ae86a0f8d816f39dc3ca25fd632ca.jpg)

### Tables

![tables](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/tables/28a7363c566ff6174b73d73dba8a02a61a7ae60a31c2ad49c8f1d1b3a375cb72.jpg)

![tables](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/tables/35f96d31403707bf482fc04bd2f30d595518587abf3eb5623063f044d65f1cff.jpg)

![tables](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/tables/54b06c2c35545386c4c08056a3f08853405a01a88abf85e6f8451dac6e30c361.jpg)

![tables](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/tables/58bdd15ed09ff4d8ec30ae6e120a2d165d6d212bbba3be375006fbca1703af97.jpg)

![tables](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/tables/8a02ff822d360e506eb8b3a703042407c343b88cde3fa2fe7f3c53799515c3b2.jpg)

![tables](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/tables/95b67d3e1a747e92b6af4ad5652e413a87d8759b1253e5e18da09e417afb786a.jpg)

![tables](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/tables/adedcf8746b97cf7b1879725d95b793fb5aa9f9002749c1a3844d1d64bd724e0.jpg)

![tables](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/tables/ec1a249ec50a2f81275488c8a3fb82bc390ab095c38b65f946e8b640876ba72a.jpg)

![tables](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/tables/ec3b5eff02a97ca75dcc7fb38949ce7e8c85cf8d36196494eb83a8534f5dbcc5.jpg)

![tables](../iclr_results/4550_Cache_What_Lasts_Token_Retention_for_Memory-Bounded_KV_Cache_in_LLMs/tables/fe15aa0c257835f20eee693749106e36b11d45e47000090fe39b9f56a2b6b690.jpg)

<a id="paper-4551"></a>
## Decoding Dynamic Visual Experience from Calcium Imaging via Cell-Pattern-Aware Pretraining

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=z9kAjjRejs)

### Images

![images](../iclr_results/4551_Decoding_Dynamic_Visual_Experience_from_Calcium_Imaging_via_Cell-Pattern-Aware_Pretraining/images/53e14d9e6fdea1766e099772302df71016348a8da963f7084dd1cecb355faf78.jpg)

![images](../iclr_results/4551_Decoding_Dynamic_Visual_Experience_from_Calcium_Imaging_via_Cell-Pattern-Aware_Pretraining/images/60b8f571801d9a9372f41e9a24482dfa632e69cf3237f5d43161245461683458.jpg)

![images](../iclr_results/4551_Decoding_Dynamic_Visual_Experience_from_Calcium_Imaging_via_Cell-Pattern-Aware_Pretraining/images/725189d5f98f7211acbfe7facccba5f456df3c4de5b9286c7a7c7586425d2c16.jpg)

![images](../iclr_results/4551_Decoding_Dynamic_Visual_Experience_from_Calcium_Imaging_via_Cell-Pattern-Aware_Pretraining/images/7df1563ab8ba189af249a83ad0fd5ca357707f9c2007c83819e9ecb05c01d957.jpg)

![images](../iclr_results/4551_Decoding_Dynamic_Visual_Experience_from_Calcium_Imaging_via_Cell-Pattern-Aware_Pretraining/images/9fb3abdba6963ee86818c98b763118523a9f1b871fb407788e6d6ca55ec00807.jpg)

![images](../iclr_results/4551_Decoding_Dynamic_Visual_Experience_from_Calcium_Imaging_via_Cell-Pattern-Aware_Pretraining/images/a7b969dadaaea16bdce720a25ce04ded56b83f2095e12b882ef13512b215a194.jpg)

![images](../iclr_results/4551_Decoding_Dynamic_Visual_Experience_from_Calcium_Imaging_via_Cell-Pattern-Aware_Pretraining/images/b6f25a24de1f40097509e8b5e6866a8f02ca602858dce353116de1e5202fb816.jpg)

![images](../iclr_results/4551_Decoding_Dynamic_Visual_Experience_from_Calcium_Imaging_via_Cell-Pattern-Aware_Pretraining/images/bf138931a6ca24e427b3c55ee224b7dd1ac7a8e6d262df3a2908a04377e59e40.jpg)

![images](../iclr_results/4551_Decoding_Dynamic_Visual_Experience_from_Calcium_Imaging_via_Cell-Pattern-Aware_Pretraining/images/d58ab5c47d6ac5bdb2b5b4b10c23af3123665b2ae1905f31d167bc2519240a29.jpg)

![images](../iclr_results/4551_Decoding_Dynamic_Visual_Experience_from_Calcium_Imaging_via_Cell-Pattern-Aware_Pretraining/images/e3a3f536ab5614cc37f72459656cd11c8610f3b08a5249cda8b32562d0dc7b04.jpg)

![images](../iclr_results/4551_Decoding_Dynamic_Visual_Experience_from_Calcium_Imaging_via_Cell-Pattern-Aware_Pretraining/images/eaee935006734408e3546a7f2a25b40294cffe240a6643debfaa78a6d72b710b.jpg)

### Tables

![tables](../iclr_results/4551_Decoding_Dynamic_Visual_Experience_from_Calcium_Imaging_via_Cell-Pattern-Aware_Pretraining/tables/13c539418306db1103cba01df38f99008f93a2b804872149ec9cc873e09392fb.jpg)

![tables](../iclr_results/4551_Decoding_Dynamic_Visual_Experience_from_Calcium_Imaging_via_Cell-Pattern-Aware_Pretraining/tables/2090179e2880dee088adb694a67c7cb59226ae1723ae1d9437b66f9d98142223.jpg)

![tables](../iclr_results/4551_Decoding_Dynamic_Visual_Experience_from_Calcium_Imaging_via_Cell-Pattern-Aware_Pretraining/tables/2413711252cbb492bba263d3cbf13501ca988dcb3d555b547c6b499ce8f04e1c.jpg)

![tables](../iclr_results/4551_Decoding_Dynamic_Visual_Experience_from_Calcium_Imaging_via_Cell-Pattern-Aware_Pretraining/tables/25c9d8be3760c67ef374920e35cb8c6b6dff2216875373fa74241ae919f70d1e.jpg)

![tables](../iclr_results/4551_Decoding_Dynamic_Visual_Experience_from_Calcium_Imaging_via_Cell-Pattern-Aware_Pretraining/tables/28a97292dfc0e81ca99204e616a2fd979466c08c848dcab956cddbe4cfd7ad06.jpg)

![tables](../iclr_results/4551_Decoding_Dynamic_Visual_Experience_from_Calcium_Imaging_via_Cell-Pattern-Aware_Pretraining/tables/8db281040addbee57dc24140a602528aaf5f6e94d0a4f634cdbdf3a2633df1d3.jpg)

<a id="paper-4552"></a>
## Critique-RL: Training Language Models For Critiquing Through Two-Stage Reinforcement Learning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=SsUjdSVdUl)

### Images

![images](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/images/0495bb0e3aa6df50f7ade6da255ab3a5a192ac2010336098087f3374e36df33a.jpg)

![images](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/images/3ea31d8dff7d48f88c89f6899706484f4fd00e4e755e9c12ec7653290ef84301.jpg)

![images](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/images/6520ebce26ab19828cd1ade46112dd79b854859959c2c6992f45d5b88ecf5775.jpg)

![images](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/images/7593b94da9da9bff7c37143716378b216a0d76c4c2803a2bf79026678eed75a8.jpg)

![images](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/images/a13b95c53910484eefb8b0eb2eb8d70ecd073011994d66aeeb600e46e7e6dfe8.jpg)

![images](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/images/a9c14366f8f604b2851e31c76c7e50c44a577c0b4642440005ef94f74b34c446.jpg)

![images](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/images/ac9c41e2d1d3b5084f55bf42916e7086ca8f1a1f84aa45c4c8693d5e6c3956ae.jpg)

![images](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/images/b514b7640fcda3af3319ae4389467c4a62033fefd0efd36e59620aa3154e99e0.jpg)

![images](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/images/be45cbaad6d3c7048adab6fcbe17dc169587d4ccf29624a922e84d77fc5d78bb.jpg)

### Tables

![tables](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/tables/14a257cef5d2f95cd6e8a045db89c77f045b88e9561c84bbe791deca7bd15b97.jpg)

![tables](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/tables/3e8e996d973024f3878ac0c9dce44397d5b0d42d382e501efeb1ff39ec128230.jpg)

![tables](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/tables/3ea7de0199d16a281435d88633dab0aea5847bed82b1ec2386a1f43cca548eb0.jpg)

![tables](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/tables/4e32b7e2b4e116b1244ccbbae6792bc2d9b2081ec03d954128115558c5aca4b6.jpg)

![tables](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/tables/52e4be2d4ce0b96a41de07e8214dce16e42e2fca451528ab5e57dd44d1f454f0.jpg)

![tables](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/tables/57403a7e536628dda2e4972811b9747d80b02e3a8ea170136446a44663563f96.jpg)

![tables](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/tables/5b962f7e02665b309bddf7c3c199a16608f68eb7edc65aba663351be8b09c789.jpg)

![tables](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/tables/5ef1c6603762a2d69855c1a44667262fdf41881bc4dff92f9e79b5d41059f823.jpg)

![tables](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/tables/868485b3cf1c893444f54c2eacdeebeb5f067e61f5e144dc20e4d851176932cc.jpg)

![tables](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/tables/955da1c7af45538458b4008a65982f7cc2f9ae300d71ed7f00f8cfa1ab2d2dbe.jpg)

![tables](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/tables/b04dc13fb991f427f576852a738f53af3f1f7fd609a84f83c98cfe537a7a3d09.jpg)

![tables](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/tables/befa1f774c29da6da1b68a191b3ef2a93e01cfa94c335a10f5d902862b0fbfb1.jpg)

![tables](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/tables/c04ae312b6334c45aa7390697a07d99d6f2499653f5453de626f65f539e02663.jpg)

![tables](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/tables/ec04e612d4688e7b0623e94678e8cb256cc5f84a3b515af50480e1ee433c357b.jpg)

![tables](../iclr_results/4552_Critique-RL_Training_Language_Models_For_Critiquing_Through_Two-Stage_Reinforcement_Learning/tables/fed7012f9e4b0047ea4c864bfc532790ce3e8136f38e4ba4de4a905922c0f238.jpg)

<a id="paper-4553"></a>
## Distilling to Hybrid Attention Models via KL-Guided Layer Selection

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=RzbsHcFqIf)

### Images

![images](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/images/2e87b85fb97b55944da218c3e59ff033f7f462b574652ae0e0a751d2c0106a8d.jpg)

![images](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/images/329edbd5b830cd4723c1f7bc50757c168fc37d6322f6c1c5d250c8b397605759.jpg)

![images](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/images/5cc517ab9d1b7936cd08d4a9f043262747c14bdcb26a349b59ccdfb0a87089ff.jpg)

![images](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/images/90419cd18c957bbd7c13cfe3e36a32a8b2f766e5de973e83f140eacc5798bfba.jpg)

![images](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/images/c0f4f69808f3b69d3c7b57f4eb2a93173c4a9af95d5488769f33ffdf13eccb47.jpg)

![images](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/images/c7deeec723dc45f83274c847a10a9c6264616769fdf9f882f1c6924c0dc0a277.jpg)

![images](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/images/d16c60f310e94d4bf32b91650c86efb766851324d75c55187051b983aaada3a4.jpg)

![images](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/images/e397c49e8ba80579a19be0ff03af2ddaf4acfe2a8a2b312be200afb2dc473101.jpg)

![images](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/images/ef997be45f1e3134825835b9dbadcfca5e9ba94417c28535aeb8cd5d3e208034.jpg)

### Tables

![tables](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/tables/0bace2be2c8e60b8de97820f0f736e1ee1b71dc8de72e875dab9172505277000.jpg)

![tables](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/tables/15405d837b7df02e0e5c378d0aa78a45609e02b75d6f89f7b8a5f7a40cc0d792.jpg)

![tables](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/tables/2539bb78cee347d5b4512766292be7e453dc97e33b2c1af0510cf725162dfc79.jpg)

![tables](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/tables/3157f9769355779cbbae180a86b622eb1aaa67162a2caa5afa8353307646dc76.jpg)

![tables](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/tables/5e38eab0bd6efbdc14c8e8b39e1ffcce3735872eff7c7b34e18c7331a3acb9cd.jpg)

![tables](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/tables/69027243a3f160a737ad5266bfc541bdcdb3dfefcd656d6a0efe1efb45e8b2f1.jpg)

![tables](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/tables/6df52ae1d38a5f84a7a44ff8fc857170a50e5e2217872f77b9be17476c2b11a8.jpg)

![tables](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/tables/7d25a0e3b1acd33a07ea0000f7545551bbc7f238d8019a1064605f1e605e8c23.jpg)

![tables](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/tables/827a15e005af8144b0e98daae32833634bb3a02561b1423d004966cd35d36457.jpg)

![tables](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/tables/8f9e75b6f382245677b64893816a4dbce5dbecf3e53a6344bc7b8c616e2f6bae.jpg)

![tables](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/tables/b330403933945c1b078500300d50a80c9c90e58e052b33608b7497010982239d.jpg)

![tables](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/tables/c468896f7abab610101322233dc15b67ab9ffa46803fd05b1ed8db8d9cfa7b76.jpg)

![tables](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/tables/d91601d950f20982bd4cfe769a65d52abd488784f05840e601cc4100d7fe5b27.jpg)

![tables](../iclr_results/4553_Distilling_to_Hybrid_Attention_Models_via_KL-Guided_Layer_Selection/tables/e8232e6503005a7bbba91bec0a5e4ddfd3da778cdd03a2cbe771ca952773f5f0.jpg)

<a id="paper-4554"></a>
## Helix: Evolutionary Reinforcement Learning for Open-Ended Scientific Problem Solving

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=2CHz6NYBmd)

### Images

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/01925597e9d60ed519a7ce8f1e702755a2c54b556105cbe11962336632cfb3ea.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/0e2846770943ba9a0dd7a952a3ebcff36dce8667bde2fe01486cd5a7b1b8d948.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/1285d29abd2117891b1b7c2e295851f1ab876d363c06e53b8568c7c672d2c6ec.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/14062c1820e9648f6cd5002e0e569192ac7510af97cdaf3ef6a0407379141424.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/1cea87d6d01f3a2112918c614ff81892b87e1bbf561f49fc982bab6e7c8911bb.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/334146b997d83e2b54d01e4440927c7b30f132559d75180b711c758997f31d84.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/4474152254f9a00fdd55bbdbab6174f320d19e2ac7ee76ff898e34e1644f9937.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/5a11079be7097ff8f6d7ba92d3e282dc961a99c8775394771126196893791582.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/5c72aade1d2bc62a8e8cf6a0ac149d9dc70b13c22d732e30d8e0d7fab561faf0.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/6c5337914752901da294b48397a7480269095cc713613097348e9be21dfa3b70.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/863e866366a2a739175ef75b4b7c522e707faf7c5881928d62c135e3f8784168.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/89e92cee90849d579d55774cc49dfeaacf479c1ca861840f3fe081833c80c433.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/9081c368645a627c2285fb1e81d2697870fe83049502de59ae618136ccc1bbd5.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/957a389a5c6b90431437b9535ad09f47c97a8c421c1a9ca2aac2f49ddfa37352.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/9b33214a56980a8a69e90c96ac8ed47fb69cc6f4d79e42370b7d5a0c42277b8d.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/ae0a4eb23f83d2e26bd6ea09eb36aefc97be5340d1558e857079fe2173e31ef5.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/d5fc99d6259c6e8d3bd0fe3456be89474c476b1d2e1d605bea8ccfdfeb1ece99.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/e19fff85e0b0a0eafa3226dbd202249eceafa047edd3afb16ca9bed34f11e10c.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/e1e98b0411cf3a1ee97c627e9d268803e0ab018009cd1ef0080e6fbaa40e78fa.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/e4dfa0ec0c25bdad83345e0bcf86512c8ecc6cfc95dd96dca44f555834dda27c.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/e518b930d6b79580158cacdd49baf89383849ebcc99fe5dead594deb61602d68.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/f06696e8cf6d0588f39eb5106f405070f71f36000656c0e8371515a815edf40a.jpg)

![images](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/images/f33b8014626b4a9433f9c02a64adc7823445bd4b673efedbc206ea1672eb341e.jpg)

### Tables

![tables](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/tables/206fee77b1f3ffe6e605c75faa38907d6165d3d37b401d2bb595b75a6ff660b5.jpg)

![tables](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/tables/4283fa3a6edf1b97fbee6efbaefcd5a887f0e42ff871a9a47ed5b08de1642c6f.jpg)

![tables](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/tables/b860534ec4173dd2ba2774999e0bf9d58303a408bd551c7614c87940c0e256e4.jpg)

![tables](../iclr_results/4554_Helix_Evolutionary_Reinforcement_Learning_for_Open-Ended_Scientific_Problem_Solving/tables/fa653918df0dc4bcbfc7a3eab42c8e9cb11a9fc5ab4bbfcc919ece53d90f2f05.jpg)

<a id="paper-4555"></a>
## SPRIG: Improving Large Language Model Performance by System Prompt Optimization

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=VdVV24KSWK)

### Images

![images](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/images/1822f4eea6d3bbbf7bde8f70ffcaa7e3f05d0592e222e7bec9be96f2e8d364a3.jpg)

![images](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/images/3b0fa38d8a2031e8aa6be91be0e6da18d8b0552412d290b29c15ee35922dea35.jpg)

![images](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/images/3d7b00a6931e4320edc4c566ecef881e39bf877e1a3959173c8d0eb7e9b0b7fe.jpg)

![images](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/images/4f8d426fd830efae528cdcad7ec8469b6bb42bf2d4923c68307f4965ea8e66ef.jpg)

![images](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/images/56e598d0a750e96fbf95764959cf0308802163e757ca8f0501cb54d5d96353dc.jpg)

![images](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/images/600c0dc704391aa0555edff9975896410d82095b198a4bed5afa2aee950e761c.jpg)

![images](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/images/667809cbbd39f823b4e1ccd4c05a7a1ce3aa120dc98bd8ec590d25294f9da10d.jpg)

![images](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/images/9d1f3ccfe7add758050e5b95c1e0f65a9d83b16ff12dfacdbfb120f4a297cf3d.jpg)

![images](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/images/a21acfc47a880c30f0636605e4ca62f20684d0d2b6706e02b057ac891939b8e2.jpg)

![images](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/images/b99a0ee50b0c7703051432f873793facae7312e8309cf135b52a05e43d91b202.jpg)

![images](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/images/f67d191acd4803ac4e8344fe3a6ef8920ecb1fd1cc3ed75fcc596c87f22df3f7.jpg)

![images](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/images/f9e0328a5d6585fc2c10ca4b506e6582f69b4d8c9b2633220abd4b2c87701d45.jpg)

![images](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/images/fb370086432535b263f267b954b8b526d4fb39705f1c2711837768792406a469.jpg)

### Tables

![tables](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/tables/66a8d2caabc13f76cd3df7a5976964a8f46da3614825494bd1e82f6bee95ca7b.jpg)

![tables](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/tables/9305555f078bdd89d959987a46e39b7032675ba347c488c88078f9977268c53c.jpg)

![tables](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/tables/9cf732e92d19762840e41f123a90ad56c8e8b886dddc03902cc5140eb38e769a.jpg)

![tables](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/tables/a0bd3daf6b5a7284cf9b137c9c0bac81129e8a209aa5793f63b347681eb284a3.jpg)

![tables](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/tables/baf99ef19573d3394b3807e6cdf945eded856503f98398dfc86477423b1bcbdc.jpg)

![tables](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/tables/bec93e0804aa2e65ed9731e8ff75cba9b99113420787820b923f7b6fc561e6ef.jpg)

![tables](../iclr_results/4555_SPRIG_Improving_Large_Language_Model_Performance_by_System_Prompt_Optimization/tables/dba21b5da404a3ce7b5b6034a5363a8d37d0bcf21db17ff21f0b45dc465c4d0f.jpg)

<a id="paper-4556"></a>
## Bridging Kolmogorov Complexity and Deep Learning: Asymptotically Optimal Description Length Objectives for Transformers

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=IwX6Gcvt0j)

### Images

![images](../iclr_results/4556_Bridging_Kolmogorov_Complexity_and_Deep_Learning_Asymptotically_Optimal_Description_Length_Objective/images/1730459e135fc429f7b51a116b15472898d37690e2e839793a6a9c1d609dd158.jpg)

![images](../iclr_results/4556_Bridging_Kolmogorov_Complexity_and_Deep_Learning_Asymptotically_Optimal_Description_Length_Objective/images/17f43fb48bbb19b9eef6139d5a26ccdb76d605d3b59abeef67de995f41b165a8.jpg)

![images](../iclr_results/4556_Bridging_Kolmogorov_Complexity_and_Deep_Learning_Asymptotically_Optimal_Description_Length_Objective/images/1d0522b9e6c03d862db94b18be43dd2033d9b4ad2c4948f229e10a3c874e1f47.jpg)

![images](../iclr_results/4556_Bridging_Kolmogorov_Complexity_and_Deep_Learning_Asymptotically_Optimal_Description_Length_Objective/images/3cd2d441f4cb1436a58bd1efe6fa905d6baf2e706a54bb088626f2bc80fc4eb0.jpg)

![images](../iclr_results/4556_Bridging_Kolmogorov_Complexity_and_Deep_Learning_Asymptotically_Optimal_Description_Length_Objective/images/41ef4bdc69d74e84d48f64c06d7bbc44841735e457af69852276f94adb6631dc.jpg)

![images](../iclr_results/4556_Bridging_Kolmogorov_Complexity_and_Deep_Learning_Asymptotically_Optimal_Description_Length_Objective/images/509d4c4619f38b8de4ca3bf09bc8b04edfb14380ec6a582d649659a79228d787.jpg)

![images](../iclr_results/4556_Bridging_Kolmogorov_Complexity_and_Deep_Learning_Asymptotically_Optimal_Description_Length_Objective/images/89ea8d595f8848f7b96b2409a7822162d95675cd0a78ccdbd1eef6ab44ee1e8a.jpg)

![images](../iclr_results/4556_Bridging_Kolmogorov_Complexity_and_Deep_Learning_Asymptotically_Optimal_Description_Length_Objective/images/c35f59d1fbcd041db3b9c65011f2775648d65283741e5beb56bd04ea4958f360.jpg)

![images](../iclr_results/4556_Bridging_Kolmogorov_Complexity_and_Deep_Learning_Asymptotically_Optimal_Description_Length_Objective/images/d79153b8385a2a53823e320f702658c693bd9063e9a41171c5d981a68b4bfe6f.jpg)

![images](../iclr_results/4556_Bridging_Kolmogorov_Complexity_and_Deep_Learning_Asymptotically_Optimal_Description_Length_Objective/images/de55487cd9aa57ce413fb9e35a1e26627d3db5cc88e4ecf5dc1aad6fe4ab22cb.jpg)

### Tables

![tables](../iclr_results/4556_Bridging_Kolmogorov_Complexity_and_Deep_Learning_Asymptotically_Optimal_Description_Length_Objective/tables/14526155a6702a839342a8fbdbcedc82a917025c166e799236ca0c41c64353a0.jpg)

![tables](../iclr_results/4556_Bridging_Kolmogorov_Complexity_and_Deep_Learning_Asymptotically_Optimal_Description_Length_Objective/tables/6be71321c2a5b0d0bab751bdb201985890fb3641b2445c00ccdfc253fbc132cc.jpg)

![tables](../iclr_results/4556_Bridging_Kolmogorov_Complexity_and_Deep_Learning_Asymptotically_Optimal_Description_Length_Objective/tables/b8467d21847a788d70996bb035f37ad0289abadf1d2df314dcaa018c25585fae.jpg)

![tables](../iclr_results/4556_Bridging_Kolmogorov_Complexity_and_Deep_Learning_Asymptotically_Optimal_Description_Length_Objective/tables/db5961b2a8b9a9841de2b87fd0760b8f3d7ae2bbf74c6bf8223da7709e756ad8.jpg)

<a id="paper-4557"></a>
## PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=qetBM8nLkf)

### Images

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/023e2b810238a0c3580ae68c706c931c3a7b79568f7a64d510ad6cf3048a36e5.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/157be3f8b6a7f1d7d855ca8e5fb3ffe120fbcf23123ebd48db8176520e6e823f.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/1741619cd8c85f465317583caae425a8c5101a69fe3ed3f3de8f8e3ac7d705cf.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/228c29555ba0ac073b1fceb1536d2c6b96e3e42faaef6f42015913cf988793ce.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/3bb606b598924cc2662f0292c4fe074ab5136f747f3dd6d5840b3c253dc701d4.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/43adf6942a18a335973a788bfa0a92a9c8e0a7446a21f518a577b5ab13f712f1.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/59021d2bc82b28bcd5863318147f39afa2ed434e84a002c780b61a8df6227c41.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/5c147d86c21f472a9e821c2e91df971982c596febeb7b7ad30a2ae86556878d5.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/5e602019556ef5ded7ed246b48c621cc29c0c223a1b931d73b9680b03f6f80ee.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/63840fe95d90d59af49921b68dc1497b4fda9297a0d8573d5b63c8ac2bce30a7.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/6b31eb254736577649d88701ff1cc143ff1ec80436e06fea726a1b57b9674aaf.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/70cee4528f907de809a6190eb0927cf26786b21089e07a9ec4a841e7831bde49.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/71bc8d54e0891a39c293ea9d18d7cc2ac71ef922b139f00a7845e7ec619e64d8.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/7561ff03ac0665e4283bcdf4ee6a2850fa4c65d05a1add10a7df2ac50a35cecf.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/800d0432ac4fb08be36a0b2311ec18c67b5c0fde20c6b3abccd7e5eccba4777b.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/830136c37ea71cc7c76bb165079d0f79a180ad5a194c006bfbdf04c1caec5b18.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/84ec71256217b98f924e8d8aac0ab110a5d36142a89e4d3a63ee461a4414e52d.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/8d6bea30834b777791ba2ac894d2ba5a7598778ab85c870af52ea63be302b030.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/8df6057d522a0a54050286aa533e99c2b1972041d9e0ff28f92b79ecba712b62.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/8e0145b4c11e8c7f6b55cd5637890d93f2d4bccec3dbdac34b4bdc10639f4eed.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/969a79d5d07326ae82176d1425085e8c7f96cebf8bc6d8a734822014c1c7f4f0.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/9ad5023b41ec82389915db40f82ec0cc82029fb9dc2b6d9527c20b574b744a70.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/ab9575321b7722d563ae63e8bfe431eeff8814eaf8beab1ad458b4d48a5914f9.jpg)

![images](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/images/ea00e5cbafa91a248c608df3d6e19b1f4eea1d2bbb0ee050802ded3a30617483.jpg)

### Tables

![tables](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/tables/0d3d6f8db4056379d1b08098f59bc2978500904f4819780989b312f12f0422aa.jpg)

![tables](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/tables/0e57f2b098b1c5a79c46f5f487e19ba294bff635a88bd54cc73c14aa78d60905.jpg)

![tables](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/tables/1c93ada2fbc3f0dc16c14b96cd88260b2134623c08b613bdd19531f14797b714.jpg)

![tables](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/tables/2072f0a1c543d22b9fd0777da9aa7860efb85b12be4cfcc29dd4ff1946cc1196.jpg)

![tables](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/tables/2cc613cf5bad2a8ac9e68af85e9863658624285888532a880d4fe872c3e5368c.jpg)

![tables](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/tables/364559d7e3dbe1315579d0d1ccb877f385dc89db927457756017c27bb59aa4a0.jpg)

![tables](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/tables/3c8a9704b3242e941220b2d91e8faf4d6558925af35c11e3ecd41e2eb6b36792.jpg)

![tables](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/tables/55efa521a7abefa6e933b8c783f77d81f54dcd5beac90f2cb74839bce4733ebd.jpg)

![tables](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/tables/7c8ec7bac9d52359000523952e05758673b2c7e3b95354b27d7e08a0abe23a25.jpg)

![tables](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/tables/7f9f595c1f72791ed847d10f8597fb62e034ea7afdf41ae68f08c03daa543807.jpg)

![tables](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/tables/8067c62c5626c4456144c4774b36ee56048c3ef629cd0360b2ff4fb884e4c66e.jpg)

![tables](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/tables/934c070c1defa0f7f6092a4b9e04ce2e7267e43f3fc9ac3f9a960673559fa1f3.jpg)

![tables](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/tables/faed70ad2735598961de7fb0de845a3ed35fde6b134e68dbb646b8d68558a023.jpg)

![tables](../iclr_results/4557_PYRREGULAR_A_Unified_Framework_for_Irregular_Time_Series_with_Classification_Benchmarks/tables/fde94434c0293911eecaa035d7cc5d24883f50b424ed5c723e12ddf2ff891b06.jpg)

<a id="paper-4558"></a>
## GraphPlanner: Graph Memory-Augmented Agentic Routing for Multi-Agent LLMs

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=ZdGB7MNQDT)

### Images

![images](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/images/0ea295591cd17a7f6fa9379b0149f529b5b999e15ceb094a13332137fea1411e.jpg)

![images](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/images/1d68b7bdbfa9e9cc256b141cb2ba7486de8a6139f7ec22b335c4dfc4c91333fe.jpg)

![images](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/images/6cf1ee3e4bb5f6b1c598a4c4ce90cb263ed845b516ef17dd49758686f1176935.jpg)

![images](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/images/c61404d82c80d96c49b7d0e379c5dad88a739aa4b6fcda04e6a3a1714f664fa9.jpg)

![images](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/images/d2a46a05a4fdf735338e71f86bb1c2003a022858c23da0f1e15c0f0f3d294244.jpg)

![images](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/images/ec98c9671ca7892bc9efa725c2f96fa762923ea754b962e46913dad127cfea7c.jpg)

### Tables

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/0202f45804b9d5c0078f65fab0a4e297b3144b60cf46dd84863ec7b4e8f9f74c.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/0e136a91d99b2eb9725e7a8c397f3a065705664f3c21a442ecebbdf03889d2ff.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/24109edefbfb136b74849ec9d62be5f0d70b88c53d998df74de97f1b9306fbaf.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/29711025547e2d9a52f74a2c3d27715a278360b7f3d36c8027692de30866bfa8.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/2f91016a35e6a77d0fd6eab7c1865ab69b330a80c4d2b34aef5220b69be0f075.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/34c8519eaf6be2176709545812692e8d4639f11547d2248b7e2091f72a75f1f8.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/383970caadc6e9069baf137f6170a16a95ae8b0ccddf9626d2a4892a002db764.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/3e64a687121101b081e92b953c5a29edb773432000031835ee25a6494e6a2cd8.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/3fac39cda86b75a1be76bc13d681ef8e1c5f57ae1355fa2e7f1239ed5d59703e.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/4034b75db244d49b8a6b4e26e70e26b71d3244c91fbe6adffc68c27b329493c1.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/421f9377caf74cf1fe84f57c2ca490b38edcaa51556c17aff003a3ed199a67af.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/54887883430280cd4c4bf85233cde612eab881f34276465357532d242f610f0a.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/60529a10720293a2cc2cef534601ecb5f973043a5d1ac2a883ad88e94379c192.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/7258990b395b91c2502aa3b33280f2ff5485bdb8c988947aed2052aec82f7cbb.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/7ac89ee6c6ff282fdd4d6e72c9a124bb0ceacbb73b610a785048dbd1e4ae546e.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/80470d9bee537b7cb357b314539b4e9fb127b8dea3997baecfc21985388ba150.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/828a0f8604915ec379afcb31947d3923d17bc1b28c3182296adb5be4da8c3506.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/9d96a1c0e1dc05882d435ea01fa97abed3518350340846c84f466e5b60a40f89.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/a0f6f26b6f6371399bc4c2446ce23493670ecc62be202b23d815466f8ae7852e.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/b0b0551ab179127004f5b165487cb864f1f4420540954800311fb47d34d820e7.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/bb37bb0d9aac9b58b197f23f40c73ec853fb5792b973d7a1e913cf2dbd009c6d.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/d8884a849f548fd340e2c71dc0b4c333c1640badeae03e62b92654d7f154bd70.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/f6ed87ecaa57d6453608f4158f31327d37452e74cac8160972c532bff16b0488.jpg)

![tables](../iclr_results/4558_GraphPlanner_Graph_Memory-Augmented_Agentic_Routing_for_Multi-Agent_LLMs/tables/ff26b8cedbd754b457126faae0672a2eb9b0334f6c2a34eb6179390f88796ef8.jpg)

<a id="paper-4559"></a>
## One Patch Doesn’t Fit All: Adaptive Patching for Native-Resolution Multimodal Large Language Models

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=six75YUGgS)

### Images

![images](../iclr_results/4559_One_Patch_Doesn_t_Fit_All_Adaptive_Patching_for_Native-Resolution_Multimodal_Large_Language_Models/images/2bf9975ac678f6f36fa61901e87130aed4cf54ca111e26e214a7ce38d12d69b5.jpg)

![images](../iclr_results/4559_One_Patch_Doesn_t_Fit_All_Adaptive_Patching_for_Native-Resolution_Multimodal_Large_Language_Models/images/30b3ae5a5b6fa4bfe2f7b674d1fd649207b0c0a32308f4168bc27e28d1854597.jpg)

![images](../iclr_results/4559_One_Patch_Doesn_t_Fit_All_Adaptive_Patching_for_Native-Resolution_Multimodal_Large_Language_Models/images/395900bf29c74d1fd264219d8967f2ec2d0094f2d0229ffff8b146d738d6a620.jpg)

![images](../iclr_results/4559_One_Patch_Doesn_t_Fit_All_Adaptive_Patching_for_Native-Resolution_Multimodal_Large_Language_Models/images/3f85231d26eba07a091649d9930ec6d05c134def04542b597267438b532b607e.jpg)

![images](../iclr_results/4559_One_Patch_Doesn_t_Fit_All_Adaptive_Patching_for_Native-Resolution_Multimodal_Large_Language_Models/images/6fd28dddfc3d9b4291e135d1fccefefbc74f47f729f17be61780210558c6a5aa.jpg)

![images](../iclr_results/4559_One_Patch_Doesn_t_Fit_All_Adaptive_Patching_for_Native-Resolution_Multimodal_Large_Language_Models/images/82096054e4323ef34e126cb9044cf38b17aa784771ac9e6d98d6ac69957ef692.jpg)

![images](../iclr_results/4559_One_Patch_Doesn_t_Fit_All_Adaptive_Patching_for_Native-Resolution_Multimodal_Large_Language_Models/images/8b490efd81dd1d4fcf021cc1b400b9479eed6dcdff0624c986440f87895daf5f.jpg)

![images](../iclr_results/4559_One_Patch_Doesn_t_Fit_All_Adaptive_Patching_for_Native-Resolution_Multimodal_Large_Language_Models/images/de538ff8583d1f579e10d76f8a51a984deb4c4bdba6e14bf9e08a23053c9092e.jpg)

![images](../iclr_results/4559_One_Patch_Doesn_t_Fit_All_Adaptive_Patching_for_Native-Resolution_Multimodal_Large_Language_Models/images/e1b2f5cdd30619b0d88562b57c9c6f3ec75f3fc4ea8a345e74f3a8228f5b84bf.jpg)

### Tables

![tables](../iclr_results/4559_One_Patch_Doesn_t_Fit_All_Adaptive_Patching_for_Native-Resolution_Multimodal_Large_Language_Models/tables/10068eb9be7dfe092e829a53647317cdba7f52c806341b3c9b9d1c07fcda490b.jpg)

![tables](../iclr_results/4559_One_Patch_Doesn_t_Fit_All_Adaptive_Patching_for_Native-Resolution_Multimodal_Large_Language_Models/tables/775478038a1161670370e95edac4da9c3b400e323db42ef1340ef46c90fe9ef3.jpg)

![tables](../iclr_results/4559_One_Patch_Doesn_t_Fit_All_Adaptive_Patching_for_Native-Resolution_Multimodal_Large_Language_Models/tables/85e4a7ef2accd071321bf60296b02fdd5e304907926cc9ec1aa37681ea0e1079.jpg)

![tables](../iclr_results/4559_One_Patch_Doesn_t_Fit_All_Adaptive_Patching_for_Native-Resolution_Multimodal_Large_Language_Models/tables/da111fdfa9a9d643df8d6f6bbd86d96f23c5f332e8505b86f2ea72b6f726d3a8.jpg)

<a id="paper-4560"></a>
## Keep the Best, Forget the Rest: Reliable Alignment with Order-Aware Preference Optimization

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=LrHfYPFTtg)

### Images

![images](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/images/05d61dcdc129ef398ad7403d4b2e238b8d1b057e031c66e77b845fed15487c5f.jpg)

![images](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/images/254498ce7aa51e06725da470a3067bedf29746e6d1065a5cfdec4ec9742716e1.jpg)

![images](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/images/6a8c69ada3794a07e93ba3eb77db3093d0c4892b05b44853b2b2592cfcda84e0.jpg)

![images](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/images/d88f9e1592abb7ca12402e0dfd253356016526fa5850102b4983a242ac87cd2e.jpg)

![images](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/images/df541b0e0e6bf02f9b98a8a2217190f9ef17d4cd93537025cc86d86920b97cd7.jpg)

### Tables

![tables](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/tables/083b1796048ed369a601648725b790b0e6cacd6b8bb7772b0848c4498922fa47.jpg)

![tables](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/tables/1a0adedbab67126537a6dad04f2cf0eed4ed1d56afcb76f8a17bad5fbf272a07.jpg)

![tables](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/tables/1a54b44901d8f6de67e97e2668542964854d3c7f1ee237a58023a34b75039744.jpg)

![tables](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/tables/223ee52e4088ed4b63b585d0736dec57ec8ef8bcc879cf1d64b14d3b195b646c.jpg)

![tables](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/tables/2537316624590ec46d0f82e4d1b0f9c012f70c4989328d640c33014f5027e2b7.jpg)

![tables](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/tables/2c8507da70a16bc4f14b1dcbfb7469764f74fa69072f05857ccf2b77c36e143b.jpg)

![tables](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/tables/339f13982665a0648b2ed92ef1666be550cb8eb81306cef2ada04526f73b63c9.jpg)

![tables](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/tables/40f5f61259b0aaa7ad79277da5357b2a7fbba71b3412659ce44b287f7862a21c.jpg)

![tables](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/tables/6d220da4f08fe91b4959f40a18acb88b615d68b49f745a60be38a23bc2641d61.jpg)

![tables](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/tables/80123f20f858216cdf3c1e67304cfc4318a4c500e8790175c1ad71ad171aca82.jpg)

![tables](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/tables/9adecce845116779ff87c910bba340bacdda1aa9f9cff9735e35c790d0aef526.jpg)

![tables](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/tables/f4e6c28a5ad719b417a9a8d0473de729b2aff33c1d673b28b38f27039f9e3b16.jpg)

![tables](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/tables/f53314ce7afdd293954448c4095fda4966c4fd81d2a34b8ddb2fbf2568d9f460.jpg)

![tables](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/tables/f86339f98b24f5a28bf7d33f9077c253d85be3d9a6bd9ed485114b41fabaa0e8.jpg)

![tables](../iclr_results/4560_Keep_the_Best_Forget_the_Rest_Reliable_Alignment_with_Order-Aware_Preference_Optimization/tables/fd7ffbaf5470560caf241f4e91c69d1df834f9df6f29a77ba98529ae5d9c31f1.jpg)

<a id="paper-4561"></a>
## Influence Dynamics and Stagewise Data Attribution

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=8epkNiuAQC)

### Images

![images](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/images/2003c01b46716f000faae8887392c48abf2c5f2157d761437aba793f1b445355.jpg)

![images](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/images/4897b9fb00a781da6f124eb0fbb579ce4d54a723143c31a9d43f8408720394e0.jpg)

![images](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/images/6cb1775571f593c72c038f194da4a19063ac35cf936bf8f193b871fd2df9b31b.jpg)

![images](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/images/6fdc3a5a4810b0d2b401e975c9dbb0ee57160f3526d40a339f0668271cc298cf.jpg)

![images](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/images/837a1231166a52f3519ed805d56804766a1eb991409a79a72c30861b14d53334.jpg)

![images](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/images/ad1a1410fb11cb78b34f38a2525bd7cf9663d88b2ec5251881700f7f8abd8264.jpg)

![images](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/images/bb325751599683e415556f0547f4927672c0907a758e20ff205d1ad43010c6f3.jpg)

![images](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/images/d54c0f7e7abe5917bd44f892ccc9239f7f3176e811cbadca3cdea7882d703e00.jpg)

![images](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/images/d8f0c5e9b8c38c861f6cc02b86234e1178753daec3a6658b58f66b67c276ccc1.jpg)

![images](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/images/da9a0b84e5d1eec3adf04c8e596dcfb5415888f8abee351161be3f98a3664838.jpg)

![images](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/images/de1dd498a85399ad3ac30c364bb83424cf04b5daab10cbd29c3bd019dad3d070.jpg)

![images](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/images/e2e28228f0ea30ef17086aeb61757b8d3b2e46d3edb1ed7809821f402c07cb4c.jpg)

![images](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/images/e38438ce5ecee105e935191c6b9f29fc65113bb5508e7bbdbdf7909d99276beb.jpg)

![images](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/images/ebfebf7f35276035b2fa6529f8c85bd8fdd6909b2a0e01f3139e369a612f08d1.jpg)

![images](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/images/ef78fc548a816ec633685bde80dd1b6abace3b845939e76d8e376309f74c9e56.jpg)

![images](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/images/f3170289eb42f40c37ccb287e1331b05290f037db2f4a5719d334a2f6ada08f7.jpg)

![images](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/images/fdc8421b760f321bda5c8e500f0a33fda34a99ee59c2279de8ea2e498d059d42.jpg)

### Tables

![tables](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/tables/33116e624ee0e581388fc3eee0f0dc557f24bbddc6bbb836134e26b1f769bf18.jpg)

![tables](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/tables/bf9e1ce101783c2bc290dbb443b678d06d35d90c377477cf3c5db6f63ac3f559.jpg)

![tables](../iclr_results/4561_Influence_Dynamics_and_Stagewise_Data_Attribution/tables/f9d67ac510ed058dd7a9809e3dc5e362164c8620e7df64a0102633371fea5e29.jpg)

<a id="paper-4562"></a>
## Output Supervision Can Obfuscate the Chain of Thought

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=JTX0iqxBjk)

### Images

![images](../iclr_results/4562_Output_Supervision_Can_Obfuscate_the_Chain_of_Thought/images/3220bd48d53c753f9c3bb07ede70ab3aceeb608b1d98160d65d21aa422a98101.jpg)

![images](../iclr_results/4562_Output_Supervision_Can_Obfuscate_the_Chain_of_Thought/images/4116a92758d30bead0f1ab7c3ce77d6523a8dbf8239a506d9c983e2ef38e573d.jpg)

![images](../iclr_results/4562_Output_Supervision_Can_Obfuscate_the_Chain_of_Thought/images/5216c61365eef5170344ebed8600105973b6d892bc7804bf83e8a0b0b6fd48e1.jpg)

![images](../iclr_results/4562_Output_Supervision_Can_Obfuscate_the_Chain_of_Thought/images/6235a63161d068285d02bc8dbb26658e45cf32e052f43c457d98449041b19d8a.jpg)

![images](../iclr_results/4562_Output_Supervision_Can_Obfuscate_the_Chain_of_Thought/images/6f1ae93d13f5a8d89f73b9987767e9bd9038627204ebec9b43a4aa86dd57ce9d.jpg)

![images](../iclr_results/4562_Output_Supervision_Can_Obfuscate_the_Chain_of_Thought/images/918e39dae3a8a51884e8224454c6d335a8499127bd101dca576292f2ff7c5fb5.jpg)

![images](../iclr_results/4562_Output_Supervision_Can_Obfuscate_the_Chain_of_Thought/images/923d0af46e409b35dbe35a305bcfbedd9da4c63a952d38bb799c45a9cb1eae0f.jpg)

![images](../iclr_results/4562_Output_Supervision_Can_Obfuscate_the_Chain_of_Thought/images/b55d6778d32a4ded387b5af2e30e7c9c8acd3cb365435fb689c877175e69983e.jpg)

![images](../iclr_results/4562_Output_Supervision_Can_Obfuscate_the_Chain_of_Thought/images/ce84a56e03b23e748b16432fc335dc30e11e5bbd885c6008ebf59a6bfa00e542.jpg)

![images](../iclr_results/4562_Output_Supervision_Can_Obfuscate_the_Chain_of_Thought/images/f385fc4b98918428e78feb11734729ee83f5a89ae7c5e78ef30cb81f6aa4a726.jpg)

![images](../iclr_results/4562_Output_Supervision_Can_Obfuscate_the_Chain_of_Thought/images/fadedf5f66320b27588450215ba38facb93d4b18656347582fe19a61274a6ed3.jpg)

### Tables

![tables](../iclr_results/4562_Output_Supervision_Can_Obfuscate_the_Chain_of_Thought/tables/74c2b436fa62cb3218a577bc75f2819f3f96976a2cecbe27bff50c7967dd9e7d.jpg)

![tables](../iclr_results/4562_Output_Supervision_Can_Obfuscate_the_Chain_of_Thought/tables/98cf0d48095eb3030fad62dd9f25cbd7c62f329942b4112f147388b167c82c5e.jpg)

![tables](../iclr_results/4562_Output_Supervision_Can_Obfuscate_the_Chain_of_Thought/tables/b3f52653ba6cd9f610a972a2692fe89c3492253c5e866a557f895aba3ce4cfc5.jpg)

![tables](../iclr_results/4562_Output_Supervision_Can_Obfuscate_the_Chain_of_Thought/tables/be9c0fff1c58a5d86d7b8b1df0625482898917d126b96f45c8357ed7cde1e97f.jpg)

![tables](../iclr_results/4562_Output_Supervision_Can_Obfuscate_the_Chain_of_Thought/tables/e3705035100d7bf957868181e13524431c7a1aa12fc34f5bbb392a33b970407a.jpg)

![tables](../iclr_results/4562_Output_Supervision_Can_Obfuscate_the_Chain_of_Thought/tables/edf30744e16a1f756bc91e233a93b2868a5785ec697d4b7100ece6ad9a1f3135.jpg)

<a id="paper-4563"></a>
## Laplacian Kernelized Bandit

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=5majEbTAZa)

### Images

![images](../iclr_results/4563_Laplacian_Kernelized_Bandit/images/23a387d94d299dc8ec6afdc6f7c46be9388e21eee152630d575ffcccdc8efe24.jpg)

![images](../iclr_results/4563_Laplacian_Kernelized_Bandit/images/6a00d9fed59ae26ebf04df841e8312b7c373d2f974a95b7b48f2e26eb83714d5.jpg)

![images](../iclr_results/4563_Laplacian_Kernelized_Bandit/images/6d41fd402eef6fd3231af30e11fdc790edd109c473f73f3bfc9861466b473c5a.jpg)

![images](../iclr_results/4563_Laplacian_Kernelized_Bandit/images/9713bfcbc7ef416b894f9f829cd6ebcd4c4e54dd07b0b5ea6c28255a50fb3774.jpg)

![images](../iclr_results/4563_Laplacian_Kernelized_Bandit/images/a12402d8f5292a116d86b65b00c0dca59165e97d1d663767334948256cf162f6.jpg)

![images](../iclr_results/4563_Laplacian_Kernelized_Bandit/images/a532856dd56c3110ef1245bb5fd6177031391efdc73fcbdf9096772599f8f102.jpg)

### Tables

![tables](../iclr_results/4563_Laplacian_Kernelized_Bandit/tables/38945941dedde585d2e07bb2b6baf09f0255eea12a6b4094afb7f1e5c10c2c7b.jpg)

![tables](../iclr_results/4563_Laplacian_Kernelized_Bandit/tables/3d094964913d8b0b7b9d7d062debb1bf1b7cd5bdc5d98b44a9f83bfebfe3da3e.jpg)

<a id="paper-4564"></a>
## Closed-form $\ell_r$ norm scaling with data for overparameterized linear regression and diagonal linear networks under $\ell_p$ bias

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=qPKTDOJ5Xs)

### Images

![images](../iclr_results/4564_Closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression_and_diagonal_linear/images/1b5d8389acb9c260b6a1852053df728ae36ddc914ce05eef5f952aab41271b2b.jpg)

![images](../iclr_results/4564_Closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression_and_diagonal_linear/images/23119294a87a2c049086bd67698e60cf52ce5e2336058c173d86628f2f5abbba.jpg)

![images](../iclr_results/4564_Closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression_and_diagonal_linear/images/483f5acc9fb873b542a484d3bd85f6d9017bf1a3b2041faf7c07f064fd7dd417.jpg)

![images](../iclr_results/4564_Closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression_and_diagonal_linear/images/4c944dc563b714cb25186d0197e30e3fd1ae94dce952a2028cec446fb2b464e7.jpg)

![images](../iclr_results/4564_Closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression_and_diagonal_linear/images/52d073cb3a6a692217a6b0b2fe2f064e8d3d16d18e0b3e52bea7e9ebf23f9745.jpg)

![images](../iclr_results/4564_Closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression_and_diagonal_linear/images/5767043eeec3dbe12b88b8ec22a71e659fa4ae69d5dd668742bde0df3e272c3b.jpg)

![images](../iclr_results/4564_Closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression_and_diagonal_linear/images/5ad6a37b286247c2fd54943f30c10cb97218578163826e77949d28a36dd0a0f9.jpg)

![images](../iclr_results/4564_Closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression_and_diagonal_linear/images/6ca609f3501084d3d5ef119413da503b9edabf0e1843fe0b6a01a2645d1204d0.jpg)

![images](../iclr_results/4564_Closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression_and_diagonal_linear/images/949b76a16f35fa3f2052dd746f48f2262653e7b21ec8bb91b740e6b3fe84cb1a.jpg)

![images](../iclr_results/4564_Closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression_and_diagonal_linear/images/968c5af7a21911f441a4506696f7cd6b9018564c5ef1742cdf5a3d98af3fef65.jpg)

![images](../iclr_results/4564_Closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression_and_diagonal_linear/images/9a0a5704fe8e0165e29b5d88a7b7c8ae988d7b4095c1396b81400a47dcb9178e.jpg)

![images](../iclr_results/4564_Closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression_and_diagonal_linear/images/b12d824075a9251d8d4c6416e70df0bcab97713f0f1ae40fe627a023804b558f.jpg)

![images](../iclr_results/4564_Closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression_and_diagonal_linear/images/b9c007c916a91fb5d1298a840d1940758156813dfc02a2fd008731aaa2da035d.jpg)

![images](../iclr_results/4564_Closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression_and_diagonal_linear/images/bae08dfd1bead45a035bd118af1f392fece42f831219c1d95d955228eac44283.jpg)

![images](../iclr_results/4564_Closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression_and_diagonal_linear/images/bfb91357968cdd85efcf74d072f3b4f4e35580d91131734130d9f42b80669586.jpg)

![images](../iclr_results/4564_Closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression_and_diagonal_linear/images/c046378cf375a14a4c337ae1ed36d0b0782e92a1f6b81b389ad29678294cbf64.jpg)

![images](../iclr_results/4564_Closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression_and_diagonal_linear/images/c6a5a4eb63c7fd4641d9378cfccd01750517d75f42b3f228464140b52e435706.jpg)

![images](../iclr_results/4564_Closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression_and_diagonal_linear/images/f6d4806479e87f112dcb3c163cf6ffd17c71bc7b138537b9b3e71425677977c0.jpg)

<a id="paper-4565"></a>
## Disentangled representation learning through unsupervised symmetry group discovery

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=I6xjMoLY3j)

### Images

![images](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/images/014df1914b722d681e8e409ab2c37f19e5b037e756719ddf17f876b818f58d27.jpg)

![images](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/images/018436e1feeaf975fb0cb91b85cadc58c484b24b91fc7994253637bebb143e3c.jpg)

![images](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/images/245447645303a1a4b323d2fed766ecb3815e4f2cfebfa5092cab0bdc1a7e8661.jpg)

![images](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/images/4b7107d0f38df2c5f9dea0917404918d4adb98b702968f2c4b816cbe787727a2.jpg)

![images](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/images/50e55314e7fe9cf4c6ccde6c9e213d0bd7ff57dd56f123f4263922b3986d6d51.jpg)

![images](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/images/5b6af6c2357eabab64ed95197d0832d84c078eb5d96dee3e6e3ccd72a469f10b.jpg)

![images](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/images/7abf41817791f08d4b2206e7847724f2cc5bb13b58b697a54ffd99c59f18570a.jpg)

![images](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/images/8d8e171bcbb8086dadfcaa0b8b47ca67d856d84a31e40f07570443a83fc0780b.jpg)

![images](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/images/9d0bf1930bfdc54c1ae0321dd3794cfd29f162f57f95c79620f92fe7a4394b93.jpg)

![images](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/images/a911f15132333b5b9825282bdd63fc56e213a477f32e2958b18d11caa6fa747a.jpg)

![images](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/images/c359116c31a8493e47b7873dcf667fa5ae003bd4676da509668a30f719fcb635.jpg)

![images](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/images/cbfeecd62a3ef076ee9927cd793ea4c4c3ae5d6d3fb5b74577b7f297479ce5d2.jpg)

![images](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/images/e4b8401b223ed8546fe319884eea0ceb0558d2b2ca33b039c96dc1559937417f.jpg)

![images](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/images/e7edd9587f8892b553ae54bb215bf91d0b18149339cde0c2e791d86a73d349c7.jpg)

![images](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/images/f779f18d2ef17008d39ef549761b2511b38f310fc3babddc6d4ebcdf2ca7da2f.jpg)

![images](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/images/fae53d85e62c49491286337ea2fa7d0f44e2ef8d724600246dd05048bf0b8300.jpg)

### Tables

![tables](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/tables/11af6c9e016f1df2b23c31d927c6c46e2fea128f5fcaa4baab8f9ef3c47d4ddc.jpg)

![tables](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/tables/293d4f4e012e5bff3ce1ac7bc78821b8ae50d04641533a5736c80a66625e510f.jpg)

![tables](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/tables/356d07115692056f2ec509d39372d8a4c9b6d8edbfecad2eabc3e7357f2dd6c7.jpg)

![tables](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/tables/6e2a694df73851a2edfb32a3e576d417d05146bd7647b3cb3ebad3efe86996b5.jpg)

![tables](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/tables/713d350de831457760a40995b108a477cd79a1e517e2453b094f436780145bd1.jpg)

![tables](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/tables/86882bedd413fcbb1cc3e4be3101602278ba6bc5796c7ae592ed7e2919723b27.jpg)

![tables](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/tables/871aeb59cafadecd4b80f48e2d39ae3671026b5bdec929547063928ba89dfba4.jpg)

![tables](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/tables/9c98237cbf88c82adde764ce78d75139bd104855407b6b8868fc5ff1fd2a3924.jpg)

![tables](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/tables/b1e1856e3693bf04274f1711c4ea9b4827deedd58628c3f5e7ff3b83f6831b1d.jpg)

![tables](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/tables/b47984a622cce65546f7eccb879e390042b6a95611e81aed23e9f2dc5fb6c376.jpg)

![tables](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/tables/bbe7479274d99bf376a93e18a7578087d083551238ecd49be3e568b4e76394da.jpg)

![tables](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/tables/c1b05cac2f697332a19941e2028408398ff4a8e17b5ee39f9509b2f96d6e25a4.jpg)

![tables](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/tables/c495b4e5056f6923d06ac85c5a73dde680f6ec8ad252af5a214f70cbf3f3085c.jpg)

![tables](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/tables/cc5d50b95627d5202fdb6613db69858f878a3d44efca9d8a0b6ffb40e655d448.jpg)

![tables](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/tables/e5f45716ac34a10e08279c3904b73b21df13a7adadfafb672ffe0fc5b6c2d2b4.jpg)

![tables](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/tables/f284d378394de3de778958f865723f89e1938d0b6268030e0351fec2f9e00475.jpg)

![tables](../iclr_results/4565_Disentangled_representation_learning_through_unsupervised_symmetry_group_discovery/tables/ff3e2946b0eae285b5506b641ff178b0a8c419cb7e8f23674a575a9cd055de02.jpg)

<a id="paper-4566"></a>
## CircuitSense: A Hierarchical MLLM Benchmark Bridging Visual Comprehension and Symbolic Reasoning in Engineering Design Process

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=WEuJlEJmX8)

### Images

![images](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/images/0b26bbe700aa63df4171a6ebe6bcb3f3ff9737c859fa6a2a528770fd204ac632.jpg)

![images](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/images/0c3d6a34a33d388ac3f5515ba2614ccbd89cae96550da81135254bae783dc1c0.jpg)

![images](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/images/3013543176e4cdbd7c1b33ee71e9c68f9a388bf4784cb5ad7485713ee78d839f.jpg)

![images](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/images/3995161e1dea1ff78cf2a3b5cf7d76e70b06b32d2ad9f0274040c4f5d196f517.jpg)

![images](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/images/4b787f1851079bc2e4434711570ce214c197d4c26adb51db7ae0370fc059a2e9.jpg)

![images](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/images/4d400ee107986fe902d28acda0f0b01c6d02970ef72cfa0c75f3048771a3cb7a.jpg)

![images](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/images/5cc1fc3e67313357c4e3f66481cce149f9a1bfb69fcd7a98c70fcb50b38b14b9.jpg)

![images](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/images/7afa5268c962644644fac5d515131c787d05edae5779f781f148258c291f9ec2.jpg)

![images](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/images/83c6447d74b26fd25a8434bdc47b73b8d0d0df9039280f12c8151d32eeda428f.jpg)

![images](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/images/a0b1ec6a37c6a238b2a4d69209c3057d355784a5d9b9ade0848ed94368e6799c.jpg)

![images](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/images/ba90163878b602e5ff23e773bacfe0881cba33cdec79a66ea679c9070bafd279.jpg)

![images](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/images/d9c76372fab4782a5c274325132f3c84fb8c923708357f6fab87a3c4bea82e03.jpg)

![images](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/images/ea2dc712c6e82bc165a05d08c193d4073c28cfbd2e89053162944b642c641d58.jpg)

![images](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/images/ee4b53b94d8bc89655f802a9e4ba39fbf52e8d7de20d6929ecb5f264e4f00387.jpg)

![images](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/images/f09773c5c2e66f49be46f0ca689786aa26b27afebb5fcd854b851fc1a70cc297.jpg)

![images](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/images/f49b41fb4c4197b89bb3d0bc1932dfb87cac45790fe1a41542b8a99bcb33bfcd.jpg)

![images](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/images/f571fb892dc71a26b4af51b2110602971499c331c2758ea720b0525b33042fee.jpg)

### Tables

![tables](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/tables/09fa13a49442f18511d85f893d45445a28ba18fcaab391a23f4ecd4c09262ae0.jpg)

![tables](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/tables/14e6290856a3a744aa7cf67dd04316f4c739e26f1fced2805aebc5ad06b3d7b6.jpg)

![tables](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/tables/3f3a3ef63d6b7b06fa1316a51176426b921c134935b85eaf975708655ab951ea.jpg)

![tables](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/tables/47c6b99c4d85fe36cddf9ab02c32f2eca9996ad6ba7629ed40673dba46bef54c.jpg)

![tables](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/tables/482ad5b182c119829068f3f9a76856dee4fb6015a27c33e1881d8e1a39ba3e2d.jpg)

![tables](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/tables/840e43f69d66123a0bc4ace8b2ac0667483f75e003b9ae11354ee435a3cf8a7d.jpg)

![tables](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/tables/bff13e8add3a43c052695979c1457bf8602f8c689117b8906f26791f9c2d331c.jpg)

![tables](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/tables/cd39f24eecb86066b2d79859d57bc4ab0c008e6d071e92dc3776c029a009324e.jpg)

![tables](../iclr_results/4566_CircuitSense_A_Hierarchical_MLLM_Benchmark_Bridging_Visual_Comprehension_and_Symbolic_Reasoning_in_E/tables/dba6e4165deeb1c8cb6ea527afc24323fb7c057f917970952e5f177bb573d302.jpg)
