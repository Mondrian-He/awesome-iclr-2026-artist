# ICLR 2026 Main Papers — Part 105 of 199

27 papers.

- [AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints](#paper-2812)
- [Enhanced Continual Learning of Vision-Language Models with Model Fusion](#paper-2813)
- [Knowledge Reasoning Language Model: Unifying Knowledge and Language for Inductive Knowledge Graph Reasoning](#paper-2814)
- [JAPAN: Joint Adaptive Prediction Areas with Normalising Flow](#paper-2815)
- [Glance for Context: Learning When to Leverage LLMs for Node-Aware GNN-LLM Fusion](#paper-2816)
- [Why Adversarially Train Diffusion Models?](#paper-2817)
- [Stacked from One: Multi-Scale Self-Injection for Context Window Extension](#paper-2818)
- [G-Merging: Graph Models Merging for Parameter-Efficient Multi-Task Knowledge Consolidation](#paper-2819)
- [Plan-R1: Safe and Feasible Trajectory Planning as Language Modeling](#paper-2820)
- [Human-Object Interaction via Automatically Designed VLM-Guided Motion Policy](#paper-2821)
- [e3: Learning to Explore Enables Extrapolation of Test-Time Compute for LLMs](#paper-2822)
- [Flower: A Flow-Matching Solver for Inverse Problems](#paper-2823)
- [Dragging with Geometry: From Pixels to Geometry-Guided Image Editing](#paper-2824)
- [SAIR: Enabling Deep Learning for Protein-Ligand Interactions with a Synthetic Structural Dataset](#paper-2825)
- [GPTailor: Large Language Model Pruning Through Layer Cutting and Stitching](#paper-2826)
- [TrimR: Verifier-based Training-Free Thinking Trimming for Efficient Test-Time Scaling](#paper-2827)
- [A Federated Generalized Expectation-Maximization Algorithm for Mixture Models with an Unknown Number of Components](#paper-2828)
- [From Data Statistics to Feature Geometry: How Correlations Shape Superposition](#paper-2829)
- [HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model](#paper-2830)
- [Strictly Constrained Generative Modeling via Split Augmented Langevin Sampling](#paper-2831)
- [Generalizable Heuristic Generation Through LLMs with Meta-Optimization](#paper-2832)
- [Provable and Practical In-Context Policy Optimization for Self-Improvement](#paper-2833)
- [DreamCS: Geometry-Aware Text-to-3D Generation with Unpaired 3D Reward Supervision](#paper-2834)
- [Revisiting Group Relative Policy Optimization: Insights into On-Policy and Off-Policy Training](#paper-2835)
- [Beyond Speedup - Utilizing KV Cache for Sampling and Reasoning](#paper-2836)
- [Soft Tokens, Hard Truths](#paper-2837)
- [RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion](#paper-2838)

---

<a id="paper-2812"></a>
## AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=zFkopTvclB)

### Images

![images](../iclr_results/2812_AutoTool_Automatic_Scaling_of_Tool-Use_Capabilities_in_RL_via_Decoupled_Entropy_Constraints/images/201a23cfb06b4da510192c897d61cb5eb0a1b1dc0de5e4fe0b100e523e364789.jpg)

![images](../iclr_results/2812_AutoTool_Automatic_Scaling_of_Tool-Use_Capabilities_in_RL_via_Decoupled_Entropy_Constraints/images/4c08a59ae347ff07a332a6ef91e01de20b374a919a2a8d6d3e92f5aaacfe1ca6.jpg)

![images](../iclr_results/2812_AutoTool_Automatic_Scaling_of_Tool-Use_Capabilities_in_RL_via_Decoupled_Entropy_Constraints/images/5d094e8841175f6f1ae16f811f43c5b5c53dae9f617965e4ec314602fcf9b9c4.jpg)

![images](../iclr_results/2812_AutoTool_Automatic_Scaling_of_Tool-Use_Capabilities_in_RL_via_Decoupled_Entropy_Constraints/images/631ae08349884957ae909ab81dd0e6811f7325c0dc60578158247d46eac590b7.jpg)

![images](../iclr_results/2812_AutoTool_Automatic_Scaling_of_Tool-Use_Capabilities_in_RL_via_Decoupled_Entropy_Constraints/images/79938c98104268c79df5de1743b06d246eaad8f64d4d366051d9adf0c5a69986.jpg)

![images](../iclr_results/2812_AutoTool_Automatic_Scaling_of_Tool-Use_Capabilities_in_RL_via_Decoupled_Entropy_Constraints/images/845c2a5880639c74f21217cb3fafac42d92305c9909599cb2d96fa0391315aa4.jpg)

![images](../iclr_results/2812_AutoTool_Automatic_Scaling_of_Tool-Use_Capabilities_in_RL_via_Decoupled_Entropy_Constraints/images/95488462a559107599f60cf4f3770292d25e9ed0817da6282f04c2b0dbe0570d.jpg)

![images](../iclr_results/2812_AutoTool_Automatic_Scaling_of_Tool-Use_Capabilities_in_RL_via_Decoupled_Entropy_Constraints/images/aafd6bfaefc4ca4d98ad92badc6b7639475e4680515847b4ed6e56416cccc155.jpg)

![images](../iclr_results/2812_AutoTool_Automatic_Scaling_of_Tool-Use_Capabilities_in_RL_via_Decoupled_Entropy_Constraints/images/c026bd5936f3b051d916f8b22fba328ae07903c7e75b331fbc888d5405ae81c7.jpg)

![images](../iclr_results/2812_AutoTool_Automatic_Scaling_of_Tool-Use_Capabilities_in_RL_via_Decoupled_Entropy_Constraints/images/f1797dde62b765bd78634d5f02af519d762d7e3655a26fa484ac5277ac4ba04a.jpg)

![images](../iclr_results/2812_AutoTool_Automatic_Scaling_of_Tool-Use_Capabilities_in_RL_via_Decoupled_Entropy_Constraints/images/f388ae610248afa75530c3a712bb4285ca3d867b35f1a5db22230a254152c828.jpg)

![images](../iclr_results/2812_AutoTool_Automatic_Scaling_of_Tool-Use_Capabilities_in_RL_via_Decoupled_Entropy_Constraints/images/f522908e1f995980782fe5af33f7410ffc8d7f2efb0e176dcdbd7f90d8157271.jpg)

### Tables

![tables](../iclr_results/2812_AutoTool_Automatic_Scaling_of_Tool-Use_Capabilities_in_RL_via_Decoupled_Entropy_Constraints/tables/033458d840cbafeae50cb133b1e989e80b77eabcd7a15c64c573b1cd3fcb5b03.jpg)

![tables](../iclr_results/2812_AutoTool_Automatic_Scaling_of_Tool-Use_Capabilities_in_RL_via_Decoupled_Entropy_Constraints/tables/308d5d2477cad4d04ec3f6fc1480315f6993fedf5451e77508a7d794052187ed.jpg)

![tables](../iclr_results/2812_AutoTool_Automatic_Scaling_of_Tool-Use_Capabilities_in_RL_via_Decoupled_Entropy_Constraints/tables/405cf3bacbf26f97d3df1e3be5a0267e7c3cc384bc4bcc258fd0a84956cf59a7.jpg)

![tables](../iclr_results/2812_AutoTool_Automatic_Scaling_of_Tool-Use_Capabilities_in_RL_via_Decoupled_Entropy_Constraints/tables/5f8efb7a2eef3fa3a96b6af789f538cba0728c1c2188211113b73e6b72e64d96.jpg)

![tables](../iclr_results/2812_AutoTool_Automatic_Scaling_of_Tool-Use_Capabilities_in_RL_via_Decoupled_Entropy_Constraints/tables/7f094f3ab151b37c16490ffb38e0a6a787208ff2ae26d6405efe16005fdb2414.jpg)

![tables](../iclr_results/2812_AutoTool_Automatic_Scaling_of_Tool-Use_Capabilities_in_RL_via_Decoupled_Entropy_Constraints/tables/83161b8060f2da76d46811172f2dfc08a7651d6b57dd83a2ee91039cc3767b0e.jpg)

![tables](../iclr_results/2812_AutoTool_Automatic_Scaling_of_Tool-Use_Capabilities_in_RL_via_Decoupled_Entropy_Constraints/tables/f95d7952fc47a95a5eea6db89645ca187813007bb6b694323cce4ca47a850eb2.jpg)

<a id="paper-2813"></a>
## Enhanced Continual Learning of Vision-Language Models with Model Fusion

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=ptFP9yT9DK)

### Images

![images](../iclr_results/2813_Enhanced_Continual_Learning_of_Vision-Language_Models_with_Model_Fusion/images/134b6ce1d2c43f9ad4b9d146f03f2b954f9e29cb723d2f4fff28483ec16441ca.jpg)

![images](../iclr_results/2813_Enhanced_Continual_Learning_of_Vision-Language_Models_with_Model_Fusion/images/311a2b368646ee1611cad7ad1b37804a60da0b867d9bc75419aa67a4f38fbff3.jpg)

![images](../iclr_results/2813_Enhanced_Continual_Learning_of_Vision-Language_Models_with_Model_Fusion/images/95bcf5e6564aa3b40fe652fd5a75d43024085f8bf404cfa232c08e71dd2fb709.jpg)

![images](../iclr_results/2813_Enhanced_Continual_Learning_of_Vision-Language_Models_with_Model_Fusion/images/bb8a17a5fc925c8bb51828153373b2629a7564d69c545e41263978fdb0425860.jpg)

![images](../iclr_results/2813_Enhanced_Continual_Learning_of_Vision-Language_Models_with_Model_Fusion/images/bf910523f5cd5f09f812942efcdbf8b6b8d82ec53e2ae2173eac9689e0e92e0d.jpg)

### Tables

![tables](../iclr_results/2813_Enhanced_Continual_Learning_of_Vision-Language_Models_with_Model_Fusion/tables/28c69534e281eb8b6bc2db27f83b13103f0c0c38577e975bd028476acd80e3e0.jpg)

![tables](../iclr_results/2813_Enhanced_Continual_Learning_of_Vision-Language_Models_with_Model_Fusion/tables/2d11bfe3de7f93203b8826c6a8569e67a3b7d768557b3f41c5729ad5e9c931f0.jpg)

![tables](../iclr_results/2813_Enhanced_Continual_Learning_of_Vision-Language_Models_with_Model_Fusion/tables/4d2df2f4d2eb60660365853d5779a7e0b94a8d5f09685e61258a1bb7dc3c23c7.jpg)

![tables](../iclr_results/2813_Enhanced_Continual_Learning_of_Vision-Language_Models_with_Model_Fusion/tables/548ec820605eb2cf796deaadf21dc17d0d049c1a94e08d8756b1f2d61c9e5dcd.jpg)

![tables](../iclr_results/2813_Enhanced_Continual_Learning_of_Vision-Language_Models_with_Model_Fusion/tables/571fc196e37e4de8734b84f745ac1000ae5bf58f1edf2c15c23bcaee7f63694c.jpg)

![tables](../iclr_results/2813_Enhanced_Continual_Learning_of_Vision-Language_Models_with_Model_Fusion/tables/7841071571ecb7d92d24c3911ee4e026bfbca93a611c16ed27d4ad73079f2d6d.jpg)

![tables](../iclr_results/2813_Enhanced_Continual_Learning_of_Vision-Language_Models_with_Model_Fusion/tables/80d2568b93bee8bcf20eb9bfc20aaf2b6361483d1a7b73654852b4bd31da49c2.jpg)

![tables](../iclr_results/2813_Enhanced_Continual_Learning_of_Vision-Language_Models_with_Model_Fusion/tables/95b35623e3f57ffda789061b7863f4f263d3031dca7b8ab5a78fc9642f4d406c.jpg)

![tables](../iclr_results/2813_Enhanced_Continual_Learning_of_Vision-Language_Models_with_Model_Fusion/tables/b3ea3b193800bc88a9d76f1a37c751ad6db0276897aa71512893868a573db233.jpg)

![tables](../iclr_results/2813_Enhanced_Continual_Learning_of_Vision-Language_Models_with_Model_Fusion/tables/c5bd39466bc21645b3ddbe642203c0315a259392405bbcdd384efc4cd426f3d8.jpg)

![tables](../iclr_results/2813_Enhanced_Continual_Learning_of_Vision-Language_Models_with_Model_Fusion/tables/d136539f9a446e3be1687973d78d9ab3d08c2849fa5d5ddb87562f3d6fd85031.jpg)

![tables](../iclr_results/2813_Enhanced_Continual_Learning_of_Vision-Language_Models_with_Model_Fusion/tables/e984e650691cf9027e0eebc2a00817e14a3eb8e63e8759d8deaa45de3dd2c323.jpg)

![tables](../iclr_results/2813_Enhanced_Continual_Learning_of_Vision-Language_Models_with_Model_Fusion/tables/fd798e883f00ccb732c7a7af03b74445b4038be7ac5bbcbffa79dec36ed74279.jpg)

<a id="paper-2814"></a>
## Knowledge Reasoning Language Model: Unifying Knowledge and Language for Inductive Knowledge Graph Reasoning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=2g8EmFwNTB)

### Images

![images](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/images/01a78395d8c8612152a7aa8ca73237b1c81355df009d04c6d3bc54018a4e8d56.jpg)

![images](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/images/51a4eed2098a8a58770df382bd835e821670e843f713be590d84e9f2ae6f581c.jpg)

![images](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/images/7926a81f41706cf6f62fded027c754437b500ba8f4e882964dc146c785401c65.jpg)

![images](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/images/81f8b9673ab6f685f37975e74a887012ecff60cfa0e2fae835ffa5559eccb629.jpg)

![images](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/images/99bc4ead955f9f83648f452eb9fdf648cc4a1c8f4a93eb167535baf0fda7c4f1.jpg)

![images](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/images/dee82c8324d03f616bae45fd5b7cfd25576550b5a1d5a53c779aa563a089ad0e.jpg)

![images](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/images/f8ed1cd62e23487f9bfac1b329cd54509dbffdc0d6c8e3e063a3124e1256a3a6.jpg)

### Tables

![tables](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/tables/01e3a647ddb0cc3c506ce0aa6b8c7fd147391d7a57d4561ba27ceaf58b0214d4.jpg)

![tables](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/tables/1b6207300c66d0fe0e0e9e387b1d80d1de427ca60879aabbe26b1af746e17923.jpg)

![tables](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/tables/1ddfffb38c317600ee62c1fd6af40200c80ec03c932908134d5f5bafc8c5e4c0.jpg)

![tables](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/tables/2cfb56f66638e5ec97d9630c64d501e937b027bca42bca1b36ddb75080ccab97.jpg)

![tables](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/tables/328df816a1ed6e041dede11b3555ece838bd6073c2c87ad478918c207d2e992b.jpg)

![tables](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/tables/622b19cd8856fbc69c8459d83b2fcb0faa5c5094db6cd747594302fe5f44666f.jpg)

![tables](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/tables/7ba6e8501cda690b91bf4a6475249de5d322f7475e1a4a4b32df9acea0bb2216.jpg)

![tables](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/tables/7ef927511f075531b095532bbe9e7e1756eebac7d36a36ccbdba9be346d7c75b.jpg)

![tables](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/tables/8e411efb76540e270e5b3db68b72193ec4f355e194952b4865bcd940c60fb913.jpg)

![tables](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/tables/915307939ed29a2c3b34083994db22ff736c9c0b090e35207f3028fcdafbeaa4.jpg)

![tables](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/tables/a1b810bf88b93af56384c657b43b3d5e3192bee471acad26fdc5ba64d13998c3.jpg)

![tables](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/tables/b75903ef86f698c20c16ddf11d560b3b08f72106e4c5818f3f4f9c2928be89cc.jpg)

![tables](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/tables/d3465f0845b4f34a6206ae5ced9a0fe7fc8a28130039fec9c2e9d6fae1104fe4.jpg)

![tables](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/tables/dc741d11f4db1ff13d26cc43b39d3e1cbe771764503a5f4227dc638107f82dbe.jpg)

![tables](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/tables/f46af9a70bb3c6cb2ccc14bc47a309491cb2a17b65e287ecdb89e87760c7fba0.jpg)

![tables](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/tables/fc7e711c9836f2b795830debd9ea7e003009dbbd3d2a3d2699c34fabedc8fdb1.jpg)

![tables](../iclr_results/2814_Knowledge_Reasoning_Language_Model_Unifying_Knowledge_and_Language_for_Inductive_Knowledge_Graph_Rea/tables/fe4437f9eddd607eb3f817133271316c70f98f289f47fd92dc66bc2d5a38abf7.jpg)

<a id="paper-2815"></a>
## JAPAN: Joint Adaptive Prediction Areas with Normalising Flow

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=4SxAu9zMVC)

### Images

![images](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/images/072bdc2d225a9b0f72cbb5c72f2cd8d61bfbc4c5334cbb2de47e8f1bd152a374.jpg)

![images](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/images/2666e1c3d1130e058b8de88cfb61f6df2d5433de512f184ac8108c6122f7d932.jpg)

![images](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/images/2b9cb9fc492a0e789bb8a88cba6f216427a04e7a025d83e50a7fdf4fd1a05f14.jpg)

![images](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/images/3fe3c9561ba5fb6ada774721a44b6842337654d72f659e485ac7a6345b44368f.jpg)

![images](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/images/460ff8ed659c16a4ff2646e60ded8d90f3ed4c3664dc59fd45ce1777aefd9cfb.jpg)

![images](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/images/7f1a75dac0d09c7a133bfb115985452684e7515313e1597352dc1c8d2ec9f3a1.jpg)

![images](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/images/850c793b10f32b4ba64060e838230c110afbfa5ebf130326793a27b872d2681f.jpg)

![images](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/images/854851498f6e871c9a03269643c006d949ea3a420386918f0cdcf834151975a3.jpg)

![images](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/images/a743483a4442fe3a10d53e8142e2050d9c1e0fd18a332d1c4995a5215f111f80.jpg)

![images](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/images/ab3dbf7f817066eee4d211a789d703ed7e35cbec99f0595a97368ada8fb1eb12.jpg)

![images](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/images/d2e93f93d68ef0b5c47471230938801912120a358104c1311ed19edfd8b10440.jpg)

![images](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/images/d7f349fe9dcf9c512a9e2372a0fece38196e0ac1e850b86f21ac0845e40be7c6.jpg)

![images](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/images/e94a91e3a0176915b4caa8e36320cd203ef6ddfef99a5607ac36dc4a753fdea5.jpg)

![images](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/images/f32bd85ce33e7185903716e907058584d437f97a01b7470102351d0d1d52384c.jpg)

![images](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/images/fe6fec2f73a9c9b6e654b0faf8d83a630ac1b62858f0c426b73922c4ddc3245c.jpg)

### Tables

![tables](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/tables/156c0a1a2d00a08ab382b4e994e844cdd70beafda1cc408fc7dfe4050156a5f7.jpg)

![tables](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/tables/1ade5b828a2c9350bd5e745b2f8f421bf3e382e36cdec61e9eadf310b7cb139b.jpg)

![tables](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/tables/2fcc9915a3ccaf64a2907f673a1900bc8d9b3f755ba5def6f1cc50622ad1af95.jpg)

![tables](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/tables/63345492cd9d9df42b2ed3f40e5f4f2da3e5c91d115bd0c48e41b898d9972193.jpg)

![tables](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/tables/6bff634b4733ffdf0f6d9b92b9fe81986d79a186cd29a01c31c57597113ad119.jpg)

![tables](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/tables/8dfd6b476b1667b75505214e00ba98b1c2f7505611fa72cd05c443441865f8c9.jpg)

![tables](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/tables/98679b367a131c219929a97f82ee3f6b85840502d64d0c72096fd156137fb309.jpg)

![tables](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/tables/b8a44cce65214122beceeecc8d450a9646f51486b181fc77b5c8a9dc20b812c4.jpg)

![tables](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/tables/d6abc9eef00280c202e720f3ca06462fb847790f16753bcbc3d2e53f5b93b81c.jpg)

![tables](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/tables/e7c8d8c45f2a8c4c5f51261dc0e4ab267749c167270f3f8f880c91cd96ba7ec1.jpg)

![tables](../iclr_results/2815_JAPAN_Joint_Adaptive_Prediction_Areas_with_Normalising_Flow/tables/fdf586cb10ad7c301d23e1757d6f685cca9ed42c059b659649a31c97a5f47a4d.jpg)

<a id="paper-2816"></a>
## Glance for Context: Learning When to Leverage LLMs for Node-Aware GNN-LLM Fusion

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=oODFyykHF5)

### Images

![images](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/images/488c419e737ddc3bd60639373d6a71010053181872371cbd5f5a2e0c8fa0e1a3.jpg)

![images](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/images/4a2e341210b565795a3dbeedd8ecb99a369665cf4438e847f587f781ca9fb88a.jpg)

![images](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/images/5022119aaeb5b19aca12ed6e19b50cf23d7bddc8e8606a349c9a9564d378bd74.jpg)

![images](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/images/50587a2e5bd00bd0de0401a7afeebe4f7f57aa90787e46eced694af1aa71272d.jpg)

![images](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/images/9d4ab28b333282b65afb1118ae8fdcec44dddfd1b86e1c4ed3eec7f7962e98d2.jpg)

![images](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/images/a38a9d6fb6f385bc8821586585a28735c4cb27604ca0761bcfc678a3eee30e67.jpg)

![images](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/images/bb8b3014de8d02343bc399f266d5b4add5401afb668e9c7f21a26bb6d603d26f.jpg)

![images](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/images/d2a06ccbdd3043c4fd66babfb1812ee426fba8aa7478d46ac5e939415c2ce326.jpg)

![images](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/images/f835ac3b3265159b0c779b74ab26b67b313127fdabbb4e6fef0dcd9c52e85b54.jpg)

### Tables

![tables](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/tables/00e76c3b0d7b8c5ccdcdc0a8105392dc95c3610ca6e2e405caddd713f4cf8e28.jpg)

![tables](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/tables/1ebe01f94c93873033db24c492bb5f5b422d1d0aff8a8c486676248032fe3433.jpg)

![tables](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/tables/333b025d51dc2e67c9ac876de2526a5c4685b07d45ca5d5c64b3ce4959b5fc54.jpg)

![tables](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/tables/4bc0ea21baee294097627741b15fb610c26faffaa70535c36420b1836e1dc4ed.jpg)

![tables](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/tables/62136fd65b00612ef05294b5c1189e5daa2575db8b54861ffd5d96930263dc23.jpg)

![tables](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/tables/6291a4a00ec7751a9f0bfb7ac6e07a4fa7ad9b65ca800142734df8c19a598d2a.jpg)

![tables](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/tables/6ab802b8b86ddcd49fa62bcbf876a00946526145451f56a8a902f97c6cac9191.jpg)

![tables](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/tables/72f641fd7523b61ddf808b820ed58ae35d79a4cf716494522182b5c2974608a2.jpg)

![tables](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/tables/7b5aecc4be7d51376b38f0c397b7f53b5d5caeea66e8d505cf393d3270584717.jpg)

![tables](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/tables/975d9032b8250be911aa272e299b6bfa3016db93f0a1fb78fa3463eadbc8f186.jpg)

![tables](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/tables/aa0d9e1511a653fa503b7346eb340cc849f5c0950b93246462a9f2e2384c4b1d.jpg)

![tables](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/tables/dfae786c667a282dd604f4349f9bbd2049d028f9d54f6604215ec7164565880f.jpg)

![tables](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/tables/e4fee17967c6206df0026fe3eb2fbc29cd41f4ad8973cd50850d129f50bdf74a.jpg)

![tables](../iclr_results/2816_Glance_for_Context_Learning_When_to_Leverage_LLMs_for_Node-Aware_GNN-LLM_Fusion/tables/fec3afa9df9f7c908a0d27ef797d32de64c490bfa52c5dfd3960f6826be85ece.jpg)

<a id="paper-2817"></a>
## Why Adversarially Train Diffusion Models?

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=lL6htAaolp)

### Images

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/1b6772555ebe32f3d2465868df9ce5d59bd9b75adfdd040b80be243eea2acd4f.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/24d6359e059c12774a52074e1acb02bafdfcdefb6465774165cd7ade01354917.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/286d497ba6d3105bcbe1abbd53809c2f8ce96b6fa3c290c900af466aa5dcbf59.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/2a83314bcdf6006e95aeed0967f85534f6b48b745673040b441883a726faedf2.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/2aa03cd1113553962c2dacf9a2cae210d5d30b0e729e9de2e5dcaa28cdc815bb.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/2f404f199b3bdf510647bbffb8603c17496bd8ac3f2a840f8a15cfca45eafcbd.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/4e281f836882123a172de28d437bfcc2dd0e9d70d0592e0a60f2404559376d40.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/58fb472c8f9349ef7ec189269c5d5de004784b32a5c55f9bca3792239c5a6ed0.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/59a252f2c4f87108f92dc6b5fd01e1220ef44d9783a03ad90800a5ca18cd56de.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/62982e6f4090fb209afc42decf9cdd1a2bed727c50f459bf2e954b0f495ea42a.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/6bac65ec1651294a118ce383253016387c069b5e1624dd9813fb37ddd3a4272d.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/6cde411af40c3fc1429f303a384ce0d287598d94fa0e04cda52247e635420f1f.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/79bc351132b5a848c3b452a291d053f910a10f67d5b5587f33f9eee31d0f8de6.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/83554f49d5d0734f3319dbee7a7799ab8d385295c74573d6a286d634c083e4bc.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/878b8522c37cbda3bcf694e13b572fbaa1d6ba2428171f9aa06e379123851fa4.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/8a99307a29411c504c40c28ba430a5146ca64dd81ce2b0ffa5b6f4ac7c057714.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/8bce1511008a8d50f95b9a3f821510c1613f14fd79b945e95472d8ea19c69950.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/95a34e8ddbbf41950da089b4542414cee824bae2e79a4292ea0976efface74a3.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/97a7a9e34578a446ce64d18fac3ede832ab62bed28f6207b0a41f94611c29e24.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/98fdc60788f364980c2aeffa060109f4e0e7a0ee25bd53c7cc7f5a2d6ff36a0a.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/a10c073415f8721489767cabf4effca1823b9aef1e3fac598905360bbf3ffe9a.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/a2c9830bddce00ef52683e692f055d32287ea099811cf4a13c34e1e50767a4a2.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/a361a3d4177394348b38bc3ac4e279cf5e48c1f38e993a75bbb75e2107a36a37.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/b82081d433c9e0ff19e9436322c4a1f0bb9275ebfe5dbcb1ec070bd9d85284dc.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/bd8dbbb080ec6306175b95ae4123e65568b40cc8bad14216779496218487710d.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/c148a4aea37376f5ca2c624b3c6da9d65c45f802134f81480cef845eb9f4066c.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/c880607cd05af6c78ecf13fc934658d22d4e3b7ca058df295589a14145559ebf.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/ed33df05a25d906171b1c4a9e6ac8aeb34b685555c7900406d742a0ed5890bd7.jpg)

![images](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/images/f5ddffd06cb8b844a171d3b52e66a0bb9ca1f81adfbd3125c94689fd6cdbc16b.jpg)

### Tables

![tables](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/tables/12135984c26d9f0cbe03712223663ecefc36e6d8f2d83b09997a05f06c5a3897.jpg)

![tables](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/tables/2c57e7cd1449c11d524023a5b38ca959dd7f024a2e788b4381b0e0e7338c00a8.jpg)

![tables](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/tables/3a2c0b09315775dcacb8f6cf2bcde2ccb4095fd158c256789679e3a0ca4cbea2.jpg)

![tables](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/tables/51581b494466b06c40e74111c9e6711f24e9871fc73cd35a500d0ed65e7097d9.jpg)

![tables](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/tables/6bb7530233819c5c4dc8cc31b42e573c940aedde0657f482d5ebc9142b126f91.jpg)

![tables](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/tables/90564373fe602af2fd5255d093b728fa1d3be1fcccc4ff836cbc5e152e5ff190.jpg)

![tables](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/tables/b1f5d1cffc3ddaa1b6d3f1f06bf35b915699cff3fb82fc70d45e17b3784b536e.jpg)

![tables](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/tables/befca5586baae2f81e50491f9eab0bd986250b38d3382a92f2c6c756fd20a767.jpg)

![tables](../iclr_results/2817_Why_Adversarially_Train_Diffusion_Models/tables/fc1c01c640bdd5efafed9ca9c674781ae984945daffbfc193e7756ec5395283c.jpg)

<a id="paper-2818"></a>
## Stacked from One: Multi-Scale Self-Injection for Context Window Extension

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=lh3Aa1u7kU)

### Images

![images](../iclr_results/2818_Stacked_from_One_Multi-Scale_Self-Injection_for_Context_Window_Extension/images/2635bafa00730e389285fb4290609246c6de6c3a56a2eb0813222b443b7bdc99.jpg)

![images](../iclr_results/2818_Stacked_from_One_Multi-Scale_Self-Injection_for_Context_Window_Extension/images/5ab50dd78b73fcf3980f5d35880b3a8052945161938a6b25be5fdb539315e739.jpg)

![images](../iclr_results/2818_Stacked_from_One_Multi-Scale_Self-Injection_for_Context_Window_Extension/images/7083f316f69afc1be1c12035f700c2bccad9bad1c2dd8f9e70fe891f3b48cf8d.jpg)

![images](../iclr_results/2818_Stacked_from_One_Multi-Scale_Self-Injection_for_Context_Window_Extension/images/8d6667bd559240baf91af03b2c9b6efb9dbfd37cccb9ef162f44df9786663e01.jpg)

![images](../iclr_results/2818_Stacked_from_One_Multi-Scale_Self-Injection_for_Context_Window_Extension/images/b69fe0520b0c3237cbc55f75419f0737d60e7cef3611c6bf88e08d3e22a32d04.jpg)

![images](../iclr_results/2818_Stacked_from_One_Multi-Scale_Self-Injection_for_Context_Window_Extension/images/b8a5e12f2cf2026b04766640016f55ca01dd3c28288d2302267d958424bc344c.jpg)

### Tables

![tables](../iclr_results/2818_Stacked_from_One_Multi-Scale_Self-Injection_for_Context_Window_Extension/tables/14694b4672f4768e7fd367644eb882eef09e5037b554be7126fce91c20ddaf10.jpg)

![tables](../iclr_results/2818_Stacked_from_One_Multi-Scale_Self-Injection_for_Context_Window_Extension/tables/1a305012fb186899bb0ac55599f00be253c67b0b854f58421095af6c90314b82.jpg)

![tables](../iclr_results/2818_Stacked_from_One_Multi-Scale_Self-Injection_for_Context_Window_Extension/tables/26fbbb8f71fcf664b9c267c1a52cfa4886cbab653ecdface2a2095a7c5a45485.jpg)

![tables](../iclr_results/2818_Stacked_from_One_Multi-Scale_Self-Injection_for_Context_Window_Extension/tables/3cacf380bec692567c2445002953c04429e7d8d39132f80bbe7f018236d9042a.jpg)

![tables](../iclr_results/2818_Stacked_from_One_Multi-Scale_Self-Injection_for_Context_Window_Extension/tables/41eee53574524d1caa29f9da534152c95c97e529cf3743c0ebec61d770643e99.jpg)

![tables](../iclr_results/2818_Stacked_from_One_Multi-Scale_Self-Injection_for_Context_Window_Extension/tables/4cbf5f904ab2a7ac58b7060e5854ee2a0f5e92f21581f2950335680eefc77a83.jpg)

![tables](../iclr_results/2818_Stacked_from_One_Multi-Scale_Self-Injection_for_Context_Window_Extension/tables/4f5775da87d2d49a2aebd5399cf91596ed9f25f88f58e3daa2ac8b69f5a60d88.jpg)

![tables](../iclr_results/2818_Stacked_from_One_Multi-Scale_Self-Injection_for_Context_Window_Extension/tables/6a1486bfb499bdac3ee2057750cfbcc6a726e1e8c93f2c072cb51f73ffd59316.jpg)

![tables](../iclr_results/2818_Stacked_from_One_Multi-Scale_Self-Injection_for_Context_Window_Extension/tables/8aae45d86113ce43633ce02ddf7d37be8858d2e61fc81c5cabe4d73f4188f28d.jpg)

![tables](../iclr_results/2818_Stacked_from_One_Multi-Scale_Self-Injection_for_Context_Window_Extension/tables/aaa8317484a9c1fcf55e76b1517ba6672c3755df5f365f0d6c9003aaca83ddee.jpg)

![tables](../iclr_results/2818_Stacked_from_One_Multi-Scale_Self-Injection_for_Context_Window_Extension/tables/cc1507c9406a161050df79200fe3a8b236658238954c574f8b96827312a2e774.jpg)

![tables](../iclr_results/2818_Stacked_from_One_Multi-Scale_Self-Injection_for_Context_Window_Extension/tables/d9d3efed90373dec435c087ddf3d296e427349bee8ce69ec993804396a1c1e89.jpg)

![tables](../iclr_results/2818_Stacked_from_One_Multi-Scale_Self-Injection_for_Context_Window_Extension/tables/fd09beeeee13b8596b216ec82fe282fa52934a5913fdd9368baab6a3eac88f60.jpg)

<a id="paper-2819"></a>
## G-Merging: Graph Models Merging for Parameter-Efficient Multi-Task Knowledge Consolidation

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=FoTtvLkkfU)

### Images

![images](../iclr_results/2819_G-Merging_Graph_Models_Merging_for_Parameter-Efficient_Multi-Task_Knowledge_Consolidation/images/010113b829641e8c7ddce73cb57a3dcc74104465ba966f18d59e4fab28b7b237.jpg)

![images](../iclr_results/2819_G-Merging_Graph_Models_Merging_for_Parameter-Efficient_Multi-Task_Knowledge_Consolidation/images/16cb2ae4ff5063dced5735190587e234ecbc8e5a6c410d8d7ced6d2f5173d5b1.jpg)

![images](../iclr_results/2819_G-Merging_Graph_Models_Merging_for_Parameter-Efficient_Multi-Task_Knowledge_Consolidation/images/23ada9549c66f17e5cfcbc4c130f7077206fa5589a0c2c89e92e01dd8f27046a.jpg)

![images](../iclr_results/2819_G-Merging_Graph_Models_Merging_for_Parameter-Efficient_Multi-Task_Knowledge_Consolidation/images/273567868221b26d76c878bbf3c553533415f15816373c5cdfc54394283d3ce4.jpg)

![images](../iclr_results/2819_G-Merging_Graph_Models_Merging_for_Parameter-Efficient_Multi-Task_Knowledge_Consolidation/images/2ace1acccde81f23655d25bebcf4004ded69ff54bdee8717132008ea0660b68e.jpg)

![images](../iclr_results/2819_G-Merging_Graph_Models_Merging_for_Parameter-Efficient_Multi-Task_Knowledge_Consolidation/images/30d31f7386d04140fa69099618311e5f0c3b9734457e64863d1e525211c9c4c1.jpg)

![images](../iclr_results/2819_G-Merging_Graph_Models_Merging_for_Parameter-Efficient_Multi-Task_Knowledge_Consolidation/images/51487841b4ddf32c84eb796a7b3a4d863a96b5d45ed8abe1f0818fd78ebaa6fd.jpg)

![images](../iclr_results/2819_G-Merging_Graph_Models_Merging_for_Parameter-Efficient_Multi-Task_Knowledge_Consolidation/images/5d6c9547d2ad786a0648d08d08a4678915acb83204cda7bc661972ecd1b6a403.jpg)

![images](../iclr_results/2819_G-Merging_Graph_Models_Merging_for_Parameter-Efficient_Multi-Task_Knowledge_Consolidation/images/5f89d090d9b45380646d80ee698de1efc001f79c88f709952709e24eaf753dbc.jpg)

![images](../iclr_results/2819_G-Merging_Graph_Models_Merging_for_Parameter-Efficient_Multi-Task_Knowledge_Consolidation/images/9e1cf512b18a3a651b886629532464299ffa7476fb954cc041de9839ce11e66c.jpg)

![images](../iclr_results/2819_G-Merging_Graph_Models_Merging_for_Parameter-Efficient_Multi-Task_Knowledge_Consolidation/images/a55453f25284b60d9d34ac79f34153f1073eaa4ee2eb5752bf48e47916f44caa.jpg)

### Tables

![tables](../iclr_results/2819_G-Merging_Graph_Models_Merging_for_Parameter-Efficient_Multi-Task_Knowledge_Consolidation/tables/13aec5cc2f67fb7060456ba9f7b0c22278cbce8cc3234e2b77ca89c021a06dc1.jpg)

![tables](../iclr_results/2819_G-Merging_Graph_Models_Merging_for_Parameter-Efficient_Multi-Task_Knowledge_Consolidation/tables/2558b3b58f1859f834d48d52bf230430c3e3a48c91149e868cb249b9c5a76d03.jpg)

![tables](../iclr_results/2819_G-Merging_Graph_Models_Merging_for_Parameter-Efficient_Multi-Task_Knowledge_Consolidation/tables/74d7ac838e329a65c23ac6ceed8aaef8a2aa4849193582936aa9b314e99df3d5.jpg)

![tables](../iclr_results/2819_G-Merging_Graph_Models_Merging_for_Parameter-Efficient_Multi-Task_Knowledge_Consolidation/tables/7b138932a3cbf491f5de65cd0f81829ef7e31b0aa476163972eb230192152745.jpg)

![tables](../iclr_results/2819_G-Merging_Graph_Models_Merging_for_Parameter-Efficient_Multi-Task_Knowledge_Consolidation/tables/86036d106d401ede6304a6fa7b409b5ed21f1a988cc401888aef348d5fd769fe.jpg)

![tables](../iclr_results/2819_G-Merging_Graph_Models_Merging_for_Parameter-Efficient_Multi-Task_Knowledge_Consolidation/tables/b7f269a047a39ad2b7863976deed751f3e32141dc3c903e975068003d3271446.jpg)

![tables](../iclr_results/2819_G-Merging_Graph_Models_Merging_for_Parameter-Efficient_Multi-Task_Knowledge_Consolidation/tables/d6a360efbd3da825ef4c9fd21865bf48c40aec6d6992bb0ea3e0b20678fa614f.jpg)

<a id="paper-2820"></a>
## Plan-R1: Safe and Feasible Trajectory Planning as Language Modeling

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=uusTA1rBhR)

### Images

![images](../iclr_results/2820_Plan-R1_Safe_and_Feasible_Trajectory_Planning_as_Language_Modeling/images/15dfa7fe2b3a27feb718f04236490830f746665ce3dbfd3331eb76cd0f085bf3.jpg)

![images](../iclr_results/2820_Plan-R1_Safe_and_Feasible_Trajectory_Planning_as_Language_Modeling/images/4cb77dddb7e756ff1e7b8f611e23c53e53ad8cd2a66aeff594137deeae3a2ebd.jpg)

![images](../iclr_results/2820_Plan-R1_Safe_and_Feasible_Trajectory_Planning_as_Language_Modeling/images/59f4a31d9dd9d2ec8a9a06bb97335b4ce9944657fef786266d2bb078f0ea7405.jpg)

![images](../iclr_results/2820_Plan-R1_Safe_and_Feasible_Trajectory_Planning_as_Language_Modeling/images/68ca8e2d03f07719fda83ecb958dd96a7f294ce12f2c3431f9257b4b85994a33.jpg)

![images](../iclr_results/2820_Plan-R1_Safe_and_Feasible_Trajectory_Planning_as_Language_Modeling/images/86e96d642f0e982fb6f89fd948cca5236d22505abb8cfb144259959dc90ed34d.jpg)

![images](../iclr_results/2820_Plan-R1_Safe_and_Feasible_Trajectory_Planning_as_Language_Modeling/images/935e813cec1aae8274a257f85e42ced61058a0364dcc74856653ac317f541ace.jpg)

![images](../iclr_results/2820_Plan-R1_Safe_and_Feasible_Trajectory_Planning_as_Language_Modeling/images/cf1e69fbc1108703d59bacafd1d1c719306860e1816b8ace6b9dbf9df94b66b1.jpg)

![images](../iclr_results/2820_Plan-R1_Safe_and_Feasible_Trajectory_Planning_as_Language_Modeling/images/e5b1ab81e6b69776ba631f44ce300218dcccc8819ed1a16167a6d4ec5faeca8b.jpg)

![images](../iclr_results/2820_Plan-R1_Safe_and_Feasible_Trajectory_Planning_as_Language_Modeling/images/ecd68d4c99472a01a9ec047edf6012db52af46745de7dd724f9a4e76f4b704d0.jpg)

### Tables

![tables](../iclr_results/2820_Plan-R1_Safe_and_Feasible_Trajectory_Planning_as_Language_Modeling/tables/1152fd7731677f4116ba80498506b38b24dbc133b0745f34ed69df6486646c45.jpg)

![tables](../iclr_results/2820_Plan-R1_Safe_and_Feasible_Trajectory_Planning_as_Language_Modeling/tables/17ab0257210de0efc86e7f1671a9633f3f32476d915606672d360d8b554ae0ff.jpg)

![tables](../iclr_results/2820_Plan-R1_Safe_and_Feasible_Trajectory_Planning_as_Language_Modeling/tables/36154dbf858d9d557ef4b22b0cf2f604b28cd1a310dc74994f60e8c5ea987a2d.jpg)

![tables](../iclr_results/2820_Plan-R1_Safe_and_Feasible_Trajectory_Planning_as_Language_Modeling/tables/49b5bcb8d3a8e62248c21b52fbe37996e2b447d208346470d0f65d3ae219b7a9.jpg)

![tables](../iclr_results/2820_Plan-R1_Safe_and_Feasible_Trajectory_Planning_as_Language_Modeling/tables/53e21edaaaeaf6fb972b599a4390214420208ab0ff71edea899f29b4571dd579.jpg)

![tables](../iclr_results/2820_Plan-R1_Safe_and_Feasible_Trajectory_Planning_as_Language_Modeling/tables/8d9489bb93388eeabcf8ddd45f14d4ed8f62e80b727e650f2cae1aaa017054b5.jpg)

![tables](../iclr_results/2820_Plan-R1_Safe_and_Feasible_Trajectory_Planning_as_Language_Modeling/tables/c13bf9c9d702a75848cae36946368291ffb33f0e92c8ca216f67624acab70641.jpg)

<a id="paper-2821"></a>
## Human-Object Interaction via Automatically Designed VLM-Guided Motion Policy

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=LfkPlFTfe0)

### Images

![images](../iclr_results/2821_Human-Object_Interaction_via_Automatically_Designed_VLM-Guided_Motion_Policy/images/377c8345634395749a09ba36aa7506623d7cb86d4c1c3a03c771447cc60ca3e8.jpg)

![images](../iclr_results/2821_Human-Object_Interaction_via_Automatically_Designed_VLM-Guided_Motion_Policy/images/4795e430f400a322ca0ebb7b9d4e892eaa5eda4f4c90feb9dcf3c8ad89a731f6.jpg)

![images](../iclr_results/2821_Human-Object_Interaction_via_Automatically_Designed_VLM-Guided_Motion_Policy/images/53930889350001bb6dda126edede2e5ebd64537b3af59fc05787c99868dbc4ed.jpg)

![images](../iclr_results/2821_Human-Object_Interaction_via_Automatically_Designed_VLM-Guided_Motion_Policy/images/578742a0b0b8c28ac5f2a181d7de9b21c4306f78b586f5aae1723015fc53b73a.jpg)

![images](../iclr_results/2821_Human-Object_Interaction_via_Automatically_Designed_VLM-Guided_Motion_Policy/images/80b078353cb74ba5601803c281d0fba2bad153d928cb12ccc3b9015302013bc9.jpg)

![images](../iclr_results/2821_Human-Object_Interaction_via_Automatically_Designed_VLM-Guided_Motion_Policy/images/a123ffd5f38e992ff5f3a2f98e31ed380c1a31da59d854e628f70f85b388243a.jpg)

![images](../iclr_results/2821_Human-Object_Interaction_via_Automatically_Designed_VLM-Guided_Motion_Policy/images/a52c44e7f70083528a5edc45a34656af1fca670363ece37970360d3e693babe5.jpg)

![images](../iclr_results/2821_Human-Object_Interaction_via_Automatically_Designed_VLM-Guided_Motion_Policy/images/c46ee7eeb0285e6e2f037327b269ebdf103f62e19ed1f086d116e2cdf37b0eeb.jpg)

![images](../iclr_results/2821_Human-Object_Interaction_via_Automatically_Designed_VLM-Guided_Motion_Policy/images/eac4113c446e1546dcbfabb48c6766d5115741883c48c3f77c39653f83ee8eec.jpg)

![images](../iclr_results/2821_Human-Object_Interaction_via_Automatically_Designed_VLM-Guided_Motion_Policy/images/f2cd3750df8deee99bb72eb6220632994cf5ab825a93db1835c50455a0443ea2.jpg)

![images](../iclr_results/2821_Human-Object_Interaction_via_Automatically_Designed_VLM-Guided_Motion_Policy/images/fba504823ba1afa9bd4b087cffe650639ef6dcd7e7321df247938feacaeb21ac.jpg)

### Tables

![tables](../iclr_results/2821_Human-Object_Interaction_via_Automatically_Designed_VLM-Guided_Motion_Policy/tables/0bf0ac5a7f7f015086fba3dd6d4a979dfa5a96c092ab395ee8f49d5145c3e31e.jpg)

![tables](../iclr_results/2821_Human-Object_Interaction_via_Automatically_Designed_VLM-Guided_Motion_Policy/tables/384cc7746a35340ff74299dab779ffeacfb628761aae67677224c58b4215ec1b.jpg)

![tables](../iclr_results/2821_Human-Object_Interaction_via_Automatically_Designed_VLM-Guided_Motion_Policy/tables/5527aa4b5dbb1c02a28d06be3ec162df20ac4623717d981ede92a0b93784d7d3.jpg)

![tables](../iclr_results/2821_Human-Object_Interaction_via_Automatically_Designed_VLM-Guided_Motion_Policy/tables/6b3972c2d489a68b379256e0302f0edecf8a40d3e8d2448c3e651f9bff4d8508.jpg)

![tables](../iclr_results/2821_Human-Object_Interaction_via_Automatically_Designed_VLM-Guided_Motion_Policy/tables/8120ae03f7fb3fd65819a4e19ac9db82b17a9ef6af0695b60ea31a4a49282c86.jpg)

![tables](../iclr_results/2821_Human-Object_Interaction_via_Automatically_Designed_VLM-Guided_Motion_Policy/tables/b7726cc710dfd31bd786a4d38e704f5ad94b4f01edbb6a4bbdc1fa771c7ecf57.jpg)

![tables](../iclr_results/2821_Human-Object_Interaction_via_Automatically_Designed_VLM-Guided_Motion_Policy/tables/d2b752a0dd1f72c4c4ece78ec362ddbcd537056e0e5a0895df5f10fcf2135675.jpg)

<a id="paper-2822"></a>
## e3: Learning to Explore Enables Extrapolation of Test-Time Compute for LLMs

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=aID0dZmMmM)

### Images

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/1dd86b1c7e5d3d2dcaa9089d959e6b59fd65dc92d33f4ce2ca695b99d0ec062f.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/1f09eac7cb2389b2d8dbac83ef65b8b5f7039bdd8502e5c718867ed1924f4e96.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/28761c1d1593a6f55b7795acb97d8b52b51f893d6eaab00034d70e5205558c51.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/29ef3b583f3930330c5c3af60264c4dd34fc721488de438caf7847307e5c7821.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/311fb8468b8fd7e17abd51383e55b8e8414b89e550a9df12acf0094f993144b1.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/3b2b7af1468e956940082b7c6ac5f8726a77880f4ae546041e551977814ec573.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/4c03011c759dcf57f0f45c42873a66819347a07f47d4197bc4618d6c886b7b59.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/50d6f52a48e4c377afc6e30f5107b9da3abf2da89008fa9bc1b8d2fcdbac5da4.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/53e73a0b423582b5884e1e328377d82d2abfa2813bc89aba8964a4096a4d4a9b.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/7df8960eb308074a4284956eb55840d3c1e73205f19a254a0453a608fdd5817e.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/82a8ddda8ebcb91ebb650eb47051373cbb736a40b27432d8a70f6757c9747e45.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/905e3bf4d63bc3f432f3f54e37024cc4fbb45d91a1ee812da5f93c3c9046be69.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/a20ac6036b1867f9eae48f5bbc1782026d4aa94e0fc522c5670e12c489c958ad.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/b61fe0168fed99f23151063dd07ef4d5c1ae9f3a92a357b6927ccfead7431d93.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/bacec950082318b6bedc44979af1aea8e0912c90860de29b83f1a849c8d0adbe.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/bbf8db5bcdfdc9bd2ca6f6efd46f90a542b03b3b0fda4567ef3a830b4d2d8bc8.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/c6359f3f8286f659aa38b3ccc5ab60841494e5552c7b68c29570d997f8ae8321.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/cff3665ff0302a7ecee75c97706f09506d7302e051f6efdd2d021aa22c48886d.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/da0f668a3d5967a037372a7e15ce3f6853137e0efbbb3c26ab89f4bb82327d73.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/eade7612341ce1d75d93b992dc26bdc69d0cba6b9c39073fc242fe53e9a110c8.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/f748bc83bf4079595be320ce0b75dec35e8425d482aa984dd51e443010900ff2.jpg)

![images](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/images/fff09b4fe3668fb5c387d5e055dbe5feb1f2ce373abd20cdff3883557abd44d3.jpg)

### Tables

![tables](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/tables/041e64354ef993a5076167a6d435aa65c7f17ccc2f7aab77598a037580891b57.jpg)

![tables](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/tables/24eace11a6c711f57308fa36e1d4e7e2de3a6f500b1f7f3c541d5d0f42d8824b.jpg)

![tables](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/tables/3083e2606b792793131d062b8886dcc20e0fe19e6ebdf54b016039a14b055bc7.jpg)

![tables](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/tables/3948c1aaebc4ce563fc3707353276a63691d79ecc16c10dd870c8d6af41ef5a7.jpg)

![tables](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/tables/4b8dff84aa8096e1f13aff038e644c01fb4ce1feb05407148e1e1ffa6f987edf.jpg)

![tables](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/tables/734a3e538d606aaf7666ef274fa9a5af887c91c67f1eb68ad3bf8017a44b48c6.jpg)

![tables](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/tables/84a0d84fb51cc0f5aec8502043d2506606e5c74dbfbb8fe1ab4b6ecefa14def4.jpg)

![tables](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/tables/ac4d5370ea02daa63a05faa2755a9cdcc34a06909e8504b576a57b47c77d18d7.jpg)

![tables](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/tables/bffa04b8e3f6a8d6e1b244d34e828823cb87a58c2572164508feee1c793f56aa.jpg)

![tables](../iclr_results/2822_e3_Learning_to_Explore_Enables_Extrapolation_of_Test-Time_Compute_for_LLMs/tables/e319ed14b8ba5dc7e50e5ae564af7e155ccee9ba03ba3f0fce855a610945c55a.jpg)

<a id="paper-2823"></a>
## Flower: A Flow-Matching Solver for Inverse Problems

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=QGd34p02mI)

### Images

![images](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/images/0414ba2aa8778d50e5d7539a23be7646f0d9c8d3b21cf926e0489474b7c7b7f2.jpg)

![images](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/images/0bd43775ab5a7ca4549a4cafaea37fc6f6f8a9f07bb9c7a41f81cc322669d6fe.jpg)

![images](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/images/0db9ea686fb6ed6bc5bc1325644160c446d093e733007bf50d6e96ff782bf964.jpg)

![images](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/images/1658ad3027757ec9a9778e06fa888bdbf162391818e6a167b0864d1d37c8097b.jpg)

![images](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/images/21e5ce301c84a729ccb4cb3f400b20a81732b88071986bb46a17a4496b252043.jpg)

![images](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/images/2b22aeb0c3478a817f82db7031d5e9bcb3e6218305c00efa589eea2ac590a35d.jpg)

![images](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/images/3a94acd7d0fd9f5e4b75451b0ad8bcf593a5b0ed62d2acc8e41bfc6e94c84943.jpg)

![images](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/images/52910fd9ab5379f37d03426a8e5bdc11daac8fd56632decb20ac2474046d897e.jpg)

![images](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/images/5524172955b5935258c5efa68ed032ec43d20bd88941cb765326a9733e5ecb54.jpg)

![images](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/images/6171b8a4de784cecd7107872f6b501701fbacc383f55d23327c371c58169fdde.jpg)

![images](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/images/682e760f8791a4ebbe5d784144f951d9e4244321a69289fca0a6bd619f7a7d7c.jpg)

![images](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/images/9900ca28fbe006b10d417171a56878c4ad0e6916eb07708275e9d7b8f1732d3b.jpg)

![images](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/images/a07ad4561d1dd47ef9934106506ab2fcc55ef9c3bc911ce044370b7aa9c9ef6a.jpg)

![images](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/images/a2f1c2306e174b14552f3910628644dcc6ca5792ce023592cd34c8d7d90d55c6.jpg)

![images](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/images/a5c2f4fd782f9e8273134e480d3cd91f2fd1b0955c81b1a70fe400ea9274e2f9.jpg)

![images](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/images/cd441bd6b6ffee7b38ade6c5176bfb2563ff205e3c94dc0639b67c6b36de1098.jpg)

![images](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/images/d11ffc44fff394312911d95b15a56ded924a3eec75654e26e21b5a15053b3601.jpg)

![images](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/images/e83baf672c327bb1d395d5ca53d6c1d0b0f535b30ebd68f92fd6609e423e7bfc.jpg)

### Tables

![tables](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/tables/18be3d7edc3a0b0a3a6504eb8b8de83b17aed20e78174e6283ea2fdd15879d5f.jpg)

![tables](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/tables/2622518a6bb1e06ef64e83951ec2633448343076faa26ba7001aa8a4b2c6f240.jpg)

![tables](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/tables/26245c211fe1e34c137d817394734c3c2a62fd1e044ab1683e16eb4d969de90a.jpg)

![tables](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/tables/27d5d59aa407d9d3edf11d8d7e6400fe0c6efebc94a79f3655a3ca7f1e7e676f.jpg)

![tables](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/tables/2e442ef1ed12341f69bcf66cbee24c864abf1a7744736766e004d1f954169f7c.jpg)

![tables](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/tables/867ab8133dca3cf332dce91e129e6ca79944a49153fbc40a596d17c9de4ea0d5.jpg)

![tables](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/tables/9d1262f04d58cea55650d7279d92f2fe53116758aac4167f62c98f6506704dc9.jpg)

![tables](../iclr_results/2823_Flower_A_Flow-Matching_Solver_for_Inverse_Problems/tables/b7e4d2d8f8edadd227c1f7150d74306f746f23772d09803aecbfba9ae69de2da.jpg)

<a id="paper-2824"></a>
## Dragging with Geometry: From Pixels to Geometry-Guided Image Editing

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=MBiMt3wp8M)

### Images

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/048219b6fa16e947ff51cc4c66d15eac8283476656d2ba801485d4455c5405c6.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/0fee6e231cec31d2a55e3dceded75784ddec82234bcb5bf0f8915637b1b9e93d.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/1f993766bc962753d0ef95545a4022f46c37a0d64b661f105af7a183f58f2a85.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/207352c080aa4a82a79fe52568b2d8ec5dc69e6d9ea4d5a3a57261247c213cb4.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/2477b5fdb27c83464d6326e86c17fde1d781a027a4c7499312fef76b7217cf64.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/2eb2c22aa04aa78822c1eac256b88eac67b31f8b97cc2e9196625190dd6b7804.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/3639ae50d0c3e3bb61c2eee8bf9c640d842ae2e077e8086b137ba4702997b9d2.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/3c9b53755941bc82069e66dcbfbebba50321003662cd74cd4bc7477d3ef71119.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/3cd395181a3db153f940560f582957f4fa9854a6b6af7ff15d6bc7fa26e49764.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/3ee1bd53fc765b641ce32ec5c1c9e6eaa6526081eccc7a97ad3be2059e2fde12.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/420803e44b715ab581f078a01c78d1fad71079c159bb2013c2915b8a137535e5.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/4359f7cfbbf7253448d89c3c8f720cf82cb7417b489a47d9b74120b86e033f31.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/4fcc7f2dd3a76099acf436c6dde0d4f82c3aa4308bf805fce3ff6d3c29ce5aaf.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/66001a9653db386f4b8da4161f534ddb5b794cfefb94e1644e745292172d8b3e.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/71bb466e97d4018222b95040c2b09c8bc89b74fb0bdc30363a465b9b53cc13fc.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/7245873474c6ffd1fc4b61142bf0473fde9ceb887151429fdb58b17321e28362.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/799190d524ed1faab4049f5b4888db881f3b1d7ec4ba13e364ac71dc6b137cb5.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/844cac61719b47b5ea1e7383ea840f93adfdc17a7efef8dca8779401dfee326c.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/9b7b8cc9c441a71e794a7ac2965eedbebd5b1c07784571e73ba495e578531805.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/a93fd09bfc56f54d920304887aaab1354664b6dece408eb10c695f5949c03acb.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/b13846d57c55a821995fcb363892fccdc048c8b56489657776697a427ca266c7.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/cdd92277968bc01ae3f96a1f2f1430ede187d154d3b1588508bbfe7168e094ce.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/e65eaea4a6c9f6275eb421bbcc01dca903b039873e826c7675745abc6c368e18.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/f8b12da8704a6897b685fc2ad0576c2455702db383e9f67644bbb0d0770c5227.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/fbc76c1ea00c8909bbaa1e3e017077d9c30cf1d9c74247aa0e69e2e41669225b.jpg)

![images](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/images/ff833a68dcf4e9bb45cf7c53e0a40af6294de5117c15030c71fa0ca51782d132.jpg)

### Tables

![tables](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/tables/564f933cbb83b8209a76dd9a3dff9c178b80f9a89cce87d12905f2f93e6e36a3.jpg)

![tables](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/tables/74f6d1f80d8c015c23c54f3b5f1c399e9d84152db1e433825f23b2ac2df2cccd.jpg)

![tables](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/tables/8cf7987ff12c7778777c3b68a4158e553642bac7180a586ff3ddaf5ba9aaebff.jpg)

![tables](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/tables/c8f2069a1b2899bb891facbb14db68a9d1e94b3512a523c247c858cd1b152c71.jpg)

![tables](../iclr_results/2824_Dragging_with_Geometry_From_Pixels_to_Geometry-Guided_Image_Editing/tables/f4d49e3fc913c53029fef4ede0c88f35d3f78625421ad1d24265b0acbd027445.jpg)

<a id="paper-2825"></a>
## SAIR: Enabling Deep Learning for Protein-Ligand Interactions with a Synthetic Structural Dataset

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=qgk2F6jxH4)

### Images

![images](../iclr_results/2825_SAIR_Enabling_Deep_Learning_for_Protein-Ligand_Interactions_with_a_Synthetic_Structural_Dataset/images/0f7faefc851e60c1e3854bea74ce312137aa756aef12b2941967fc263ba10107.jpg)

![images](../iclr_results/2825_SAIR_Enabling_Deep_Learning_for_Protein-Ligand_Interactions_with_a_Synthetic_Structural_Dataset/images/4d33c2056de435e06d355eca168c0a5f6c660059946e5a2d0af8be316bdb6831.jpg)

![images](../iclr_results/2825_SAIR_Enabling_Deep_Learning_for_Protein-Ligand_Interactions_with_a_Synthetic_Structural_Dataset/images/5e9247dac88661b8f8fb95e17e00f1ddf658e0e236b441e5e80dfd416b735b4d.jpg)

![images](../iclr_results/2825_SAIR_Enabling_Deep_Learning_for_Protein-Ligand_Interactions_with_a_Synthetic_Structural_Dataset/images/7107f2a5d3da9e4154b821fc4c243a39453dc62cd5464b466e7dcc7633d45d40.jpg)

![images](../iclr_results/2825_SAIR_Enabling_Deep_Learning_for_Protein-Ligand_Interactions_with_a_Synthetic_Structural_Dataset/images/a0413393e74e6b731335161bce37171e6759eeee275d32d7a8bd436ced1bd536.jpg)

![images](../iclr_results/2825_SAIR_Enabling_Deep_Learning_for_Protein-Ligand_Interactions_with_a_Synthetic_Structural_Dataset/images/bb987e517185e38e991ed9974ca7687b4b277f36b3888f4f3d6b0131b06e02fb.jpg)

![images](../iclr_results/2825_SAIR_Enabling_Deep_Learning_for_Protein-Ligand_Interactions_with_a_Synthetic_Structural_Dataset/images/cd22bb23387f18dfd4e5a36eacad761f63f3af36a47e5d4e2c8f438c34a2d54e.jpg)

![images](../iclr_results/2825_SAIR_Enabling_Deep_Learning_for_Protein-Ligand_Interactions_with_a_Synthetic_Structural_Dataset/images/e5a4df39936ba4560c16ced23ba5955a5c0d93f081ded36dfc0950a9909adec0.jpg)

![images](../iclr_results/2825_SAIR_Enabling_Deep_Learning_for_Protein-Ligand_Interactions_with_a_Synthetic_Structural_Dataset/images/fe934340de992f64a7e5d8feab388ddb7fe9ebe9b775b5f6308334c2c2c8514a.jpg)

### Tables

![tables](../iclr_results/2825_SAIR_Enabling_Deep_Learning_for_Protein-Ligand_Interactions_with_a_Synthetic_Structural_Dataset/tables/17a2b778d1d30dc603ffafc48b9592956d5305fe97ebe257956ed4e8f183bb8e.jpg)

![tables](../iclr_results/2825_SAIR_Enabling_Deep_Learning_for_Protein-Ligand_Interactions_with_a_Synthetic_Structural_Dataset/tables/338a02509d9b33593338f223ba816ba27ed77533e81eef03b751bb81a554938d.jpg)

![tables](../iclr_results/2825_SAIR_Enabling_Deep_Learning_for_Protein-Ligand_Interactions_with_a_Synthetic_Structural_Dataset/tables/62d700f2b1074fadc24510a352e70b26ab1548b11fc24bebab47bb0d69e68cd6.jpg)

![tables](../iclr_results/2825_SAIR_Enabling_Deep_Learning_for_Protein-Ligand_Interactions_with_a_Synthetic_Structural_Dataset/tables/74b288eb37a24aa824c79e5ee74bfdc30ee4298f0a723a5dfc32482c4cc26294.jpg)

![tables](../iclr_results/2825_SAIR_Enabling_Deep_Learning_for_Protein-Ligand_Interactions_with_a_Synthetic_Structural_Dataset/tables/9a7295984fb1e368f6afe08c776bd76ee7dc554d02bed07c5a77370de095f6da.jpg)

![tables](../iclr_results/2825_SAIR_Enabling_Deep_Learning_for_Protein-Ligand_Interactions_with_a_Synthetic_Structural_Dataset/tables/bc18b42cd2db8417409d3af3c9521e183c25b00c075435d24936a176c03fa6dd.jpg)

![tables](../iclr_results/2825_SAIR_Enabling_Deep_Learning_for_Protein-Ligand_Interactions_with_a_Synthetic_Structural_Dataset/tables/d4b7650e714af7dd9181da9bfd682660c945954383712e101e5f2f4af2e58737.jpg)

![tables](../iclr_results/2825_SAIR_Enabling_Deep_Learning_for_Protein-Ligand_Interactions_with_a_Synthetic_Structural_Dataset/tables/f9838c11602bf31af05cbec3a2a41e98068bfd09570eafddf0a0ca402aa77e1f.jpg)

<a id="paper-2826"></a>
## GPTailor: Large Language Model Pruning Through Layer Cutting and Stitching

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=yCTpYe3UOL)

### Images

![images](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/images/7c3dbe5e7ffc7fb89255e6781d69a8958d5fefd58febbfcca6af3715ce33fcbe.jpg)

![images](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/images/88213528a775e8761c9d906a090c09ff7ac58c1c508a605725a447450acb1b74.jpg)

![images](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/images/8b8f44428cb7ab35ce01ca4c90bd8a6f5b43fbed38fbca5c0818610cd223fab9.jpg)

![images](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/images/8ca871fd97b6e5851eb87732a693d82c95bfedb52ff85269944640a88ea749e4.jpg)

![images](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/images/91ecc43f1a8e150c658c4e2acd2786382f53dd41be5276628c8051c313aee15e.jpg)

![images](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/images/d2a32f545fd232a63ee878d6a0ebc0f385fdbdd84b56d16731047481544510b5.jpg)

![images](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/images/e381be2e5df417e5e4545c0ecb893f932d7940a794477d14c1e0b5b3edbbd486.jpg)

### Tables

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/0a16c33e1217c37782767186814ffbc060af17a0c6458d3fbff72ace28bdd46d.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/0a6ddec573e96270c237b75c4d325156ddd8469de541e90e9db9786c8032e72a.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/0f436a38041fbbe0f239850dedae3356f36118b1c1e8516e5fa1e90e55a489c8.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/29d34d29cef0d7a2ce4e5d59ad48dd965029e327f105e1a31137647c6404e7bf.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/34df62718935c3a98a415726a4ea57ca59e04dc88beb236c2ec710da342c67b8.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/3ddcc4a4d221ccf0b090780e070a377c956e4b6bcd8c7295ce73057aab0714ae.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/5d564d440f201e1704700b17ac1681b84f2bf554b95688adc634716f2c5d927f.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/625fe8256b78a2f77b977a9b77bfd71539e25973bad91252273706bf4553f6ea.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/72fdf72e1c25211d48e0350002f0c6230096775bfe32fa2833ba18cda23eba47.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/7f646d3b2608882ea3edf2af4b89f98bf511f90b0731a15f62dc66fb6ed65233.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/9f10c01e08a0e99d4d8c6c814bca024f24ded65f2247e193fdb21de5ac1707c9.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/a0adfee10af3b9f0d5db201671218e00d0f1c0b6158cc1c43908f73b94ff788c.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/a11b1cc2905b41ce741204b7efaffc57b2330fc2f6920c1207f733169d6b36b8.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/a32f4a9da405fdba78d3bf7302aaa8ba6f63d68f8c154b549eab391da6b7223c.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/a7f549b6c1baca7bbcacf2a6de39ab3e472b5fc08771e7118579d1c688adf40e.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/ac76979eedd961de7e421d85a84029cd0745c9ec27eb8d2f9e39aa50a9121d6b.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/bc020890b663051f8abda3bd3f7162c6f358c61379f6275c31daf6f751d78e97.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/bc3222679becb5c6c254e6f6314f83f3c06a536e60e52592b5a545e3d3cb8fe4.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/c1f449efe0efea0a256677de33f18cb07da37bf7095d9ca8df09d15c8ebdf2c1.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/c479ed497c1111a33d2dda9c938e1c7e49325b5f35235524055e84aece92f52a.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/ef38d5b89cfea0d1758f6453a53a068ecd3d577eb48615f4aa77dda5cf742b4b.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/f0ebb5b0954cf3a4277331e2021e2dc7271b97ad6b967b84f8c9560395e14e4a.jpg)

![tables](../iclr_results/2826_GPTailor_Large_Language_Model_Pruning_Through_Layer_Cutting_and_Stitching/tables/fdb6e8b3e607cdf1038a14d803360eb2b3eda59ddda7ea425ca74b4ec36427e8.jpg)

<a id="paper-2827"></a>
## TrimR: Verifier-based Training-Free Thinking Trimming for Efficient Test-Time Scaling

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=ofEkphaqg7)

### Images

![images](../iclr_results/2827_TrimR_Verifier-based_Training-Free_Thinking_Trimming_for_Efficient_Test-Time_Scaling/images/1d7d3ab61f877eec24789325777e31666ac194993ebe961e6c9f61c20a0b31f4.jpg)

![images](../iclr_results/2827_TrimR_Verifier-based_Training-Free_Thinking_Trimming_for_Efficient_Test-Time_Scaling/images/28d447327c4e305000f07d9f3f2833469ab581db7d051eadfed5fe156666098f.jpg)

![images](../iclr_results/2827_TrimR_Verifier-based_Training-Free_Thinking_Trimming_for_Efficient_Test-Time_Scaling/images/8c95395121494c361f5b007891244fab323d54b8dc9c4a6718d7c4c4fb23007f.jpg)

![images](../iclr_results/2827_TrimR_Verifier-based_Training-Free_Thinking_Trimming_for_Efficient_Test-Time_Scaling/images/a90a82f85f84ae4d0580b08d1435473eff0de7052246db28db774ef11afce559.jpg)

![images](../iclr_results/2827_TrimR_Verifier-based_Training-Free_Thinking_Trimming_for_Efficient_Test-Time_Scaling/images/bac4e994a528ceeece02be37e0d5f15d2d6552c91f25e1c59e28ca1a60576031.jpg)

![images](../iclr_results/2827_TrimR_Verifier-based_Training-Free_Thinking_Trimming_for_Efficient_Test-Time_Scaling/images/be7479fade110c007870c1563bb5c9e6c90713e70c9b946eba9b6f88c48173ca.jpg)

![images](../iclr_results/2827_TrimR_Verifier-based_Training-Free_Thinking_Trimming_for_Efficient_Test-Time_Scaling/images/dc0244a1d71ab82ef0b334c82f5fec5a9c7f22f13fcef5da22332c65111d366d.jpg)

![images](../iclr_results/2827_TrimR_Verifier-based_Training-Free_Thinking_Trimming_for_Efficient_Test-Time_Scaling/images/de47f31ba4411170ebe724ca26025b46598bfa91eb58f2b7fb2bc02204d01ac2.jpg)

![images](../iclr_results/2827_TrimR_Verifier-based_Training-Free_Thinking_Trimming_for_Efficient_Test-Time_Scaling/images/fc2c03a20bc1994d8e39dc1d25b61125f43e684087d528e9d2185b42e44dcd33.jpg)

### Tables

![tables](../iclr_results/2827_TrimR_Verifier-based_Training-Free_Thinking_Trimming_for_Efficient_Test-Time_Scaling/tables/316d79782046fbf5ffbd5ab7bf8cdf8ce105335a3c4e150e62af559ed97376dd.jpg)

![tables](../iclr_results/2827_TrimR_Verifier-based_Training-Free_Thinking_Trimming_for_Efficient_Test-Time_Scaling/tables/42a1ad4064958d0e6c9a43ca36c74057bc09f217b3d195ced6b193805f5cb310.jpg)

![tables](../iclr_results/2827_TrimR_Verifier-based_Training-Free_Thinking_Trimming_for_Efficient_Test-Time_Scaling/tables/48f99f828aef8f2661f4d101388b4dd5ee6a98ee708c909b548dc772455fb69c.jpg)

![tables](../iclr_results/2827_TrimR_Verifier-based_Training-Free_Thinking_Trimming_for_Efficient_Test-Time_Scaling/tables/494330a5d9dda35f62729c433431d2e33432d2b5050be061bfc3ce8ba6acca52.jpg)

![tables](../iclr_results/2827_TrimR_Verifier-based_Training-Free_Thinking_Trimming_for_Efficient_Test-Time_Scaling/tables/8dfb83c0e38e76b4d1076016e518c1f8d5cbfe9eecc717dc9ab30b4f03be45b7.jpg)

![tables](../iclr_results/2827_TrimR_Verifier-based_Training-Free_Thinking_Trimming_for_Efficient_Test-Time_Scaling/tables/a7e775a1b72f56c6a5b898b0ee91f0b90c2dce3fbfe0530e5a9d2d4555ede4fe.jpg)

![tables](../iclr_results/2827_TrimR_Verifier-based_Training-Free_Thinking_Trimming_for_Efficient_Test-Time_Scaling/tables/b71926b579527922555735421640150008dae20c3a37c40b224adcab383dc185.jpg)

![tables](../iclr_results/2827_TrimR_Verifier-based_Training-Free_Thinking_Trimming_for_Efficient_Test-Time_Scaling/tables/bf171e8257270f2921e6c71eca587a2e5fc5ef1a4eaa69303ba854d1d8c98e43.jpg)

![tables](../iclr_results/2827_TrimR_Verifier-based_Training-Free_Thinking_Trimming_for_Efficient_Test-Time_Scaling/tables/ca47016ca632b26a9f46d0f9a9f7f9500283b63c7dc31ea8b1e009a19b5d5738.jpg)

![tables](../iclr_results/2827_TrimR_Verifier-based_Training-Free_Thinking_Trimming_for_Efficient_Test-Time_Scaling/tables/cfa6703f5d69ee6e812ff676f1d40b1e6b9845d475536e54c085546f74f7381e.jpg)

<a id="paper-2828"></a>
## A Federated Generalized Expectation-Maximization Algorithm for Mixture Models with an Unknown Number of Components

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=CELYMnherN)

### Images

![images](../iclr_results/2828_A_Federated_Generalized_Expectation-Maximization_Algorithm_for_Mixture_Models_with_an_Unknown_Number/images/1d2e1cc9420198a8e6bee3012fe6baa1025fe908f85cc675e6f795f00fdbfd03.jpg)

![images](../iclr_results/2828_A_Federated_Generalized_Expectation-Maximization_Algorithm_for_Mixture_Models_with_an_Unknown_Number/images/32a8d0e80ebcdb95bb222227917e8fe812ff115e75aba3e2a789ee36c23341f3.jpg)

![images](../iclr_results/2828_A_Federated_Generalized_Expectation-Maximization_Algorithm_for_Mixture_Models_with_an_Unknown_Number/images/6cee75adb0dd2f0538a4b4cc3ab2ecec3652c52aa722b0c41c255813a28d8774.jpg)

![images](../iclr_results/2828_A_Federated_Generalized_Expectation-Maximization_Algorithm_for_Mixture_Models_with_an_Unknown_Number/images/77279633ffdf6db77b2b528091c9c833f0467f67481882eb8155925bfa688462.jpg)

![images](../iclr_results/2828_A_Federated_Generalized_Expectation-Maximization_Algorithm_for_Mixture_Models_with_an_Unknown_Number/images/fd49ecbc327ac282563c5abface6d52ff3f2d65e73ee45c476a8837d3521334d.jpg)

### Tables

![tables](../iclr_results/2828_A_Federated_Generalized_Expectation-Maximization_Algorithm_for_Mixture_Models_with_an_Unknown_Number/tables/5119accb3d6571633f29630b47f9844273576a8e57326a74b552cf3111d00f80.jpg)

![tables](../iclr_results/2828_A_Federated_Generalized_Expectation-Maximization_Algorithm_for_Mixture_Models_with_an_Unknown_Number/tables/8496a854d92b481dd85a35094b53e57771b31ce7a1bc420ca9f6348dfb7cc4a4.jpg)

![tables](../iclr_results/2828_A_Federated_Generalized_Expectation-Maximization_Algorithm_for_Mixture_Models_with_an_Unknown_Number/tables/928472835fd7febb2ef513c8b25e1fb6d11376a7fddcfb43a386ef50b0896c36.jpg)

![tables](../iclr_results/2828_A_Federated_Generalized_Expectation-Maximization_Algorithm_for_Mixture_Models_with_an_Unknown_Number/tables/b97a109c93605232f7ec533ec8f1ad7fb0bcc488ff96a34d2605f14bb0e76f39.jpg)

![tables](../iclr_results/2828_A_Federated_Generalized_Expectation-Maximization_Algorithm_for_Mixture_Models_with_an_Unknown_Number/tables/c373a04798f739ca1625461caaf4c4e58c5641f1e7190f1526f94a35af4ba269.jpg)

![tables](../iclr_results/2828_A_Federated_Generalized_Expectation-Maximization_Algorithm_for_Mixture_Models_with_an_Unknown_Number/tables/d80a901a2963a7c9e8a7119234f85b97cc6567172484b1cbd4c4d6d84de1c939.jpg)

![tables](../iclr_results/2828_A_Federated_Generalized_Expectation-Maximization_Algorithm_for_Mixture_Models_with_an_Unknown_Number/tables/f8f5c233bb7c1840f65d91c5a25686526ce43b7212786e507d8fe053b35dc804.jpg)

<a id="paper-2829"></a>
## From Data Statistics to Feature Geometry: How Correlations Shape Superposition

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=7akSRQS5Xh)

### Images

![images](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/images/0b26e2f61460e44ab6e4d86905b4843a70109e588fd2be2214dba81d16c85704.jpg)

![images](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/images/0b3ae649d45fca0ebb3834b53853a81bda6e48816589c69a2728a9cee3749e93.jpg)

![images](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/images/0cb009b4d880779dc667372b9be2a83817c39353f029d5c397626959ea7c36e1.jpg)

![images](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/images/32a137971566e68623a648ef3967eb6016685ee43ae76e563a3135bec79e1ac4.jpg)

![images](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/images/3c501167ee2c20ce3b329728f95f04bc0a6858ab842b31df92c62c9fc9395eef.jpg)

![images](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/images/44b4cc3224223feb5785e88b87c69774c469dde3041ffae12706afc380821ee9.jpg)

![images](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/images/52d535d46c94b5e42eacf979637abb8ef1eba42836e2f883a5fe4926c50e854b.jpg)

![images](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/images/52de0ba61be43ecdeb45661eea22d3ea122cdafbbaf06c381cb85ab62bb3bc43.jpg)

![images](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/images/72ea3ab8ff0b506b23f0a942f0eee217b8a2d04884eff3b27a32062490ce5acb.jpg)

![images](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/images/9e07b8bceebb25c1a0124f4cfa1efcbdb25be2796b4bb7bd782a96a36b651c3d.jpg)

![images](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/images/af9fb2cc09d5bdd1d90f70bdaedcd0745f39522f17f8db129310e31ba25f76fb.jpg)

![images](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/images/beffd23399a965a22951ab14db58912204f0ae0f089e39bf5cc6c9711dcd7011.jpg)

![images](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/images/c3bab8486cfb74868f078e2ac9d17168aa532541811cbf88b7d54fbec02c6cee.jpg)

![images](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/images/c85a82665b1a7bb5e0d838e4d8a23f3ca77c33240a4ed1c086c0318c7439eaf8.jpg)

![images](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/images/c86be51b342e92bf820cc094a4fbd1b463081ef142b6bdf8922c2f20e2c17e49.jpg)

![images](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/images/d6419736da38b34cbb6aea5dc30471843ad78b4aec8a698f2f5d59b9c7d5ee88.jpg)

![images](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/images/db2f6453d3e57057ce611a2e61e450ae25e73471caff5e63e644a68ac208be97.jpg)

![images](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/images/ebd43876092226f309c2db7440548e7ddc0e23a8a642359b7cfc7d143473f530.jpg)

![images](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/images/f65ae4aef145b8f2431e9be6df24f9f36ab233ac43889eaaa6fad99cf20f7755.jpg)

### Tables

![tables](../iclr_results/2829_From_Data_Statistics_to_Feature_Geometry_How_Correlations_Shape_Superposition/tables/d441d3f60b3446c865efd892a264a54083e5fb2a71654998a62ce206fafc024b.jpg)

<a id="paper-2830"></a>
## HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=H1KDMNOKQn)

### Images

![images](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/images/172a5ecf247c2f888b1d46ed971d1f056c653295e606690b2a725d8b478afb7f.jpg)

![images](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/images/4f595c1ddf4c7f5c14705bd2654505761bac24579f14c315498ebad56a377266.jpg)

![images](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/images/66c63c4c620e281f86d8b569eb21014f3271786bf73c8fe74872d2137a346816.jpg)

![images](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/images/886aa88aa8ee624d9afe38430f303af34ea965807b726cc83e7c8704bd432b2e.jpg)

![images](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/images/98cdef6dd4f05e40fc0110afc425ef748682102fbcf712652d9f030685ed6c0e.jpg)

![images](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/images/9d546e7447bfe480c205bc4749f937aec03437408bcfabd9533ae66c6bd09c17.jpg)

![images](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/images/9f72a7e6a3c4c816668fd2b16ea74fad55cfd8a40c21d0cdaca9a47697e7eb23.jpg)

![images](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/images/bea128dc2e07a40e0f76cd8ffe387f6382fe63853f7a5af848989785f464ed39.jpg)

![images](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/images/c91bc3e752812084770e6396960162aac3403b81e9cb3ebec67d9965809343bc.jpg)

![images](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/images/cfc4a163f7c9387ce9940b0382c7537ead061e47260aee73e4608377863711e4.jpg)

![images](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/images/f6989ac571f7d01a3e02cf1d9a00126c91bff73e0c719e3d1a8bd751172e1b14.jpg)

### Tables

![tables](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/tables/19350141a3266fa86168ec31499c907bb7a739b17ac83f47fdb80e02c21a6827.jpg)

![tables](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/tables/21db5dc6de50005992be8215f4b7a5f1e0f130dc6f8907334f42fbe88178e90c.jpg)

![tables](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/tables/5c15d814744d2cee76de13690dc1a7aa5cff07539252a8835297d7bc9d3cae81.jpg)

![tables](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/tables/7285389f4e77b079a8a077d784f2ac080db1d2534568a784bf8a41c36a601057.jpg)

![tables](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/tables/7d1c7c61197640f8fe09f4691b4cf2e895b63f5e7106b2deab874023312392e5.jpg)

![tables](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/tables/80dd3df11e51e4b909e75ecffd22dbd82be044b77c91162b3a7a487d0165490b.jpg)

![tables](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/tables/91cb06a53e5f4474fbef35fff68053da45038470b70fd2d975fe75c9c1e1ba3b.jpg)

![tables](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/tables/9c74c024ef9425ae50e92c0df46804b01f2727e06ef71d06f1a58e5bdcc796b7.jpg)

![tables](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/tables/d2a4b0e3b68a77d4e18f3dcad804cce60de3ea86857f17ddd5238b103477fc06.jpg)

![tables](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/tables/db9842fe5c945a7f24703c059f3100bf20c4b9dd072ac88fba1422892a0903cf.jpg)

![tables](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/tables/e00d9c05e49e754f70747cfb80e6128520d172e764a1530e24ccc0a86b6da28e.jpg)

![tables](../iclr_results/2830_HybridVLA_Collaborative_Diffusion_and_Autoregression_in_a_Unified_Vision-Language-Action_Model/tables/fec32792eb90f9b84260742926d3dcf9e843102cab066aba300c2a2ba5ee3dd1.jpg)

<a id="paper-2831"></a>
## Strictly Constrained Generative Modeling via Split Augmented Langevin Sampling

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=aDJcWNmfce)

### Images

![images](../iclr_results/2831_Strictly_Constrained_Generative_Modeling_via_Split_Augmented_Langevin_Sampling/images/0e4065d3a618ead6b7aa5dac66f9fb409c9d05f5d2ed0c3bfc29b6d4dfa0d1a7.jpg)

![images](../iclr_results/2831_Strictly_Constrained_Generative_Modeling_via_Split_Augmented_Langevin_Sampling/images/302b712af5e21c1259635ad608cc56488e5d65d8aecb54f5eda4fe768d28d45f.jpg)

![images](../iclr_results/2831_Strictly_Constrained_Generative_Modeling_via_Split_Augmented_Langevin_Sampling/images/8a9ab2d00bd54b606e79b7a28e3e48d8a25616c8b91b8e2705ff0001257bbce9.jpg)

![images](../iclr_results/2831_Strictly_Constrained_Generative_Modeling_via_Split_Augmented_Langevin_Sampling/images/8e8d6f46acfa1c3bcfec85dfc2a89f0eeae37cc6b53220ee732fbd99f18d06a8.jpg)

![images](../iclr_results/2831_Strictly_Constrained_Generative_Modeling_via_Split_Augmented_Langevin_Sampling/images/8f6de17bc24b6b41c66428a6de60bd98173feb2b5b06ac170f2b76d61ec27256.jpg)

![images](../iclr_results/2831_Strictly_Constrained_Generative_Modeling_via_Split_Augmented_Langevin_Sampling/images/ec2bde95f565e1fc140c78578c806b622ccaf004c112db5c0a63e0a727e8821e.jpg)

### Tables

![tables](../iclr_results/2831_Strictly_Constrained_Generative_Modeling_via_Split_Augmented_Langevin_Sampling/tables/0e24f9283656a701bd0f445f9d272bc2fdae20bcdc8a1bc91d23c348b543bff9.jpg)

![tables](../iclr_results/2831_Strictly_Constrained_Generative_Modeling_via_Split_Augmented_Langevin_Sampling/tables/56e41afe8afbca89a89504d9eb488b3e6a1ca5411b474af39c0ccaef8a36fe60.jpg)

![tables](../iclr_results/2831_Strictly_Constrained_Generative_Modeling_via_Split_Augmented_Langevin_Sampling/tables/8c0c6e308cef4bb69f0368b307d4fde1d3b3efd0225b82dc3b2bc088402866c2.jpg)

![tables](../iclr_results/2831_Strictly_Constrained_Generative_Modeling_via_Split_Augmented_Langevin_Sampling/tables/911a59df3b287453ff5d5c2d3c15c4e707c8bd76cd3ca4e37e4b92758a486c8e.jpg)

![tables](../iclr_results/2831_Strictly_Constrained_Generative_Modeling_via_Split_Augmented_Langevin_Sampling/tables/cd0f0bfc687958251336000a7aeef4c02928688ec5d5ee043a5f07d07a76234f.jpg)

![tables](../iclr_results/2831_Strictly_Constrained_Generative_Modeling_via_Split_Augmented_Langevin_Sampling/tables/d0a2389ca59e8fcd768cca7597f02ffcdac778b97301c28667e2c86e596ca87b.jpg)

![tables](../iclr_results/2831_Strictly_Constrained_Generative_Modeling_via_Split_Augmented_Langevin_Sampling/tables/d90b77724c9daafb0fdf387db7be7d52eb6f02329b77d5e95bb37e050b3f76a3.jpg)

![tables](../iclr_results/2831_Strictly_Constrained_Generative_Modeling_via_Split_Augmented_Langevin_Sampling/tables/ee345e0c11bd3aeefc41af7bb4587febc2a3a415aba3e2e729854c9863ab619e.jpg)

<a id="paper-2832"></a>
## Generalizable Heuristic Generation Through LLMs with Meta-Optimization

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=tIQZ7pVN6S)

### Images

![images](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/images/1a37b66ed16e04b8e733c0d478d644ba496a1fd5ca9369f7a07b44ac33c01104.jpg)

![images](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/images/2d61b96e25bfe45598f8734b86f80ecdf3c1d88fddb9f42dd62d1b410d2ec253.jpg)

![images](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/images/4251427520b25b564f6ac91ad3a7fdc83111a5938b3f1fa8f4736b53745eeaec.jpg)

![images](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/images/53ee7497291194eceace2b54679299716eb5f3250d7fa0b66a561c1a31e18d7b.jpg)

![images](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/images/9672e1c1fc17f4ed478adf719586b8f16b0374de1f9e7938c911901362e026d3.jpg)

![images](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/images/c863ee33709256b944a6e11a5f769c28a20557619632f967b4ef2a032f10b251.jpg)

![images](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/images/cca7c8b3a01400f3ecc2af9f3092b89684563709a3d31c1c42c7d25d76f27832.jpg)

![images](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/images/d0793deb28f618e27ba8e66f4430071d406dc44f68b59f9ef7b63d7d0d50295a.jpg)

![images](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/images/dc218b57c29756b1507d4cecb9ce24e451dfbf826f09dc57d5f2b46b662f782f.jpg)

![images](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/images/e5be97b36400a3341b0ca90d548b2140aee0cdf3220ebb46cdc822685473296c.jpg)

![images](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/images/fac8ca052eed8c484ae2768d503c1d5356e5182e06c00f952b69a6583483a602.jpg)

![images](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/images/faddba65f9af5799352d7485b748651b8b3499ef6795b282f649c5e978ffd955.jpg)

### Tables

![tables](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/tables/0d5bd8e5f949aafbf7970c23facb35e0c5ce3f3ea194ac823e6d4f487c19c5cd.jpg)

![tables](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/tables/0e84b877a3d964dee47b4645e05edac4da163896ab916303fb8d77153da356dc.jpg)

![tables](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/tables/1d4597f7f39a6ced48184cbe11db03a15e5d0cb3ec6bceea7e56d3f217e563ca.jpg)

![tables](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/tables/21397ac10b0af2d1d344e9a128f21d07718d851bb629e75efbbe3669021be27e.jpg)

![tables](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/tables/251b27e467f96771079768960e8efd1cadf0f27707ff08b04ffce71ab2a1f26e.jpg)

![tables](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/tables/2aa7efa4acf075a8a661fa912a0dbbb3ba15a28d0a8ad819c07e0e604fc43f74.jpg)

![tables](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/tables/4eea4e333254eb6817f6e4b85a7e9b53314f4b8b85157e67f8d81b45f18f1c8f.jpg)

![tables](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/tables/6659e115a4612f4ecc9c8120e076d5a7c43b330319632b268434ef18e004fbf5.jpg)

![tables](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/tables/7d84d2caefd11a2417ad64462b94089691a1007d5c9385aad95bcdbab9015137.jpg)

![tables](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/tables/7f23eec15a69f073c196ffc02b21c4f0bfba54e341dc7e423db60cec4a305d77.jpg)

![tables](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/tables/8b1f7a8d64c33992497214118c3aced85404c2f6343b3ef9bfb278fe2c93bd64.jpg)

![tables](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/tables/8c2bec0b4cb439695ea1dfebae1b29f697d0f4d0a3705b1d19ee83bb54b3d6bf.jpg)

![tables](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/tables/aabf8996d73a8a90dcd42f524184f8e362b90b317de75acf37271e7cdab8944f.jpg)

![tables](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/tables/b3bd983c4937f236b97ab725ebfb376f7248629c32b16303a99605e40a6948f2.jpg)

![tables](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/tables/c044ea98fe24281d038de418680ebb60934d7db55f8fcd1dd04ea8a30d903621.jpg)

![tables](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/tables/d5ea8b89c3f04f0245490091538cc7e01e6c190279fecf8b590c1811e7c8c17e.jpg)

![tables](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/tables/e36a97bfbdd3359f4c81b19bbcad4b3beb767320141624ab29f950ed39247fda.jpg)

![tables](../iclr_results/2832_Generalizable_Heuristic_Generation_Through_LLMs_with_Meta-Optimization/tables/e8e021e132a0e2cae4b0bb370092bf48ede19bbb2590b1f6b5227e5326eab3d8.jpg)

<a id="paper-2833"></a>
## Provable and Practical In-Context Policy Optimization for Self-Improvement

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=TAthdtPe7k)

### Images

![images](../iclr_results/2833_Provable_and_Practical_In-Context_Policy_Optimization_for_Self-Improvement/images/1a1e9d59b7df54b3c6e4abcd8133f0098726b1cc659cb2fbe93758d4ceadd3e1.jpg)

![images](../iclr_results/2833_Provable_and_Practical_In-Context_Policy_Optimization_for_Self-Improvement/images/83ea4afd8940af8285ac3e72ec36d3cfb76048881e55d9b9c67a3fb3deb51b03.jpg)

![images](../iclr_results/2833_Provable_and_Practical_In-Context_Policy_Optimization_for_Self-Improvement/images/b5e0ba6f45024db5761523daa29871a2e3c043bb0c248712d944306ca3450b29.jpg)

![images](../iclr_results/2833_Provable_and_Practical_In-Context_Policy_Optimization_for_Self-Improvement/images/fa911e7b4ca9822315b3c23b7d789a36147e7651c4e2d76fc2eb4b99ef6270b0.jpg)

### Tables

![tables](../iclr_results/2833_Provable_and_Practical_In-Context_Policy_Optimization_for_Self-Improvement/tables/1f05ab577e3bcf0a7753c25fb81f4c56ba230ca0e3c8cb9f6109f1e7e05d2187.jpg)

![tables](../iclr_results/2833_Provable_and_Practical_In-Context_Policy_Optimization_for_Self-Improvement/tables/40dddeaf73ff4879f14babecba7efc1c2a5bdb4bec6e31e08abe648441f80447.jpg)

![tables](../iclr_results/2833_Provable_and_Practical_In-Context_Policy_Optimization_for_Self-Improvement/tables/6c06a98559ff5a75604e90c0e8c0b039eee62df194b5fb3853f21558129f5bb5.jpg)

![tables](../iclr_results/2833_Provable_and_Practical_In-Context_Policy_Optimization_for_Self-Improvement/tables/9bbc17a70e801a442ae56791c2eafee022a2a1d92a90545c7c37e8c7a2d32c2d.jpg)

![tables](../iclr_results/2833_Provable_and_Practical_In-Context_Policy_Optimization_for_Self-Improvement/tables/b06c2fb4ea7679b1624b6f39e3a6c637f6489afd31a4d8f6c416ef231932768d.jpg)

![tables](../iclr_results/2833_Provable_and_Practical_In-Context_Policy_Optimization_for_Self-Improvement/tables/bf12e85723d7a04e176e50aeca8083824a3bc50c766d368cb4657be2a500c8f9.jpg)

![tables](../iclr_results/2833_Provable_and_Practical_In-Context_Policy_Optimization_for_Self-Improvement/tables/e31618fe74ec1c53fe19d21411a491e489e6001a6fcbd80887ffd1b5d1dd00c0.jpg)

![tables](../iclr_results/2833_Provable_and_Practical_In-Context_Policy_Optimization_for_Self-Improvement/tables/eb7a44af104c0735f3e9195e5297247062fa7919cfb3ea84109f62c055dfee1d.jpg)

<a id="paper-2834"></a>
## DreamCS: Geometry-Aware Text-to-3D Generation with Unpaired 3D Reward Supervision

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=PoU33ZdtCP)

### Images

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/005ed3b5aac1831c6065155c545d2f907b23dc548ba7b45919c510e955982566.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/11bd5ae89f5a8bfd4345edfd427037fc182a2cc25409d9d9a3458c837078c91a.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/167b2f467085419735c3ae83b884432d475ad3cca4f8d85258f130b383c926a4.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/24f32ab84c87cf7d2aa99e92061f961d0839d7f9ecc8c17ec222639fcc271528.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/2c295787bb739925b719f51e21e63b3d58c982d401550346c16cd8bc2703d2a2.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/341c761050b097d2b03e49b48e50deb99e4ec3a455cc5ec31ffdfdfac23f65f6.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/3d03d398f92616ef36b095b386e62130202911af4177b7716c4845b35b2fdcee.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/44212a04cef1992bdb807dc4cdc98a67f78be51ecbb1e110a46b297a44874812.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/53142012a82a9bdd530b9499b0826407f08b0bc579ebf5728d775662b1bfd501.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/53d5f804f70a1715af536b5405e9c00050f511bafe3bdc3adfbc79507b0bc6c4.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/579232a5f46eea45b5c3d305ae24f881f313fc7b4776eb765d892a94898fd7ad.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/591d0c38a7f751c83f68f17f376ac1eb6ca5ee1ad2df64d205b0b73c2cdcbcc4.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/5e21c38a2e93aae6e871c702705f063cd9c3720dcfda7c7ce3153c7ec6ca3d8f.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/60f7014c189860a5261fa17727781e834af052428c647d6c6b6402304b33022c.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/6c6a342addd28bd166011bcd065f57e1b6693f676a46a31eed2e3a8d6a76fb8f.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/6ca0c5690d7a6db367c1c98c4d8ad5c2a2dcafda47226605e0e809f304674848.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/6d3d79174852265ce92fdee385554c4ac498233a2326d50ff64b81d18c263e3e.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/70034e9ad308e0f038c3a4f34dafc776a5960b7cb80df83176998bba06b27fb8.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/724d5762b83e24fddfa62c8cc2e0eed5ecdb2fdfd63df75919d9b37d096b8971.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/79e72a6c571b4e688cc989f994891e39e07f9ac6fbce80140ba6a8c825235d8d.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/8f8d5b9aa88bbac701addeef6aeab5d8857b01cac6d57ae2b820729bb3485be7.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/90d9163ed5be3435236f6502d555de444bef858c4fe9e2960dcc5544bb3c5206.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/929ca1b31c7ed42b552a121f362f8675d38f248ccb6abbf80cd57a0c1c1f49c0.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/9e3ef00e70a0c0da2ac1735f3a23e42c76681ce6858e4d67d19d646347f35cac.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/9eb871f1ba0aa10142f19ca73c0f8b19c4df922e4fb575a767c09ae3d8d4e2b0.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/a42a1a984e774878d9113fb4e94798c6fbc7bf4062eb31a9315e2c854689b497.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/b170f3e2a7e35646087c3bdbd183304fdacebca1b30d8e06a7e118fb9ec20a60.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/ba22d7c88f95ea944b01378b6f06422e8ee2d5ded1650a30aebad6b09184b3e6.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/bab824e562ad384cf6beff9839f6788618d6d69cd1305f426ec87cd214d389e7.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/cb44cc56795f5293aa23dde0663fbbf0fe8f97ab1a3ccf2025014c5a460c7331.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/cdc27c2e2e46a7e64ad30f7ca3907e5a74cce1d6229002610f02282441851869.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/cfcb613f4f9b99749bedd286927a4fa12bfb9d1c5d9f2ffb3e4d94a394342236.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/ed6d96336297279cf77a18e214c99e12385f30f7578ff280f17e9147528cbebc.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/f47764484d8dc948034ec52bd1017c1115bcb7f16cee658ebba5bde09f4d225c.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/fce808946f84ecbcaf859ff586ce6f7c7774667bf62827e12cc514cf3bd241a5.jpg)

![images](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/images/fec7ca6855eea76cc41469520ac98fc97caac84f10ca0e7b59e3402741ac9be7.jpg)

### Tables

![tables](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/tables/07ba0c88fab17314af85d2b21b7313be6cdf63878644b3b488db73323b697ad9.jpg)

![tables](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/tables/10186a0427cb7f05c3224e29804a38eda13ecff00eee8262de9e2edc3d6098fb.jpg)

![tables](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/tables/13768936e3b49d9a7a70309079544ea43d87f6702d24a166a5ec9e0bc71c08f6.jpg)

![tables](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/tables/2d7427ae53e23ea8031896922f4cbdab96c8980b4b6d40e815b23b0b0077c91c.jpg)

![tables](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/tables/36f99048fb121926f8be143dca3fd04791fb7be467cf5d855cc50ef097c8a72e.jpg)

![tables](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/tables/3d4c822a76f8a0525237739565ccb3e10faf8fcf2a6d77e217d0d723ba8d761f.jpg)

![tables](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/tables/3ed71c18b77629081b5efba4511927017382141ba9609cdc7b5f3109af6dce1e.jpg)

![tables](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/tables/78443fefb6ce073059f62ce67db1d51b728e733c6e7c7768c5234e5ecf184f65.jpg)

![tables](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/tables/7a491fdfa6a0e5ddd09cd7e559aba513cac79db45c59a9d2cd75758a295839c6.jpg)

![tables](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/tables/bae03f397bb41097e3497dcf50c00d7bec94bf75ca090b63caaa308879f14315.jpg)

![tables](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/tables/df4e5d002ce089a2ed2d765cad8bc27e8ead8d20b51a6ce0a0a15535d14efcc4.jpg)

![tables](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/tables/e626b8a816ba9a128ab5535456b66d7a914eb109bbf7852fabc0fd89cf4f57af.jpg)

![tables](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/tables/ed439f39e36fbe86bcd4711b6fc536ace4d76c625496a47539e4a822cfe2ceca.jpg)

![tables](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/tables/f6b9f6b66ce25b2bf8ff679e7d6832c4a928c12f33e9be41b180cac3c6021860.jpg)

![tables](../iclr_results/2834_DreamCS_Geometry-Aware_Text-to-3D_Generation_with_Unpaired_3D_Reward_Supervision/tables/fa4aeb2ad3207363a1b30a7e0a47b6a6d4d88a1ca26b9690c8db990751d99137.jpg)

<a id="paper-2835"></a>
## Revisiting Group Relative Policy Optimization: Insights into On-Policy and Off-Policy Training

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=XXfOf22o3K)

### Images

![images](../iclr_results/2835_Revisiting_Group_Relative_Policy_Optimization_Insights_into_On-Policy_and_Off-Policy_Training/images/158d785bacc71d2004f2324ec68d7b954d38c917904f8e340d67efa50fce45f0.jpg)

![images](../iclr_results/2835_Revisiting_Group_Relative_Policy_Optimization_Insights_into_On-Policy_and_Off-Policy_Training/images/6ea42a4b44abc30678f0290a0e6c6629a5c1249e986f9bf779f831baad79e54d.jpg)

![images](../iclr_results/2835_Revisiting_Group_Relative_Policy_Optimization_Insights_into_On-Policy_and_Off-Policy_Training/images/b754f21bf36007ada87b4102d8c2c1052b33f6b4f4e18c4f6bf5aea8acc735da.jpg)

![images](../iclr_results/2835_Revisiting_Group_Relative_Policy_Optimization_Insights_into_On-Policy_and_Off-Policy_Training/images/d8d4e1535a49fa3727d51037e4bea5ba607c2a0b3a282eb82c581b6aed6589f5.jpg)

![images](../iclr_results/2835_Revisiting_Group_Relative_Policy_Optimization_Insights_into_On-Policy_and_Off-Policy_Training/images/f65580875aca4ef146e765a32e5f2de0069f9e68d6bcc0113f53be14ae5855d0.jpg)

![images](../iclr_results/2835_Revisiting_Group_Relative_Policy_Optimization_Insights_into_On-Policy_and_Off-Policy_Training/images/fbfed8e75bd775f4ef84a52371cf145994d0fa2f435aa69e39986124994f62b3.jpg)

![images](../iclr_results/2835_Revisiting_Group_Relative_Policy_Optimization_Insights_into_On-Policy_and_Off-Policy_Training/images/fc6c3ade048f3ad05214582397ef8e28f1ef1f92aa074975038e88c78fcf6e41.jpg)

### Tables

![tables](../iclr_results/2835_Revisiting_Group_Relative_Policy_Optimization_Insights_into_On-Policy_and_Off-Policy_Training/tables/11a24f73c687cbc5d28992cdbb6a0d95eb669523a1e575950dc908e6ec0c70c2.jpg)

![tables](../iclr_results/2835_Revisiting_Group_Relative_Policy_Optimization_Insights_into_On-Policy_and_Off-Policy_Training/tables/2913e786e53ecdf5a031316d7de202c94c96048fed89a26eb1794b5e5ce4f06f.jpg)

![tables](../iclr_results/2835_Revisiting_Group_Relative_Policy_Optimization_Insights_into_On-Policy_and_Off-Policy_Training/tables/80c241e5f356f625f564ea2c550cbe9ce71e791a18fe2c4a7e6ec8f0a341c396.jpg)

<a id="paper-2836"></a>
## Beyond Speedup - Utilizing KV Cache for Sampling and Reasoning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=GUhmiJaAzv)

### Images

![images](../iclr_results/2836_Beyond_Speedup_-_Utilizing_KV_Cache_for_Sampling_and_Reasoning/images/358695363b85d697df5562d41d963293b5aebff4b908ac8a3170bc9b279e0483.jpg)

![images](../iclr_results/2836_Beyond_Speedup_-_Utilizing_KV_Cache_for_Sampling_and_Reasoning/images/79b75a99c1de7f8a7f5a1f1f69d09e3b61a9f0549799732359a16c01b617b679.jpg)

![images](../iclr_results/2836_Beyond_Speedup_-_Utilizing_KV_Cache_for_Sampling_and_Reasoning/images/c4adfb991c99f9bc61010be67cbb22092451af5de0ebe03a6015b3eb600747ca.jpg)

![images](../iclr_results/2836_Beyond_Speedup_-_Utilizing_KV_Cache_for_Sampling_and_Reasoning/images/d15a74653c12fbd303d44bd9253a608b6827d859b1bd51d63240749fd8c6109f.jpg)

![images](../iclr_results/2836_Beyond_Speedup_-_Utilizing_KV_Cache_for_Sampling_and_Reasoning/images/fdcc23f5349f43b341ab80eb3276765390715f9fc8e4ef4303a64ff9f9a26454.jpg)

### Tables

![tables](../iclr_results/2836_Beyond_Speedup_-_Utilizing_KV_Cache_for_Sampling_and_Reasoning/tables/0fdc1c5af2b989bf00fd6d6688f8b7db2dedf3cbda4199d7bdefd2e9eed22c2f.jpg)

![tables](../iclr_results/2836_Beyond_Speedup_-_Utilizing_KV_Cache_for_Sampling_and_Reasoning/tables/267b380e38f02f5996d42e3772819ad1ddb6fe4d84ce99e1cb5d6056f83d3265.jpg)

![tables](../iclr_results/2836_Beyond_Speedup_-_Utilizing_KV_Cache_for_Sampling_and_Reasoning/tables/42bdeb610d7ec216051b278eb32cc89ad1bb02c10a03e0118b4f9f1854390991.jpg)

![tables](../iclr_results/2836_Beyond_Speedup_-_Utilizing_KV_Cache_for_Sampling_and_Reasoning/tables/4cf011c13808934ba5e3a36ca7bdeab13d9bf4bacd01ab834def8513025ee8c1.jpg)

![tables](../iclr_results/2836_Beyond_Speedup_-_Utilizing_KV_Cache_for_Sampling_and_Reasoning/tables/561d26cffea533dd7c02c525ac6ed6ce145a6bf2d8f67c585ad85e5a275946c7.jpg)

![tables](../iclr_results/2836_Beyond_Speedup_-_Utilizing_KV_Cache_for_Sampling_and_Reasoning/tables/66cad7239766fd9071ff5c00fd2f1219042c7d873e6f3b620600869148f5c022.jpg)

![tables](../iclr_results/2836_Beyond_Speedup_-_Utilizing_KV_Cache_for_Sampling_and_Reasoning/tables/79e8d19328ed19427cf61b6b02afcdfe39ce5f049c08c86f32000dd081203f20.jpg)

![tables](../iclr_results/2836_Beyond_Speedup_-_Utilizing_KV_Cache_for_Sampling_and_Reasoning/tables/8e99ae462a558aa1264b7368d213cc2ccff12aa166849d5f73bc084dbf7d6f70.jpg)

![tables](../iclr_results/2836_Beyond_Speedup_-_Utilizing_KV_Cache_for_Sampling_and_Reasoning/tables/a891d6d42277428b107e9a469dd7208b50a064c0b2e0ed235701c80b40294b4b.jpg)

![tables](../iclr_results/2836_Beyond_Speedup_-_Utilizing_KV_Cache_for_Sampling_and_Reasoning/tables/dfab85a4392c9921c2f208e756b87fa11926d5621df0537d6c3446d17b7353c8.jpg)

<a id="paper-2837"></a>
## Soft Tokens, Hard Truths

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=9JjKTp8Jmy)

### Images

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/03611179dc6329bbe0a8f75073870b214731f7d95410f2312d431b4bc1e9c219.jpg)

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/0ce1e346569954017a3d3f72de2794f726516662175161ed7d2e77a68db8943c.jpg)

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/1b9506e8a64ba92bf5dc3ee92ede45f59848f6fb7535d99ed846d9e3b4907bdf.jpg)

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/1d828c800c418e7788ef58e8674aff51bebaf74772524aa53b90a44f0386bd57.jpg)

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/1dff5e336e16e9fd894066446423325bbf125b399bd05ce6ce8f56ab2c79509a.jpg)

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/3abf043427854f507307bf7c646fc3f24f447023220fda096d5b946df1043fb1.jpg)

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/3ee8f4aaa16ca445906645656418adc12516b4243f293f0d109ff0209eba6c14.jpg)

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/58f3eb82a01c75cb0668cc195d975e9b938cc98e618d3ab65dd8bbf481d24ba6.jpg)

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/5c8e8ecd4f9a4c38a2c34b6f2ae1c480e026a78d0ad6f365dd074c352047afdc.jpg)

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/7524833c76ce269468d46f62c8daf54183d959f3f4443f1422aae1ed646fdb68.jpg)

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/7b9c3436c075f8c705dacacff186f7c6e3a73c47f7a10ddc2d4d0d232b157b37.jpg)

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/8aca3e59ca0e2d9f312bfe4f15dd3e3dee21175187ff29fcc86d328134388c68.jpg)

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/8e635ea5cabbaee44c48d0af5b0aca81bc0aa9053898a26b134b9b379c6a4dbc.jpg)

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/8f205b7d10287349bb004ecbfb9de242ffad744ba2ecab6993bffa8565a7c092.jpg)

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/a8559b95a48440172f6ef6ac11b7bb8fa8da277e5a1ac7948cfe27f0b49b5f75.jpg)

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/c9ea784847d53874804e068c6dc69537f59a5c5c1703909849420c298f8c8513.jpg)

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/d6fe2238b5f2f3132b185758499214b531b2318d488938b7b7c31e3fdd31272c.jpg)

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/e2ae6f39444fdee72f4b5d10985896f3209c849cf6f763291726825febced615.jpg)

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/ecf5f4e5d0dcedba6edf13dc3b506e32cb0c013db7462bab86457875cd70007b.jpg)

![images](../iclr_results/2837_Soft_Tokens_Hard_Truths/images/fd14823a169ae7541cf3dc70f43bac98c9c3cfd066807910092c8284926888a6.jpg)

### Tables

![tables](../iclr_results/2837_Soft_Tokens_Hard_Truths/tables/0adfc9bbc8cceb8de6374b0e4e7748ce238ad73ac10e94122fe4243763631e2b.jpg)

![tables](../iclr_results/2837_Soft_Tokens_Hard_Truths/tables/3def44e1ba917a8192c2235a581ec8e2710cf8b765b56ea64959a66acfd66180.jpg)

![tables](../iclr_results/2837_Soft_Tokens_Hard_Truths/tables/59a2a6ba07e6469b9da055c15151acb4bb5cb9676f11967b6eae86d262b93d5b.jpg)

![tables](../iclr_results/2837_Soft_Tokens_Hard_Truths/tables/82bc6924c613af2f829dbb76f914a9fa34e9735025b122f5d67210a21930442e.jpg)

![tables](../iclr_results/2837_Soft_Tokens_Hard_Truths/tables/9838a3e086caf760a74889f9b88ea932bded94a101b124b4b983ab31904d87f9.jpg)

![tables](../iclr_results/2837_Soft_Tokens_Hard_Truths/tables/e31eeb7d688dae276d628a48dba828a7172f022ac9791a9e569f272f8bea8861.jpg)

![tables](../iclr_results/2837_Soft_Tokens_Hard_Truths/tables/ef458038050a2690de07e62ce14067cfc83f0aa4730ae4eab6228c9f6e3a4ab2.jpg)

![tables](../iclr_results/2837_Soft_Tokens_Hard_Truths/tables/f4ca24141cccc7b466c824e7ae7308e8e08d308b25e67ad8cd2cb9333d528883.jpg)

![tables](../iclr_results/2837_Soft_Tokens_Hard_Truths/tables/f61ae6adf2ebcb7f2ab1e74504b6896d4b050ab4bea06532c59f583bf36157dd.jpg)

<a id="paper-2838"></a>
## RIDER: 3D RNA Inverse Design with Reinforcement Learning-Guided Diffusion

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=lDyS4Qg5Ww)

### Images

![images](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/images/0623a0a6b4cdcec14ee977186a0b436960e5d58ee7eeefcb83468b34eff8045e.jpg)

![images](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/images/13c2768472085b085a4360925126305aa11597b97c04691ea734c3ae8a5351ac.jpg)

![images](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/images/174c824b562a789a09148fe3e5d83b225367b0440f431ccf5b49e01bb2b455b2.jpg)

![images](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/images/1cb41e7a62dadac28f2f7f4861cecc77500cb3117b4f155116edd9ad16fb77ab.jpg)

![images](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/images/1da591089851b5ed71359d89f8b047c61cbc03fa1792bf8067c030504de5aebe.jpg)

![images](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/images/5ad81571a82ca1c44977d3187ae2ee9d458f01b37fb601fcb3f3782d0874f216.jpg)

![images](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/images/6350f65251c6693e3a6a3f999aba25acf85ad173627d88c30d381c597d36546c.jpg)

![images](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/images/6894d82a9b85dd752ca35257fab4902502bfd3b39f3b6291d759753a5433ed7e.jpg)

![images](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/images/7ae5649e74bc39229a8879be74d1380639d40f562f8c4e4444f7b8b3df852825.jpg)

![images](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/images/7cddd2b0530a15a546a15f2fdb3b1669eb93a179e1f7a113304850fd60d45241.jpg)

![images](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/images/a5a4727f19bd6cebd1476ac9976f7f8e1a25ff8872105f742f84f7278debd29c.jpg)

![images](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/images/c85e4cd591fe98abcd50d49d804478fd4bf0146020460ba65836d2123ae29307.jpg)

![images](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/images/df4bda0ed7ccd09dd3f64e28b99ce6fac4ef204fddc1755bb667071e9ce6096e.jpg)

![images](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/images/f3156a5d7103a0e372925680f412cbf144b247c8b559d45d23860a94004e1568.jpg)

![images](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/images/f37af55db8e3e3feb97d3da2d732978497983665808e42dd5b2890e26b71b05d.jpg)

![images](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/images/f4644cc3c67c44d33fd15bf3ad57c3d2cbc631930ed3028649d889d723ff382f.jpg)

![images](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/images/f593c90d8e6f8815ec6fc5d0dcaa07017b243266484066ac1c9aae580433cfae.jpg)

### Tables

![tables](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/tables/069fd2a0e230fca03464a4baf777b8f5519b38c0b618464276a4513c38e5de9e.jpg)

![tables](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/tables/1296c0c9a8831836c7e20d11a597435a86f53a615fc480bc7a40d3cf7f714398.jpg)

![tables](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/tables/219394fd44f3c98c310f56217615fcb969eda1dd33c3027eae86c47a56d85ed6.jpg)

![tables](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/tables/4e564aa3d561edb25b390f960d02af0af7671f0e15e7fbdb3a0ecea080ad435d.jpg)

![tables](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/tables/559290f4d1efe263386486736a2212b4d1f921f2aa415d5347ceef6aca302d04.jpg)

![tables](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/tables/991e66c16ac0eae1d51c9189194a2726bd906211eab4dd528950ad15687b5408.jpg)

![tables](../iclr_results/2838_RIDER_3D_RNA_Inverse_Design_with_Reinforcement_Learning-Guided_Diffusion/tables/bb1026eca558a3ffce44a823584634ecc46d73aebe8cda77abcde423e174ead0.jpg)
