# ICLR 2026 Main Papers — Part 39 of 199

27 papers.

- [ImageDoctor: Diagnosing Text-to-Image Generation via Grounded Image Reasoning](#paper-1027)
- [RL's Razor: Why Online Reinforcement Learning Forgets Less](#paper-1028)
- [Unlearning during Training: Domain-Specific Gradient Ascent for Domain Generalization](#paper-1029)
- [OmniSTVG: Toward Spatio-Temporal Omni-Object Video Grounding](#paper-1030)
- [XModBench: Benchmarking Cross-Modal Capabilities and Consistency in Omni-Language Models](#paper-1031)
- [MMPD: Diverse Time Series Forecasting via Multi-Mode Patch Diffusion Loss](#paper-1032)
- [Motion-R1: Enhancing Motion Generation with Decomposed Chain-of-Thought and RL Binding](#paper-1033)
- [ECHO: Toward Contextual Seq2Seq Paradigms in Large EEG Models](#paper-1034)
- [Denoising Neural Reranker for Recommender Systems](#paper-1035)
- [FLARE: Fully Integration of Vision-Language Representations for Deep Cross-Modal Understanding](#paper-1036)
- [Step-Aware Residual-Guided Diffusion for EEG Spatial Super-Resolution](#paper-1037)
- [CoNavBench: Collaborative Long-Horizon Vision-Language Navigation Benchmark](#paper-1038)
- [Joint Optimization for 4D Human-Scene Reconstruction in the Wild](#paper-1039)
- [EgoHandICL: Egocentric 3D Hand Reconstruction with In-Context Learning](#paper-1040)
- [World2Minecraft: Occupancy-Driven Simulated Scenes Construction](#paper-1041)
- [Scalable Training for Vector-Quantized Networks with 100% Codebook Utilization](#paper-1042)
- [Native Reasoning Models: Training Language Models to Reason on Unverifiable Data](#paper-1043)
- [AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning in LLMs](#paper-1044)
- [REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?](#paper-1045)
- [Mitigating Semantic Collapse in Generative Personalization with Test-Time Embedding Adjustment](#paper-1046)
- [Mechanistic Detection and Mitigation of Hallucination in Large Reasoning Models](#paper-1047)
- [An Open-Ended Benchmark and Formal Framework for Adjuvant Research with MLLM](#paper-1048)
- [Optimal Transport-Induced Samples against Out-of-Distribution Overconfidence](#paper-1049)
- [PixNerd: Pixel Neural Field Diffusion](#paper-1050)
- [Bridging Draft Policy Misalignment: Group Tree Optimization for Speculative Decoding](#paper-1051)
- [UniLiP: Adapting CLIP for Unified Multimodal Understanding, Generation and Editing](#paper-1052)
- [Decomposed Attention Fusion in MLLMs for Training-free Video Reasoning Segmentation](#paper-1053)

---

<a id="paper-1027"></a>
## ImageDoctor: Diagnosing Text-to-Image Generation via Grounded Image Reasoning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=04HwYGgp2w)

### Images

![images](../iclr_results/1027_ImageDoctor_Diagnosing_Text-to-Image_Generation_via_Grounded_Image_Reasoning/images/01ee7bd7634130832f0291d7a90fe9d0e6d67f46a62cb17555fd6c3d1eca8f31.jpg)

![images](../iclr_results/1027_ImageDoctor_Diagnosing_Text-to-Image_Generation_via_Grounded_Image_Reasoning/images/15c2c627bdb19b0884ad13b2e2613076c2424e3cee29481063829cd09fd2981f.jpg)

![images](../iclr_results/1027_ImageDoctor_Diagnosing_Text-to-Image_Generation_via_Grounded_Image_Reasoning/images/778f46e6a574e1906e5641f5e5d02dac79c3e3122f517990924f1a0a0a1e5b8c.jpg)

![images](../iclr_results/1027_ImageDoctor_Diagnosing_Text-to-Image_Generation_via_Grounded_Image_Reasoning/images/7b6bfe89bf1ad48b1c6738b5dcbb2878da98d87dc9b18d4dd7cc118cab917a08.jpg)

![images](../iclr_results/1027_ImageDoctor_Diagnosing_Text-to-Image_Generation_via_Grounded_Image_Reasoning/images/9399cb538f07707e80318f9d0d4bc0af813819dab9542f49fd1158006750e3b4.jpg)

![images](../iclr_results/1027_ImageDoctor_Diagnosing_Text-to-Image_Generation_via_Grounded_Image_Reasoning/images/940b438b3fc5095687c2f9cbb7801ff1f5d8b8771268450287a3107f3d5d4137.jpg)

![images](../iclr_results/1027_ImageDoctor_Diagnosing_Text-to-Image_Generation_via_Grounded_Image_Reasoning/images/bc768d7d5fcbfd013f8e2f8744587df0a9d226aeea1a4ab95be547d6f0971374.jpg)

![images](../iclr_results/1027_ImageDoctor_Diagnosing_Text-to-Image_Generation_via_Grounded_Image_Reasoning/images/c5977dfdcedab7b42f163d8cbd852c934a13bd6cda18132471f12d8f56502e37.jpg)

![images](../iclr_results/1027_ImageDoctor_Diagnosing_Text-to-Image_Generation_via_Grounded_Image_Reasoning/images/c7f677be0bf13677d08d368f07e4d19648fc63107489600c824ab57088807ffd.jpg)

![images](../iclr_results/1027_ImageDoctor_Diagnosing_Text-to-Image_Generation_via_Grounded_Image_Reasoning/images/d8ff32b55e8b9d8ee9bc8fe695672743cac6f7dd518611100e11065ab3a27c77.jpg)

![images](../iclr_results/1027_ImageDoctor_Diagnosing_Text-to-Image_Generation_via_Grounded_Image_Reasoning/images/ec8f503981a9cbd3e9b8a3357709d86d9c99f21168d3f80e55ff0159b141fc78.jpg)

![images](../iclr_results/1027_ImageDoctor_Diagnosing_Text-to-Image_Generation_via_Grounded_Image_Reasoning/images/f21c9a259c3f4869cc2a68f6e628e76050a51ce2f2a73bdd9a5ea254f40231bb.jpg)

![images](../iclr_results/1027_ImageDoctor_Diagnosing_Text-to-Image_Generation_via_Grounded_Image_Reasoning/images/ff19be37ac50cfe1337882940897b0d07a4c5be7b892b933cd54bcb7576c1261.jpg)

### Tables

![tables](../iclr_results/1027_ImageDoctor_Diagnosing_Text-to-Image_Generation_via_Grounded_Image_Reasoning/tables/3d2fc40338837f75590010706ea3f602d81be0f26905f963c8d157c90e3a518f.jpg)

![tables](../iclr_results/1027_ImageDoctor_Diagnosing_Text-to-Image_Generation_via_Grounded_Image_Reasoning/tables/588e2945af71c0ad4a9fc44141b950e283fbf1b468d22a002a9352ec260fd399.jpg)

![tables](../iclr_results/1027_ImageDoctor_Diagnosing_Text-to-Image_Generation_via_Grounded_Image_Reasoning/tables/76ecb0f4dbcd54603601cff50eb6a7fc13b452fb4d2285014e8348705ef62f8d.jpg)

![tables](../iclr_results/1027_ImageDoctor_Diagnosing_Text-to-Image_Generation_via_Grounded_Image_Reasoning/tables/890218bb315d597e67e7028ea3d85f888e742a270ec8a73069bcd3226bcbfd0e.jpg)

![tables](../iclr_results/1027_ImageDoctor_Diagnosing_Text-to-Image_Generation_via_Grounded_Image_Reasoning/tables/b4e2be299023680c17fd929de7b9176d5bc06e5f4cd4a9432a89db5f715a01bb.jpg)

![tables](../iclr_results/1027_ImageDoctor_Diagnosing_Text-to-Image_Generation_via_Grounded_Image_Reasoning/tables/b55b3128bfc0ad1cd905c710c6f9e40d0629ca598c5295b27e103788b0fa4fec.jpg)

<a id="paper-1028"></a>
## RL's Razor: Why Online Reinforcement Learning Forgets Less

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=7HNRYT4V44)

### Images

![images](../iclr_results/1028_RL_s_Razor_Why_Online_Reinforcement_Learning_Forgets_Less/images/0e71eb09de46cd9b2a7bd0a9dca6302a9e7e1418547fe8c38d729ec09256e7eb.jpg)

![images](../iclr_results/1028_RL_s_Razor_Why_Online_Reinforcement_Learning_Forgets_Less/images/1b9a2e3d57c8de59c31b2b97114f64aeff1216a85ee271c00c853cf80ec6abe3.jpg)

![images](../iclr_results/1028_RL_s_Razor_Why_Online_Reinforcement_Learning_Forgets_Less/images/27f9445139c1954aaf6c687050c8cf412772390427c8267d2999f458fda4303b.jpg)

![images](../iclr_results/1028_RL_s_Razor_Why_Online_Reinforcement_Learning_Forgets_Less/images/2dfb86d7cebc1c8e06ebcbb23a525fe0792c3d12574bb0656fef44a29a435e06.jpg)

![images](../iclr_results/1028_RL_s_Razor_Why_Online_Reinforcement_Learning_Forgets_Less/images/3ed50a63d2eb0fd151f5a726cc113e07e9501df65805c0726d7bc9eb9e4255f1.jpg)

![images](../iclr_results/1028_RL_s_Razor_Why_Online_Reinforcement_Learning_Forgets_Less/images/4136774dc81675bc3938bf5bf3e9a193afba811f2fefb7b05a99c88569643350.jpg)

![images](../iclr_results/1028_RL_s_Razor_Why_Online_Reinforcement_Learning_Forgets_Less/images/59be2db05819bc6ebdaf6fd055ebdd799720cd7343792a14af73dd9e2985a027.jpg)

![images](../iclr_results/1028_RL_s_Razor_Why_Online_Reinforcement_Learning_Forgets_Less/images/6bf7631ef1e7d43f112eea07dfce2a82321f858088b832b59637f219d2abd2d0.jpg)

![images](../iclr_results/1028_RL_s_Razor_Why_Online_Reinforcement_Learning_Forgets_Less/images/8e03ca12ae84ee06aa55c5b3f62ed2f77f505a5c8ae317b4f90e1eeb808b6f09.jpg)

![images](../iclr_results/1028_RL_s_Razor_Why_Online_Reinforcement_Learning_Forgets_Less/images/a469f323c89f8b1405f65bc96e6fe84b77e544cf3efaab7597894cde909e49ee.jpg)

![images](../iclr_results/1028_RL_s_Razor_Why_Online_Reinforcement_Learning_Forgets_Less/images/a683cec40224cc8a708149731ff5316df2552de73f31860e7cb3cf817ed719fe.jpg)

![images](../iclr_results/1028_RL_s_Razor_Why_Online_Reinforcement_Learning_Forgets_Less/images/b055e66a5bde2fdbf954d241010e713df5898780e2f6c35e45d4da32418d9edd.jpg)

![images](../iclr_results/1028_RL_s_Razor_Why_Online_Reinforcement_Learning_Forgets_Less/images/f0811fabe47a8bc11b98c5359f83a919f5a0f12b08cb59f104ad8aa7689b9c68.jpg)

### Tables

![tables](../iclr_results/1028_RL_s_Razor_Why_Online_Reinforcement_Learning_Forgets_Less/tables/0c70bb7db4b2825aa4804a71e2e5c32a3cf1469ec0f916713030538af6dcc088.jpg)

![tables](../iclr_results/1028_RL_s_Razor_Why_Online_Reinforcement_Learning_Forgets_Less/tables/95f540e6c7234e83588474e3bb13be6e8811fe0901ff78fdb40cad914d85bb72.jpg)

<a id="paper-1029"></a>
## Unlearning during Training: Domain-Specific Gradient Ascent for Domain Generalization

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=9ufS5Jl0O0)

### Images

![images](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/images/048c8aef22af391ea7342118892de914d3d5e51f5eef3275dafdbe8d0b735e66.jpg)

![images](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/images/1b0654ccad400d78691dd1e594c29c559e7e02a5e64797d4d06d81b0633de783.jpg)

![images](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/images/1b5865a465750bc3be8cd42d10fcf741beb37828699198a434f9da33d951cc09.jpg)

![images](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/images/347b55e8f708057db6f621a4a71d0e8834468b49e10a16c89bb55ebe80d013c3.jpg)

![images](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/images/367912734060ebd6954e0dbf85b1df1d7f1e96ad9c070a099f3f2a97dee88cbd.jpg)

![images](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/images/4934c79e8591d618a4c2093a117fba2fe36d0908856900dbcbff03421f022af1.jpg)

![images](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/images/73a60976ed4db91d595f24834caaf959c1ada0f955cf693d277f9a45791825ba.jpg)

![images](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/images/7c1316b6fccf4bb4872e415a40559a764c47fefbcbf9fdd16d1527f467b2dc17.jpg)

![images](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/images/a7175195b5ee90c92d9959936d26de73d18fb196e71d5805b56000b9a85d60de.jpg)

![images](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/images/ae871c549ac6fdfc4258c3884d6285a2335d55b4f8459e6140c9b2f343636d53.jpg)

![images](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/images/f5cab881d6d300ba83343547de64910f0bb17b66b78d62887f0b5c24da482d2d.jpg)

![images](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/images/fadc1cf07dcac642b44907afc5c612e50f858f70af79b5e25682da74f3e01951.jpg)

### Tables

![tables](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/tables/200f7032e736f0f9bde9de2893b41e46926016fae6a4e593bdcd15a4eb1fef46.jpg)

![tables](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/tables/2af2ab789ed68d2b46e49a0781ebc0c459ab13fb159c6e71f9d31cd6ce4a2a56.jpg)

![tables](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/tables/2bff8aa6ef0328b22b8e6a90e44e919e769030344818455d3d94df4107475c32.jpg)

![tables](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/tables/3d72f86b4050a4aa2453cd6e68278d859c0b632b48ea6f025a31daf636741f7e.jpg)

![tables](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/tables/43609b4491f4b961a589270dc938159ddfedfcdaa9b15617e5a47d67ceeca94a.jpg)

![tables](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/tables/4db1c78f19d8e966b1cd5cf3215f45190dded3f08c070d6bd0a479357fbd691d.jpg)

![tables](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/tables/4ed48338d55fab51f7b9e4a92401e114cad5434076cc6f181d718c10c4e385fd.jpg)

![tables](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/tables/6e95f4843438af58cd6b24cfc3e2730a10664e2581581db6d904067d3b317a3a.jpg)

![tables](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/tables/77a716aab326a41b2c75a737cc0f8a0f72d67d82abe67fe96107b5e2fe6b18a9.jpg)

![tables](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/tables/b1bac587d68b5a6bc9563feeef36f31f0a92337713a2c22da1c734cb2a25e56b.jpg)

![tables](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/tables/c16ea7e88a6c7dd7b72b1f615eb1df10e2ef58b2bdcaef59524f7eff65da94a7.jpg)

![tables](../iclr_results/1029_Unlearning_during_Training_Domain-Specific_Gradient_Ascent_for_Domain_Generalization/tables/d77e3c4dcc2b3180eaaa0f8a4b61ac2bdbd22b92990f776f0d2514545f4c9836.jpg)

<a id="paper-1030"></a>
## OmniSTVG: Toward Spatio-Temporal Omni-Object Video Grounding

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=azcQJtcYTE)

### Images

![images](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/images/2dc03469276c4d7fc28f3520c97c3d2e086351284c97290c41fe39ce26a4fcfc.jpg)

![images](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/images/3310b86761e1ee8e5bb70e33385019daf019ed57d4c64f868c8461be7a897ec5.jpg)

![images](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/images/40bf0140328dfb0510b71526d1832181d7523300d8cf20c9f89bd0b26a9a05a6.jpg)

![images](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/images/46b18a3e90b57b422eddb44fa7bd4da1012db191a5fb8be9a8ef6981ee56c6bc.jpg)

![images](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/images/527d4db766c5674c2aa05ac241e9a7acbff33727288074fb3ee9ce136f845953.jpg)

![images](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/images/564e200e3c47220cc85f47284b4c462b38f8721a8c19caec46c92b0eb3212528.jpg)

![images](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/images/5c65b29741adb582833bda6af42455161d194edbea57af6c89c69ec3f6e251cb.jpg)

![images](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/images/656b682f81bd11f68949689e09ed45985de37a8be4d4f3c95721be6b15fe212d.jpg)

![images](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/images/bfd808b5b65afccb5e2666853f68eed99febd7dfb856c37978fdd09443fdc6b7.jpg)

![images](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/images/cadf61f3128b9e79fd7620bc70e12bc15ab176c671a963760c43c434f635841f.jpg)

### Tables

![tables](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/tables/00155f443c62845700b771c58c7914ceec58848134ba2beb4285baac34d4b4f9.jpg)

![tables](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/tables/030b2dcee56e7bd9c848176d602553135cff7926e5a1a8f5c282739bbc09e272.jpg)

![tables](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/tables/053cfaae5be4d9e2857ea6d9d1cbb823c138145c849e44d5cfa8b36f8e90de7a.jpg)

![tables](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/tables/1273f42aabf2a983c3341d21978cd9ea4dd4ebe6edbc86bb787eca649be10653.jpg)

![tables](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/tables/322cb5552e3feb7f7b9aebc700a1c16b2611f65234dbf3acb1230bff77c68744.jpg)

![tables](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/tables/41bc1a3e39e46da9a9ce81bda058c31e55b5d776a200fab1c5a1490696a87094.jpg)

![tables](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/tables/49a2b80b7f5eae7068bd2a3140e44f71e001b2660c86a26e831d97e30856bcbb.jpg)

![tables](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/tables/58815c4df5680e35c4bc0060d261caf40f5e348e7a1a206a66c5c195a6d65441.jpg)

![tables](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/tables/68519a4787a9258627461ee9bb62e39df9105798f5a1f74c5d748678f74a946b.jpg)

![tables](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/tables/7bfb7f5859a3658366c926b543346fde4da9e46b149531520acb9c1f3d10168d.jpg)

![tables](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/tables/a25ad6e55fc4c87789ac1221f315d3b9086fdd3431d440ca0671f129a512cc0c.jpg)

![tables](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/tables/e6f3d17708623c4478d9e48e932e163a778bf11ca07b97f784bf9230c8aba856.jpg)

![tables](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/tables/ed57713c46124d15fdef0b02dc2b2c8263f2413793542c14353fe27ee17c6dd1.jpg)

![tables](../iclr_results/1030_OmniSTVG_Toward_Spatio-Temporal_Omni-Object_Video_Grounding/tables/f2d7fac5827c1b947d1c0832679dd406ce3f2ac21f25df15b649e87eb0474753.jpg)

<a id="paper-1031"></a>
## XModBench: Benchmarking Cross-Modal Capabilities and Consistency in Omni-Language Models

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=HaL9EZovFg)

### Images

![images](../iclr_results/1031_XModBench_Benchmarking_Cross-Modal_Capabilities_and_Consistency_in_Omni-Language_Models/images/1b380ff4abe83388b948265e60278814c65d9b1c98b3b5bbd2cb39093dbad319.jpg)

![images](../iclr_results/1031_XModBench_Benchmarking_Cross-Modal_Capabilities_and_Consistency_in_Omni-Language_Models/images/2b3db40af635acfdcaacfb374751d527bc87ad9b45a82a9d11e70bc740b1f84c.jpg)

![images](../iclr_results/1031_XModBench_Benchmarking_Cross-Modal_Capabilities_and_Consistency_in_Omni-Language_Models/images/2c430dac29e3567fdf08164ff72af78ce2d741c68df0a3c73ea9d4501f144529.jpg)

![images](../iclr_results/1031_XModBench_Benchmarking_Cross-Modal_Capabilities_and_Consistency_in_Omni-Language_Models/images/2db6fc42a29a6a618d92a45360e9849258d59662e5a2415e94579d36b503d393.jpg)

![images](../iclr_results/1031_XModBench_Benchmarking_Cross-Modal_Capabilities_and_Consistency_in_Omni-Language_Models/images/40cd3a5050d0e9bf70a3758e2d038aa80b74a1ac2f6c6dec620e7a37e569d088.jpg)

![images](../iclr_results/1031_XModBench_Benchmarking_Cross-Modal_Capabilities_and_Consistency_in_Omni-Language_Models/images/eddaf503ebac5ca4a6f706b27d83f3fe54c6a6aa93db86b13dcfb875594a8387.jpg)

### Tables

![tables](../iclr_results/1031_XModBench_Benchmarking_Cross-Modal_Capabilities_and_Consistency_in_Omni-Language_Models/tables/086eb65020e09ba3ef531aa13658c72122623ba7d540a84d06c7eb6a1bccb036.jpg)

![tables](../iclr_results/1031_XModBench_Benchmarking_Cross-Modal_Capabilities_and_Consistency_in_Omni-Language_Models/tables/6c480165bf84b93511a90175ecbf972d3490d5b18ddafa8a3631c3690fcb4e29.jpg)

![tables](../iclr_results/1031_XModBench_Benchmarking_Cross-Modal_Capabilities_and_Consistency_in_Omni-Language_Models/tables/efd34502b9e2a513b80e08f34d41faaf285ae75c98b13927ece263c76a93d826.jpg)

<a id="paper-1032"></a>
## MMPD: Diverse Time Series Forecasting via Multi-Mode Patch Diffusion Loss

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=NEUgHT8dvH)

### Images

![images](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/images/0194451e020692bd6decedb69fededcf6808b107af07570f1e6e732953e5fbc6.jpg)

![images](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/images/05d295f580955785a650edeb29de1523b3fda22b19a2ac61f71cb3c078600703.jpg)

![images](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/images/29ed3912ca949a6f6fc544dbfe54ed19d21cfebff36f7a49b71b5e1f32051d79.jpg)

![images](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/images/381cf6a251725fa898dffd625a1f417385083205bcdc62062b606bca4b8264c5.jpg)

![images](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/images/644596864cdf8bb5e5bc8831c75526bfaef45e49149dbd242cf8043c80d1291e.jpg)

![images](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/images/78a6d7e028b7f829167c651f7530e493f67edd01e6c7220fbb2b9027026fddbb.jpg)

![images](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/images/a8b66fe01ff7865694cf51cc1893553f0ab6142cf4aae7e7f181b381a4988277.jpg)

![images](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/images/c925a8a2d9b47a99b756e726840f9ae114be6f74802ae36907c5ffbc7c4f08e9.jpg)

![images](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/images/ed6912a2ef445661a4a1688af47b852deef2be8ebecfd40086a77679849fae71.jpg)

![images](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/images/f6a7347c795769daad64117ab0ce5e1a6e6ee6593a37e48016dc2a0bf2a95b55.jpg)

![images](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/images/fe133149a3cd16daf2f39dea108c26abff7711a745b9b17edb1df9ddb19b3251.jpg)

### Tables

![tables](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/tables/035fb18968f7f487ed147f8208bea2a295fc0f37a2c0a1e965eb8511bfc4a5ad.jpg)

![tables](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/tables/0fd5e97d4143bdbbcc23eacfee0458db2bb9da22e22e29e9b7047d195ecf224c.jpg)

![tables](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/tables/10351d3495492e72d63beaa13a162dc71f12ab548dd7d1ea9be41a97bd237856.jpg)

![tables](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/tables/2867688cc3dbda9a0301365281531b1d70e1bb0e76d85d6ff8a9c3984db31a89.jpg)

![tables](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/tables/2bbf95cd550bd5cba9bb8a41d31b1553e5cdaeed986d25ba66b04e006068517f.jpg)

![tables](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/tables/420d6fde73408be8dc39ff96dc73f475d5f0a86bacd2313f8f9dd3ce3c4ef8af.jpg)

![tables](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/tables/4a6ecf8a91388305f7cfdae39c129ec63bf90e353206da29a823c8f0acfbfc8b.jpg)

![tables](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/tables/4c1d41e4389b232bf1110dcfd648e6d7f95f44ebcd548da184f19094c3de36b0.jpg)

![tables](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/tables/4e7e154249b6fc0ba1358d60da4906c6bcf0ede92ed0a53e44e9527bf4b09931.jpg)

![tables](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/tables/76e728d317374602c35ecacc3077394394172ae6f177c1590a70e7e704db1cac.jpg)

![tables](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/tables/8e2b1ee98d61e86d582ff7b1372cc2d0e63c38e348ceffeb114ea6ff1357324c.jpg)

![tables](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/tables/8e7948c76ed32a6e50e747bd5f6bb5148a00c683dcc044201e08982d8a017a75.jpg)

![tables](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/tables/99a4322c39fc2416ece7cdafc965cb1229ef0a838b1754ce82c0b22872e0620a.jpg)

![tables](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/tables/c940ee795a6040b04fb3e27478abf602d9cd491c99947810d85e141daf627787.jpg)

![tables](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/tables/cf171d8e12d7a5529830e83d6dc6d24d2b77589f10204eb44731d808866a4f49.jpg)

![tables](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/tables/e418eb79d47430ebae8cc6fdab372e1b789088d29acaad893c69af6051070611.jpg)

![tables](../iclr_results/1032_MMPD_Diverse_Time_Series_Forecasting_via_Multi-Mode_Patch_Diffusion_Loss/tables/ed4dcc497627823d13ea27d09c2b15e099a448aa1f6ddcd6f012d103a8b8768a.jpg)

<a id="paper-1033"></a>
## Motion-R1: Enhancing Motion Generation with Decomposed Chain-of-Thought and RL Binding

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=eXXsUer975)

### Images

![images](../iclr_results/1033_Motion-R1_Enhancing_Motion_Generation_with_Decomposed_Chain-of-Thought_and_RL_Binding/images/874b7e96ff928be82078587742b21cd83ea68318fc9ef0a18342ef711b14d0ce.jpg)

![images](../iclr_results/1033_Motion-R1_Enhancing_Motion_Generation_with_Decomposed_Chain-of-Thought_and_RL_Binding/images/8fbee4d5f27d427bfaf47624387dbbd948e60d831356de0eba4b73072df7656a.jpg)

![images](../iclr_results/1033_Motion-R1_Enhancing_Motion_Generation_with_Decomposed_Chain-of-Thought_and_RL_Binding/images/e157515a61a24ca5885671cf28bd32cc2cd73ef8fdb41c850ace6c4056614abf.jpg)

![images](../iclr_results/1033_Motion-R1_Enhancing_Motion_Generation_with_Decomposed_Chain-of-Thought_and_RL_Binding/images/e6e5d85ec9b20eaf9b43b880b3b677874152e35082a80270b604b0683e58d302.jpg)

### Tables

![tables](../iclr_results/1033_Motion-R1_Enhancing_Motion_Generation_with_Decomposed_Chain-of-Thought_and_RL_Binding/tables/1b4342dbb76b771e3992180a768133d4ef1b9e19ea0244f1b1f34bb9db471921.jpg)

![tables](../iclr_results/1033_Motion-R1_Enhancing_Motion_Generation_with_Decomposed_Chain-of-Thought_and_RL_Binding/tables/81161af9543e3f0c6d783e769dae9e541163d18936aef9aaa29a889d2a396230.jpg)

![tables](../iclr_results/1033_Motion-R1_Enhancing_Motion_Generation_with_Decomposed_Chain-of-Thought_and_RL_Binding/tables/8c0c57fe0911122bedeea822af34192ef0f202bb6f3c23c102e5a32f1d6b196f.jpg)

![tables](../iclr_results/1033_Motion-R1_Enhancing_Motion_Generation_with_Decomposed_Chain-of-Thought_and_RL_Binding/tables/dd9f8380bcafccb411d0a32c72a27c781bf1f4e755ecd257159ae6af499cdf2c.jpg)

<a id="paper-1034"></a>
## ECHO: Toward Contextual Seq2Seq Paradigms in Large EEG Models

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=ClLQ6cLkoR)

### Images

![images](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/images/08e179e2194dd6d067631c754f8893fa8f024e1bacaa0229a939665c45ae865d.jpg)

![images](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/images/25293eca76876257f2617663102ae48aab521c105ed0587a776375e13e6ad9af.jpg)

![images](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/images/4c83d0dd4007f831e1aab23285c0ac10c5824bc908756265c635ffe28f7cedb3.jpg)

![images](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/images/50af72897907e74f73602961e08cdc77bd2127448b3facbed0fe156c39087d86.jpg)

![images](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/images/52f4d810b0cef6ee3411722ec1cdf5ef4f8182b91cea002699031e53cc833fa0.jpg)

![images](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/images/7f073bb94a0a912348326f0411687d74146e0521642315ff37e82f6e9786f44e.jpg)

### Tables

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/09a205ea702444e056f8958465298d3394aef4476f51b56bc2a81014516bd3c8.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/0d8cabec4f5399d69712e63faf3ca2f581891c20244dc5fe617bffe045d52a43.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/134b3c83c67811b6eb78e79f204c4c91cd04a1ff30056845e1c18fdc88425936.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/1b0231cec92cb9b5baa8dbf0974aaba1103eb394bec5c61a35e35d37afa06ffa.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/2644d849eacd266386c11c80a09ae1a63eca6e6b040587250df4c35ae59d3755.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/3d833727e5aa323dd52d767fd481863c529501ee490cf2ace86af3e5d78e4398.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/3e757b45708c9e8a8ab382ddf50078f1a824b4473142f94c7f81e4ee69ae460b.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/5f37e8509ba0bcf8204a2ba05b80c42881fa4c4020e7625d703d1a3e310fa0c0.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/7cbf95e94af8bbab1b2a4b4eaa610bb691180a9bbc9b96e1639bd768b49aa37a.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/809baa3ff09b82e802fbd09ba5a31bb25cb4c6553b09efc062185c2aee786575.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/81526384d5762b81aeffddf9270c1b0abb2f03be94e3cd4d99498b7c4fcf9bb4.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/a73cc501c1d0ad46bef1eda836d11ca4a6d41560dcb70b8041066bb731d5dddc.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/b4c93e9abccdd66066772fa9339d161b50b41205fde240c71e1b5a90053e0be2.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/bc622747b2d923452f2b2eb362f4f9cbb2a636ec6c7bc9f6ab8b0caa786dd752.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/c18e88018efcbd54f1f7929ed376842b84ef59fb53d8c318d8f04b7fc14686a1.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/c45e437564b5bad8e789f336e68a90e7dcdc2bcd9dcda908c68a746b5a542430.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/c8c4a3914dad8dd036b11d8e65cc8ccda6823281e7705af2277cd257fdf8232f.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/cbf75896a573acccb3fe6fe490098bf6ac8e462b27dbe4720cf981bb044272ab.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/dd2fa839e81b339f54886d29e245e9281348b01e21124d3761923a7118350849.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/e15f2755b43bf76d481f248871d6db9d74fb3a84e28c6cc7ca7b9425ec78e4b7.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/e452b9dcc24531bdf16fd9a664d1123b918b2595a1337a12e1b24bf3076fb2b2.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/e6a30fe6434b3b870fa0056eef0c73cd626ad29d448b95a522dd56cf54a6148b.jpg)

![tables](../iclr_results/1034_ECHO_Toward_Contextual_Seq2Seq_Paradigms_in_Large_EEG_Models/tables/ebba80d77e95ef68872110c83e2c086a864dbafac4f137422491e8d45a691d30.jpg)

<a id="paper-1035"></a>
## Denoising Neural Reranker for Recommender Systems

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=JlwYkFm91F)

### Images

![images](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/images/24d62b6ea0c319bdb11707671ce457de4c74c4dd854e0d0275d3aa682a7908b9.jpg)

![images](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/images/2c81d676630a379033b490008f228a12fa01ec6ba8d8fe4e67198f6e115314e6.jpg)

![images](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/images/4559f291186e4cbedc6e9ff6cb5296e60ee314f127383fe2f7911904d076cdcb.jpg)

![images](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/images/4a6f15fa5910a6fb826eff8dedcbee3d2ccb8a8fd3a9b5629363cd1bbce23318.jpg)

![images](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/images/558c4787a4d9bf4e50cd61b02db286a2aaf41ac99e8c027145ae0a0053acea7d.jpg)

![images](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/images/56d5cc1b2ebc3632374b0da31fc4371b89aef28378de5c2d20f9d0fce870de17.jpg)

![images](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/images/5c00641cf93449418732bb582fc1af65b0e0d9c32bd7a5a9659c2c045016b505.jpg)

![images](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/images/921ce925ea72ba4904af43e59a6cc289f28c347c6035d66d284bfa1cea9505fc.jpg)

![images](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/images/9bfec0cf1283664b0758ad87aa9d647aba460128693178248181b0d669db5c35.jpg)

![images](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/images/acb594caf76eaea08cd4f11ffb3ac56b7c2747c5ef579ef7a398b8af790a1f6c.jpg)

![images](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/images/ba883a6f6ee776191964bb19e95d84540dd1c709d195da698d5a0e7bcf9eb05d.jpg)

![images](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/images/d56716258eb42d07f4a9628ea88db3a6537f5fa33d82db3dcbcbc9b9b211e616.jpg)

### Tables

![tables](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/tables/0765ef2a3e492eebfcfa65103430f9c85d938bc3eec4739dee0a690799cc4c2d.jpg)

![tables](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/tables/2a4068ebe39662743ae4d45451f75132446c6444459b31ba7e7552a0aae86dc2.jpg)

![tables](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/tables/3f28b800c6e28259da1b75fbe5fa2eb6dc3667a0e3542a6ec7e88884946dc5f2.jpg)

![tables](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/tables/51a7ca46ad06d29f36158c054a012a0217a57849c954d7b69df290147b8bb753.jpg)

![tables](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/tables/6e9732552250c0867951ca8240fb4675b29b468e105ffca35e277f2c0d5c021c.jpg)

![tables](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/tables/7a72c10d65ca97e64e2202776f40b8598496c14a35acf5a7bfa0ed163617cdb3.jpg)

![tables](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/tables/805168ca83ea3ed9f5a002901eb92259720068f641f10c02212ae45bdfa02e3a.jpg)

![tables](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/tables/98a0727a100f40103722c03b9e7449dc2c5a8af4b1191bfa6f427be0f2a4eb1a.jpg)

![tables](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/tables/9d96de019055183ae4572c1e60d623cb1bab62dc3cc8eb63f6e6a1107fdf25c8.jpg)

![tables](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/tables/a3dbe1f0c8c078c97a098a821ee53fcfd4bc660f3d08a3edbd13f128eecc60a0.jpg)

![tables](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/tables/c86f32e4647b9512fe416833450cd343b74e80127df35569b80232bfcdeb314e.jpg)

![tables](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/tables/d0eb5ed698cde1e15c91f90659729119d79b625b4de1a353160179bcf12c42fd.jpg)

![tables](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/tables/dfce2206146c708e1af062b422487a9d737ca3fd44dc3558d186dc86058f085d.jpg)

![tables](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/tables/e9e6e89594e2e20e371a44a9ac54f48c94ad78c7548dd0b1b071b31b3d01f38f.jpg)

![tables](../iclr_results/1035_Denoising_Neural_Reranker_for_Recommender_Systems/tables/f59cb0af81223deb66429d2feefdaa9530a3787885763f5274ef0ad0ec815689.jpg)

<a id="paper-1036"></a>
## FLARE: Fully Integration of Vision-Language Representations for Deep Cross-Modal Understanding

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=LWw9yLNQfx)

### Images

![images](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/images/00627f2d49b353068600a61be2dae726ccee2eea59f1e9f30391453bff10fc7f.jpg)

![images](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/images/05e0577dab542466c6ed507e48663ee2a6f059b21ed8a726cd328c0aeedfabf2.jpg)

![images](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/images/062164c544affd55e725508dda2ae9f7d5110893c4f4d81634f1321be2ea08e7.jpg)

![images](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/images/15c908310341d1d271f3ba0f9796fb710d79d39afde8cebc78333d6858f5600b.jpg)

![images](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/images/1e13badad32916b1581f8d05a56d93a7697b35fde2e5389690e6287f6251cd1b.jpg)

![images](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/images/245bbc20c642f7ee452962c25918176e87ca9608bbf826f49307cf4faa6a9ae3.jpg)

![images](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/images/2dc8e65d247ece52e2e9925e31f6b1e574b79167a17c6c0bc28caa6c3f30189a.jpg)

![images](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/images/2e1a3a923c65c56c8f52dc85ac0b6ad4132c2ea92ff9b10bb9394beb3d1f15bf.jpg)

![images](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/images/37fef5faa51bdd800c96001379731d73dab764ace73df1c4cd133a0000e69fa1.jpg)

![images](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/images/3c911ed17d146805f705a45db4a29766559dd3c4328f2b745c0900c8789bd441.jpg)

![images](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/images/5e7f4e3aea5947c93d70e03c5c55e4d5566bbc560ed5ef399878dcc751a58253.jpg)

![images](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/images/7445b5ac9e3661c4be33eea9938c9fae2ceb25d64e28c6db996dd7f3a46292b8.jpg)

![images](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/images/81642a39b0be7d65bedcf318f88991ee3dc7b8ada2d9590a5564e2d8e9233d89.jpg)

![images](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/images/983e3f36b5416652f68f7bf1336bf7879b39a2014befece904994e77133ebed9.jpg)

![images](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/images/a4cccd4980842536002c2462d6455c40108c91da392629301ecab0be8bb27ee8.jpg)

![images](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/images/ad0dc69b8fc2433737b96923d7e038d7128140f315b770d119a3b0a317ec2b31.jpg)

![images](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/images/d8e24f65589a78a4515d0529de0d88164061d5578945ce9894db4c9356ade99b.jpg)

![images](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/images/daab46dc2ad712b9f652b04655558df49a295a38f47670a91b4882514479fb93.jpg)

![images](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/images/e2b6fbb4d375f718c119a23a6bc8cdec89c3fa818399e99fd04af0a05fb2396f.jpg)

### Tables

![tables](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/tables/1c17429dde74f59352bd791137dedc5480d2178e47a72b8cb00811536c08f762.jpg)

![tables](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/tables/2b7cb66eeb1018e366c86bfbfefad2ccb0bc95854434af422742d7a656772b6b.jpg)

![tables](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/tables/303d59bb715b4a5d5a594129ef1301d4050747cc0e3fd46709b9bd6b06677d75.jpg)

![tables](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/tables/41937ffea063530088f0952a13ec7b7d19bc8724d69d308c8fc3aed7ab7c2e07.jpg)

![tables](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/tables/4ae5480acd655d6ffb30f2a463e35ced44381407918f5ac6dc530b62e4802d65.jpg)

![tables](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/tables/59b8a78c2682dbc739097e8073f9a46c25c9fa58a5f9b1a26e8922bc288597cb.jpg)

![tables](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/tables/845bacf2d6ec406ff4ea6c34a084a9103ea84e16d4af87cbe498e0814e8be3b0.jpg)

![tables](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/tables/94605770436411d76eccd2099393f4197784194433396b950cc4b7a380b2ce24.jpg)

![tables](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/tables/bcc6c250accae2652a41b9b1b1690de04dda23f71fd768aef3d05b51d78a819d.jpg)

![tables](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/tables/c4a4f99141c0c0a17c586173876175cf2d7cdcad3d8830dc574967490cf46693.jpg)

![tables](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/tables/c553dc3a876733c69e124b29d02099034c6f791666757380e42c45488331336a.jpg)

![tables](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/tables/fbf38c2feb3248c780da67331bb7c9913a8d37241fac5af4e1f373c024e8b99a.jpg)

![tables](../iclr_results/1036_FLARE_Fully_Integration_of_Vision-Language_Representations_for_Deep_Cross-Modal_Understanding/tables/fc58d9c680e086b6d5f45428b7df3ffcd4f622645671c3e6c899612b4ba4140d.jpg)

<a id="paper-1037"></a>
## Step-Aware Residual-Guided Diffusion for EEG Spatial Super-Resolution

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=2vN2GlfV6T)

### Images

![images](../iclr_results/1037_Step-Aware_Residual-Guided_Diffusion_for_EEG_Spatial_Super-Resolution/images/6287d2797b866ef85b6be2c2da971364f1f0b533c8a216a3270dd54eed15cb8d.jpg)

![images](../iclr_results/1037_Step-Aware_Residual-Guided_Diffusion_for_EEG_Spatial_Super-Resolution/images/9cec95db19cc02bc7328c69a2f257a7937f71ae285e9047cf55f0f20f80138e2.jpg)

![images](../iclr_results/1037_Step-Aware_Residual-Guided_Diffusion_for_EEG_Spatial_Super-Resolution/images/9ff807755872eb2b8519839907b5bcad6ee0bf8dec9aa7e6db8bb0ae3fdb3323.jpg)

![images](../iclr_results/1037_Step-Aware_Residual-Guided_Diffusion_for_EEG_Spatial_Super-Resolution/images/d71ac6776110cd9183ecd2128d07cac6b04523f5636613fd429047ba9539d324.jpg)

![images](../iclr_results/1037_Step-Aware_Residual-Guided_Diffusion_for_EEG_Spatial_Super-Resolution/images/e5bc3fe777f344443a0bf2c2864b100ecda93851ae425025ca18b495cb6f1b6c.jpg)

### Tables

![tables](../iclr_results/1037_Step-Aware_Residual-Guided_Diffusion_for_EEG_Spatial_Super-Resolution/tables/8630c4fa29edf328b2859b699a290169cd0f66084f8e60ea43e17c35a7ffa4ed.jpg)

![tables](../iclr_results/1037_Step-Aware_Residual-Guided_Diffusion_for_EEG_Spatial_Super-Resolution/tables/9f216a9bacf1a9a2fe6f75782109d476834d5e50739147cdefae6a512859606b.jpg)

![tables](../iclr_results/1037_Step-Aware_Residual-Guided_Diffusion_for_EEG_Spatial_Super-Resolution/tables/be64bee10a53e6557bdd6cfba425ed32479e9c9ca61a49805206fbbeecdc33a9.jpg)

![tables](../iclr_results/1037_Step-Aware_Residual-Guided_Diffusion_for_EEG_Spatial_Super-Resolution/tables/ef09c3bfe1041c1b3949686a4742be2fbd181b53ba145c08b92da816aa9e0eb0.jpg)

<a id="paper-1038"></a>
## CoNavBench: Collaborative Long-Horizon Vision-Language Navigation Benchmark

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=bMrH2PFMsi)

### Images

![images](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/images/11b6cda8238a80c16f0445328795f4a3dac7243f13e552adcc8fefce104e4f60.jpg)

![images](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/images/2731f92370284700dfd338f42b9ccb62aeff4a33855e3ed9dd420b7c816dcf2b.jpg)

![images](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/images/2a70226b759d7119ffb63c2f99772d8ee6b319db917ded0e7f8daf29e26ba620.jpg)

![images](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/images/3c863d69e463b99b09e6702c9796858d0c32520a6d52efe52463f9d0f8d8670a.jpg)

![images](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/images/3e133076c2d730d59b73105567ac5e2c746cf9a5cd6b5bb1809689db6654f814.jpg)

![images](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/images/68ce15255fb96844fa771c2ef5ac817703d66fce1742d58caeb9ac48ae449c19.jpg)

![images](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/images/70ea3f0538699c10d07f1f0c09816867a97a15d192a9a91c0a614e98b8d4fbee.jpg)

![images](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/images/81de62c2ee386986a2d9c86a0122c6d3396bc6fd4a4cb3fce58b073c42cf6ad7.jpg)

![images](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/images/8cde273f9a6324caeb613371773bc4f7cae4f1fd45f90a55c0f88d92a260b524.jpg)

![images](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/images/91b394c851a0253d87f2f37f9212ad4eb2be4af33bef6da011330a6fa8cee4e4.jpg)

![images](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/images/a386c0f59c5f09a468ab381d14a68d692f256e72ba4a49f94ca46a8387764b76.jpg)

![images](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/images/a64a624792709d8b7ca391ea55d78645d6d5c905b4e1429d841b827950f77c27.jpg)

![images](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/images/b6c5b17068e10638e98abaa0081eacb6f6f0511cebee47413ff720e3ad4325d2.jpg)

![images](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/images/b93729ef4751254715f7c562d5c8e36229e81580c367cb2f3b7d797c47c71020.jpg)

![images](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/images/c2d9ab341474fd3ce051770599fff3617ff7a0e254eeee7a77a84b8241469580.jpg)

![images](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/images/d3f7de2ed0b3af6bbb6e11a7edc1175bc675a56c707064aa460f144b2c55e860.jpg)

![images](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/images/f4d9c483d047b9df88a5efe62f8564f077153176d0a6ed1eb4d7c61726f34243.jpg)

### Tables

![tables](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/tables/7e9d158953b146f9322486a478f8c7c717d4c4d480ac139e1c8fb69159020998.jpg)

![tables](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/tables/ac9ba9704097f4b4f6068d632bd055a4c68baaf65cfa523c1aeac386eb0c5b77.jpg)

![tables](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/tables/b923b527dfd4c8a3da438d76edeb9e53c3e2b2cc5965b964caf3c4db790e4023.jpg)

![tables](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/tables/d0147e29b62d4c728833fe7c921a4a5b9af939b808b56cf0ccc394a3ffa6fb01.jpg)

![tables](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/tables/eed9d073f592d9592adcd873d88d76d84caa60513ff52e9f9c0b1cd5c718eddf.jpg)

![tables](../iclr_results/1038_CoNavBench_Collaborative_Long-Horizon_Vision-Language_Navigation_Benchmark/tables/f938ac5e109db43c7cc00aff9d8935c29b915968de8b2fedfccc5a00e3412b8d.jpg)

<a id="paper-1039"></a>
## Joint Optimization for 4D Human-Scene Reconstruction in the Wild

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=7eLE4mfEpz)

### Images

![images](../iclr_results/1039_Joint_Optimization_for_4D_Human-Scene_Reconstruction_in_the_Wild/images/037ee1f8fab52149be6fe735b7afc8f43748625b191aee106e081089a934c429.jpg)

![images](../iclr_results/1039_Joint_Optimization_for_4D_Human-Scene_Reconstruction_in_the_Wild/images/07e4762a49b421f81b9f303e9b9daf40f6bbe8d8a806933cc5ec04ff538c2735.jpg)

![images](../iclr_results/1039_Joint_Optimization_for_4D_Human-Scene_Reconstruction_in_the_Wild/images/1772dd6c979167634eed0035c2944d6aab61825be1cd1261f80135adff88e852.jpg)

![images](../iclr_results/1039_Joint_Optimization_for_4D_Human-Scene_Reconstruction_in_the_Wild/images/2f7f5705d34e4e7a0912fa50b48d06d9554c47a6213fad141f3f7d19e1193b54.jpg)

![images](../iclr_results/1039_Joint_Optimization_for_4D_Human-Scene_Reconstruction_in_the_Wild/images/7833411701af8aa8db2864b901fd6083370c10efb32fcafcff096635a07d6fa9.jpg)

![images](../iclr_results/1039_Joint_Optimization_for_4D_Human-Scene_Reconstruction_in_the_Wild/images/81099956a924f18d3f216b83f9fb05a6885e96f1f84f7e5fb8e1de52cba57166.jpg)

![images](../iclr_results/1039_Joint_Optimization_for_4D_Human-Scene_Reconstruction_in_the_Wild/images/acd98d1267f9d1e8667411318d86a23f784c760d2cf2bfe62408d014889bacca.jpg)

![images](../iclr_results/1039_Joint_Optimization_for_4D_Human-Scene_Reconstruction_in_the_Wild/images/b84747805c593e68cf2bf34e5848eb2a585462d8212d87258b8974191215af15.jpg)

![images](../iclr_results/1039_Joint_Optimization_for_4D_Human-Scene_Reconstruction_in_the_Wild/images/d330a563987ed0e8c46ba4d2d3eaeb28cb6dca4fe54834feec2cafada282cd92.jpg)

![images](../iclr_results/1039_Joint_Optimization_for_4D_Human-Scene_Reconstruction_in_the_Wild/images/f09e00edbbb72e95e84efa1ee047425a8d8f5e152a394880b3cc41f549a63536.jpg)

### Tables

![tables](../iclr_results/1039_Joint_Optimization_for_4D_Human-Scene_Reconstruction_in_the_Wild/tables/0b3e76635377ea68027cfd0763dcbec6a21504df99ec5cde8e9c30bb25dd2cbb.jpg)

![tables](../iclr_results/1039_Joint_Optimization_for_4D_Human-Scene_Reconstruction_in_the_Wild/tables/87f77b5eb6e16152683013bac81185cddd7b344130f3ec66a803f0de9d238d05.jpg)

![tables](../iclr_results/1039_Joint_Optimization_for_4D_Human-Scene_Reconstruction_in_the_Wild/tables/9994c6d9ab0d944e841e7c1162aa976508453b312007382ecf642af9b44ea4aa.jpg)

![tables](../iclr_results/1039_Joint_Optimization_for_4D_Human-Scene_Reconstruction_in_the_Wild/tables/b5fd0c0142775fc427c5403a5d12fac45046f2c9e49ccc723581e1075f77e0dc.jpg)

![tables](../iclr_results/1039_Joint_Optimization_for_4D_Human-Scene_Reconstruction_in_the_Wild/tables/c5cdbb1549ea44bb3087542edd8e5ff64f727124c2dda9fa4ce81d7bcd5c50d1.jpg)

![tables](../iclr_results/1039_Joint_Optimization_for_4D_Human-Scene_Reconstruction_in_the_Wild/tables/e8f44c158bf9acf0f8e5346c433185f49006f6c5897c31a4671579a946b0257c.jpg)

<a id="paper-1040"></a>
## EgoHandICL: Egocentric 3D Hand Reconstruction with In-Context Learning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=nwjy9BeorI)

### Images

![images](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/images/0b3ec9e621e4891e7f0fd632ddd318ea8d0afdc229b71fab69798acd28e3b70d.jpg)

![images](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/images/341616213ee6da0b870f122007904b0b8a5911cb151e6ed7a604fab9b2fce455.jpg)

![images](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/images/37a046f3e230adbbd58b528a63d728da71f1f05eb973a8f60c0a1f631e9b02a4.jpg)

![images](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/images/5774b98951a8d12b5d36624236dd4e713eae6c8860b9d56196b159de42f545d9.jpg)

![images](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/images/66bfc937f6f43166e97c430aed6d5d67b79726a8c534742d5a4b1c04bd36e895.jpg)

![images](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/images/718d0841283a4267a1e8027ef62740dca44d36c6dec0cfb2783ac706402415ef.jpg)

![images](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/images/779ab31ede61a2645cc91b51d03e2fb4e3e64ea96cd4fd07a0566302f46dbd2e.jpg)

![images](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/images/930d5657911947e34eac9db11fa10a6a4ba4bd285316319d4b01f53c4cde17c0.jpg)

![images](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/images/99b1b90e28b7817c948099f631f44585e5c580687ae62d12b143a2129e983aba.jpg)

![images](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/images/b8891e018affd6db2eb883020d3d6cdd29217ab0cce72ae039c8f140691f97ee.jpg)

![images](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/images/bac776c0870bd1f18032b392b8a0384a853d9f377852338a913a651d82f98f33.jpg)

![images](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/images/cd1149cab0b09514db5adfa58a42575fecf58765a7aa8cf3a004ab6e9d360c04.jpg)

![images](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/images/d917b8765a34db723b59e312837c7c640f94b66b86e5642d36431f46e8fe862b.jpg)

![images](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/images/dfe7c9a35fa713e11b21b7e30998651ed95cff20c3628ef5f855d2217aba2a0a.jpg)

![images](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/images/ea3f67193e583fbf9b8474e6db46afffd783ab0ee1c871f1d0ac089e8fd7909b.jpg)

![images](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/images/f0a7460798b5730c3a8cbf47a6931cadccfdc634fab0310d5b63a2177a6b8554.jpg)

### Tables

![tables](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/tables/1731f28722b11482c74a0a9cb3624500ae67cb690b278f94d3b267b84cb822be.jpg)

![tables](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/tables/3adfce4ca1dec9c45b7000d81633df1df5fc655299c74699de1220f87130ca06.jpg)

![tables](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/tables/491b12fbbcf6cdbc92a37d35484c46756d341e684a301255eb708a2a5126eaf4.jpg)

![tables](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/tables/5c0a9a1edc36dca00f861f4743ca0372a606a6b33f33b2c99cc000b7b86f2a9a.jpg)

![tables](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/tables/6f7d3b6ac1031a873919cb60e06a2cd583030184a59af8d9b8ab86a5bc86d16a.jpg)

![tables](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/tables/81771d6a11619478964b6e0482cfd30850bdf143606db1a853a3e00e806888a8.jpg)

![tables](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/tables/980d5803b9e48558a1774f0d45b9249aee72ce3172634931dcbbf116b2caa80c.jpg)

![tables](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/tables/ba5d4453fea01bcb8e61b40a656e49431f6be33cda3f660436c1f77b40b74102.jpg)

![tables](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/tables/cc14d15935771689c6bf8d972abd93670d79d1b325689b9d9a43bcbb8eb80fe3.jpg)

![tables](../iclr_results/1040_EgoHandICL_Egocentric_3D_Hand_Reconstruction_with_In-Context_Learning/tables/ff72bc98a824ca874e2ca152f5f4691710fd1c4ea4a9e8ced9e5deede89f1f9d.jpg)

<a id="paper-1041"></a>
## World2Minecraft: Occupancy-Driven Simulated Scenes Construction

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=dc90uPqxWF)

### Images

![images](../iclr_results/1041_World2Minecraft_Occupancy-Driven_Simulated_Scenes_Construction/images/7e1a3f5f00c18a45464b435c33a72c7861f65e31a86c198e20435d535c0f6ec4.jpg)

![images](../iclr_results/1041_World2Minecraft_Occupancy-Driven_Simulated_Scenes_Construction/images/900b14d22d3929871adddf5c765af1f7b88dde8d432c7ff59e1a2934715e5b5d.jpg)

![images](../iclr_results/1041_World2Minecraft_Occupancy-Driven_Simulated_Scenes_Construction/images/c23f95a4eb6610fbb0f703c8e88fc11c6ec5e1f6fb22aa20bc8e4d16910fe1d1.jpg)

![images](../iclr_results/1041_World2Minecraft_Occupancy-Driven_Simulated_Scenes_Construction/images/c290ffa2044f430ddb1938065bf64ad20364327244fa88a5dea5e67a1cd2051d.jpg)

![images](../iclr_results/1041_World2Minecraft_Occupancy-Driven_Simulated_Scenes_Construction/images/e07b3971923fd6c94a077a235308f904053dbc9d1e63ce1babdb2162f7d0daf1.jpg)

![images](../iclr_results/1041_World2Minecraft_Occupancy-Driven_Simulated_Scenes_Construction/images/fd7fd9e50667ffac01a12a294e1c2142142bcf62904f4194741e5e088c08d57a.jpg)

### Tables

![tables](../iclr_results/1041_World2Minecraft_Occupancy-Driven_Simulated_Scenes_Construction/tables/1b2d09cf6c7a5fa872005bed14bf0b2acbf91856a090971d9f249013f9d934e4.jpg)

![tables](../iclr_results/1041_World2Minecraft_Occupancy-Driven_Simulated_Scenes_Construction/tables/2ab116a42b18aa59d2db3bf6b6f0e1a2036d3ece7255daece764de04891f75fa.jpg)

![tables](../iclr_results/1041_World2Minecraft_Occupancy-Driven_Simulated_Scenes_Construction/tables/4a42d5851b429fad8068f177670a1cf7f35560bae60cdf91fdc59d0a545afa50.jpg)

![tables](../iclr_results/1041_World2Minecraft_Occupancy-Driven_Simulated_Scenes_Construction/tables/818fa29d7a03f608d6e8fdc0a815d6ec5ef0010c8488d13d9a76ef3d28c9d037.jpg)

![tables](../iclr_results/1041_World2Minecraft_Occupancy-Driven_Simulated_Scenes_Construction/tables/8e002ceb09043f5598038d086d9ef70a6cee6d50ecfe557a9d8e91cb97aacc27.jpg)

![tables](../iclr_results/1041_World2Minecraft_Occupancy-Driven_Simulated_Scenes_Construction/tables/a108519f731f2381a10c1c7dd608b79622ed921176681e3b7c143dd0feefb987.jpg)

![tables](../iclr_results/1041_World2Minecraft_Occupancy-Driven_Simulated_Scenes_Construction/tables/a898c2ebce47b6c7cd032751c11307b7c4eb9f2808b412798c1d03b87c1e22aa.jpg)

![tables](../iclr_results/1041_World2Minecraft_Occupancy-Driven_Simulated_Scenes_Construction/tables/bfcfaa40b14c73d87ddbc41ed46c2be06baff24b1df8d5a94ddbddaa755b235f.jpg)

![tables](../iclr_results/1041_World2Minecraft_Occupancy-Driven_Simulated_Scenes_Construction/tables/cd2bc1e9aff233d9c6a4a0eb9a3fb032eebed1b755274296561db2cf22bd5d39.jpg)

![tables](../iclr_results/1041_World2Minecraft_Occupancy-Driven_Simulated_Scenes_Construction/tables/e87ccf92e1e838bb3509d2e0380ff3745055489d89c73310dcae8c0cdbefbda9.jpg)

![tables](../iclr_results/1041_World2Minecraft_Occupancy-Driven_Simulated_Scenes_Construction/tables/f67eeea91b283b2df9e5078b547d5d44cc7e1358f676251a5c68253e8cc46e51.jpg)

![tables](../iclr_results/1041_World2Minecraft_Occupancy-Driven_Simulated_Scenes_Construction/tables/ff968f7d05800fe8745ac23d6c210f7680f12870e5e061c3cda21e9515c113c1.jpg)

<a id="paper-1042"></a>
## Scalable Training for Vector-Quantized Networks with 100% Codebook Utilization

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=juM14y0caI)

### Images

![images](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/images/0b1228de5e771bc0772410402971ff553cb5090902cc89cd1f89b9abd4677474.jpg)

![images](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/images/17da430165305d89b2cd26a9589942e3943be0ca6e18d32ce985fe9342f2ffd0.jpg)

![images](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/images/2ff594c10947b75c6400c003b85f2f4fd06f625c981c0cebf31f9ab90af771da.jpg)

![images](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/images/4cb73d18e29847ce7377fd20b0083d2256f4473ac1529e0316d551583e0f30b3.jpg)

![images](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/images/4e38b1205acd42a2ad79e804aff53fb6694422e75084aef001eea3c350a1597b.jpg)

![images](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/images/6254192a5fbd84e7a245039ab3ec270308c172d5f7979363bf75cc5a3eea1d06.jpg)

![images](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/images/82ddb0b0b0552b6ee1ac6413c2d892f83d5b872631826f9d34e7f0c66b3a636c.jpg)

![images](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/images/a4111f7f1a9b79eaecce44dd438d2456c5088ade325eadc393729717b3fbe7a9.jpg)

![images](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/images/b5c005c145f7fa9c83e6e7d1c8c864172e90017178b5f1b9f7c69d72d30f667f.jpg)

![images](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/images/bc667075c481c8915819d0218f9c41e848b50359d1db2a4f71af43542a9cbce7.jpg)

![images](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/images/d896a404e9a88d2428712e1f8908a4c4dfed1cf517910c9bde4978418ad9f259.jpg)

### Tables

![tables](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/tables/2073b5fc8ac4ce0bf42e415ae0b791f9b2d96da0780a9602759fc72823c74337.jpg)

![tables](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/tables/31944420fe239010ba455b8513edf48c7947bce89809e8fd0ffc745f6cf9a036.jpg)

![tables](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/tables/35a3fd63439435e17b5430668001a6d2bcfa1e0ab3df07e4e5f3893ad0335d66.jpg)

![tables](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/tables/4917bb780e830ce95be891d1b28f76297638b8f1e7489afef17b495d771b536c.jpg)

![tables](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/tables/560c7facbded3d812d586182c713d7e30cf467aaa89eb0052b1361687c346d81.jpg)

![tables](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/tables/73325141d7f608c13fd366ebe29c33faf46d431afa8693a11c660df9decba355.jpg)

![tables](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/tables/778fd70aa4294d720effa114edd5995e2fb9b8b21602291882026ff908524926.jpg)

![tables](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/tables/8b5a477f2c4425a2f0e592f7458c627ed0e87ddb64d7451575560e1825094bf0.jpg)

![tables](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/tables/9ffd7bd1c83de0f6266a6596fdbec23e98fcc5f43c3384c8177c79b2267965d1.jpg)

![tables](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/tables/d192c8503fbcfbffe4405155787da3655838c3301d0d8da6254561ada1af7942.jpg)

![tables](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/tables/e2212d68c7c2fc63738621ba2659c4a747022b7fa34571331a685e68ecd040c1.jpg)

![tables](../iclr_results/1042_Scalable_Training_for_Vector-Quantized_Networks_with_100_Codebook_Utilization/tables/e2877613fec3ca1d10f0a5477c3c8c57c72305710c6d3cd96c4bd7c0d3d4921c.jpg)

<a id="paper-1043"></a>
## Native Reasoning Models: Training Language Models to Reason on Unverifiable Data

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=abAMONjBwb)

### Images

![images](../iclr_results/1043_Native_Reasoning_Models_Training_Language_Models_to_Reason_on_Unverifiable_Data/images/a68964572dde634d46d08d1acc057a83c2f4e60a4f46836fdc3688ea8e6261d1.jpg)

![images](../iclr_results/1043_Native_Reasoning_Models_Training_Language_Models_to_Reason_on_Unverifiable_Data/images/bd824c50636e1c71005a3ff5145318fb7e23b944ab77414352feda3b20d045dd.jpg)

![images](../iclr_results/1043_Native_Reasoning_Models_Training_Language_Models_to_Reason_on_Unverifiable_Data/images/d0641f27d20a669484ab8c693587e3ebadc752ab2b137ebc3089c0d3f21ca7e6.jpg)

![images](../iclr_results/1043_Native_Reasoning_Models_Training_Language_Models_to_Reason_on_Unverifiable_Data/images/e093ca98c07856bfbe67abec601d29f4e55700b8d2da8e657f1ef2b522061135.jpg)

### Tables

![tables](../iclr_results/1043_Native_Reasoning_Models_Training_Language_Models_to_Reason_on_Unverifiable_Data/tables/0f9889b9c3042e97349594d6cb987b4cd51eb37051ddb21f8079de4e445911fb.jpg)

![tables](../iclr_results/1043_Native_Reasoning_Models_Training_Language_Models_to_Reason_on_Unverifiable_Data/tables/16521c3da3ea9dc76dd934e98ef8ab6b2bd6357b25c97644ac0b170d7af1d215.jpg)

![tables](../iclr_results/1043_Native_Reasoning_Models_Training_Language_Models_to_Reason_on_Unverifiable_Data/tables/18e62c7a33979e2b35fe6319f2b5a6209e2fedfab4a589f67983b7e4a8ef5727.jpg)

![tables](../iclr_results/1043_Native_Reasoning_Models_Training_Language_Models_to_Reason_on_Unverifiable_Data/tables/491bf0b338948674ba7b521a7e731fd7db56c3527e760d5983b4769003702fa9.jpg)

![tables](../iclr_results/1043_Native_Reasoning_Models_Training_Language_Models_to_Reason_on_Unverifiable_Data/tables/4b9f397f869bfc3c995f0842570d908101b66a7faa9552a3382b9d3a9e29fe4d.jpg)

![tables](../iclr_results/1043_Native_Reasoning_Models_Training_Language_Models_to_Reason_on_Unverifiable_Data/tables/7284c5a17047d76c1603ae5947a95e292d85cb9aa7ac984d032c5f09e6db1f56.jpg)

![tables](../iclr_results/1043_Native_Reasoning_Models_Training_Language_Models_to_Reason_on_Unverifiable_Data/tables/93211ec1449cb68d9ba773ad7a3cb1d4215d119497bcc08fe34e68dbbf207458.jpg)

![tables](../iclr_results/1043_Native_Reasoning_Models_Training_Language_Models_to_Reason_on_Unverifiable_Data/tables/b3ad425b9701e2a937d7e71f98d261688e1f885909fb4702a0d3f12d0bdadc6a.jpg)

![tables](../iclr_results/1043_Native_Reasoning_Models_Training_Language_Models_to_Reason_on_Unverifiable_Data/tables/b41e290a93af7818f52bfd2cfe797126144a97bb1d611d8ec45e9acfb924e84f.jpg)

![tables](../iclr_results/1043_Native_Reasoning_Models_Training_Language_Models_to_Reason_on_Unverifiable_Data/tables/e3f322e9aae74175bfa919861f8e5dd52c4b7ffa8ed5d10f0ec6fbbddd330ce4.jpg)

<a id="paper-1044"></a>
## AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning in LLMs

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=iKRQMeC7yO)

### Images

![images](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/images/199b6e89d7df00937a001ab5697efb7c572df89d76c14127d3c20f0186eeded0.jpg)

![images](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/images/2513f78f21a8d6c29da1aa054b5b59cd5ab298d898ccfeefc9db39416d977b45.jpg)

![images](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/images/2fd986e909831d67667397e1a164ef521c10ce948546649e593ddf87bc22d549.jpg)

![images](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/images/533fc9bb80223a52cd43987e1df248d510d36b373a3f97de719a50df0756ce5d.jpg)

![images](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/images/5d4186682d24ba6592ae0f252e1c2ab0bf80fd2be2259bf886a6c75aa96f8ff2.jpg)

![images](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/images/65c8ce5f629430f7dfb3905eb0e357f895a50604b52642b59acb49a71570be9a.jpg)

![images](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/images/71eef941f89053d745e196ac6017aca1e51c17de8f1a74d6fb8e554959b3f574.jpg)

![images](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/images/925e77457326cda4b2a25e7f5c87339e9eb58104ccfd04fffaf79bd7c90d5d84.jpg)

![images](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/images/b0d9fa66a221e8de331e332ee50b096231a933f526d2eac539e2c0ec014a7f12.jpg)

![images](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/images/d9b884c75327dd30c379fafeb63ca4e0c62e22b8c8d2ffdc2684283b18ce4947.jpg)

![images](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/images/e619bf8ed7dfbe8e60dcf65bf3f45b0c15b2e22e2f8ff44c868a86a3cd2f55ab.jpg)

![images](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/images/ebef59cbb50c98f2ecb25cf1dda71e54e86bf7fbc53c1fc86e4f11ea8780b7cf.jpg)

### Tables

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/03fe7bb864c0df49a221725b83e72d1759d8a8e15b3166d953b642f8ce8b19ba.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/0c5f2d8c443a301113aa31856892639906c1889a4f37b13cabe1f151a7147cd5.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/13c1f0502add7f05dec65148e77796847c5154753b60c87a4f008cd332db84e1.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/22467891ca7eee0ff7fe68925f7956971c97a3b8d8bf8f5a3e22b9f193b78d47.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/25824c104690c54f38382c1d03cc06185340a8a66ef290a9b1fd1eb513437a8d.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/29986a1448a14cd0368375ef6734432c4539babf6b5a0cd0f6cf6b676b835ad9.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/362d584bb1e8b07df7279ca2ea5c6c687abd233f390071c5171027ab688d823c.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/4da7ea2f5e59004ed0796f7e372302ac6fc9506caf0b024cadbc1c2db3cfa64e.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/582f4f4b8c9a68662db9d31d4ddb9194c56b6098da552141be9ef79478cd421e.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/58831e1c2334d6e9ecaa7797669f45e41a1a4ae3293b289b3a9727724d3ede34.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/5ed06627cdfbf16198baed613ebae80d9081790a2c2770ac1f0efa6f09f2af1e.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/7f7400ea55644bf6f40d33862e7d6f5689c3387c3db78582b5fe24295d231a8a.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/823ac3dd725c5b63f61d766b4f97f9c25d13d2f1865b7d4f2fe2c53546f81d6f.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/9ab14ce0919589e5e3ea747e312f5c228b916fab1cbc50a6bac646adbef68b1f.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/af145b775189c71639ac238fecf863fd998aa6b041d4e1acfea56a0857b8af26.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/b205db48742c7f106a65b7f03c3648f045ed935f7532bd2502acc073986d2006.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/b92df06c08f2d9b5a586e47a199e15e89ba0c8b3bcbfb045180eda3ecdac91c4.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/c332d953ecd549c9b525254625e22d1dd447caf6b685cf1aa8d469c23ebf0cbf.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/c6deda2aa71e14482821e111b850addd54a1ace9add09e34b2c5e6ae54e1eb96.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/c967467099a5cc542ad8ee9a175a510c04251d9cc75fc7038053a9431b5b43f6.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/d7de19737717be60930c2b04ed2cfaea87720d85bdd416e2107f9f0858a33dcf.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/d7f590c784ba17dabd78ba254e491d27f9fa5ed0f53529caec64813abb6d12eb.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/ea34eaeeeaa4ab05120dac8f0460e9a4f0cd3e6108c80d7b540469f478dc5bfa.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/eca6ad3c066a145b4901f56717199748497a28439f7017bc0e74f936b2921e2d.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/f84f56ea3dcb6a6dceff72c05d2d61e1f961b2ea8d2026b4833a8398ca74e268.jpg)

![tables](../iclr_results/1044_AnesSuite_A_Comprehensive_Benchmark_and_Dataset_Suite_for_Anesthesiology_Reasoning_in_LLMs/tables/f9ee2c2e01906601acbec2ff16bca0c71cff7ddb6381b7bf1f08d89e7122a7ca.jpg)

<a id="paper-1045"></a>
## REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=vmBIF25KLf)

### Images

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/022031449523b4a2024364430d174b3ae3bc01e430564389781f0f0c728000f2.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/1f45dbd660b524efbc0c2c539177432b021216399b952d19bfbc84f80aaad7cc.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/2df0d0b0093bc8ae015d137d8dd8f1302a476082e5c1b9931759a0d9b667695e.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/3a355bba39d7c939e3ab1184d28da9ec9ad622879c1d1a4044b205ddefbeebff.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/540ef11fc1270cbcea5618526b25a5ce3b1615beb6e473ba641322c4641d5c10.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/5c9a5d28c5d654384cd1aaa9f676796f46906fe3a94ed964d3a12e80ed69e4c4.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/6066d04a09abce9f24e51f45bff8abefede19595cd815a27f5d668e011ef65ba.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/65d08ce9cc94b0d6574c355b11aded7f516e65cf6a8ea5064f1d30f675657223.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/7507accc8c339144ac694ddb0820db3fc9bd5b99a6f860494c38994bd2ca3f86.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/8b26f4ea581d9fe6969d30d17b89b62cbb27f7fb97f2974062200215bbc8e076.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/9fb3daaa71a4c0754342fa2212fc5b3a3493a39e27b0a9faffd3e8c9909fbeda.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/acc7797b64d7aa7b6461bc5a762154be773ca8f94a5d06c5d99d8eae6f756822.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/b61c342d82311986b8319221ff4c7462877d4e3ce4fa77ed7165a585fb3748d0.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/bacdc662112657d8ab10251e90694f05f31aea92048ef72c7a8a7392c99a8da7.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/bb7b8413a99a8892cdea5ddef9d1e144d3c1a01f36e39c7155b723d774ee0f5d.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/bfc86adee79453845da1cae958bbd66840b2b43559f53f7af386938763d0f15f.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/d1e7300b51cf14893725479aec601a8de5c99805b886b55129d81760e1c4941c.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/e89f8812670304aa726505b9f46177e3b58dfc6a63683020709e1c7cb2b2f863.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/ed959453d845283e4d31f9b732a2954e4b4442d910cd5b0fe67f002b6734aa53.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/f1853fffd420717f731f3a171672860ace710d157f5c046f233102e04ca0f154.jpg)

![images](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/images/f2517bda814a69293ebc5b5f7b696ad992eb3d619fa77e4bb72a3ba1e04d9d21.jpg)

### Tables

![tables](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/tables/15f37fbb895a362618e937bdcb3e664668c2eca2f2e780fdab7a7ebc145d75c6.jpg)

![tables](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/tables/236053f4c6f23bf6f301563baf2116ed78ca29b4e064ab68346c7d8e01c56888.jpg)

![tables](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/tables/362d97f917815d8c967fd2587fe449a3ecf1b8d5984e18ad0fabe334168e96bc.jpg)

![tables](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/tables/4788360a17b02380f52da33cb2015122daf818fd682a3fc95a0374019b224187.jpg)

![tables](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/tables/644151e3e58675ac3fa535da003f23539fb131c683534e59e08b804a03966846.jpg)

![tables](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/tables/96344799af93969fe18b4ddf84a2112e7a261ac8926bfe8817535ae728159fe0.jpg)

![tables](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/tables/add69782f71f5b2574f8ffb5e193689e921188a35f7cad0bb9253720dd6a1357.jpg)

![tables](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/tables/c8752342fba0b8a97c134ff9313d14c50f031bfbe429ba9d73b23d5a0869dc61.jpg)

![tables](../iclr_results/1045_REI-Bench_Can_Embodied_Agents_Understand_Vague_Human_Instructions_in_Task_Planning/tables/eb9516dfaa5d669c2ed21d749682ebf662fbdcd21abdfa49c077683a68938e81.jpg)

<a id="paper-1046"></a>
## Mitigating Semantic Collapse in Generative Personalization with Test-Time Embedding Adjustment

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=P7sPfvq7Ih)

### Images

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/01d79c2b12861e45f4112b30f83886df71ba1d9873ff75dde34c2257bf66b444.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/18b6df36355a86d41c542c7e2e18dc2ab0f36dea02a7ca764e12c9f258f5492e.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/1941d9b3b77863e49bae76c52badabed77dffb458bb131f2b66421ae50afad13.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/272c226f880e836ef839c0b2a76a26901864913e827d6a067586baf86ed19838.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/3376342f006e88996f978bf77929f627c706bc8d214a2cc0d87a5ea75c43ab26.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/3951b311cf913e44698cc34891476135473c9a3ca2eca61c9b87fdef50111f0e.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/4b84b55b26ef5a950a968cb6d0c2544c8b2f63536f53eb9227e33a4ad70b5a8c.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/4e2a2958454219ac5817e807a0bbb86738e61f3a2188375cdf8fda7e8bf124cd.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/64bd13dc97f42f9e1d536ffea12f3d1978cf0f33b228300e94b0622de433a80f.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/73bcb36fafb069ae529e1cc463066af3c1ff1583f9776f28bb4d223169749ac0.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/783e4b7486e9b8ab35686cacb2726846955ae75691d5ae89f58fa011d44f7320.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/7a1ef9885b2bf2f0ab62568e1605a901a897c655d64a76543b9709f91fd0d057.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/7b8362ef4a4cc67fd7756d8099b7b035e86407fded19b1368dd454167c7e5fb8.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/7ea5969a980dcf9907c92f923c13e53e16f2616cddd9ede792c0b896a091bd1b.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/7f209c6d2dd5ccfd66608e1963d938f0d34f97fd6e5fa90072d6c5f80a24c5f6.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/8491f622176c465fa0ee5654eee58e1a277a15859512607cfd00b5dc19c960f6.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/8a8a82524d87d80aab0d37febd6c1e11002fabcd1dfa78fa358fe8879a2fb6cf.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/93f97faa7e4a255b9ea323b2fa1737228668fd94a447c0d0b72d88667d219e36.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/9872619f6eaa740ac2d22984ea0d4f431ec2c4f2f2cbfe9635dbdecf34f6017e.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/99b56dc351c9ab389c11623beb1e0ac083a8fe5cf387ca5c26d0bad86b91ffcc.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/9ff5e959e1d8cd432b1294a359f5b95fb630d0f666a00987cafb0bbb6a6c72d5.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/b3d1e2da074aaf4ae0e8180cf15a1053a9e7d628332d1844ca5a89f90fc2a063.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/b9502f51403f106682c34b90a98c3e254d0435bfa08ece939b95b5117f58f994.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/c852a86bf0ccc0f53a61a238c454744eecfbd967ca2ed35a34fd67122474c550.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/cd65b9dd21653f5c59d39430dc31239a40b62aa92b7db9321916da0905c29f4e.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/d2498bd07e9e1949f1f363e804ebc916c6b0fdd209c765217096bc690e4736e5.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/d820b4e1c001a131de60d26870f509a3766faa06f1206f6b2736ea73e68181f2.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/e12ab31cebb2ba4fef67f193dee3216fbde06a09d9498db71df2e4d6c3f29462.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/e3a36082d0e1f371100005a8096f0c38a10e8c266d8c4ea06a05369248bc51a4.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/f0ce5eb1aff0844823f0d19ab36e80c83fb297056746bd554e0255283c04fec3.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/f26a5d34b8cf6f599eb88b1660351b4abc48f07a51bfd81c134de0cdd07296e6.jpg)

![images](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/images/f8aeca717d851c23ba046d5f27785b8a16291f62283da6cddaeef070d5ab1809.jpg)

### Tables

![tables](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/tables/03b7f14d8806a504ec3641c6972bb634ed36b7e47000acc1f2d9af21931e98bf.jpg)

![tables](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/tables/1a696974b7eb9591bae6802abec06835d585936a886c716c9d888ce3b0d39708.jpg)

![tables](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/tables/3c677d19b0de5c17250decfb55d831d4b3a1f7197706dba2053b221e5ea52ca1.jpg)

![tables](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/tables/496be117e3621180ab4be3144a88bbfef771d61665501a9c8720bb18c234aaac.jpg)

![tables](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/tables/96c25c7741137a313a21a9928c8d54164d0b04afd9fef51fcecd4790308d17ef.jpg)

![tables](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/tables/ab18906524ee67c492cd847a46ec5471e1d911031e2e440f88a5124e70e64e48.jpg)

![tables](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/tables/cb23fb6212c8cf080e7ae5b9527495fdc81953b4bf97c1843393de7d01fcf8ac.jpg)

![tables](../iclr_results/1046_Mitigating_Semantic_Collapse_in_Generative_Personalization_with_Test-Time_Embedding_Adjustment/tables/dab6bb0172601ca03c4701d79f74ef4ffc8382d2f43c7247625ef3487ed1dc62.jpg)

<a id="paper-1047"></a>
## Mechanistic Detection and Mitigation of Hallucination in Large Reasoning Models

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=XU2STJa1Fi)

### Images

![images](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/images/0cf4c2c64c50b7b431b3260e393a13af4239b3485413c91ec61453805974ac48.jpg)

![images](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/images/19a3e92f5a5e9930a2e843d467eaeab7d3d89b2dc11d9c548561789f85050707.jpg)

![images](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/images/4313ba473d578700136b97c8b28600170161cc9a5f56d0d5f231043d4d68893e.jpg)

![images](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/images/70b2c6a1d29137ba814afa53bb736b3e83486f244070c97e2e179d35b3208653.jpg)

![images](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/images/7dfd3962864ed3a1af3409551d3837109c431daff386e10dc44e85919d9c4459.jpg)

![images](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/images/82a3cebb4b24e8cf58e9f2813bea09b823a3668ca301e7a4d72c0657600dd512.jpg)

![images](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/images/8d9ca7abeab3f41e499251e867c83937d097b401b9da8f8c46cb861cd0d07d8b.jpg)

![images](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/images/cea910c0cb6755fcd1d5e5978d46274af2e7b61b2e1120ea8e9fabdd4d3e97c2.jpg)

![images](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/images/d4fb6a4cbf6956e1d7c59b60c8252112720ff8e1eb7635b6e34de776e0d922a4.jpg)

### Tables

![tables](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/tables/07fa4d147c75f4f0fda4a62686cc077d4d3c069705da7d2f66f9feda8a33e6c9.jpg)

![tables](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/tables/0afb29db1d0662fd1e52e1ae97b2d01168b6a794991e4eb78317093012e17e0f.jpg)

![tables](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/tables/0b48ce794b42cbd6f69b43eec6f559ea40d4734ce5188bc000619ea36f59ef47.jpg)

![tables](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/tables/0e608bf3cecbbc037a310c4bf42a87548ae53f65efab110457a2fba762934c77.jpg)

![tables](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/tables/1801b03821a0cd9d004e7e52eafb7b59deab7bd137ffbfc888ee2c178172ca4a.jpg)

![tables](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/tables/18728d6a6a2817d07194bd8038b07e7914a21417aea45900212d499cc6dc1057.jpg)

![tables](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/tables/351d551b8d62097e3dff005f9af20624fb64f556dfc7081ecf595e5902b8ea19.jpg)

![tables](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/tables/49722cd8fa4a2396f70a4c9a29499a2baadcd8168ab26ada19a31c083f25be82.jpg)

![tables](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/tables/64aab1ebe88ce30627f57f1b87996a6a250596f5e7333b94df0801b0401d9e13.jpg)

![tables](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/tables/86db9050763a48f023af4632416c8c87fe318c5b66ca73245e0c61c15297dc3a.jpg)

![tables](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/tables/91e7e7a52e4fcc6c9f8b69a47259504baf03b926fd89ef02210f64ff9ed50317.jpg)

![tables](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/tables/b3a8876a04bdf9610f687222cca5d837cd256dbe72852d734dd9235726076b16.jpg)

![tables](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/tables/ba502ad70f8aa2394bb79d35ba719df40267a2dffeaeecf4a17951fec5fdac93.jpg)

![tables](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/tables/d52e9c8d85ebc47cf0925a6fc3585e51ad244d94ce56446158ca5dd53fa1e0fb.jpg)

![tables](../iclr_results/1047_Mechanistic_Detection_and_Mitigation_of_Hallucination_in_Large_Reasoning_Models/tables/d9147eba8f62248558445bec71ad3d1d8a2f4d5c48d11746277c620cb218217f.jpg)

<a id="paper-1048"></a>
## An Open-Ended Benchmark and Formal Framework for Adjuvant Research with MLLM

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=moeOrHkDg2)

### Images

![images](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/images/20c8e1d944578e670c0eec8102657eb0eae75ce87c3b633dd4f8c5a840be2cb4.jpg)

![images](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/images/2a9a2ee1e78d7aa69abee5286db5fccc4a028c3f49609aeaf66c79f4fd23b127.jpg)

![images](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/images/3949b067d58afdb532c65db238afa98688f5cbe377f6257d0acb7b5a9b01c889.jpg)

![images](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/images/54885868969b50f3a1f9efc4e77a8bed532e5b7ac29af6e35725842e47ebdc16.jpg)

![images](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/images/577ce21c24b5486fb0c60a218bd262e3cfa13d2e192ff8a972975759ee8648d2.jpg)

![images](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/images/7283e2777015578249577e2ddb505d314ad048d78149bc95b8f46f59f3a63709.jpg)

![images](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/images/99230d1b23d7723108c7b132541172ebb2ff04d3c7aabdd3c41446163933c59f.jpg)

![images](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/images/b30d13e8602ada9de85f97f0e94b9d272bc0cb4238c95677d0d8973b114d57c2.jpg)

![images](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/images/bb0461b1639e406978303304ff2a5585a0510718d92ea4670c7081e1c04ae853.jpg)

![images](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/images/bf1a30e457d923a76a57ec9c5f0542289369739b0dd820b6f95007a3cf09349e.jpg)

![images](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/images/ce6512bd1d53cabcf2459d517176c0a0a13c0a85815d7a4890d9871221c00626.jpg)

![images](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/images/e6d92609c54d35f852c6e48f871d169ca8d3cfe4d5c6ea3c9411b017dfa6a254.jpg)

### Tables

![tables](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/tables/0efcc2ba2d83d287840753e855b8860cc84da899b139dc8abffe09d3893a018e.jpg)

![tables](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/tables/2ba9828d87f080ec46b9b003fb0d259fcfae83187c7b740b10a8232d76772986.jpg)

![tables](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/tables/384071c34c16fe19292111767faa11389d484fc7e99291edba71aa7c05a878e5.jpg)

![tables](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/tables/3887c78261be44dbaedc457189024705c21c3b3ba76fe204c5180f0ead95a4d9.jpg)

![tables](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/tables/4f41031ae7c8ef0db1eaf8b94804b9f4ec6d87e1c090976550e8a40e4a42aead.jpg)

![tables](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/tables/5e7b88d8bd7b72b9d9f2a8e07c06ea8d1aa9b9156cf5655d61e9a76f0cbdfcc2.jpg)

![tables](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/tables/66d4aafe6bcf8adce3aa9844262c81ec100badc9105e619971713486f9b8a248.jpg)

![tables](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/tables/6eac1d37bce365701a831c3d31553ab401ae895f3a1b5b7c75e1bc5d4ac6a7e6.jpg)

![tables](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/tables/7241cf336e01bdbf4351e38af0b289dce43517acea5b80a2aea7745b66393d7a.jpg)

![tables](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/tables/85d09c14da96e26caf0b452792709da416da1c9bae4d4da1c7eb98a6e33d3907.jpg)

![tables](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/tables/bfb48547420edbe5f1e7d038a29ba0a75a5b0dceb22f8463d232cc9d046a122f.jpg)

![tables](../iclr_results/1048_An_Open-Ended_Benchmark_and_Formal_Framework_for_Adjuvant_Research_with_MLLM/tables/ef457730f6383d2f4909cebbffcef6f5d84bdbf71a5f0d4ec4543bd75189f405.jpg)

<a id="paper-1049"></a>
## Optimal Transport-Induced Samples against Out-of-Distribution Overconfidence

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=r83AdmvULT)

### Images

![images](../iclr_results/1049_Optimal_Transport-Induced_Samples_against_Out-of-Distribution_Overconfidence/images/363010cb0d61a5cfad4b85ca16500f548926bc56893e437b4052395d1d4ac3fe.jpg)

![images](../iclr_results/1049_Optimal_Transport-Induced_Samples_against_Out-of-Distribution_Overconfidence/images/3d58a3322c29fed0501e93e1765346ede7e5532a74d80e9eec14de1e8fa1ada2.jpg)

![images](../iclr_results/1049_Optimal_Transport-Induced_Samples_against_Out-of-Distribution_Overconfidence/images/6e782521fb23470496e5c6c60862cd7a59d7f88bb196f24e2806cad502900a1a.jpg)

![images](../iclr_results/1049_Optimal_Transport-Induced_Samples_against_Out-of-Distribution_Overconfidence/images/83ef1074200de8e1eef6683aa2304be12628bf0befb24f1b72e5fe53057cf32e.jpg)

![images](../iclr_results/1049_Optimal_Transport-Induced_Samples_against_Out-of-Distribution_Overconfidence/images/847402671c07e3ee61e82c88fdc8696353bb923921d61cacd7d48da129927bc0.jpg)

![images](../iclr_results/1049_Optimal_Transport-Induced_Samples_against_Out-of-Distribution_Overconfidence/images/8b7fc46a08fc03defa690343b036441ccc818fedfb660b1593a56227f5136bc6.jpg)

![images](../iclr_results/1049_Optimal_Transport-Induced_Samples_against_Out-of-Distribution_Overconfidence/images/a821faad7a6c4f99d3588f939dc9ab4e84a06cb2f2fe15778446b8f9d0155602.jpg)

![images](../iclr_results/1049_Optimal_Transport-Induced_Samples_against_Out-of-Distribution_Overconfidence/images/ade5fc5e86037130090decc471f1642b342f168a4515fc552bfa3b7d92864476.jpg)

![images](../iclr_results/1049_Optimal_Transport-Induced_Samples_against_Out-of-Distribution_Overconfidence/images/b3608b497d2648a0e08b8a91747c5e82b37bd0aa05908ce1febb00429d52e0d2.jpg)

![images](../iclr_results/1049_Optimal_Transport-Induced_Samples_against_Out-of-Distribution_Overconfidence/images/bb038863b318412a0a10eefe55994b19391071ec47b13f368b9438762dff9ee4.jpg)

![images](../iclr_results/1049_Optimal_Transport-Induced_Samples_against_Out-of-Distribution_Overconfidence/images/c21fb81a0015156730640a6a421fcbde6aba9df6338cdb952f95ec285528bac6.jpg)

![images](../iclr_results/1049_Optimal_Transport-Induced_Samples_against_Out-of-Distribution_Overconfidence/images/e94a0ab9a2101eb4fe1e8774b27e2910259a1335402a3f4d6e53e1ea6f2cf5b9.jpg)

### Tables

![tables](../iclr_results/1049_Optimal_Transport-Induced_Samples_against_Out-of-Distribution_Overconfidence/tables/116df461358c955f49a618e3b305682e3ec16a1571a523f8ccdf585b918c7467.jpg)

![tables](../iclr_results/1049_Optimal_Transport-Induced_Samples_against_Out-of-Distribution_Overconfidence/tables/6c700ad76b9a9d0e0f35b094fb3d183747b321194239b2a9b84ab559c78aeca7.jpg)

![tables](../iclr_results/1049_Optimal_Transport-Induced_Samples_against_Out-of-Distribution_Overconfidence/tables/7afa96f50246198ac8633ef501762afa828cd8d8c7b2d4714ebe2c5f4b735ac0.jpg)

![tables](../iclr_results/1049_Optimal_Transport-Induced_Samples_against_Out-of-Distribution_Overconfidence/tables/8c2be130a7ce8f5cd4e02a470820d92820cf819f75d55352820ed8dce36bea03.jpg)

![tables](../iclr_results/1049_Optimal_Transport-Induced_Samples_against_Out-of-Distribution_Overconfidence/tables/a4a204383ac4a3e55fb9f8c2d553a8b2cd063c5bdca492db1175a20883f782c2.jpg)

![tables](../iclr_results/1049_Optimal_Transport-Induced_Samples_against_Out-of-Distribution_Overconfidence/tables/b64efdc551fc98cb21c12c862b6d6c5a11bd8df53fea66f6573b180a2c29efb4.jpg)

![tables](../iclr_results/1049_Optimal_Transport-Induced_Samples_against_Out-of-Distribution_Overconfidence/tables/fd18b194cac67d8378e58e917d13db61c019984c49e0c94beac107a0b72090f3.jpg)

<a id="paper-1050"></a>
## PixNerd: Pixel Neural Field Diffusion

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=BDnOrExHmt)

### Images

![images](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/images/1a2a332fc6f4d377c4a7027c51a1f0b5beb3f5e44809b2cb8d1a92441ee45443.jpg)

![images](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/images/42f4a504da8e83718bc2286f4f1c60a3a55c568c8db5028a080486071f872859.jpg)

![images](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/images/45897992b9a75e9194fe5ce73e5eecb94f20b77a04c106213e9db3937bef5308.jpg)

![images](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/images/6555c26c98fa67797cc88ed0c4c9990f7d3e5ac667222699e02d3a964be16f94.jpg)

![images](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/images/81e2b2cbbd2c3acb5d379ad017b354e6c18c5d9dfabdca73e5ab57a0576c04f8.jpg)

![images](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/images/91fec28ff9cd981faee6799a60c3f807a2340bd825a19ca024a60361069342e2.jpg)

![images](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/images/b26904ad6a6de499902ee02bb8aa77929e3066a1a4b1b75631798a269b4d8691.jpg)

![images](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/images/c80ed32212a67bcf2155adda84e714003b7ceb1b7d3e2a881f7ff3e7147416e0.jpg)

![images](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/images/d8505e5d0ad70c3f7c03a8baba344b3d110ab4fa5f70db85d097933aec143429.jpg)

![images](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/images/e49260cc0516f6eb20dc51f0bde548cc92f8e85819cc8d88e01cf253f409b94f.jpg)

![images](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/images/f0baaf0d2589140bcaa4902c4db2609d20ae64ae4d8149bfef148a396b59f4f5.jpg)

![images](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/images/f3ff341541dc68519fa93c05c2ec3603b74fc73807b11e4c884acf7ea917e724.jpg)

### Tables

![tables](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/tables/2944aa8a8d1e40c3a12726893aecce56157377341b8a4993d9d920cc958a0157.jpg)

![tables](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/tables/3c7a76b3fc9b4234b51be323285f19fe75ca434159f21710b8c7689c048f21c9.jpg)

![tables](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/tables/50822a61470b3b0feba87091d65d15595866336268ba804a7489d7b40946377b.jpg)

![tables](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/tables/754dcfc69894a5bb254cca8297035c67bc395e2d602283d39ca89e5ebc5fa047.jpg)

![tables](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/tables/779ba081f18d216918ba221911905c7219bec1b7e1a07c1b462c37b85bbe8ddc.jpg)

![tables](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/tables/810777d508f6eb4d7f59261e845a9573f7de9b6b1092ea95e0027cd0f470e036.jpg)

![tables](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/tables/961acd036baf8e5355841eef16ffa9e23097c0c7dd6e42cc794c51e8cfe6c6d8.jpg)

![tables](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/tables/ae10c9c2b9b005cdc36914fce374a397ea5ee73e8d07d30e22b712c77afe3de2.jpg)

![tables](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/tables/aec7776d331148cb2bf2da5311a48f97f35f0016ebf311044e9267db38b0bf35.jpg)

![tables](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/tables/c47f18b5d6072498403bbe3b8ec57f3cb77bb32b27f1a6a36a69405692745183.jpg)

![tables](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/tables/d522eae25f755ba0effb13adc0492c6cc9d2311c14fed1eb18451716d1f39602.jpg)

![tables](../iclr_results/1050_PixNerd_Pixel_Neural_Field_Diffusion/tables/fd76a6758cbe44f3f1555233f42fefa3cc13f3e165bed94f5f014a14121eb8f5.jpg)

<a id="paper-1051"></a>
## Bridging Draft Policy Misalignment: Group Tree Optimization for Speculative Decoding

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=dwPdYFqVWO)

### Images

![images](../iclr_results/1051_Bridging_Draft_Policy_Misalignment_Group_Tree_Optimization_for_Speculative_Decoding/images/e1f7f62821a3561c1463d0c8852c226fe81d423d165c4bdd08824d8f3895f88f.jpg)

![images](../iclr_results/1051_Bridging_Draft_Policy_Misalignment_Group_Tree_Optimization_for_Speculative_Decoding/images/fd13305996eb635dfdae6fb12544209338c496e7bd51aa068064351995603fa1.jpg)

### Tables

![tables](../iclr_results/1051_Bridging_Draft_Policy_Misalignment_Group_Tree_Optimization_for_Speculative_Decoding/tables/118525c880a3f425aeb4bfee3c8d7089baa0f5e419564f573cfb2790e9466fb5.jpg)

![tables](../iclr_results/1051_Bridging_Draft_Policy_Misalignment_Group_Tree_Optimization_for_Speculative_Decoding/tables/307864b99c91a2f2b777cfb81cdb30dd424703ff747200f5267086bdd75bf888.jpg)

![tables](../iclr_results/1051_Bridging_Draft_Policy_Misalignment_Group_Tree_Optimization_for_Speculative_Decoding/tables/3f1f4d659eaee2a1afa9f53fc69a3ddb6afcd52d6a2fe5be316d7050116ad9d2.jpg)

![tables](../iclr_results/1051_Bridging_Draft_Policy_Misalignment_Group_Tree_Optimization_for_Speculative_Decoding/tables/4cf55ec1dd49b9f6b13ac23da32da14564a6441c0cf971a329f706309b0829fd.jpg)

![tables](../iclr_results/1051_Bridging_Draft_Policy_Misalignment_Group_Tree_Optimization_for_Speculative_Decoding/tables/4e5dc7e06983ba9cc77e587c2e74159454e9a12095ede29ecc3b491bd7c002c1.jpg)

![tables](../iclr_results/1051_Bridging_Draft_Policy_Misalignment_Group_Tree_Optimization_for_Speculative_Decoding/tables/7d15346bf9d70b7778ec31f135df5605113255361d0ae509dd370a72c0007b27.jpg)

![tables](../iclr_results/1051_Bridging_Draft_Policy_Misalignment_Group_Tree_Optimization_for_Speculative_Decoding/tables/cb2cb958c6b227619c3822db772b89c73fee9550a13fb4200e7dfe4f0547f966.jpg)

<a id="paper-1052"></a>
## UniLiP: Adapting CLIP for Unified Multimodal Understanding, Generation and Editing

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=6tx4BGjwJP)

### Images

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/032295a44bf3bbf6d68fa2b06ea090fbce9020d7e38c0861398b4ada122537ec.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/22b9314fe2a7f7895ad4e12f8ffb6e183aa3b39b5611850e15a8f932a1fbfc67.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/2a51b0329b330fbd5669716bf83cb3bc89f98b6278789b0f3fe4409fed838934.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/2bdff8c76d33a5a7f59e2b15f1e5f7cdbc4767cc428d1f89c5e8bfdbb6b35087.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/32f961be0c368d39279e2f9d782e3efbab75f425b990fc18e5b9615d70cdfd9c.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/365340ed6e6f99a07a9999716cb4d102cb1e6cbc95b1a8dd5365a5278c372641.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/599a77f3141f7007f541d2e92ae48123ed6e3503d762682931c78f1a0097a11d.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/5e84f347fda469d40483f00d926992776695f8e12d11f7e0d926dc5590a2045b.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/67956ec91dba145a73654bddf60cc67aae0af5a701825a0e089cb7a9d2f38a8d.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/6a9f9aa4df7ded38edef20027a77c0087ebfe15dab29fcab87e6f1b2dc67372a.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/70149f33eb60872ed4d805ab0eaeef9cc5fa5727b34613741a34d7e2cdb4ac79.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/79547927aeb0020c1b55361bb74403bec8945e175d2455431a1a428518179c29.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/7b5ca151cd412d2354367368a45ca29c22f2ee0d26d88483f8ddaae5db716651.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/7d79161230054c0dcb9b874f6a6a026f9177c03e56e3fb22aad5218c6ee37ef3.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/8044fc9ea47d36bdc008fddf84f15a3a177762a06d716e2402de096338c4a402.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/80f46559b73d7b584a7a8964b56443bd7b04a72329742b20ec35f68ff5517dee.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/923abf224ad673c87f73f721584e54ec4fe27fe025a2b7260cf6288477f2a10a.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/9fb848674e4b14f2f2491b0a7d09ab1e824bf2e2d34433ee5844fe9e9adf946f.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/a46fc6179ed22e74b2c7127fb776b90f7ca453ba534b39a00a204c5fb0e9e9f6.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/b86b32eb2a5d35380938c731b3f02e31a9e78234911de993786d74c834600d17.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/c2218d583b764b817719fb340bd5e469d6e6fbfce28c0f5156f3f8415cf58549.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/c81b8a1b846166430f48c6b2349ac5b6e5d771e9ae277d49a6b31f465bf9753c.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/ca51fe3474ec4be43599ca4b3811294217baf1553f3bcdb1c2b072a3a5deedda.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/caebf6bcee31aefbd7eb7dd80b9e3c809ff91c21a80327c9a149e4614198feda.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/cde2c4449953e42d4784341a70eafd61418244bec7693d8703dd28e28438652f.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/db3c5ac6878a2a5cfcf8035d0fdbf7ebfe50fc2fb8b29d6a999dabc95c2960fd.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/e27c4bf58a27e5db64c844ab2506eaac647372fe9b05658f220c25e73998e068.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/e7446551467529d791ee098091b2dc0f76cf641255dbdbdd42034a7d564c3974.jpg)

![images](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/images/f08dca76141644f5796b53655f08dfbd9e6806b2dd4ed27a8dd247a72087bf26.jpg)

### Tables

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/00603b9734dab3362afa506a3a35d985728e39a8e2545ea8ca257418fd3e74cb.jpg)

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/057e622bb6587682d30e8d0d1f41832f654f60b727e6aa8a21db7fe2b5867bf2.jpg)

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/0c0f3cfdc04d14a044fab9297a74a9387c9f20c00de870389c3e8144558f8623.jpg)

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/1a3c0edf9dd6d5ac7199adcbbc7f27f0af798c8b33b1af8f5a27203cfe0248e2.jpg)

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/1fb2e584aea33ca8a3222bcf68151094ad97e281af81b4d547074a744d5415cb.jpg)

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/2ce2fb50890f79b468f1f40bbe5b7d44537fb6e65fd28a37379617b7728d222b.jpg)

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/40bf6893079abb4f105f2c4be40e2dde2ad3007fb7f3e3a7458dce6ef91f22ef.jpg)

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/48729bd3b8ddf89757987cf183fc9530b6bb273689190f459f9054c2e2114ec7.jpg)

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/537bfd4a317d7ebee37c87ccfc55115e4395107ded148a88b4aef7a5322441d0.jpg)

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/846d1445da6e843c82f947b1cd3a01f25b78e0528d9d122b36ea3ee165b38d16.jpg)

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/8918943dbd4b96e6c27badeb53fbfbce99100493c6e45993ce569f54f2d9c6ca.jpg)

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/8b55d6eb65401ff1de872d0b4b053e102820501ce15d8cd656b42fdc55b281ca.jpg)

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/8ec698994b6f7de594e2c10947e3b8f43ec9d2f800563fcc898dbb7480c4b172.jpg)

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/91d12873d2d7b76dd3e60814459ce2bcb6bcc4d9dedbf109d96e9ebb1c354512.jpg)

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/97f0de4099d706b85c1c6dc5019f92970d2b7d2f2e8dfbc9bd237cbacbff45d2.jpg)

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/9ee0049328c1eddc52394304ea1341f6db7a21beb2bc20f9b1c5d419819ad7f9.jpg)

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/acb47af728231beba35983f3d7dc5ecc68a6ebdb150d1c6f7e669595ad0ca7d5.jpg)

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/e12321c116e9cdae9033aa288e924e9d231fe559db4a0193eb1f080e378bc5e6.jpg)

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/ea6516b8852034afa671a38fe088183306ddfbe534d6c9f89fa876e7b5b78a10.jpg)

![tables](../iclr_results/1052_UniLiP_Adapting_CLIP_for_Unified_Multimodal_Understanding_Generation_and_Editing/tables/ebecab548c3e8cf7bb7e75c594f1dbfec796cadd22106523f4751dd8fb039c2d.jpg)

<a id="paper-1053"></a>
## Decomposed Attention Fusion in MLLMs for Training-free Video Reasoning Segmentation

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=79SSF3ppjS)

### Images

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/03d4e7eed0d65f87b93ba896bd37254195c8a0cd1eaedd55efb929e353295917.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/10668a6bd4e852c089c1df4c702e1c089939dae1264db2964398d2048cffd4ff.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/21ff2d0789bf469ebd25cbb2c536d079e04436413c73c2888f5be15502446428.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/3bad7167b8f729083ae83bdd91a071c38031edf9d3750e9e3e4fa8025090e55b.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/47eea9bd5a7b878fafac36b9f94cf828cbc17df6a40e4b2f79b2689996b6facf.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/4a412e7cded1393758fe06f20597a6772b691e85a5d34ab7199e140967607f2d.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/52d2ae0a53c7d31706a4e4fe99653ff7094503524d475aaff516c4769268de2e.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/55eddcb79bb374d0465c932a20a34573f781bbed302f4e4a3f29ec3a8c25f799.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/58409d7864b877fa054ed9f41f33b64e43657973e4e8b1d34d78ec7c72ae3e3f.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/84167883fd43f84cedad8c4e367594e551078330dac2661168fe0786d28e206e.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/918f3da6bea329e54b1307e5899215bcb5675d72924349f6191e34c1d0bf80a8.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/b04fa0959b126b86105d0cc6a2f88ae270589dfbaac9afa21aeb4c232cee5e08.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/be5e464c2f579821900b99cd6986879b888890e43fcaf1afa90cee4bf274c329.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/bec4c247ce193322760e8c6258b74df21834281a23891c5cc4b345bbd048148b.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/c87317482e4c0ab382e486636f3b1a62dd341591db83d9978de3c95aaff004cc.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/cac5788b19b4b634812d03aa511d4de544484fc646aa2dfc0548c9bd09a25868.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/d2f09f5473ab277e93969ee11cdf55b28d22c548bfecda5b1529547dcd390128.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/d90092080d156222b3272a35a172b2ff31d29cf2ff5cac84f3ae7a245743f47b.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/de181b652c0d20308942d35b0efe4df0245d4f07c57382adfd4e6d0f68cb04b9.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/de3811d34f514af69f132ad2c29b7eac6f3719acc1a7707b224a38f91d8625b3.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/defb62a7ec174afc45b8ca2baad55262dd39767846d18bca6795abc595a8a5d0.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/e2cba74f23c03900808bb9ff63fc3e327efd4fe8000bf8ac9cff20f20b8c6560.jpg)

![images](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/images/f4218af31a99f74a32098d8a273b2d3fcbee82b015e4ab8537b8c1b1bd823f44.jpg)

### Tables

![tables](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/tables/101be43f50ac2a3f1eed142121fb38215d1e677f2ff7dcfe167886679a2777f8.jpg)

![tables](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/tables/1b9ccce09feaad076c34062c3da2fb55867159807c88f7a0b0b543e0619eb981.jpg)

![tables](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/tables/1bd0652e32b1bd1cd40131b7aaa720e28776623981088270748b7f3ab41b8c4e.jpg)

![tables](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/tables/4aedbdfaf2d034d432e4176823a7f5ec5f06ca83bcb8adef94ede664137cb9c7.jpg)

![tables](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/tables/4f1d439b0fa9a477afd9dabf82108a8a214eaa3d68172a7441291f516c534d19.jpg)

![tables](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/tables/544591662d223baae6ba1a7d4e259d3144c353a57e0d40f5cb897b5db0e10f54.jpg)

![tables](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/tables/5bbf9303c0acb75356ffd0ef181809a1b4ca788d0a906a3ab94b6e3d55a70c99.jpg)

![tables](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/tables/7180d8067c4eeb616eace15e91992779e9429fdadbede8f50e4d8d2f64f4d6c7.jpg)

![tables](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/tables/7fcf5f9e41e4d6be9e5e7a7dfe1a5a407ee05f17c64f0b23f32853db9abc271a.jpg)

![tables](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/tables/b894d3e61ead7bb6450531ef8fe3061cb3105408d28a869e451cff9c717edde1.jpg)

![tables](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/tables/b9e7ebdab1c4489c8d99ed69288193a75cffe851c9900eba14a36f8cd384184e.jpg)

![tables](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/tables/bf1973f41ab2e4241a73634b99ab09c6a3eca26b417ded861f2a86a91bf8ac20.jpg)

![tables](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/tables/d47112a220e41eb7b8e519aa44f7c90028b8561c01955a2b79479eaf393efd63.jpg)

![tables](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/tables/dd4e3f820c9f8b830cdce695420c73c021653e1d065344651c28ee743453ce1b.jpg)

![tables](../iclr_results/1053_Decomposed_Attention_Fusion_in_MLLMs_for_Training-free_Video_Reasoning_Segmentation/tables/f59a7f19d7c9ead06173722120065fbd2f2d82a13b465bae557464ac054f5b3d.jpg)
