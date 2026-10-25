# ICLR 2026 Main Papers — Part 6 of 199

27 papers.

- [How Reliable is Language Model Micro-Benchmarking?](#paper-136)
- [FALCON: Few-step Accurate Likelihoods for Continuous Flows](#paper-137)
- [Global Resolution: Optimal Multi-Draft Speculative Sampling via Convex Optimization](#paper-138)
- [Energy-Based Transformers are Scalable Learners and Thinkers](#paper-139)
- [CyberGym: Evaluating AI Agents' Real-World Cybersecurity Capabilities at Scale](#paper-140)
- [Rodrigues Network for Learning Robot Actions](#paper-141)
- [Spherical Watermark: Encryption-Free, Lossless Watermarking for Diffusion Models](#paper-142)
- [Omni-Reward: Towards Generalist Omni-Modal Reward Modeling with Free-Form Preferences](#paper-143)
- [AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference](#paper-144)
- [VibeVoice: Expressive Podcast Generation with Next-Token Diffusion](#paper-145)
- [Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability](#paper-146)
- [Seeing Through the Brain: New Insights from Decoding Visual Stimuli with fMRI](#paper-147)
- [mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules](#paper-148)
- [WAVE: Learning Unified & Versatile Audio-Visual Embeddings with Multimodal LLM](#paper-149)
- [Learning to Segment for Vehicle Routing Problems](#paper-150)
- [Reasoning with Sampling: Your Base Model is Smarter Than You Think](#paper-151)
- [SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents](#paper-152)
- [What's In My Human Feedback? Learning Interpretable Descriptions of Preference Data](#paper-153)
- [In-the-Flow Agentic System Optimization for Effective Planning and Tool Use](#paper-154)
- [Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding](#paper-155)
- [Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention](#paper-156)
- [Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series](#paper-157)
- [Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces](#paper-158)
- [On the Wasserstein Geodesic Principal Component Analysis of probability measures](#paper-159)
- [AutoEP: LLMs-Driven Automation of Hyperparameter Evolution for Metaheuristic Algorithms](#paper-160)
- [Fast Escape, Slow Convergence: Learning Dynamics of Phase Retrieval under Power-Law Data](#paper-161)
- [Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments](#paper-162)

---

<a id="paper-136"></a>
## How Reliable is Language Model Micro-Benchmarking?

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=cReExMQLiK)

### Images

![images](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/images/14ebf015bfd369a081f92a46fecb38b900d5059b270a320152b879beabdfdb66.jpg)

![images](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/images/3e7079e455260e9670306e5b5f4c47de1d71eb42febd4eb9c4d02eca977655d2.jpg)

![images](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/images/4d28be583e8e4480953b1ff4d383ea2236a777f5e43a6e9edc85cfdd89c145c8.jpg)

![images](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/images/62879a79d117fc15990aa095147e3cafdfa0807532a08ad53f629b471e83ca63.jpg)

![images](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/images/65b72288f90f605d148e9eb512970506ec77ec1f7b24bb20449af4c1d4f067ca.jpg)

![images](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/images/6a4784b1ba5e2e831c441470104baffec613b4afe7895badffb6923420763cfe.jpg)

![images](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/images/74d14168ed08354a6d0a541ba11683c44127e19d88005bca97c33f5aa20df3c3.jpg)

![images](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/images/90217dec4eb888c3fdd96c1d44c12662af5d7ccb5104b97e2c5cba88d8711308.jpg)

![images](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/images/94baff83614c0ad4fcb29ffb732c8743fdaa1ef5d2455f4cff3b837b16927f38.jpg)

![images](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/images/a6a182eff30f007bb9257646627bf6c3209b7b8998a7e7ce03cad4129e857a04.jpg)

![images](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/images/bee97af74c9684427275c23a1b9fdd0e0aad2f413244f52b2917379eb2a2df52.jpg)

![images](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/images/d9fb3e354bbb9a3354ae9e124b92a75098d2ecd5ceda4f8d9a48344f94b42076.jpg)

![images](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/images/da1b35f666cc497cceb244165f001c88595b2a0066fb8b94eb89eef7165f1ae0.jpg)

![images](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/images/dce14a6e570bc2cd4bf9c06b671fdc670277e294e1e62b0a763f4bf8b3f6dfde.jpg)

![images](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/images/e6cae6f9cba59bd00e3c2dca35e17f9a7dd3bcdc7528aba0775bf0a98b91be77.jpg)

![images](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/images/e8f16a08769a6084931487f0a89f1d1ff39d343fb70524271e999ff1980d7eee.jpg)

![images](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/images/eb4b931ec203bd0683de6ff05ce9a396f283db9896772b33e7e16c03e4f74a37.jpg)

### Tables

![tables](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/tables/4b10065d966fe21c5dbb13fa053183257a805c99224dc0e0db841bf528c7cb5d.jpg)

![tables](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/tables/88e7a247b8bd04680064d7207f84fb6cf4e2647a4388a8b2e0d62e17e4dba844.jpg)

![tables](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/tables/913305a4b7709ac7d19e6a5e35ac450aafeea35c8aa145e2e37e8083e97068b9.jpg)

![tables](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/tables/b94faa4c083c80e09da1616e62de462d9e4b7f5ef5ed8d42a06830db01ae076e.jpg)

![tables](../iclr_results/136_How_Reliable_is_Language_Model_Micro-Benchmarking/tables/da5a2698292b3781acd469298f073a6bece110120b34e65351a1beb172ea9d88.jpg)

<a id="paper-137"></a>
## FALCON: Few-step Accurate Likelihoods for Continuous Flows

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=FbssShlI4N)

### Images

![images](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/images/0d654af2d63cebdc8f8875b2479a37edceb06816822a07339d2d913893d3c277.jpg)

![images](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/images/3ea761406bddbefca08a2d4125919213a76e640e16afb8e21753ff3c80131ab3.jpg)

![images](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/images/403c76364e3e11413d2f6198c025883de53af9cfe204be88a20bab011b32cd2e.jpg)

![images](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/images/6754e635abe83ab61831e9cc8c483c93e6442c0ef6e69947b30065c190bd626e.jpg)

![images](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/images/6b5c1ef4abfdb2445585a6c7fb9c43822f5dd4c958f4518ec37f3a3b702f84f6.jpg)

![images](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/images/7ee874a81d2405cb69f91028235f8435b06c77bb0990777fadded49e17059807.jpg)

![images](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/images/7f7ab9a735da34bcb54caaa9d4fd80a7b16ea992c36ba02c25b64f37f9ada19d.jpg)

![images](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/images/8a046a1f5481a47b580ff801938ccf017ff12735b8cc2e1832e45e4d4f10e737.jpg)

![images](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/images/96da5fc89cc304822ee8f7d5e15e3feefffedbd9cd4670972b59ffe087e76feb.jpg)

![images](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/images/9feb0677e268ce925cce5643b315a8f70ec92802c90fd801c1c0c8df40641de2.jpg)

![images](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/images/a2ae1c5de01b14d8ef0cff161dab236191fa6fb5bc3ca34f2a59c9a2bdf11869.jpg)

![images](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/images/c976db1ada6fbe60eb036aff13e91c94dfb4a6b938c8e5a6886d3e5566d7728a.jpg)

![images](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/images/e0345deada21a644d9b7c5395b44893ad8883acb3e4ea4f1ee5c76a29ff6e985.jpg)

![images](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/images/e38c1964124fd69e9ed2e7562f392fa1f837eb5e5d59fade3720d27c26f315e5.jpg)

![images](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/images/eb2fd05b44d19baf0be72bb0cea0b02accf82a6cdb2387fbdf3acad603d31aa6.jpg)

![images](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/images/ef5baf6a0c5ab85573012845662f35ad891c920c7c2c01b20a24bc1abf5aa3e3.jpg)

![images](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/images/fe6e850b51916df06f1c441942b8395ade404c628b114f7f4632f65ce9267c20.jpg)

### Tables

![tables](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/tables/7c8a4815f97faa136c73602c1e03e6e8f9b4899b3f2a92d81543a3e887956300.jpg)

![tables](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/tables/8d35e0e84e769c6643d404bb11192f4180a2f747f3250dc08b2e68942369396e.jpg)

![tables](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/tables/970bfc47dcbb4da393d068dc5d81307871d4d7a042d4442f375b82da25fc2212.jpg)

![tables](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/tables/b102e5684b805c253dc8b3b19bac96a54d7e392d13686251c9108c34139c7d86.jpg)

![tables](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/tables/c435edd5e0a6c49e470bd00ee9ea394c389213c45609f445cac8d96d2d7b26be.jpg)

![tables](../iclr_results/137_FALCON_Few-step_Accurate_Likelihoods_for_Continuous_Flows/tables/d9682ce8079fe1d29c9a9a82ffbf4f5a72b4a43c530302147d9f49b334dce020.jpg)

<a id="paper-138"></a>
## Global Resolution: Optimal Multi-Draft Speculative Sampling via Convex Optimization

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=gpsczXOsHn)

### Images

![images](../iclr_results/138_Global_Resolution_Optimal_Multi-Draft_Speculative_Sampling_via_Convex_Optimization/images/b10353520f1ad40cea9691f3e714eef3bc79d55ef62cd13b2e451f1f31b162a7.jpg)

![images](../iclr_results/138_Global_Resolution_Optimal_Multi-Draft_Speculative_Sampling_via_Convex_Optimization/images/b1170e637f11f1f19cbfc1dc2793580e37c585abfecdeb395c51e32c01a63636.jpg)

![images](../iclr_results/138_Global_Resolution_Optimal_Multi-Draft_Speculative_Sampling_via_Convex_Optimization/images/b2bb6ecf5e078ca768c29ec459e37865b4b1317e001c861d61f2257fd78526b9.jpg)

![images](../iclr_results/138_Global_Resolution_Optimal_Multi-Draft_Speculative_Sampling_via_Convex_Optimization/images/c0dd8f2464b4f779df81fc6c5a56f88524418b207e5b6eaa9f3872a4d648437d.jpg)

![images](../iclr_results/138_Global_Resolution_Optimal_Multi-Draft_Speculative_Sampling_via_Convex_Optimization/images/c921338b21b2644ce73bdd9beb88bc0e67e6318647c0336219f3fee0151a24a5.jpg)

![images](../iclr_results/138_Global_Resolution_Optimal_Multi-Draft_Speculative_Sampling_via_Convex_Optimization/images/ca193128e068869412faac241c27c5626cbdad3b97b7e122b5bbbedc312f1b82.jpg)

![images](../iclr_results/138_Global_Resolution_Optimal_Multi-Draft_Speculative_Sampling_via_Convex_Optimization/images/e11cad6f5db4ea6f37b38ce7a6a00ff05011424affad2f71edc46a026852721f.jpg)

![images](../iclr_results/138_Global_Resolution_Optimal_Multi-Draft_Speculative_Sampling_via_Convex_Optimization/images/f558194bfe6c99b4fa05945d392b0a691ba41a727c374531157002405322fbef.jpg)

### Tables

![tables](../iclr_results/138_Global_Resolution_Optimal_Multi-Draft_Speculative_Sampling_via_Convex_Optimization/tables/04e572b2397696a01b75d2a535bcf0fa25e8b9b394249c0d28384d051156d85b.jpg)

![tables](../iclr_results/138_Global_Resolution_Optimal_Multi-Draft_Speculative_Sampling_via_Convex_Optimization/tables/0547eaf3d8fc1039670495979a761a75ded841db3d2c27049c4b93c7f6520881.jpg)

![tables](../iclr_results/138_Global_Resolution_Optimal_Multi-Draft_Speculative_Sampling_via_Convex_Optimization/tables/6190b4f7797c74e56ab05d59fb6eabb346d05d1cbbe6824452ff991c3463a583.jpg)

![tables](../iclr_results/138_Global_Resolution_Optimal_Multi-Draft_Speculative_Sampling_via_Convex_Optimization/tables/69dcc21ffc1474082863cdbc632e248575ad86fa76ca7b34c24963ffd5d17fa4.jpg)

![tables](../iclr_results/138_Global_Resolution_Optimal_Multi-Draft_Speculative_Sampling_via_Convex_Optimization/tables/bde952b51923a9de268636b8a4e7b6ceb8811cc975dcf840fdfa655b82a093b3.jpg)

![tables](../iclr_results/138_Global_Resolution_Optimal_Multi-Draft_Speculative_Sampling_via_Convex_Optimization/tables/c249c6e115ded8e35d72160d831833e68e06ca7cf0e76df7031c910a85c8a417.jpg)

<a id="paper-139"></a>
## Energy-Based Transformers are Scalable Learners and Thinkers

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=ZBj3Qp1bYg)

### Images

![images](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/images/016ead2743b6c228f3160c71ea3ff1d1961bebdab3237395c4a2b35a48c4b779.jpg)

![images](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/images/0189e183ecb113af5f6198c6ca33a63e37cace871c830a945d1ec94a3ea75ae5.jpg)

![images](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/images/04e898baa425802d422bd551df181bc0b1002e3bf852e845cf0ce43f17889c13.jpg)

![images](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/images/097e22e3225cb9162f2c97dd2d8fb12a580e215143eb4256975e9e10d8abf34c.jpg)

![images](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/images/0a26737bbc3380f8a814de15a13a0bcb6a057253318035110d6d3eec7c7a4bac.jpg)

![images](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/images/3105dde367b45d9bf6207d4abd3334e6a9bb7950617bc8c08ffb57abf009169b.jpg)

![images](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/images/3249b0a730b49ff5dec5e6190cbc3fb17bf796aa3d31f949b90589e90b8e4d1d.jpg)

![images](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/images/39ea2b6e75d0eabaa275a442477b2af5a8fd5db6fc84984d2ac045f913bae2a2.jpg)

![images](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/images/4750d51dac7dc43f8ff9a48ff9bbc597077db164360653c63daf908a1428250c.jpg)

![images](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/images/4926ea9e03a5c912138af02a24c7b3ca5e6b5071dd8a505fb341de90a7edfd8b.jpg)

![images](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/images/5ae49d95105a7121707f1684467f284b1cf812d4fafd5938b65ae7b2cb24cf96.jpg)

![images](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/images/5d95178abea4dcd56c257ae2bdeb95dabb09dd16aad7b03bfac8759f8c0d9173.jpg)

![images](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/images/723c87253a5fb0e6a68924e2846c8e2c85328c2d6b1086297fcd69f4f2acca07.jpg)

![images](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/images/7f3e7f2e474c68e4fd50e1b8f55adacbdf61fcf5bd0b1f6873a9705de81cfb09.jpg)

![images](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/images/99cfbc4cf8b016a272bc21b006894b61a6c3ce3cdd78cd44ff15b2037e9958f9.jpg)

![images](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/images/a9b4a4490a17e742b5a2f7f3c0baed3158cd6d5880b16924bc05f5b61c0e5bdd.jpg)

![images](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/images/b67a4fe586ec4694bc4acf0a791540adcfa3354c513aa8dbf4ac186367c4016b.jpg)

![images](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/images/d084c3f807519abaf2b3e71592739f03b657ca00d567bc66c8f6ddbbd7660f01.jpg)

![images](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/images/ea12af6ab0341b97427940f2eebeadfd36e5ae6e4afbf3bdc2f0226aed1a9142.jpg)

### Tables

![tables](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/tables/175ff5db1986607233ee3e99d1625df31c29cab5b84049d612b1422b3659d4c7.jpg)

![tables](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/tables/292ce9322a38cd1a3f0e85327bdfb99f6171de47001a6af35aab8a931e0815a0.jpg)

![tables](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/tables/3034d770c683eef5ebf587dd51a5ad06f52a7f0608f140398b78460fbba7cac1.jpg)

![tables](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/tables/3ad251a57935d158d5a5090bcaf84ac7087582b832dfcfeee65e530d5d16d84e.jpg)

![tables](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/tables/47e290a0022766babcfdc87f73fcbcbd1ee1a444e2cdf6ceece3f91a1c699bd3.jpg)

![tables](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/tables/5687d11bec3e03ec66f4a870636d613f5b0579b1d1c8492c894ddd4cf6baf18a.jpg)

![tables](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/tables/8a3beac37420f5acdb49c87094036d1976349941122bdb0b5ec47868574149d2.jpg)

![tables](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/tables/8d664fe1ae69a5a301b6ebf824bedb1f91a3b078597d710cde264b850eb16e3d.jpg)

![tables](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/tables/8dfab8ef8f9f470f1cb9c0b3894f5f6c7afec56d0afa7d3fd736dbca138f7d92.jpg)

![tables](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/tables/a2d8d7da6d5d99e5b47eb53cedeeb7229480ab5920b964d54eee61ccaf83ee4b.jpg)

![tables](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/tables/c90a8e44252449c613dbb3909c78e892197edd380af3bbbab50a2ce3b9e7797d.jpg)

![tables](../iclr_results/139_Energy-Based_Transformers_are_Scalable_Learners_and_Thinkers/tables/edc1d21324ff6405e76d048ba77c064b2aefe9656d2170ded3dd368373ad2266.jpg)

<a id="paper-140"></a>
## CyberGym: Evaluating AI Agents' Real-World Cybersecurity Capabilities at Scale

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=2YvbLQEdYt)

### Images

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/03bb829ef5f9ccf693ccc77d293f76c0f661e1fb55689444db0b8c34f75f2044.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/05d4c1a031274a2d48767bec5ca082f6663f4fe14ea7df908373585a6f40885a.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/05e90737438141aa1897916180197509a7635826733460f34ed1d9828b1a7c63.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/0b06bf656d3d9f5261cd769fb06ac30b871f8f3fe9a3e212b1f4b78fd7bd6d1b.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/202185b1ba7a914eb27f9c4d655548efc5dd4b5e7a9b471ea032e818b9eb0d0f.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/3465ab205b21709f3ca4def6cb569cfef69583cdbad2810629e436ccdab5592a.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/3da6f42fa224a7d17f26d140d6309122421a3c22dab60b7e8a5e4721d7295464.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/473ecb3fa04abe6e3db84ee392732135235e0ef51c711587326e0db533734a66.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/50e519f342fcab37ede80c7473d3be1bef0396e4cefc28f4a42aef8152428e46.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/53577f9526bb5d62c955aa0420b7a19d04c08de98f87113525b25f1802c7d50e.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/5f90be034599142fd991f831245d872afc73eb79869a261d6d715dbaebaa36f0.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/5f92b20759f910d44f2915271cb2ccb0958632357c9cac1756212415cf9b5e33.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/7876b34290daf35af72978aa16235d0472f600d35ace0edef10b7c275d92cde8.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/8cd404d0196f935024573150eb34b76d327bd31c3ae08f2b221f1b24b26537d8.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/8f098206ea459ffa038493e9304977c7075d43124aa7ca71cfec2053687bb975.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/91b002ca8dac88e4064f29d6ae3cf86e275d4082df5364b5950a06ae2942226c.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/93afd55da2ee71a49dc393ee599a587f91691859ceb89a31dc6fb7b0438e788b.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/98ec3101b9e41017b602c0340aa1b130d2f55bf0c070683f97564f7652d04ba0.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/a40061bf10de13e5b41496b63d128093580deec0bd2bcc45ad8df633571337e5.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/ac751572351b2f95ef3bd5b9617f999f4cb01c5fe4cce14e9d1d8e6144eb30d2.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/b058a6b4c1b15d3e5cb8a643349b99444891d34de2ee7689b36bdebee0d5b572.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/b7ae997fc13f9857ba1dd0002bd6f91d1fff8b13083039042d4ee48e44afca66.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/b86bc2d0b3be2d3d6a98f4ff1667069928aca0a0c604428093ec75175376e4eb.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/b9d7656d73fe1a5191fc382fb88dcb153c2f4f92600942300df33be55bcc8dca.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/bcf0bad5b52e8ae56c1fdec8fff4e9a4d685ed4772a231e122984bc071e8934a.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/d962e570151b56591246e59a42ecefeee4d17331b6c818f43a9fbfaff5e472e5.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/e73a48b8b90948651c1e8ab6e0cfedd05f65c7255fdce8b351b22c4681737ec4.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/ed999306eea0c572f9de3640e941bb3c834137189a206adf8841f5c2a7451372.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/f29675d2d86d49c65ffc2bd72afdbee651b604642ec4c3b35e7eb0c576d80202.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/f2a7f61e765b931209718c7cdb2780f47efe93401d399039355b1a55c5fac88b.jpg)

![images](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/images/fd6dbdf85f07e6a4cee404e88661f843bd2804bf8253837bbc183869931064f9.jpg)

### Tables

![tables](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/tables/08cf64ed6f20da1c08325a5e4d618103296d9e1391e0bd79cd4257d1b696fbe1.jpg)

![tables](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/tables/24fbbdc6504530af91a485a14a7f19f361e3cf685f691eb3dc3e91abc7a1b3b5.jpg)

![tables](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/tables/6dc6a13b51aab2e6408b9591c483397c654a1fd5ea112f47704c714e9237ac5d.jpg)

![tables](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/tables/6e7011cf60fa04542eeb132878ef98e98027d64c0edecc944b2d32e3ae7093a7.jpg)

![tables](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/tables/70620be594b3f8778bf4d8b94619081be7ce6855f75854408ddc5cbddb373057.jpg)

![tables](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/tables/adf2f7c3e9708d9285cd04d58ea6bb69a8b21e4ae825e5acaa7855e40e9bd48c.jpg)

![tables](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/tables/bbc5d51ad43a9e6ef34027579eda0b3bed456f365bc955b306cf3629862e271b.jpg)

![tables](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/tables/bcb06b97556a1415c56ce75a8a2eff01038dbfd6af81bf1b924218855a44b84a.jpg)

![tables](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/tables/ccbd72db226951d99aecc8547f3d333b280646ce0e0e093456e558686171ab40.jpg)

![tables](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/tables/d596e310ee905519a25946e5498eb48c48e9c6131b7637898378b904b2423c98.jpg)

![tables](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/tables/e3c662486b6e231001ffde54d533dd89e717b7cd4fb8df0d42d0e7eb970bca8c.jpg)

![tables](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/tables/e7611dae9375f65b4e31d8bc551ce79673dc4297f90d40db56e75d40f01240c4.jpg)

![tables](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/tables/e7d8744363b8c36852c35e49abca0e67b8a0c1c30c100731e805c4fd46bc360a.jpg)

![tables](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/tables/ee0805f65117098bc160ce3dc9b3d6095e203ab55171876cd17da17e88c14246.jpg)

![tables](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/tables/f04645b06457b0321df6f977d83421dfe88c900036920d26db3fe0a30a8a711b.jpg)

![tables](../iclr_results/140_CyberGym_Evaluating_AI_Agents_Real-World_Cybersecurity_Capabilities_at_Scale/tables/facd089b7f7213c5e43f4ca3a7566bd9e03a9262bbac307a7c0907d7b8dd32b3.jpg)

<a id="paper-141"></a>
## Rodrigues Network for Learning Robot Actions

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=IZHk6BXBST)

### Images

![images](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/images/10c6a059a0394d6936c7d0171dc7522f727b44d2aa4fa76062ae87b5085ee8ca.jpg)

![images](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/images/1fd079263af5938b3fb7884a89d13e621a1cf6f1793f83603c6a1138b966e7d7.jpg)

![images](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/images/28095bdeee111d14114c24677bfafe5a5233009e7147e0b5df7fecac349f076e.jpg)

![images](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/images/4ba91efa46c37980516e8f16923810035a39c3a293502aca4fa699c3a3ccb667.jpg)

![images](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/images/57ec311ff3801e949be509e4ebd8e210bac17d491ef6b6d5c66e8b7d78b3ab54.jpg)

![images](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/images/bf1f160bb3e509c02217178cb09caec85b2b11dba4c25f265067efece9fd45d2.jpg)

![images](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/images/c0501565fb96e1afb77981989f0c88a337b93a4577175bc5db9eb9481b87f68c.jpg)

![images](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/images/c6ee055400882cb6b793dfc679af876c177ea55aa2832807ab8965a23b5d1955.jpg)

### Tables

![tables](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/tables/1afa0bcb404faad7258dc3a177c6f2d8723ca63fe40da936243400828fd365ff.jpg)

![tables](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/tables/22f306a8075150719701cd68c229cff901962073594ff39a60c382bf798a9f97.jpg)

![tables](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/tables/23a598f0c39fa4ff50a0bcb228d6135deba56e36eb02bf5b07f8b692d3eaccd9.jpg)

![tables](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/tables/313bd4a352ca1c4c9e9cffcdf0c205cd5ea089d1ee4827965d590cf0f49fc3e8.jpg)

![tables](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/tables/3fe58ec4f16f9b9cba8c8abc127eab112536272c7772b1100713c2991a6af559.jpg)

![tables](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/tables/8d0a5a709d5ba6c58b4909cc8c990f3a8ad790e3c069e619c87efb4590f3ff70.jpg)

![tables](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/tables/9a643202474b3b85f33011080c87ce765889dab04976195576a02d957e9b93f0.jpg)

![tables](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/tables/c280de4776d6f562be7188a67825bbacacb71d3e1d1ff9f5117c0e62ff68b350.jpg)

![tables](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/tables/ceb2ec1ba2e5c4c3b0eb3ba698db6ecd5003fa27defa56afc6c8b83cb9443bf3.jpg)

![tables](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/tables/d82b44e57d863f574781a67fc145732f15fe60d848c5a94093b122ad752c267e.jpg)

![tables](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/tables/e095f307f8382a94cd43c14425d97e2584242536e9594f986147f510eccb1fd7.jpg)

![tables](../iclr_results/141_Rodrigues_Network_for_Learning_Robot_Actions/tables/fdc32d989e77ea036ace0ad7b39aaca18c19e8ecac6a7f3e551a412882f60222.jpg)

<a id="paper-142"></a>
## Spherical Watermark: Encryption-Free, Lossless Watermarking for Diffusion Models

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=2eAGrunxVz)

### Images

![images](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/images/00c8ba57282b195ad09870a2ea6f047c1ae516296057eed24d4b8989ebf79a9f.jpg)

![images](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/images/014e6ff6fae1c3629ffd299c00c7c2f0586cebb0deadc52ec05d3b01c1724981.jpg)

![images](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/images/0b4a64ab9ddb13d26eeab42fbf97c5cab07cff3bd26b234719463d8ca309de42.jpg)

![images](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/images/1c519aacb34416771b72075c94bc155dcb78ca1d8794760be881504eac23fbb4.jpg)

![images](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/images/1ceec738eee57479638605942e3c6a4d36ac2d1e37e67cdcdaa013487fbb6564.jpg)

![images](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/images/5a59b1567c1f80d376ba9bcb806befef66d40a9e8370fcf38e32305937235792.jpg)

![images](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/images/6440e8b1db15f75a47425e777e0f7e65c6490fcbabae000c55d23045b289d52e.jpg)

![images](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/images/660842ed15902fd41b4b3d5db97f10c9ae93bfa4c34906a61231741127867a22.jpg)

![images](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/images/755231273ec91cc3452f4aff765596b35712859d624786ad0165facf320d39ae.jpg)

![images](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/images/92025b85cd11b244cd14f2486c9030f842c7c24e0eec718f0217877d1e6a5292.jpg)

![images](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/images/999f28da9f18f2872f6d43ebdbf1eef51db723a3bd4ad083ae3d58f7a50a68b9.jpg)

![images](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/images/b67c88ba2e9eb08b98c9a80149d162e69f3a6f7438216f67e0d7e41dbff26fdf.jpg)

![images](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/images/cb88eac2f212ba42016c655e9f3e2574e38b24230f93ce25e4acc2b0c94fe680.jpg)

![images](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/images/cc7427a13efd92a2d135b65da19e116debee42fdd24eead69747d31f0f0507dd.jpg)

![images](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/images/de8709faa648f7b87dd19c6a322a822274bb6fa06f2bcbe5cfd34eba52ceb7f5.jpg)

![images](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/images/f011ea58d233973895c9573f0fbbedc059861f6bd7c2ffe37554b6c5fb06d2d6.jpg)

![images](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/images/f6ca4437f95a6dcd43bfaa38ae9cb9eb8aca62a0ecda84200f7d4a0422cf2d11.jpg)

### Tables

![tables](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/tables/0079f7b4632afeac201defd2567930d6ea10a514761d66a030305996f7cf6ef8.jpg)

![tables](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/tables/17ad9bef57ab0b806f6f8f467b5242c5c6ea3c861a8dfc9fa15f0b044f6b6d67.jpg)

![tables](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/tables/1c41879521914a83c096f1823e7ed4af1d8fbf490b996b4120af243c73ca8d33.jpg)

![tables](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/tables/249c50c27b5e9805effffd53fc4b40c4c1a4073713113779efb9f3e62968f704.jpg)

![tables](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/tables/51297a9ec1900b28b0c38d3dfb1493c62c1f5d7b9215fca53d7ffa8d3abe75b9.jpg)

![tables](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/tables/60dfc5f0d46a3d7a38821943cfcb371c13bb0b8a1b9bebd2621ec766dcf46a7b.jpg)

![tables](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/tables/6186b9b1bba87d4d66421132eeb146d009503aa73de26512bbbcdbb675d6b9ef.jpg)

![tables](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/tables/7006547a08511a2a3ab09dde30327a6841f672d447a3371d1d12a96ffa851769.jpg)

![tables](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/tables/994ac566fc78157f7bbfaa1ef92b6902daa7d8736aec4cff02c0437b9ba383bb.jpg)

![tables](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/tables/a706f0b669ea797712e851ab48becb5181e818472cffae9f3ae8e9a68be74fc4.jpg)

![tables](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/tables/b1a7d0fe42955cac814d793dd95d58d8790a5fe7097e893d08f71ec2261a7b3b.jpg)

![tables](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/tables/b333d216c1f717662f1d7b2dd0a54e9ed83e7b908744dc510bb87dd057f2257a.jpg)

![tables](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/tables/b518dcc071c4cc0f604e077bed221fd1a2f6839904b3475f530eac0ef73634be.jpg)

![tables](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/tables/bcb7f4c760c7557183e8df422076dec29cad833a7229a7f29b7c69ce6e3b4c7d.jpg)

![tables](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/tables/bf7ddc6dc5f64b8032360e5d359d69066450567db27ebb94869da9537dee5c09.jpg)

![tables](../iclr_results/142_Spherical_Watermark_Encryption-Free_Lossless_Watermarking_for_Diffusion_Models/tables/d4686ef0c01167cd7b2f3f7f5b67ef0040541cd2764f8e9300d6843482919591.jpg)

<a id="paper-143"></a>
## Omni-Reward: Towards Generalist Omni-Modal Reward Modeling with Free-Form Preferences

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=9C4gVbPqSy)

### Images

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/0a88b18b47121d4a088f860488ab73a8cbf719fa6f288c87d93826279cde7ea5.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/36585ac888a4b1a09a1273079599ca786529d0108c80f9bb1811add1bd1e36fb.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/37011232edba2f119ef05a08f23c6440aa0d7c3f53a5fc04a57e6a27b764502e.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/572cb938d8b163234f8cb4b94a513f5a4f5aaa587effcf8c249a0bb2ca4c89e8.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/5bf5c05b18f82ee0071978ccdaa76d73952d265085900bef1ec32143a5b05a1c.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/65963cc70c184b93697cae8f904bcc8864670ac93009d0150d422c23ce83ab3e.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/65e7a0d743e116b8a7c70fbc4c9d5f181082b30f671f3fe7b754bcad6aecfdb4.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/8130be2d1720c94fc816e48715dd23a7368c3cca7d99082a645b517dfa020c68.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/8d56739b13f89d2fb2bda2a7302e39ef6dbc19e334efc7ecedf700be57f33b75.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/8d996c623c9d743cea75f1cc4361fe98dc3d1dc55b041a2a451c9b3865df4788.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/95778530d1702019794516834a5d28bb316a81cb9f1469213bd4e4baf386c9eb.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/95fc50768a4775b0914fa279f7d21168be00321cd7555fc523ec41249ff3e83d.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/a8e883325e94791ad21f0ba586c0b5de062f3d46344b7e1a85c9530c310c3607.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/aa55d93fd66094f2f0b656869a0b06093e4ac9498624744bb6952068ca8f5fee.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/bc89ba17945ca025a1b42e43b0d2ee7513448c91ebb9fa419f45bdfb4f40020b.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/ce5eb662d7c5dcbc01d322780d56bd6edcdeb5e09f45fb563860cee1d865c242.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/d4156b9247c1727fe5561a01fe206fddfec576534de50c93ce1f5d5e11cd4843.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/e39662dad1aa22f6331b6d32a49af80135bfc4ea0cc1ea2c1b851d0203c329aa.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/e4dcfd2c6073d18c0e3b98438ca961ab042c2fca482091aed82d6a7d0f8a77d8.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/f4954acbbfc4af45b4e9da1763143ec2aea917da58daa3af0e0e24d66ee0af71.jpg)

![images](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/images/f9e6b71727ffb3befa38dfbeb6f11fdb7ac9719dd9db25a28947eb3ac2a74317.jpg)

### Tables

![tables](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/tables/2dd695c4c73b31f73beee62de6d3fa522e3b07beb897f39fc88c920bd890262f.jpg)

![tables](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/tables/2f2e95122a73710f523068daf873ab8dcd11c5f865890b5480436aae5cf4bbf4.jpg)

![tables](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/tables/30fe7e476b48f0e37f98af769515fef025f03979c14066a1e8fe846a97509f73.jpg)

![tables](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/tables/34b181133ba2a5fbd1997f8cf76eac3608fbdac6d5ae7be8a0edd5b86816c8f2.jpg)

![tables](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/tables/5c52d008f9b40d2971e8bb8a22f7fe5139ffe45bb8ebbec44ee9ab89568448fd.jpg)

![tables](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/tables/6032f614c8f9f18a0fed44ac452820659fd7de5379282a9f1176f133aea5d244.jpg)

![tables](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/tables/7eb9f17c90e01c063ec9a3724a25b73afafa0bdef9ce6779246b88a290f7c8d8.jpg)

![tables](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/tables/9d4ca48be349fd27fe5507cfbe6bda58ad24e0af6339d8fef497885b4a874685.jpg)

![tables](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/tables/9ecddbff306609264ba502b759f17ee026879dc6f982fecc8f65b9f434552853.jpg)

![tables](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/tables/9f31336b2201d7a24defbfb4558c1f99dff4826de5c77b5b0212866de5561da1.jpg)

![tables](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/tables/ac1ea955490598bffe89b5cb871fd6f14ae1896d64843d33a7bb5c0d03064073.jpg)

![tables](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/tables/b33be98dd1082f12a05e8064166880038e828a09e98453412095bc7410e023ed.jpg)

![tables](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/tables/b79db3ef73df3b540e5b9df4a9b7df2d4a00f3a28002bd58d6fc4c7fa43f70e6.jpg)

![tables](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/tables/bcbe3216d1075f29beacfb5db6c713613908975119a1c3303406bd6281b66ae4.jpg)

![tables](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/tables/ca1ec8c9c63469ac1d499b0de1ff5538134fba422dc203ef673c22c113e8d5bf.jpg)

![tables](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/tables/eb62ab501c2534a8fb8f68f2623a741e1856806e87a684a14e6dd946dbbdd643.jpg)

![tables](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/tables/ee6af3bbae2581cfe782595b341794c64db95dd07c49866e5032d00c18959234.jpg)

![tables](../iclr_results/143_Omni-Reward_Towards_Generalist_Omni-Modal_Reward_Modeling_with_Free-Form_Preferences/tables/f9d28b16c23a3591123902c92201410ec3e13e14761ee82abac0ac5a083f96e1.jpg)

<a id="paper-144"></a>
## AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=qNlTH4kYJZ)

### Images

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/07fc036c6d10a54e783928519a10932b1963539e0de731f1302892a30abc23e3.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/17489505c2f7861cdd9b2c1d3b2f6b0832973492ffd26f23ec16ff4ad27a07f4.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/17ed2046a8c6a6ba4228513b264ff06eca6d28e46c7ddc98df9f5287c89948db.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/1b92e71cdf434027a2989d736da8afce33bd25a9afa1d5f995aa4939672af228.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/230c50933bbfce06848b07f6cff79ddedaecbef7fe65229e161e380a28e44952.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/31a19a3f6c242306aa023e47e18ed252e5d074c0b4b529b61c0247745e9e7253.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/3de122cce0a932c2c4ea4d2093484eab871ddc07672be3a591c8f21c195282a9.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/45831b0481e27cd16f68588607ccdd14ef038f11993621f07fb7454d72f30207.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/46a452217d4d2dd4497de5fff915736f252db05af20ec4b29495af4e699f5861.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/482cb7e1b1bf8a7444208acbb657376ad7d39b0d38116a170911a1b8036efc1a.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/48fd6fc20efcd0fd90c7fa3e5ef3f6a9886c66fe4b4ee3bab784fe84d41b43da.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/6c80cb49d1d421280ccbbba7524406bf59298f2184792d7e955a08bd52402899.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/6d9cea7e7325c92f277f8aa6fe1971fca37474aa75e6055ec8f0c6e74ca80efa.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/8291e452679ac7b4e163c2f0f7cf1ccbb3ac3fd564a6e4475d18cd7eaf278017.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/82b2379dddbcb52c6016b56593c39bd5d48b9ee7099ed87a801c2a31fa448b70.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/84508267aa528c0f77abcca06285c470b98849f4ff7b02adb2bab75b5a318178.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/86b76453331675cefa7f471d0b14d14c28651e89ec68ab1c564d79262c5ed180.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/8f09d1a3a737e7e95538f6a31502b9d6dd1173ce4398364f9cd1a171841826ad.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/921c927b513c68c40363ab4723253ac65a7c895777c53340e6a58529756bb186.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/a1ea3b58e7ad6f5a4f2a53acfff3a0fe35e208fe84fc5729b61b6bc25cd4527d.jpg)

![images](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/images/cd3567e8be1f295f4c757fa98b2b0f76f3d9903cb70902dc60a278ff93bcaddc.jpg)

### Tables

![tables](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/tables/0907ebf3ff23fa7a094c7a00f9433964689b3987aca55f6f3bc6a4d0a043160c.jpg)

![tables](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/tables/1db01656fb594b7b91988e8c47f291450685ab78df592cd733ad91ef243b0444.jpg)

![tables](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/tables/1ecaad486cee78543a4615b6d2dbceb024ea3ee8bd75751549000dd52f96f9df.jpg)

![tables](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/tables/2d5189af5fe4ca159147118403dcd990ae024eae0835ea7642032a7bcc0448c0.jpg)

![tables](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/tables/2d933301b853ceeea39ab607c0aa7a7f3852fd073092f4161065b4eb4e090e56.jpg)

![tables](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/tables/339df1f02c136a30a18b3c8a2cb1d8ee639f21ee75d5193ab58ffa37502e4a37.jpg)

![tables](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/tables/5033f8e45932fa154a4763e2fc2bf2f34ac55217ec778abffebc49f92c47839c.jpg)

![tables](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/tables/5a70ebfcd903fe999b958081125d40a463e13db87dae29687e8be598be701650.jpg)

![tables](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/tables/6b1b13e03382ddd528cccb94c083dd8d6f66a1058643b6298fb109961b021d98.jpg)

![tables](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/tables/7b54531604848dcdf3131acf68892c0ddfe1b6a84bbd0443f40feed4d5739c82.jpg)

![tables](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/tables/83d1ce25a1bf5e72c834085491b70f914314eddb0acc949435a438992a4afc2d.jpg)

![tables](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/tables/95f7b5f75759b66ffc1af20eed534503ebc0f37ab0e635c4e3aba71ab69035b2.jpg)

![tables](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/tables/997e1fbce44a7099609537941b67f49d0d1c4260009d2b165d8193d2cc231892.jpg)

![tables](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/tables/b38c8aec5d9f6d7852d4f1aeaa8e616e71b943baeabd0b6f87747564f8a84855.jpg)

![tables](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/tables/bc2c34fc84a453cdafafb8cf040196fbfe523131609050cb29a373dfd7aa1e2d.jpg)

![tables](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/tables/c83becf8301b85106924af702631e2e09fb7f1ca814b79c2e73417423dd8f20f.jpg)

![tables](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/tables/d004daf3da1e6131f7e8fd5ed4d1f96aab7e294935636f2b95fe3af2ccd1b29a.jpg)

![tables](../iclr_results/144_AdAEM_An_Adaptively_and_Automated_Extensible_Measurement_of_LLMs_Value_Difference/tables/dac12d745e0d4a0756bd345beaf8c19f2e053031f3e2dc18d4b9f97aa4d32824.jpg)

<a id="paper-145"></a>
## VibeVoice: Expressive Podcast Generation with Next-Token Diffusion

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=FihSkzyxdv)

### Images

![images](../iclr_results/145_VibeVoice_Expressive_Podcast_Generation_with_Next-Token_Diffusion/images/06eff157549f696227a4da6ed1ca9f3ea39637c8b8584ae904fea98cafe7cc20.jpg)

![images](../iclr_results/145_VibeVoice_Expressive_Podcast_Generation_with_Next-Token_Diffusion/images/53db85d583a5125c2ee266a2fde50511a40ca8322b330e0d6e84d0b38c3a5a1a.jpg)

![images](../iclr_results/145_VibeVoice_Expressive_Podcast_Generation_with_Next-Token_Diffusion/images/55acb3eecd125d84ba608128abcc5b648d594e033e9707994ed5ece4231a1ee2.jpg)

![images](../iclr_results/145_VibeVoice_Expressive_Podcast_Generation_with_Next-Token_Diffusion/images/62efbc7b2ed4f3d354a3cb0dd56428d0c5175565199746666aa47124654a5001.jpg)

![images](../iclr_results/145_VibeVoice_Expressive_Podcast_Generation_with_Next-Token_Diffusion/images/7e22a5828783b94bbb10fa3c2404900e9e2d3a1d21c52a87351419800bb7e781.jpg)

![images](../iclr_results/145_VibeVoice_Expressive_Podcast_Generation_with_Next-Token_Diffusion/images/ed84052d5debd94557474e972d20d1dfa3657671b717251b684f1ca9ed51703c.jpg)

### Tables

![tables](../iclr_results/145_VibeVoice_Expressive_Podcast_Generation_with_Next-Token_Diffusion/tables/1c6ff14e4c621a22aaba1bf42cf41ceb48c964f3865740ad48315d2fdfe05d2d.jpg)

![tables](../iclr_results/145_VibeVoice_Expressive_Podcast_Generation_with_Next-Token_Diffusion/tables/1c73b2cfdf2d42b58ad04684a680759cf005512f66bfb0297c4a52b7860c4f3d.jpg)

![tables](../iclr_results/145_VibeVoice_Expressive_Podcast_Generation_with_Next-Token_Diffusion/tables/23ad637cd3b2b6c9f3bc62e32788367b0cbccbc9f8f21712cece4812a4f79d27.jpg)

![tables](../iclr_results/145_VibeVoice_Expressive_Podcast_Generation_with_Next-Token_Diffusion/tables/5b2812c42c020d517f5463f4303cfd00da070b133b650862570b092156f88b6a.jpg)

![tables](../iclr_results/145_VibeVoice_Expressive_Podcast_Generation_with_Next-Token_Diffusion/tables/5d3c01977f0a3ec475d4b116921e9424c12b1c3f9a9ce0ae90aa9d2f00fca8db.jpg)

![tables](../iclr_results/145_VibeVoice_Expressive_Podcast_Generation_with_Next-Token_Diffusion/tables/79d6d453a3ee0a5a4276580616c66de9f3cf4bb8b294fcc3c58b87abea49aa56.jpg)

![tables](../iclr_results/145_VibeVoice_Expressive_Podcast_Generation_with_Next-Token_Diffusion/tables/bf92fbb7fe24c4caa82fbf6d80415c4abffd7a4b5d1591ebc70db42916118891.jpg)

![tables](../iclr_results/145_VibeVoice_Expressive_Podcast_Generation_with_Next-Token_Diffusion/tables/c12e66f99fe63e26559c1ac511759a7cda903a784bb814f209de83b6415f229a.jpg)

![tables](../iclr_results/145_VibeVoice_Expressive_Podcast_Generation_with_Next-Token_Diffusion/tables/dd31cadd65033ba3fd4a01d7b00bbc00804e7cbba3a7edea720953ddf127c580.jpg)

![tables](../iclr_results/145_VibeVoice_Expressive_Podcast_Generation_with_Next-Token_Diffusion/tables/e1ef4e6851d3724ac19bc44a7d787bf1ecdd70a2bf3e6fc439fdb4719f37fb41.jpg)

![tables](../iclr_results/145_VibeVoice_Expressive_Podcast_Generation_with_Next-Token_Diffusion/tables/ffa225f19f895f33befa71c8f6d43e724e2e72af16fc446f180456368857b8b0.jpg)

<a id="paper-146"></a>
## Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=bojVI4l9Kn)

### Images

![images](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/images/01e28603867bed67b99c4f73f71aaa90e1b9ac563c78ae51911af0e534c88440.jpg)

![images](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/images/4068670775b913068f163c499ea508ac8f3fe3dc446279b1c06564c442956b37.jpg)

![images](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/images/4733a7bb7fff3d3ad77948a722c66d6ec82387d49557be731178ea299a375ccc.jpg)

![images](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/images/4c0c6ffaf70fbbe000a3c6152384386967e55c4e790f854500a214563bc99dfa.jpg)

![images](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/images/557c80759a847ff12a5685f78af8b5e4dc8994cd0cece375c322ce3d0e73cd0d.jpg)

![images](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/images/620353e5b6c25108828df99f6e2e392692ccf2c9e6320077e80528d4ef831535.jpg)

![images](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/images/7fe0b48f4d67ce2b7fa6aaa9f52ecb1fcb822d585ff92ee4aab2fc16e7288347.jpg)

![images](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/images/868f8c8feab2e80efe779e9dd6343c09fc3735a9d72cc09b08b69feda6d85fd4.jpg)

![images](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/images/9104d9c0407b003dbae8eb5353e9bf93c34fe7e90e48c14092f0352538f3a8f0.jpg)

![images](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/images/a2c04645f7f0615a0adf0e5d46752d7798de4a5dc67f8cc2ac8cd568d7406233.jpg)

![images](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/images/c18c6f4dbb843e7deb77219337833231241ff1951da1a6c25cb170d5ec0903ed.jpg)

![images](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/images/cd7fc373bd150a08e81dc928bb38a7caa99cb5258d5f5754b047c676e20c3701.jpg)

### Tables

![tables](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/tables/1415f246af3b48d9431c92fbb4b2f92f8370e2726c4a7c0912836807c960c306.jpg)

![tables](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/tables/2af437ee7adbafcc9e62b5eed8ded06f0f2191ae7e067928edcc55e4324781a8.jpg)

![tables](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/tables/33673ad438c0b6690ccdb59b59bf4b04de19e90111a93c71f9e5423e059bdcf5.jpg)

![tables](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/tables/3a73d09a1498ecf424dd3cd46123a6196251ecc5709b7c9168baa7c694391a9b.jpg)

![tables](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/tables/3eaae84399df3c078a03960c7fe7bba2a723c2af7dcd57cda84eba3a059d95b3.jpg)

![tables](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/tables/5883340e8a2c11e95440d7c1f397fd908e961acaec22559d85fcf5e267de6916.jpg)

![tables](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/tables/6cdb6e878454e9e13a552b2628235eeaaa6771969e59d9fe4d14ee0e9e596613.jpg)

![tables](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/tables/6d7432329debacc217c693aa0eb4a83f9fe86f5e18d53fd6b6077f628c440e9b.jpg)

![tables](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/tables/6ed2d7422ef423f707b85118483f7443d23d0661eceb4076a6b8a1b8e1951dfb.jpg)

![tables](../iclr_results/146_Temporal_Sparse_Autoencoders_Leveraging_the_Sequential_Nature_of_Language_for_Interpretability/tables/71c70c3c9700a6c73acd704c6f96f4b4dd7b40db52577fd103b6d6164514d619.jpg)

<a id="paper-147"></a>
## Seeing Through the Brain: New Insights from Decoding Visual Stimuli with fMRI

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=88ZLp7xYxw)

### Images

![images](../iclr_results/147_Seeing_Through_the_Brain_New_Insights_from_Decoding_Visual_Stimuli_with_fMRI/images/119be8ad3a609e53bfdf47bb4493b7abef4e3d33dff78a496e1fed92735197fa.jpg)

![images](../iclr_results/147_Seeing_Through_the_Brain_New_Insights_from_Decoding_Visual_Stimuli_with_fMRI/images/173d5cd96085354735ae249aebda8ccd3a1df74b35ba6a58f2f378b9b89230bf.jpg)

![images](../iclr_results/147_Seeing_Through_the_Brain_New_Insights_from_Decoding_Visual_Stimuli_with_fMRI/images/528654570472e11d2d8a2de8d6b4310b55b04e551369ec3194ec924eafe5df35.jpg)

![images](../iclr_results/147_Seeing_Through_the_Brain_New_Insights_from_Decoding_Visual_Stimuli_with_fMRI/images/690eb127c63818e367d5b7666c59fd3a352c18829bd92a50499aff6460299da9.jpg)

![images](../iclr_results/147_Seeing_Through_the_Brain_New_Insights_from_Decoding_Visual_Stimuli_with_fMRI/images/8e7af5ffa19afe51f59c9f708d7d43f6bd58f4d13aef5447eb89323d03cfd90a.jpg)

![images](../iclr_results/147_Seeing_Through_the_Brain_New_Insights_from_Decoding_Visual_Stimuli_with_fMRI/images/b6c7568a224f2d96802c6d4bf2d5fb0b97411562456735e9887ec4717b8417d9.jpg)

![images](../iclr_results/147_Seeing_Through_the_Brain_New_Insights_from_Decoding_Visual_Stimuli_with_fMRI/images/dbaf332d3f1a4b6589f59a11451cf5082f6c5fa0150c903f852593aa3918a3c7.jpg)

### Tables

![tables](../iclr_results/147_Seeing_Through_the_Brain_New_Insights_from_Decoding_Visual_Stimuli_with_fMRI/tables/204eba2091e859ae900a1340f331a33a3d81a7c011a7077fc7d6d39524f3104a.jpg)

![tables](../iclr_results/147_Seeing_Through_the_Brain_New_Insights_from_Decoding_Visual_Stimuli_with_fMRI/tables/2d52b25d87414ceefbdeec0452821e34387cd1c17d3bb0ed9f3a6b28e7ad315c.jpg)

![tables](../iclr_results/147_Seeing_Through_the_Brain_New_Insights_from_Decoding_Visual_Stimuli_with_fMRI/tables/39bbe71a81865ca6e8add1e05e58aaa9259f8dcbe9ec4c303c4361b471579686.jpg)

![tables](../iclr_results/147_Seeing_Through_the_Brain_New_Insights_from_Decoding_Visual_Stimuli_with_fMRI/tables/8f680c8fea4231db749051582550f1298431661b85ac2902ad9afa3c9f862e10.jpg)

![tables](../iclr_results/147_Seeing_Through_the_Brain_New_Insights_from_Decoding_Visual_Stimuli_with_fMRI/tables/9329e0837106ad577533d1e8cbaeea037f007779b0e583e6e043dca470471564.jpg)

![tables](../iclr_results/147_Seeing_Through_the_Brain_New_Insights_from_Decoding_Visual_Stimuli_with_fMRI/tables/a4427e5fea41f2fb1262984a9856c882697932db825807567d2994eacf79b6ed.jpg)

![tables](../iclr_results/147_Seeing_Through_the_Brain_New_Insights_from_Decoding_Visual_Stimuli_with_fMRI/tables/b4b71d1b9bc1c7785b234630b9cf188ab8763c07e68871d7043b1dbc5e798fdb.jpg)

![tables](../iclr_results/147_Seeing_Through_the_Brain_New_Insights_from_Decoding_Visual_Stimuli_with_fMRI/tables/d8d5e4e3a920a07500fd61d0884733a2e36582581a41280bc251bca12bba864e.jpg)

![tables](../iclr_results/147_Seeing_Through_the_Brain_New_Insights_from_Decoding_Visual_Stimuli_with_fMRI/tables/e775718911b0c384693392832b6d615fff9d523f632a23ea6c77290e5bc36e2a.jpg)

<a id="paper-148"></a>
## mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=r2HG3xOMJI)

### Images

![images](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/images/02f987de61ba4c3c7b7e05e0311f0bc74de63e637e1e45912bb3f5b7c7f5e68e.jpg)

![images](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/images/07fbdaa978eb8f18868f7b7c0576c893ebf6bd3f6ca10ad0dc077d475c21083f.jpg)

![images](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/images/0aef684efa74168c1c95ae08f7c80e79222c077338a02565c2b66e3d9d940e92.jpg)

![images](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/images/50ef2cd6345ffc3290d4d6b4b8b5d21c8f918880bedd09ffbcb8de4a03e42ca3.jpg)

![images](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/images/544f56bf310e87e62810d8b8745822feb7bda65d63ec34ad1dfe402c5a17be0b.jpg)

![images](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/images/5607b49aa1eb5f17a2d65435efdb6bb671b2cb8cea9876f03d7b1a618f4a278d.jpg)

![images](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/images/74e4932c383fb223c5fa744553800e6beab0c7d85a5836510642ba61da7a7761.jpg)

![images](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/images/8a92842c7173eb269b0252f4ac48bfbf54b1b5c51809b87083da419058626513.jpg)

![images](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/images/8e5532ad1ef3a679a1a102167844fddc5d5b72e18a93ccbad9fed496789f5ac0.jpg)

![images](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/images/cabe440de31cd653f7f3a430d3ea903abef05f934627504abc9fcff12d33b78d.jpg)

![images](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/images/f1b35e64b81c402b538dcd8dd86c27df7d7ef492a6703da0a78fc0febfd068e2.jpg)

![images](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/images/fb281bd4fc490145b0c9df12f56514257d5c12878844eb9efd2ab191ac6d6e2b.jpg)

![images](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/images/fe68f9689e60d6d26948e690822d5b0cb58ee3f1a511e43d693a1615f22047af.jpg)

### Tables

![tables](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/tables/23d71a305b47215412376e4b79b193586c4ed4667822ca3deb67eab965687d89.jpg)

![tables](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/tables/278b2ef036f3871b00efcce8ba10726dedaf72c83d308c67977d3f38bbcc8d95.jpg)

![tables](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/tables/2c78a50e2e37c1989875aa57a95080e6a1d3537c1a79dc8be8f9a610898b5882.jpg)

![tables](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/tables/3c8e89a7fa67c77f9d97ae110cfc0e43b6e51b0d247136ec43cfc41ca2a818b0.jpg)

![tables](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/tables/497c1906aec3ccc9de1274c92902ada866794d61e13f6f51391846adc34b337c.jpg)

![tables](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/tables/705ece49235d606f2a390d53f746fc1abeaf6b15b33d96a237de936f63e0f031.jpg)

![tables](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/tables/886e9c445dff0830d57170ce4b1c26da2f1d3fe4a8557d824f776659c7331692.jpg)

![tables](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/tables/cd7e875de57ca67ee36b733efedc9a476ce031f94309a5d6db1ec255f30844e7.jpg)

![tables](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/tables/e464d1cb6ad1bf2422451c34ae42d56d1edfd689769127e328cccd80d69ae96a.jpg)

![tables](../iclr_results/148_mCLM_A_Modular_Chemical_Language_Model_that_Generates_Functional_and_Makeable_Molecules/tables/ed96b29ea6908660cccfdf737b31c189644dcc63b005062598970e6d04c338fc.jpg)

<a id="paper-149"></a>
## WAVE: Learning Unified & Versatile Audio-Visual Embeddings with Multimodal LLM

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=MiV3WXDYJb)

### Images

![images](../iclr_results/149_WAVE_Learning_Unified_Versatile_Audio-Visual_Embeddings_with_Multimodal_LLM/images/67ca32c0539d95392ce4aa4a3bcaed1a2243bc5a5356c20c4754ec2623012d8b.jpg)

![images](../iclr_results/149_WAVE_Learning_Unified_Versatile_Audio-Visual_Embeddings_with_Multimodal_LLM/images/761654c41244d2d19da08f6789092dd8e0b722e7555cedbfd4d6931111cb280e.jpg)

### Tables

![tables](../iclr_results/149_WAVE_Learning_Unified_Versatile_Audio-Visual_Embeddings_with_Multimodal_LLM/tables/0f4fdc20fbf948681cbc799edaebff1fd66fd93a0b5cc896ee22768eca6c5b6a.jpg)

![tables](../iclr_results/149_WAVE_Learning_Unified_Versatile_Audio-Visual_Embeddings_with_Multimodal_LLM/tables/3d7db3441c3d35be6d2aa7743153c63c622388757f4ed2cf3fd35f1c33bf8dc9.jpg)

![tables](../iclr_results/149_WAVE_Learning_Unified_Versatile_Audio-Visual_Embeddings_with_Multimodal_LLM/tables/807250d9d41c833f825426215f7da240880bee75efd2f97c43200984712c1efd.jpg)

![tables](../iclr_results/149_WAVE_Learning_Unified_Versatile_Audio-Visual_Embeddings_with_Multimodal_LLM/tables/8355cce44079488e3281db15c6ae8d149ffb315e2b539d4af1df0b127dd8c4a2.jpg)

![tables](../iclr_results/149_WAVE_Learning_Unified_Versatile_Audio-Visual_Embeddings_with_Multimodal_LLM/tables/c4efe14dba78966f023c41951ab6eb4ba76c63f94b06af228e387f8a34b938f5.jpg)

![tables](../iclr_results/149_WAVE_Learning_Unified_Versatile_Audio-Visual_Embeddings_with_Multimodal_LLM/tables/ca10e24e1999b17b3d85bc2dbd92192e26e739d60731093915f312e8d750a27f.jpg)

![tables](../iclr_results/149_WAVE_Learning_Unified_Versatile_Audio-Visual_Embeddings_with_Multimodal_LLM/tables/cacfc3b4f555faa9efcd1756268aa14d20c2c17130f449638b0f54531c2537e3.jpg)

![tables](../iclr_results/149_WAVE_Learning_Unified_Versatile_Audio-Visual_Embeddings_with_Multimodal_LLM/tables/cbe223367983e4370e6d14a758b1e290bce2a1ab062598c7646e8085a44390c9.jpg)

![tables](../iclr_results/149_WAVE_Learning_Unified_Versatile_Audio-Visual_Embeddings_with_Multimodal_LLM/tables/e4cec5a73df1d9c37ecb7e860b755b4b72e4a5dd7f9a3ac4578930b2107faac2.jpg)

![tables](../iclr_results/149_WAVE_Learning_Unified_Versatile_Audio-Visual_Embeddings_with_Multimodal_LLM/tables/ecff68feaff3bcad82bc17b02a84bf98f2833db668592cf1280ba05eb6dca678.jpg)

<a id="paper-150"></a>
## Learning to Segment for Vehicle Routing Problems

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=pN261iTKvr)

### Images

![images](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/images/04937c1443f0a8674295521db137b1b48958eb68d455526fe27f7d41554c8a5e.jpg)

![images](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/images/4d276bd4b0f0bb6692647f8ad6facfe3b2a3cf929692f07871429cdf206d6d8b.jpg)

![images](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/images/4f7b9e8776159fa0f680da0e2230612cec28364315e7f3170fae68c61d339adc.jpg)

![images](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/images/59f8a3e9883705467ffb55e33041234a2910f96fabfd8f26c265d3f1b0b775bd.jpg)

![images](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/images/69c2416641a9a63c5d445bfda59c499f7df6a8915797e0a35f2aa23a2de0d9a1.jpg)

![images](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/images/904e163448a00c46ea73152b272950e54dfa0eccc6fa8422bb0873e96d762271.jpg)

![images](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/images/906dfbe537d73e2155fdbd5095e62f30964d3cab9f198e628f253c82384378fe.jpg)

![images](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/images/952a087bfa6fce6492ef2110db4e6da7d4402fff0f822c04cd0358adbd9c7a05.jpg)

![images](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/images/a3e17b1c0027d0160d059547836d47493901c164bad00ff00f8c5fd3916dae05.jpg)

![images](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/images/bb9fc1ecbf26c4d4bd3cbbe21107b0ecfa28561c1a5a6bc514f2502f7b0761bb.jpg)

![images](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/images/dbef419d63ce17430432feac9963bcfe72a8856bf792a02600369119068c60f5.jpg)

![images](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/images/ff1c47a2b43f22d579ffc92a5a3f4bab5c1c13f0df5b606dcb820360490baa99.jpg)

### Tables

![tables](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/tables/1202ce40b325626dcc0cab5ddf21e6890e548c92ce6384c64693362310a4a4ea.jpg)

![tables](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/tables/128621a7702a1af2ec237bb6663c11d1629e12acf2691a70ab46137924179c4f.jpg)

![tables](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/tables/1e005cda9f6392f2aa6592fd4f784c791bb3c8dcec5ac6fd18515d9f46d77bb5.jpg)

![tables](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/tables/319c49f81854a16775e43c133effed1a959a47d50e4035cb027ddefb7e316c8f.jpg)

![tables](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/tables/3a6b5027d3474c2449690e0414153ea8de397477245c57bd5eadaf99c66cc561.jpg)

![tables](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/tables/5a1b523dc20c779190520c891016940cd4f1eb4bea40f12e2a5b0408bdb33c86.jpg)

![tables](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/tables/7bb925a7223d845fb0e9bc92a9538110f6779e37457739ebbf53ae5358483bf6.jpg)

![tables](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/tables/8733622452a70bf9724ff44e2a97c95ded61bb98a68e4f8a82d6d1b591898360.jpg)

![tables](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/tables/89722c6ead666348de72ec5cf7f3a59579a26b5640ebe9e4dcbcdf6deb767b12.jpg)

![tables](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/tables/9e364c18a280ffcc16c03867e9660181f92630a49fa523a948f78c4ce9783e10.jpg)

![tables](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/tables/b4795bbf2cc8bc6abbcbc378e6ea26b970bcc98cf214d55c5733f234e59a506e.jpg)

![tables](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/tables/b59edf5085ffa2261b51da5e5d904ed6eb949913991f20e083d79b5fd1db3340.jpg)

![tables](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/tables/b79a292af9d82b057de365b65859a65ad2470d28d702e65d9471625f69a5770b.jpg)

![tables](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/tables/c151ce81c8c6c035910d7f2ed73a41a191ec4276dbefa89be8b5d6057e26be07.jpg)

![tables](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/tables/e75404458e03f969adc0e03c2a6e07d24303cc35708e09b78e435486f27f7b40.jpg)

![tables](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/tables/e882c6fbc398076b8aa3a97325830d1a7348b0cc13dbc6b6a1bfbfaaa5b561d8.jpg)

![tables](../iclr_results/150_Learning_to_Segment_for_Vehicle_Routing_Problems/tables/fe0b2d5259154530d2c6d779ae0bdb8120ecb160a7e428da467bde5e3a109cdf.jpg)

<a id="paper-151"></a>
## Reasoning with Sampling: Your Base Model is Smarter Than You Think

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=Vsgq2ldr4K)

### Images

![images](../iclr_results/151_Reasoning_with_Sampling_Your_Base_Model_is_Smarter_Than_You_Think/images/0d15240a1f94122a135e22d6aabb4f690f2c22762f1be9ba417f2ee76104cc00.jpg)

![images](../iclr_results/151_Reasoning_with_Sampling_Your_Base_Model_is_Smarter_Than_You_Think/images/490204edcd105c93939411e5491a8912c1255878640b8f5dfd061ac0cd4dc523.jpg)

![images](../iclr_results/151_Reasoning_with_Sampling_Your_Base_Model_is_Smarter_Than_You_Think/images/55e2ccd3f5c82f935692eba17770d46999ea612b632c63c6159adf145e646c85.jpg)

![images](../iclr_results/151_Reasoning_with_Sampling_Your_Base_Model_is_Smarter_Than_You_Think/images/8fb9bd9583d70594a04dc4820ac07ab16c6c263f1d23245484eab94ee37012d9.jpg)

![images](../iclr_results/151_Reasoning_with_Sampling_Your_Base_Model_is_Smarter_Than_You_Think/images/94551ab6d840178b2efb163915731bdd27117ebc2b30761b5a5406e90aab42f4.jpg)

![images](../iclr_results/151_Reasoning_with_Sampling_Your_Base_Model_is_Smarter_Than_You_Think/images/95093a74f61eff9c1bd8bb756dbe2496d1c71feb8dd1ee7a2990060acbf78a82.jpg)

![images](../iclr_results/151_Reasoning_with_Sampling_Your_Base_Model_is_Smarter_Than_You_Think/images/af24f7e4ece1ae1b7c6d559ed2c6a01d1f2da8ea33bef350a6aa89050e67220c.jpg)

![images](../iclr_results/151_Reasoning_with_Sampling_Your_Base_Model_is_Smarter_Than_You_Think/images/ba690eff137e0060aae10b1afb22b119e129a3d24aee985682bfc22f973de2eb.jpg)

![images](../iclr_results/151_Reasoning_with_Sampling_Your_Base_Model_is_Smarter_Than_You_Think/images/d14b58282a0cc09735d7c269327fe2c94201ccf78199d5fe85958583c1725fa9.jpg)

### Tables

![tables](../iclr_results/151_Reasoning_with_Sampling_Your_Base_Model_is_Smarter_Than_You_Think/tables/2fcfceccd5ac246cafe10d2b1e0c1bc5b197e6bae3be6bb0711c7210d3826fac.jpg)

![tables](../iclr_results/151_Reasoning_with_Sampling_Your_Base_Model_is_Smarter_Than_You_Think/tables/78ddbb8673682bdf67a1bc0059ffe58d6992ba0e6b6522b3570fcf730c41bb42.jpg)

![tables](../iclr_results/151_Reasoning_with_Sampling_Your_Base_Model_is_Smarter_Than_You_Think/tables/81bed47d1e7fe4dc1c3edf09b76c90e64887a4aa91d59bb7932ee969040e1386.jpg)

![tables](../iclr_results/151_Reasoning_with_Sampling_Your_Base_Model_is_Smarter_Than_You_Think/tables/aa091f1f331ed2bb381a4c67584a39a18b30b6fc07e558dbcf2e4a429c8f2d48.jpg)

![tables](../iclr_results/151_Reasoning_with_Sampling_Your_Base_Model_is_Smarter_Than_You_Think/tables/e09ea221422b1622d0ba933216448ebfe5a8b02c6d726a939e8106925f07aff4.jpg)

![tables](../iclr_results/151_Reasoning_with_Sampling_Your_Base_Model_is_Smarter_Than_You_Think/tables/f97ceea65e9a13245e32e3091f99cd6a701a6db00bbe4f7aeac989adc3f4341a.jpg)

<a id="paper-152"></a>
## SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=LCS1WsGvha)

### Images

![images](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/images/184ec077e976a4a624391181db7e70ec33713f765a9195537607be53560c1f1f.jpg)

![images](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/images/2ccb1260b7ef9f0fbfc606edc2a3e6cb6aef811f3ad1ffb1a8aff1ea4b05a668.jpg)

![images](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/images/48e66906d60ef25eb912cdbd447c19014bfb9bad3dc873b57168f16970ca2ab6.jpg)

![images](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/images/a839a3a344ac09d0eb6b3f36d2145c45101dc50750810f0912457a9f4992b464.jpg)

![images](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/images/c5f899883885f0cdb090c9f89e2539fb92db83dfcc3ce742a2d386f8a58cb469.jpg)

![images](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/images/e39e815c8b3d462d3017d127e4bf904c5bd3f536a2d844e5d66d3caac63f0893.jpg)

### Tables

![tables](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/tables/1f81fb72ffa87acba22efb39685a190fe39305918547b6b5bedc4a3e748a4ce1.jpg)

![tables](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/tables/3b4b3206e4f1d6d4c5db7a2198b90fe2231b88ebae26f02e7b1af230de9a45ef.jpg)

![tables](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/tables/3fbc99decda38552cd160c2566ed0f3f1a9f3695ec5c5d114167d4e0bb4d5aa8.jpg)

![tables](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/tables/54b28105c84bca00a12d1233bf00e5bf6bb84e74c4ed04670515fb112975c22e.jpg)

![tables](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/tables/56bee3ea2997170356139b9724d3e70364f7327fb0976c128041e338b4ca903c.jpg)

![tables](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/tables/5e1b943599f4cc9c218c95bb4730b19fde9ac455087c3976fadb5d24d98ba2eb.jpg)

![tables](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/tables/63fa97a376f38fd579b4069a83c82fc34397714fa362882c85eff54b6dc22c71.jpg)

![tables](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/tables/683588b41aef9b6c93ce5f371fe10775e38e737db7e95fc62b10a6d090746304.jpg)

![tables](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/tables/72aca1f27674959f3c0e6d5865414c36f26929660620b736ae4a3179f99cd198.jpg)

![tables](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/tables/76667c4c9116ceefc2f54f0394438e538966964e5666a180d0164679d1553eef.jpg)

![tables](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/tables/7cf0bb4aafa5055f1bc84c6c6810c10485242ff9fa5989fbf379845332c2d47f.jpg)

![tables](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/tables/824009a8414f60c065cd441b97c1b9fa28196070211cf153f603958a35b4104c.jpg)

![tables](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/tables/88ca6341ceb2987c875dbb85ed0618671df86c7d1758fd5c4d89e1148a650147.jpg)

![tables](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/tables/8b09fbe45d84a7149a3dfc406aafec8509e7ccca9f7679fb2e726d988223423e.jpg)

![tables](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/tables/a8babdadace00640e62d43ae00beb810029839222f5a02fa67fff1ff31dfcfda.jpg)

![tables](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/tables/b6be18ae8ff5c4ba3a95f2991febf3f34633ff593ea30630319cc3f4ba9d3675.jpg)

![tables](../iclr_results/152_SimuHome_A_Temporal-_and_Environment-Aware_Benchmark_for_Smart_Home_LLM_Agents/tables/be98f52d9869483cb6c54de2f568bfffe324f089dc7d6003bb1a28102a4297a7.jpg)

<a id="paper-153"></a>
## What's In My Human Feedback? Learning Interpretable Descriptions of Preference Data

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=sC6A1bFDUt)

### Images

![images](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/images/028b3cb1f3b2a5d7f61b5aa72817a7bb3662303b714d97d3fc14a4629bcf5283.jpg)

![images](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/images/4bed480e3f2ea82082e2e01f426a856529eccf8cac2b705c28e316fbe3ed305f.jpg)

![images](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/images/5860109f8bbdcfc09ee747ab2731d575ac0ea2ea6cb415664d4dab59871e4d51.jpg)

![images](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/images/5dbfc224a13ac7ad624f8bbdcb6177019f2087be161691f09e748dfb5f3d7adf.jpg)

![images](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/images/c16166e5f7714a94ae68c46e9d157532ec796e8a71b22d5ba2857851297803bb.jpg)

![images](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/images/c55ebcaa612a7a477f4c2e3157acef70a9263e77e74db5ab280b673731d50ada.jpg)

![images](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/images/dae95838f23363ba0fbfbc10244a64927e68df5fddccc0a66eb6f7f04831d584.jpg)

![images](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/images/f6a7c775b128cad010b34007074a6c99d3a738160eb8da4fac094a8c739e0b9d.jpg)

### Tables

![tables](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/tables/0bbb879c3045f2602713ecb5cabf1d633cdeceeb8e3839647e4c239403a37de7.jpg)

![tables](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/tables/1acee41e29110a38eefab69a206ae3397c695a9714c630dba6b2895874afcbd4.jpg)

![tables](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/tables/2c97eb0a93ec36c363940151e3b505be72ad6e83dd3eb48521890ffd292cd7ac.jpg)

![tables](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/tables/2cb6e5a588199d359395fb7d17deeb679f1f558cecae003683c643880854625f.jpg)

![tables](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/tables/455f53e79a51264e0d56f9a27f4f5a684ed3dab6373fd059f74cedeb4bcebf4d.jpg)

![tables](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/tables/4badea15162ece4a62062d92903ef07c8fc55aedb95a4f50cc932d09283e3724.jpg)

![tables](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/tables/611a26f19f9ed55e77f7678233ff451edf245c390561b828a187d1685fe573fe.jpg)

![tables](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/tables/62236a4ee4a942404405c30f289669aebd4592eae6fe65acdc05412a9f033fe4.jpg)

![tables](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/tables/62c2b61286bc903809d47629bd4ecfdb7640c5fa117d4fecead3a04423603139.jpg)

![tables](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/tables/6ef4352b69af4d07b6bf6637bbf175b440d6c8f2f156b986352dcc18c41d58b2.jpg)

![tables](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/tables/81c40ed3d4f9a84902b3c267e315105f22a3baccb77fa0a8c81fc1d2aa39ef38.jpg)

![tables](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/tables/83781eba52173c5886476280024ebb7d169b7b5f8512ae6bb7437d8487a55b59.jpg)

![tables](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/tables/9aede5d657b54b7627350053483a5d1962abb940a6a89a8d5e6650e96c5d5fa0.jpg)

![tables](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/tables/bdfc243b559626dfe40ccd822c9caec6a2c4fe7c66d629ba2916edddef04bdd0.jpg)

![tables](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/tables/c342d49a26235cc6b069cb563e4ac9b66f8924974d35ee057c4f41be10a4dc3a.jpg)

![tables](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/tables/c6881da9012a2889e59d85a0980d30dcc89353f2e072fca87a386853ec13e0b2.jpg)

![tables](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/tables/e037324a03bb207c70668aadba6cdb053feb89c3b27db05fd63c4ba376b3b62f.jpg)

![tables](../iclr_results/153_What_s_In_My_Human_Feedback_Learning_Interpretable_Descriptions_of_Preference_Data/tables/fe0a83ec4ad205821abe7a32c0596e0183d3de75ca2db692a1460e301e773c6a.jpg)

<a id="paper-154"></a>
## In-the-Flow Agentic System Optimization for Effective Planning and Tool Use

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=Mf5AleTUVK)

### Images

![images](../iclr_results/154_In-the-Flow_Agentic_System_Optimization_for_Effective_Planning_and_Tool_Use/images/221d3aedff3b0e0eab8208fead07155006465228c184f2d80d3807e23fccba47.jpg)

![images](../iclr_results/154_In-the-Flow_Agentic_System_Optimization_for_Effective_Planning_and_Tool_Use/images/2fa5aeb9232eac576b65bc29708d181ac7f35ac453e3d297ac0de52ec5f2293a.jpg)

![images](../iclr_results/154_In-the-Flow_Agentic_System_Optimization_for_Effective_Planning_and_Tool_Use/images/2fd9ae2d44f20d93417025180b9230f7c0f6e5b307d674b33733f8b9c1cfc265.jpg)

![images](../iclr_results/154_In-the-Flow_Agentic_System_Optimization_for_Effective_Planning_and_Tool_Use/images/36598cf44e8246b0b903d0255c136fc0b1f5b6e569a8956333d66c95c4a2dd48.jpg)

![images](../iclr_results/154_In-the-Flow_Agentic_System_Optimization_for_Effective_Planning_and_Tool_Use/images/445fe9433fa7589d7a4445ce9113e1245ac94288f890e66ec38e7c8c72bd96a9.jpg)

![images](../iclr_results/154_In-the-Flow_Agentic_System_Optimization_for_Effective_Planning_and_Tool_Use/images/72a0baa6521d72250c41cac6e78871842d466d6609f741aed9d0179ed324d44a.jpg)

![images](../iclr_results/154_In-the-Flow_Agentic_System_Optimization_for_Effective_Planning_and_Tool_Use/images/7ef2f9ad1d706d8f0f8708f080c6243a5f3fafa789cb3a7b55911c169c856c02.jpg)

![images](../iclr_results/154_In-the-Flow_Agentic_System_Optimization_for_Effective_Planning_and_Tool_Use/images/93cebbe72ffe668388c266d8c872a2bddf0680f3223c90d14c68b9ce51ba054f.jpg)

![images](../iclr_results/154_In-the-Flow_Agentic_System_Optimization_for_Effective_Planning_and_Tool_Use/images/abbc20e50ebb7ee38a2076a334e3ed4d9ecc40bffc2645315330034b1b1ec57e.jpg)

![images](../iclr_results/154_In-the-Flow_Agentic_System_Optimization_for_Effective_Planning_and_Tool_Use/images/b76014bb9d5e92bd7e5435056d1099e7863e649beacd17476d697f24e38634df.jpg)

![images](../iclr_results/154_In-the-Flow_Agentic_System_Optimization_for_Effective_Planning_and_Tool_Use/images/bcac2a89dee0d6b54af4ecfa78695e675ff6291177513749872e8ed9b15e9a75.jpg)

![images](../iclr_results/154_In-the-Flow_Agentic_System_Optimization_for_Effective_Planning_and_Tool_Use/images/c763b3c8788b67d3be21ff9fecc0f0ff05364bef534b3b14bf93df08fb67346a.jpg)

### Tables

![tables](../iclr_results/154_In-the-Flow_Agentic_System_Optimization_for_Effective_Planning_and_Tool_Use/tables/2fbb70a49a794a18f3db69e4052a6f64b9daa8006fc8c315064863f77b0bcb03.jpg)

![tables](../iclr_results/154_In-the-Flow_Agentic_System_Optimization_for_Effective_Planning_and_Tool_Use/tables/5b633156e576383c9c620c830e349c024d7c7fec5e086ca253b1e47de7c686a5.jpg)

![tables](../iclr_results/154_In-the-Flow_Agentic_System_Optimization_for_Effective_Planning_and_Tool_Use/tables/5be28a0b7f4df067883320f4504792fb83c352e6760b41ca02101b14d229a66d.jpg)

![tables](../iclr_results/154_In-the-Flow_Agentic_System_Optimization_for_Effective_Planning_and_Tool_Use/tables/5c018d991b5d7a364c73378ea4e9e2f38515b3fb475496ac17196fb5bdfd6acc.jpg)

<a id="paper-155"></a>
## Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=LaVrNaBNwM)

### Images

![images](../iclr_results/155_Overcoming_Joint_Intractability_with_Lossless_Hierarchical_Speculative_Decoding/images/2a97ceb8d38d6c1f652d6c9cb06e99a84c9018260337c165c179d0de9dd19eeb.jpg)

![images](../iclr_results/155_Overcoming_Joint_Intractability_with_Lossless_Hierarchical_Speculative_Decoding/images/813941d1d161f7032e7bf362babdda93ef09f5a5cb0a8b09dceefa505901993e.jpg)

![images](../iclr_results/155_Overcoming_Joint_Intractability_with_Lossless_Hierarchical_Speculative_Decoding/images/999491ac678b81591f85be81a36222543ab261d664d502e37967258adda09cb9.jpg)

### Tables

![tables](../iclr_results/155_Overcoming_Joint_Intractability_with_Lossless_Hierarchical_Speculative_Decoding/tables/1163e086301e78d53b08c06dc6fe855c52e5b8b1560c409741f3dd80dce69f28.jpg)

![tables](../iclr_results/155_Overcoming_Joint_Intractability_with_Lossless_Hierarchical_Speculative_Decoding/tables/33472a5e4f55639ce66cf8db1aaf966140d9f04ca3e0baac3a139481a25d9bba.jpg)

![tables](../iclr_results/155_Overcoming_Joint_Intractability_with_Lossless_Hierarchical_Speculative_Decoding/tables/3467315c160229d572d5d81ae94e31880902fc94d547f3f720c4960318a6e13a.jpg)

![tables](../iclr_results/155_Overcoming_Joint_Intractability_with_Lossless_Hierarchical_Speculative_Decoding/tables/43bd547c0fe819253517c5fc021fe9c1d2c8061251988277fa71cd41c021c276.jpg)

![tables](../iclr_results/155_Overcoming_Joint_Intractability_with_Lossless_Hierarchical_Speculative_Decoding/tables/5d3ae5b8edeea737c8b9b53077ed4be34a03075c9c245ec5a7636cee804b5033.jpg)

![tables](../iclr_results/155_Overcoming_Joint_Intractability_with_Lossless_Hierarchical_Speculative_Decoding/tables/aa5422b3e89f6a970735ceeb61a6197857d5df5fc995c449847de28835c3d759.jpg)

![tables](../iclr_results/155_Overcoming_Joint_Intractability_with_Lossless_Hierarchical_Speculative_Decoding/tables/bf7945933050e044611e286b6ace96f7ddf8644dc1d5ed73a1e516596f82b02a.jpg)

![tables](../iclr_results/155_Overcoming_Joint_Intractability_with_Lossless_Hierarchical_Speculative_Decoding/tables/bf8ca1fef627c81e6095fab5d0ae4062573f68dccefece22474ad0a80f61911f.jpg)

![tables](../iclr_results/155_Overcoming_Joint_Intractability_with_Lossless_Hierarchical_Speculative_Decoding/tables/d1c101b6447cc45b74bb06686a0ba83d1955ea5962adc8058ae57e91d03339b9.jpg)

![tables](../iclr_results/155_Overcoming_Joint_Intractability_with_Lossless_Hierarchical_Speculative_Decoding/tables/f521a938b5ebaa21a394b540bfafabbd557cd17b6f236cd65bd9edce82d08fec.jpg)

![tables](../iclr_results/155_Overcoming_Joint_Intractability_with_Lossless_Hierarchical_Speculative_Decoding/tables/fe7f1aa0ffe03c3c51225ebfbb199f502029673e065356e32e948332ac40c7ef.jpg)

<a id="paper-156"></a>
## Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=0jHyEKHDyx)

### Images

![images](../iclr_results/156_Why_Low-Precision_Transformer_Training_Fails_An_Analysis_on_Flash_Attention/images/0a6ccb149a23c9fd4459d7ba3c52e60552f44d3fb4fa6eb923190ae27cc30958.jpg)

![images](../iclr_results/156_Why_Low-Precision_Transformer_Training_Fails_An_Analysis_on_Flash_Attention/images/38fe39515320957ab7a13aeab179e3b2172c10170e33462ed705e433c919e001.jpg)

![images](../iclr_results/156_Why_Low-Precision_Transformer_Training_Fails_An_Analysis_on_Flash_Attention/images/3d535f16aefb3b90a1d43ee65cdabdc66438c3bdc66e5e4c7221963c8cacd2e6.jpg)

![images](../iclr_results/156_Why_Low-Precision_Transformer_Training_Fails_An_Analysis_on_Flash_Attention/images/3e549fd61aeee437beb5953dccc1c6a0d7f7f325c479b8b9876ac5ff948e3a32.jpg)

![images](../iclr_results/156_Why_Low-Precision_Transformer_Training_Fails_An_Analysis_on_Flash_Attention/images/413c99501f57ea5ccd405f49faae575e29880962e07a218337733572a659fa3f.jpg)

![images](../iclr_results/156_Why_Low-Precision_Transformer_Training_Fails_An_Analysis_on_Flash_Attention/images/5c90c7c76ecca8b07c145e88afacac3c296c71df15a95b9e25f5e7bd5038b50d.jpg)

![images](../iclr_results/156_Why_Low-Precision_Transformer_Training_Fails_An_Analysis_on_Flash_Attention/images/65ef9590e88211c183fda877023841272b17a82b742e422b72a37c699e4716a3.jpg)

![images](../iclr_results/156_Why_Low-Precision_Transformer_Training_Fails_An_Analysis_on_Flash_Attention/images/76c523e8f550676e5f4b8220b255bfa9bf837324a97760ec1cae142555b0f27c.jpg)

![images](../iclr_results/156_Why_Low-Precision_Transformer_Training_Fails_An_Analysis_on_Flash_Attention/images/809f02f6ad327e9e9c8c84c9ab27243483e850756b422bcf9103889ff72baa43.jpg)

![images](../iclr_results/156_Why_Low-Precision_Transformer_Training_Fails_An_Analysis_on_Flash_Attention/images/b01093351bf00de01b5935fd339dc88b073015bb8bf8f960cc925bac03469ada.jpg)

![images](../iclr_results/156_Why_Low-Precision_Transformer_Training_Fails_An_Analysis_on_Flash_Attention/images/b090493950cd1231a2bf2f850f0830ed20dcf95e4a41324f40a1c88910b6e3e5.jpg)

![images](../iclr_results/156_Why_Low-Precision_Transformer_Training_Fails_An_Analysis_on_Flash_Attention/images/d4c376c335f1d83c29c371ad7629c5690296c9d906ed686a3f471cd5c90e1057.jpg)

![images](../iclr_results/156_Why_Low-Precision_Transformer_Training_Fails_An_Analysis_on_Flash_Attention/images/efd6bf0057b70808de8e126dc716ac91fea7dccffb186b247cfa31ca85bc19e5.jpg)

![images](../iclr_results/156_Why_Low-Precision_Transformer_Training_Fails_An_Analysis_on_Flash_Attention/images/f5dd71e945930af7e0ebfb728aa2bc8de456a40d687419e92f5423e6be88500c.jpg)

### Tables

![tables](../iclr_results/156_Why_Low-Precision_Transformer_Training_Fails_An_Analysis_on_Flash_Attention/tables/2f168b4858c08cd8c16ee75adb3cb7a39ef070b7f704ba47350f6d3c863b2fae.jpg)

![tables](../iclr_results/156_Why_Low-Precision_Transformer_Training_Fails_An_Analysis_on_Flash_Attention/tables/fb7f48cee43c415c08e8727c5d9b8a92180479d622c19673b3186f33c026468a.jpg)

<a id="paper-157"></a>
## Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=oZJFY2BQt2)

### Images

![images](../iclr_results/157_Decentralized_Attention_Fails_Centralized_Signals_Rethinking_Transformers_for_Medical_Time_Series/images/09dd1fdbc4d9ddd3305171c7a5bced8fab139d27b00fd274f9bbd43caac849bf.jpg)

![images](../iclr_results/157_Decentralized_Attention_Fails_Centralized_Signals_Rethinking_Transformers_for_Medical_Time_Series/images/9451f6b7143429be5f5d9651cc9ec0964b1da1db0125b2f478c5cb6db3508844.jpg)

![images](../iclr_results/157_Decentralized_Attention_Fails_Centralized_Signals_Rethinking_Transformers_for_Medical_Time_Series/images/befd071cde4f8b3878b9cffd7f3d48a9d3ce03b519b9d6af38bb3e6543f3fb67.jpg)

![images](../iclr_results/157_Decentralized_Attention_Fails_Centralized_Signals_Rethinking_Transformers_for_Medical_Time_Series/images/c6666a28381a85ea52e4aea3d4f174c7635b24e21eeaa9b493ce503c60217dca.jpg)

![images](../iclr_results/157_Decentralized_Attention_Fails_Centralized_Signals_Rethinking_Transformers_for_Medical_Time_Series/images/d5d6c608b2a6dddc121eeef7edcf82114442030ebfa23304629ae7b78fc89c6f.jpg)

### Tables

![tables](../iclr_results/157_Decentralized_Attention_Fails_Centralized_Signals_Rethinking_Transformers_for_Medical_Time_Series/tables/176b534e821ed64c33655f5bbe6db01f4009c1c20f596e6a331f08a4edbc3ff9.jpg)

![tables](../iclr_results/157_Decentralized_Attention_Fails_Centralized_Signals_Rethinking_Transformers_for_Medical_Time_Series/tables/283c52271f58d557b34539eeefbfe83a52ccb46a1b82099fba2881637f7b9b22.jpg)

![tables](../iclr_results/157_Decentralized_Attention_Fails_Centralized_Signals_Rethinking_Transformers_for_Medical_Time_Series/tables/29884fc6642a607d210d8ed5823bd1e225e86c64b2837c5535ad789f045ff240.jpg)

![tables](../iclr_results/157_Decentralized_Attention_Fails_Centralized_Signals_Rethinking_Transformers_for_Medical_Time_Series/tables/2a63eefca4c7f63168476a6959d8f6e603151d9a1b813c493472338337990153.jpg)

![tables](../iclr_results/157_Decentralized_Attention_Fails_Centralized_Signals_Rethinking_Transformers_for_Medical_Time_Series/tables/387c28f34d9638bc8dcd48cb82cbf337034a2bb21ab37962a16dcc9cb1dc4f1a.jpg)

![tables](../iclr_results/157_Decentralized_Attention_Fails_Centralized_Signals_Rethinking_Transformers_for_Medical_Time_Series/tables/3de7b54b9305de6af30f68b5cad01a104c5ed98bcbd634421f17219404e8f135.jpg)

![tables](../iclr_results/157_Decentralized_Attention_Fails_Centralized_Signals_Rethinking_Transformers_for_Medical_Time_Series/tables/6c1c4faca6322795e1e42b52e90208ceeaffd355cc356637133418722c7c9bd1.jpg)

![tables](../iclr_results/157_Decentralized_Attention_Fails_Centralized_Signals_Rethinking_Transformers_for_Medical_Time_Series/tables/6e3c3786f8e00b6b25f3cabf000cf47e737f5a142c2afe04a4bbabd6c6007273.jpg)

![tables](../iclr_results/157_Decentralized_Attention_Fails_Centralized_Signals_Rethinking_Transformers_for_Medical_Time_Series/tables/afd76bc64b3d0f8dd05ec330e3c85d75dc6c27bf36cc9d153c81157c28907775.jpg)

![tables](../iclr_results/157_Decentralized_Attention_Fails_Centralized_Signals_Rethinking_Transformers_for_Medical_Time_Series/tables/ce88eaf292b3b67d9414f59cc379fe777d4379e8f678f235c573fd75238a64ae.jpg)

![tables](../iclr_results/157_Decentralized_Attention_Fails_Centralized_Signals_Rethinking_Transformers_for_Medical_Time_Series/tables/d0ba2ba72832602d2cd1931858ae905cae9b2f744d636166c7a91b46b37d4c8f.jpg)

![tables](../iclr_results/157_Decentralized_Attention_Fails_Centralized_Signals_Rethinking_Transformers_for_Medical_Time_Series/tables/f3edbb5e43605af71bb105afd6a27b9a01e6053414f9351ac9e48fafc76317ba.jpg)

<a id="paper-158"></a>
## Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=m6Hv0yZO3n)

### Images

![images](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/images/0a11e1c7d554fd862d0cf01ff210109a8c148c720d0ef5f22b6f901f641964f1.jpg)

![images](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/images/0bacea6fc9752255b1b3708b94e4a36c4e17a521d8223fd81ef80c5c511584dc.jpg)

![images](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/images/57963240a87f3c8bc6f3a94e0f891f49d1e0b2e944ba20a57c05f02ffdb91883.jpg)

![images](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/images/6d10882e74fef8517ece533a94a4d107305a8faf627b7a65836ee98af3590ec7.jpg)

![images](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/images/7845304e159c0df31680ae51a3ce3634baf8b88111b1f984d1e4132a794cfda0.jpg)

![images](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/images/85a3731474fdc1ac3b790bb90aeb763c2075ae74a2a1cacd7dbdd591c3229ee6.jpg)

![images](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/images/891d87007ac3da508459bb83d578e24999e806091659e1b2b2b165ad0dc2694a.jpg)

![images](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/images/8b170bbb1b63af3cdd8f90d89f669dc9313e69ddfdd5e14f49a07fcb8185b727.jpg)

![images](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/images/99961399f34c923a42c62f141a5974d348b1adbe951a23bc2ecfb268717858f8.jpg)

![images](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/images/c82d2745e833c624f5ba655f5248650b9a39ef955a2ec507871b0e7f929bd652.jpg)

### Tables

![tables](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/tables/3a1bdc3a90d488f5094e7980a4d9aeb40994a61378d930c6e21ccbfb1204dec8.jpg)

![tables](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/tables/5b77e5ab18cfb337cb86de1ff54ab6a908d800506ff0b3211167265b1729918a.jpg)

![tables](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/tables/64d1dfa889b86e4fa301a4c67ca60962e9245cbbafbc1671b1ffbee0dd26dd57.jpg)

![tables](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/tables/6ba5f3e8902a671538998be591cbb6e545ede5acf55313116addfba3a86b4eaf.jpg)

![tables](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/tables/8a41f7746de00c6e60aac02bf12f8204459014985427e1484193f3d809dc796b.jpg)

![tables](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/tables/8a54c912f150789902b504b255ca224e6311e2ebfb86ebe1a8ebdf1c30a3aa6e.jpg)

![tables](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/tables/c6836028487bc6506d88a67916d94333befcf43e7540262f652ed876b331530d.jpg)

![tables](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/tables/ce9b6f88e2fb11d698ad81302f20b04c8425000466fc28ccfe2efedc003083ae.jpg)

![tables](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/tables/d78965605d9e4396cfeb3af436e89665e1beda9e111b06217e50e93136178eaa.jpg)

![tables](../iclr_results/158_Discount_Model_Search_for_Quality_Diversity_Optimization_in_High-Dimensional_Measure_Spaces/tables/fb8fa2b4dd45be1a810462a0e5d1f2023eca278d90b50802382e9c05aeaa5c91.jpg)

<a id="paper-159"></a>
## On the Wasserstein Geodesic Principal Component Analysis of probability measures

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=OJupg4mDjS)

### Images

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/06284736d928d3e68e42ec21e666523e762d284b42aaa6f7c65a11fd7fc83347.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/08b5fe7f0844f2c2bfbaae30abaaaf1e2c2123c5fca4019e2875db0fc3de4122.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/19dfa9eda028a7da2a335ce12c7980abafa71a4536ece653365ddb109824f628.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/23a707cbcf655e24f0142b2316a09179df292c4389071202f71a7d5c8cdd6180.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/268a8421f85a56cef9880b6800fc1150cf1e4dad0302ec24aaecb9052af835b2.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/42417aca85ab9746f86888d4fa39e595be2d0684476709f355708cf3a4fb831c.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/4454c89e25be9ce5fe147776f3654c291bad9502b7a8e7f98429fb2cdd5a6c75.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/4a72212c6c7bc6091c78b0ac4eca7ae72b9a663e6a0167ab40e5783ef68095fa.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/54ed2c15ea16e9c25c9c4f26f56d4f4c5f38eac723b9df76b9275337f4ba3de5.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/5fea3eb98b7aa72060224854507dc306b449de5747adda32593687cebee91c77.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/66981271232400e39088fb151d013388769d641945d86e5edeb975344a919c61.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/6bad620b434488b6737d11fd5831ec853606428b320723a52e240fe045510b65.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/965e52a9a947f244f9ecdb51af1da4d32ea215ab959f2c262042b9a9a566aa3f.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/a6ab3f308ac333c124cdcbeba76fa2f4b55be4221bf9f77705bc32c35c4b760a.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/a87c09e567cdd4c54e40f801c77a24ce1c1c216e03aac2e266ac11845d6a9f1d.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/c4e7755f2ae692cf4ab49f092c6de45ad1908bbd21a87fa2cb04c40b5a6b2875.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/c8c8d9d907be9efdb5d92eb69e8faa24c83c6b00494e5a63fe7929cd0d3a39a5.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/d05b21a37af566de3fa54a6b5b8736405fe34558454cb05a19b221b174f0abce.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/d36ff689618fd7dd9aa2f7e6ce11bd367f3859e0ffc11b395177e9096a1ab0ae.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/e19407e7e8d317d42dcfda061aa6461fff77f05a6ca114d589114dd28ae74f80.jpg)

![images](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/images/fd2a4d514879308d9b17f09b83b62bee38a33b4018657d39491dcaa917aa6ac4.jpg)

### Tables

![tables](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/tables/9bb2bdaa3ca81a7d2cb8134a309bd975919d2b0482eeff3d1abfcc54eb323b50.jpg)

![tables](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/tables/a1c3d75f742aa2e1056c896745ca5d36a4218053348fda4946d59fdd24fcc0ac.jpg)

![tables](../iclr_results/159_On_the_Wasserstein_Geodesic_Principal_Component_Analysis_of_probability_measures/tables/b18b8cbf8d8988fb49a67d84419e932dc3512af219abc570d94d137b32af89ce.jpg)

<a id="paper-160"></a>
## AutoEP: LLMs-Driven Automation of Hyperparameter Evolution for Metaheuristic Algorithms

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=hit3hGBheP)

### Images

![images](../iclr_results/160_AutoEP_LLMs-Driven_Automation_of_Hyperparameter_Evolution_for_Metaheuristic_Algorithms/images/722ed3cd3a4f2679b3ce677854a71760b3f997fdc51990ba374f559d93a7c3ac.jpg)

![images](../iclr_results/160_AutoEP_LLMs-Driven_Automation_of_Hyperparameter_Evolution_for_Metaheuristic_Algorithms/images/7784c1e7a4bdaf2db9bac7c4b3c133653cbb7aa66bf1fff4cc1cddb84ba6e731.jpg)

![images](../iclr_results/160_AutoEP_LLMs-Driven_Automation_of_Hyperparameter_Evolution_for_Metaheuristic_Algorithms/images/82aedc5679f126c2243fcb4487e2a752c4c5bdc2720d026cfbc3b3be903d4c95.jpg)

![images](../iclr_results/160_AutoEP_LLMs-Driven_Automation_of_Hyperparameter_Evolution_for_Metaheuristic_Algorithms/images/8fc4bde81a89e43e660501fbd8b57eab4d3e902399f8535e4745d50ff6fb81c8.jpg)

![images](../iclr_results/160_AutoEP_LLMs-Driven_Automation_of_Hyperparameter_Evolution_for_Metaheuristic_Algorithms/images/d331c6bf8afde3aae5a5f01f55f7e4fb25c9e27764b6277be6eafcbf01465408.jpg)

![images](../iclr_results/160_AutoEP_LLMs-Driven_Automation_of_Hyperparameter_Evolution_for_Metaheuristic_Algorithms/images/ea94ef765aa067a04ee62df5347cec9afc21304d555d71499b62164425477c3a.jpg)

![images](../iclr_results/160_AutoEP_LLMs-Driven_Automation_of_Hyperparameter_Evolution_for_Metaheuristic_Algorithms/images/fd890e542cc031fdd919c5cd9ab223e3cd1649f1a2b38ef69c12758a6a81c370.jpg)

### Tables

![tables](../iclr_results/160_AutoEP_LLMs-Driven_Automation_of_Hyperparameter_Evolution_for_Metaheuristic_Algorithms/tables/140e37a7827c25ea1d8beb0b8578150f5e4a523a4ae1739447538e444b25779c.jpg)

![tables](../iclr_results/160_AutoEP_LLMs-Driven_Automation_of_Hyperparameter_Evolution_for_Metaheuristic_Algorithms/tables/2e4bfbeebf1b76f9d70eb324df5c750127f6eb22306bafef5840c9750dcce3e5.jpg)

![tables](../iclr_results/160_AutoEP_LLMs-Driven_Automation_of_Hyperparameter_Evolution_for_Metaheuristic_Algorithms/tables/b82e972308bfcb16f81c7fe4f2cc38f4f692e8b7124ef92899fce199dfb40282.jpg)

![tables](../iclr_results/160_AutoEP_LLMs-Driven_Automation_of_Hyperparameter_Evolution_for_Metaheuristic_Algorithms/tables/c174a34722e24d60eefb56a74281d428563feb27ca3913a5db5672bbaac3aa83.jpg)

![tables](../iclr_results/160_AutoEP_LLMs-Driven_Automation_of_Hyperparameter_Evolution_for_Metaheuristic_Algorithms/tables/c86e58cb1ed0af1b8b23f89dc62c7af0ca525b5cc3c456b4d060613251194013.jpg)

![tables](../iclr_results/160_AutoEP_LLMs-Driven_Automation_of_Hyperparameter_Evolution_for_Metaheuristic_Algorithms/tables/d3a64641f0cc68f9af53be68120bc7679b3a0666a666dc31fe23f9659978e032.jpg)

![tables](../iclr_results/160_AutoEP_LLMs-Driven_Automation_of_Hyperparameter_Evolution_for_Metaheuristic_Algorithms/tables/d71bc90686ce59203adaf42e2fcdbe8c588460aaa405738aec4ada345ae17803.jpg)

![tables](../iclr_results/160_AutoEP_LLMs-Driven_Automation_of_Hyperparameter_Evolution_for_Metaheuristic_Algorithms/tables/fa0ff2989c8238ed9659923f25c6636390decbf0d58f17157dc2c363faf00056.jpg)

<a id="paper-161"></a>
## Fast Escape, Slow Convergence: Learning Dynamics of Phase Retrieval under Power-Law Data

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=Ae4eZpkXBX)

### Images

![images](../iclr_results/161_Fast_Escape_Slow_Convergence_Learning_Dynamics_of_Phase_Retrieval_under_Power-Law_Data/images/059622e422368c8285f951cee0a93a7be31e99625a528d4ec459b3e173787c47.jpg)

![images](../iclr_results/161_Fast_Escape_Slow_Convergence_Learning_Dynamics_of_Phase_Retrieval_under_Power-Law_Data/images/7ad1fc9cbc3188ee964d996ae77d25d362e49477b1375ad1e4e737f78a7cece2.jpg)

![images](../iclr_results/161_Fast_Escape_Slow_Convergence_Learning_Dynamics_of_Phase_Retrieval_under_Power-Law_Data/images/aa6ebe00f956f70fb10119271234cdcdfbe1c437ac278bf9cba4d0b37afdff9b.jpg)

![images](../iclr_results/161_Fast_Escape_Slow_Convergence_Learning_Dynamics_of_Phase_Retrieval_under_Power-Law_Data/images/cbfb69379177874fdac3af5149e51fe00cff921bc873d562677e131e77e47bfc.jpg)

![images](../iclr_results/161_Fast_Escape_Slow_Convergence_Learning_Dynamics_of_Phase_Retrieval_under_Power-Law_Data/images/f5a9c829b9c793ea850743dba0603c05c53387b84c6bd5abee75bd20822e8ac0.jpg)

### Tables

![tables](../iclr_results/161_Fast_Escape_Slow_Convergence_Learning_Dynamics_of_Phase_Retrieval_under_Power-Law_Data/tables/c93b2821065d7854b7c262004b4cd81e4b524b96673d6264e5f14bf6ae06618a.jpg)

<a id="paper-162"></a>
## Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments

**Type:** Oral | [OpenReview](https://openreview.net/forum?id=9gw03JpKK4)

### Images

![images](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/images/10cf8a7953d5dd67e1311142861a9e279a2dad30ba4e303b8b97c749c729d432.jpg)

![images](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/images/13d1ad232c94cbf257fa305ba63d1fbbe83a51077e8b479bee93b82877d9ca1e.jpg)

![images](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/images/3011f3e5b632d3c76701f7906aca98eedcd647456d2db1995fa1d774dafd1101.jpg)

![images](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/images/3ad7eee795081d62ff61ab1c3a19de96e35a8c106d58c50b18f0ae1cccec270f.jpg)

![images](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/images/4258cf8c6958db0e841d48ec79c53ef8076568b061c7057f650da2aaf2acd876.jpg)

![images](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/images/5f5c2e82497d81ad780b2d0785d2aecd17e185129d6d0d66132be1563819d75a.jpg)

![images](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/images/8610e8a7938c8fbc66bf6881dcf5cf54a0afc46bd0e2fd03928e54b4e69bb2f1.jpg)

![images](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/images/8ad35fca5b3f18e499b38679e5f98bde3fd0a2af9b1f70b806b47de179e0e6b0.jpg)

![images](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/images/9b7f8a20b516c3b7b28986b1c908d012657e69092dcf1292640875bc9f0f9f2a.jpg)

![images](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/images/a591d0f8dbbb96618cd4b01a48600b54571cbbe51d026c8b54402303f54874a4.jpg)

![images](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/images/a993320898035460c858e07369c4c5ebcaa085c8b936db93725f713871170288.jpg)

![images](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/images/ae3f7dc868b7b76bbcdccab43188db79c0f25196cb1495b8367015c859d043f0.jpg)

![images](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/images/ae722d1d07705978799e1de3a2591e57bd4b6a9cb17c0193904a2201efcc675d.jpg)

![images](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/images/b67bfa95632e18976e003a3b959a882b42b067895d598b21fd897242178efe52.jpg)

![images](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/images/c216b0113485d44daaacae088c5f68155a256de86395faa3f5b1cd153c375043.jpg)

![images](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/images/e88fe7bf46b42e86e91e56c0dc3051b00ea52dbc2d5183cb176b859d249d869e.jpg)

![images](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/images/f79efce68d103a3f33ff9575e1306fb5dcb0058de3bcbbfced28954dd910b6c2.jpg)

![images](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/images/fcc853cf66f34ba56e8cc6a5c83bcc1d688ba5690919eddb267d06df2fd7232c.jpg)

![images](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/images/ff5c485f5e512626e28aad384acf0649867b3cd56999d3edfbabcca74206fdc5.jpg)

### Tables

![tables](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/tables/2ceaa8ddbdd74b7d3690c056ce9cfc979d7171121cba0c87736ac6bd2d0506af.jpg)

![tables](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/tables/6319269349d2b4a37ff9c5d514778c7fd4093282f5938b2703b2dba173bc3829.jpg)

![tables](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/tables/8102b531116f9c4fd876da613410943202ed7b6b85775fd8742186d4ce6f8971.jpg)

![tables](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/tables/9a9de8bebd564fe182ec9fe2318f48ffb52fdc3839ad8be7520081fd913b1a11.jpg)

![tables](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/tables/c9c59c2ec19634c2cd467d4e372891b8f805caeca7fbaf7369b45418d7e0a2c6.jpg)

![tables](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/tables/d814cb360ae470a6b518cff391b2f6073b99ca0a69369be21f2457d1da6d68b5.jpg)

![tables](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/tables/eb7247beca175be039011354607eb84982f9092653d8ebf585be9a5927e67bad.jpg)

![tables](../iclr_results/162_Gaia2_Benchmarking_LLM_Agents_on_Dynamic_and_Asynchronous_Environments/tables/fd996b0b6b5fe91d5d44be5c6a66e03c75c5826bc2fd2ca6587936974dce6ac5.jpg)
