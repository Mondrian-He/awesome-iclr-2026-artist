# ICLR 2026 Main Papers — Part 178 of 199

27 papers.

- [Occupancy Reward Shaping: Improving Credit Assignment for Offline Goal-Conditioned Reinforcement Learning](#paper-4784)
- [A Physics-Inspired Optimizer: Velocity Regularized Adam](#paper-4785)
- [Better Learning-Augmented Spanning Tree Algorithms via Metric Forest Completion](#paper-4786)
- [Jailbreaking the Matrix: Nullspace Steering for Controlled Model Subversion](#paper-4787)
- [Antislop: A Comprehensive Framework for Identifying and Eliminating Repetitive Patterns in Language Models](#paper-4788)
- [Sharp asymptotic theory for Q-learning with \texttt{LD2Z} learning rate and its generalization](#paper-4789)
- [Selective Rotary Position Embedding](#paper-4790)
- [Canonical Tree Cover Neural Networks for Expressive and Invariant Graph Learning](#paper-4791)
- [A Scalable Constant-Factor Approximation Algorithm for $W_p$ Optimal Transport](#paper-4792)
- [When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis](#paper-4793)
- [ProofBridge: Auto-Formalization of Natural Language Proofs in Lean via Joint Embeddings](#paper-4794)
- [In-Context Compositional Q-Learning for Offline Reinforcement Learning](#paper-4795)
- [Improving Classifier-Free Guidance in Masked Diffusion: Low-Dim Theoretical Insights with High-Dim Impact](#paper-4796)
- [Flowing Through States: Neural ODE Regularization for Reinforcement Learning](#paper-4797)
- [Predictive Differential Training Guided by Training Dynamics](#paper-4798)
- [Improving Reasoning for Diffusion Language Models via Group Diffusion Policy Optimization](#paper-4799)
- [Octax: Accelerated CHIP-8 Arcade Environments for Reinforcement Learning in JAX](#paper-4800)
- [OpenEstimate: Evaluating LLMs on Reasoning Under Uncertainty with Real-World Data](#paper-4801)
- [STAT: Skill-Targeted Adaptive Training](#paper-4802)
- [Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains](#paper-4803)
- [Feature segregation by signed weights in artificial vision systems and biological models](#paper-4804)
- [SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models](#paper-4805)
- [The Lattice Geometry of Neural Network Quantization: A Short Equivalence Proof of GPTQ and Babai's Algorithm](#paper-4806)
- [Zebra-CoT: A Dataset for Interleaved Vision-Language Reasoning](#paper-4807)
- [Improving Code Localization with Repository Memory](#paper-4808)
- [The Limits of Inference Scaling Through Resampling](#paper-4809)
- [Don't Shift the Trigger: Robust Gradient Ascent for Backdoor Unlearning](#paper-4810)

---

<a id="paper-4784"></a>
## Occupancy Reward Shaping: Improving Credit Assignment for Offline Goal-Conditioned Reinforcement Learning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=EW8DskWQ1K)

### Images

![images](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/images/072ccf1a0e6acebaff959ce664fe94699586eaf880f18ec72115337a7d8acd60.jpg)

![images](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/images/1125a7445d7069323ecb44c75da7616b6bb91071859ab75851669c9ff1d10c69.jpg)

![images](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/images/1304abc6de2ca2a5ac637781c2d0e29d28545602586167693085b03c0163235a.jpg)

![images](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/images/19cc66c82ffdf345ff8ed570a4e0c93341deaf954a2c5da2acab821937a51021.jpg)

![images](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/images/33d709ec5a09721a3f9c7af2b9449a36d875e6e7ceeb6577799184eab437e3cb.jpg)

![images](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/images/6220cb6f879a3e8cac69fc00c038fe7ac45c3ebd535fab5009eb2f27518dd870.jpg)

![images](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/images/ac1ece1473f7724cc99d6774ec0ff6766e5c33092d1a8b97269af8fe781df1e3.jpg)

![images](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/images/cb7fe70d3f47c624e027d165f43bf9cb44189ff1b14dc5a447fcff00700cbdf3.jpg)

![images](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/images/cc3d3016254c70cd1e41aa14a9fae8a1c730e7acfaa1b7ac22ff8f3ffc0f5a63.jpg)

### Tables

![tables](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/tables/27062a47d2dbfc064b3defb7f773b8fb2f8b1c3cab3a768e461e7e1955f0733d.jpg)

![tables](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/tables/3a91be48d3820fe0a58cf70f4f331abc1ee580abf3d6d9355901de8c7b4d1e45.jpg)

![tables](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/tables/4e880823930b4d139b9234c2eb03c3c952730c4519ae958930f8cdf6ea543101.jpg)

![tables](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/tables/7c795e38dfc63005a4a428eac4408d9335bc84c125176f331a9bffe7d7489519.jpg)

![tables](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/tables/828eadf1de3f641edd442db8b123a72a2f00358af212f8bc27f393b6434e344e.jpg)

![tables](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/tables/8ee42d032edf9b27d7c43779c76a3ccf80cb9b4c5cad9a5f72e0d057c28fd46a.jpg)

![tables](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/tables/c76c408f4362e706a7106d6657b29f2bdb2419f14b06ff883d92a4ea7e3d91ec.jpg)

![tables](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/tables/c8831c4ade3aedec1421cdf6b6023d7fc43866b20c0a90afb67053975e01eb64.jpg)

![tables](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/tables/c96897cb0b8cc9e4a903803eab01e38a5421e55fc23ea8cdcbd476a079e9237f.jpg)

![tables](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/tables/ce57ffbd5fadb09ea02683fe6d140030c41885e243db95dfb89899cf3505350a.jpg)

![tables](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/tables/ce9ed44983eb4be08c927a15f134fb5ed92b8fef1bd7718f0f10f0b7339e8961.jpg)

![tables](../iclr_results/4784_Occupancy_Reward_Shaping_Improving_Credit_Assignment_for_Offline_Goal-Conditioned_Reinforcement_Lear/tables/eeaecab154d35295529013c3252322a6819110821fdf145ae6c7f77b12a4a79f.jpg)

<a id="paper-4785"></a>
## A Physics-Inspired Optimizer: Velocity Regularized Adam

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=6BhduwrCp3)

### Images

![images](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/images/06427e55673331ee26e9dcb3a2a660dfbeb9e028ff5b59f9de2d1d8a21853324.jpg)

![images](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/images/2c4395427786809d207532145a63f398a8c8f91b966ed498d75adb675e4240d0.jpg)

![images](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/images/426bb3ce0b43896b6d88cb51423d529d311925edc8406d79526e29aa9b103100.jpg)

![images](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/images/4f402b9e465fd202fd6fee2e6326494abdc01209bde2cb556372a41511ed30f1.jpg)

![images](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/images/532074fb43c1e21242c8fd25c876d4f15aa12b6d4bc61cd23bf81948703c3266.jpg)

![images](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/images/7b465ef04ec2b34c3a603fcfac3749795fe55a9b4389ea019fd8f4bcbc4645a1.jpg)

![images](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/images/9a2a585040d792a318cfbfc54e37a9e5bb4f52149ee48da9308f80c9312f380d.jpg)

![images](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/images/ab306cc250214b7db6c526addaf648d64adb3d978b418a2d2f812678e542a012.jpg)

![images](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/images/b17b3cfeea60b19f8dcbc1ab7df079fd101b12216ff5c3a9cc14b616294776c6.jpg)

![images](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/images/c9c37d42d34e85b92bdaa039e725da975e92b1d2d5ec1931d0e0f31b9bf978f3.jpg)

![images](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/images/e04eab5a763951388b5fb9820e71b8fa741c125cf6d536cbf928d08cdeedc1d9.jpg)

![images](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/images/ec1b8ec6884ecd0f1d7b7ff7a75bbf84ae0680f92af9450c95c2c54011c01996.jpg)

### Tables

![tables](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/tables/13d43f75d4ff37c62f0495a62ffe2f21d0adf745a9ea2907355f2d66aecc5734.jpg)

![tables](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/tables/1727073e6e2c6f8a9baceae02f23f0eaaaaf8e50898d631c77c843ab2a820b57.jpg)

![tables](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/tables/2b1d44356fb38e26cabd7d62410c790bb9a499149fa1dc353aa5476bfee57b3b.jpg)

![tables](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/tables/2b7de6eb342872042995a66a7a7448d8e13828059043177e5d4976bd4f8297f4.jpg)

![tables](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/tables/2d1a68f62e4958b26c810f99b71796b097d099a3e3affb07da190e5ae28e2127.jpg)

![tables](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/tables/49a3d96506a62035a61669aa39f8fae6b1672365a2e1445e4477c5783b71d79e.jpg)

![tables](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/tables/50ff031e8ed01b69550b2adc271db25dc674201fb3de87f21f0204654510a518.jpg)

![tables](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/tables/57e1706c57275c5d08d468cee244071ef961fa3107f1e7d87c1b7c91601e8c38.jpg)

![tables](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/tables/b1bfd3c1f1e8f6e26cac814f6187dcbcafa62ed824aa1eb3ff3dd802c0fbcfbf.jpg)

![tables](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/tables/d1d181b6765f66360cf5cd4d6deacb6965e74786c09b434662dadc90be43ff5b.jpg)

![tables](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/tables/d406881c11b009d1a24febef519877686ad517c81092fc46fe34c4069daa023e.jpg)

![tables](../iclr_results/4785_A_Physics-Inspired_Optimizer_Velocity_Regularized_Adam/tables/ec0a824fdd896b6471c801bcc8ddd05efc4a53b18b5df79ace2d314e21d93221.jpg)

<a id="paper-4786"></a>
## Better Learning-Augmented Spanning Tree Algorithms via Metric Forest Completion

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=TWmS4o41oA)

### Images

![images](../iclr_results/4786_Better_Learning-Augmented_Spanning_Tree_Algorithms_via_Metric_Forest_Completion/images/7ac7284ec217dbb9d09ac7c86e560b307fe9b592fdd4a6d66088edb4f7b7b3c3.jpg)

![images](../iclr_results/4786_Better_Learning-Augmented_Spanning_Tree_Algorithms_via_Metric_Forest_Completion/images/b0cc3975bda7bbe6f2fe630acb921f33a8d136c9fe311eb4a93eed4f7096cc61.jpg)

![images](../iclr_results/4786_Better_Learning-Augmented_Spanning_Tree_Algorithms_via_Metric_Forest_Completion/images/c471a71a781ce010fe0928437f24aa35d5f008f6546c175b6911548b66331724.jpg)

![images](../iclr_results/4786_Better_Learning-Augmented_Spanning_Tree_Algorithms_via_Metric_Forest_Completion/images/cee0b44625f2aeba9a02f7df0bc691b7bf00ede536908830b034609431b982c1.jpg)

![images](../iclr_results/4786_Better_Learning-Augmented_Spanning_Tree_Algorithms_via_Metric_Forest_Completion/images/f4f498bdd1a5f79acb9e3c810d3fb235af59a8be94ffd2c7e8b792ad34391944.jpg)

### Tables

![tables](../iclr_results/4786_Better_Learning-Augmented_Spanning_Tree_Algorithms_via_Metric_Forest_Completion/tables/061e1c5d9b4f9666b44cecb7e567fca5303b15729e7a25231263ec1faf62497d.jpg)

<a id="paper-4787"></a>
## Jailbreaking the Matrix: Nullspace Steering for Controlled Model Subversion

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=qlf6y1A4Zu)

### Images

![images](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/images/6b225ad9922cc77e991b1bcc62bbbcf7bbb1a44ab289b5be140975b62d5ec0c7.jpg)

![images](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/images/88bc980be5b7e0a9d327fc45dde04e0cb3f638b6962ae1fb696a49dadaac78a8.jpg)

![images](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/images/bd6b92d643575eb9c46ab5476a4649ce0ac3b1c3fd39068000c5eaca0f53a1f7.jpg)

![images](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/images/e5529b170a9b312a12ea36b396924f128bf328c0c78b0ecc0dd7a0b0c69b6d8f.jpg)

![images](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/images/f66fe98c8e5a54fa4a5c52c037494d11b8df8e7e48ba22938bcd7fd4fb63061c.jpg)

### Tables

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/0302d3e3b9e8bfeb9ca4dba72c2800ec4be7d1a76ccfd125414969cdf1301d88.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/06598386e8612549ae0e68083d324f8264d4b210148a178e3988b9088f2a3f84.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/0c2cd76b8808109ac2ca7823f1698a60ee4b888041bfd3977b7867deed1baaee.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/0f10314c20e669a47f78df2d69e59c1df7db4b9e1fca56fc0df402882f28fc48.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/321714635cd4ac94d7b588f2718a28c22eaf2d24ae32225b708ab46e3b237d1a.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/3e65035a683380dbca4fdc8b22e6dc7ab8c754f7faf00a1e7b7aa7f35c43025a.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/4bd49a5eafd9d9ecef4888e43d0141fbd46d232b1068014b2af2855ce35a214a.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/4e5b101e80f03f6e727f83868b8d53d7388f5934ed8187771800e9f4f9a63d3e.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/511a4f93cde415aae457bdf0d88b681cc1a40f213ed052cd44cea4ea7ab43f43.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/700e05c678557b95860108897dfe3dc784bd4fae29771a6daa22dfc368eb009c.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/89a01531032689284e91bb6d28e68562ac17bc7050a450997f2fbea61dccccf4.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/8c91678280847811bd0da6d160e5a3a0d8acb671b100da39c496f50b67f5247a.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/9db3c7cdece4b372ddff365451e685dd3d3a6436f742d166f56bf55161ef6c58.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/a707e97f4d9d05a1970963d593e37b05f992a98bee500923a5b712332e0e3bf6.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/acbb90deb69683f70e8d560d56f3503fb67a7c3c354736f693128c50f468cbc7.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/b082b68e5e1e7b8d3f31b2d67f89f14c0420cfc04ccab01544cddac9f5436149.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/b3abe6c83940bdbd9566c50c78eaf776c9765cfe604dcb8beb3add12ec682a1c.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/b9814255dbd2774aeb9f82384889155dbe341e07fe62844a7caa28f145c84995.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/bca4addfa8d286449cbf7a92dc45a3795f9bc1e1a85b4a06f62068893543cd0d.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/c6dfa99f047099837b496b83d39218e7f3785bb06e13c20ae4e5bb5f24000202.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/cf2e46f0c26a61f7e8e14c019edc881b25a864de0db29e35500d0f86bb785161.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/e36ed64a13bb7bf7281da0c09cf6df7c16937eb40c3d564608dda7d65d2cafd3.jpg)

![tables](../iclr_results/4787_Jailbreaking_the_Matrix_Nullspace_Steering_for_Controlled_Model_Subversion/tables/e9ca059f4d5d362398435e8509343f4b73bee2e76dc3ff2db341cc8d5bf54893.jpg)

<a id="paper-4788"></a>
## Antislop: A Comprehensive Framework for Identifying and Eliminating Repetitive Patterns in Language Models

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=gLcyM1khyp)

### Images

![images](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/images/08d729703f663b4b6e81e4cb9c6cff0c4d227977fa19fba44b55293e84017600.jpg)

![images](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/images/1216a3448c796cf3eb2ad16ea06b6da02d75e5b0f3a264c99fbd61d0c7a1e624.jpg)

![images](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/images/126a970d1bb2b9d99bc5bc70f375cd09ac3957ffb872c4f9440fb8346a402f5f.jpg)

![images](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/images/15961cbdfb20546c253b5925f02624f59e3035f633879e6c3bbca0f01ff03ee1.jpg)

![images](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/images/17475d2b4912afd629435ba10b83501160b803f1f5acada1bba8d7590b9c6109.jpg)

![images](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/images/1f25e06604f8ebae6803f8d5781e4dd18939292e31ad003a549c6803a2391a2f.jpg)

![images](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/images/40ea2d1828fb003314f1916020ebc8160ed2b8cb971e61f9aa3ff6de50b98f48.jpg)

![images](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/images/5121844251afbbd472d294edf8d72793da4965f2978fa795589f0a7fc927afc7.jpg)

![images](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/images/5546b0b01a117cc8c95ff9fd30ad4c0188378ea87c8a6a0ae6f2442256685341.jpg)

![images](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/images/62fd46f6273483dce9854d7266ef524509fd3110d61f79b015564452d0335b45.jpg)

![images](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/images/7e230f88572b08c65b0f349e989ea3d530be9309168cb145a471ec74418f0fa9.jpg)

![images](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/images/b2b2bc20e48b2cebcd9146a269996746d9bb074435a27f1ba78a21f78880fcee.jpg)

![images](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/images/c4db62edcd4115aad18e3afe0c35d9fa699c0d38f129075167ecb673c3ade0ca.jpg)

![images](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/images/dd98e34108d6ff11125eb2b7a4232ffb1e94d2564dd80c6e7f193908945201e6.jpg)

![images](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/images/efc074da0a5f9339d6b91e13b697a492c0ba38e814e12843fbda5ad515245c42.jpg)

### Tables

![tables](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/tables/07c9f6e6a3a02aada1181ffc5b05beafed7781fb88d2872e77421d3ac800b826.jpg)

![tables](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/tables/1905464c0c54f278c29211f2fc8692cb0cffb1b8d061760a47f2f81ef5548a99.jpg)

![tables](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/tables/297581498fccffab864c2d3af46e1de226ddcf98d04cc8899517a9e90e7cf24c.jpg)

![tables](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/tables/46c00dd1c8d53a817a207ac357e79dfcb251d260413626b360d814b33f166736.jpg)

![tables](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/tables/60afe80effbd9b6e25f2d0daaf8e5e22ff93e57fb07da1262c847ce4424292b3.jpg)

![tables](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/tables/868e88cb0f0e4592f2b83c7c72e2d4bf1dc4cd370c8d410932f9597fc7f6818d.jpg)

![tables](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/tables/973fbf7762d361d6551c5dd8c3cb24fb124943cf797afa8a68c55c1e032ccd47.jpg)

![tables](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/tables/bc34bfbbc415244b88e94d95e599ea9f816b9410729f4f0a75ec489e5f5272ca.jpg)

![tables](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/tables/e560c0eab95fbfdff3bd868fd550ffb392ff541fdb4e59d667bb2564dbb646bf.jpg)

![tables](../iclr_results/4788_Antislop_A_Comprehensive_Framework_for_Identifying_and_Eliminating_Repetitive_Patterns_in_Language_M/tables/eb9d099efde807d54c403680cd4a72dd9da5ca3e78ad41ceaf9885d8827792d3.jpg)

<a id="paper-4789"></a>
## Sharp asymptotic theory for Q-learning with \texttt{LD2Z} learning rate and its generalization

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=WjEAMyLDoh)

### Images

![images](../iclr_results/4789_Sharp_asymptotic_theory_for_Q-learning_with_texttt_LD2Z_learning_rate_and_its_generalization/images/0edb31be778eedac6b37ba8f98196eb6017c448e01b1a8fe20cd94bb648ce406.jpg)

![images](../iclr_results/4789_Sharp_asymptotic_theory_for_Q-learning_with_texttt_LD2Z_learning_rate_and_its_generalization/images/69d8021adb02ced563eb2f2819d9a369030fbcc61609ab2dd0a9e41758b5da75.jpg)

![images](../iclr_results/4789_Sharp_asymptotic_theory_for_Q-learning_with_texttt_LD2Z_learning_rate_and_its_generalization/images/9d4d7b38e26ee3c8a7ab1df1bedf2ae6c94a15815f85a12655e27e65dd191643.jpg)

![images](../iclr_results/4789_Sharp_asymptotic_theory_for_Q-learning_with_texttt_LD2Z_learning_rate_and_its_generalization/images/c6b5ff10f0e366ed0951d4511cfa600c0dfce48e0ae9b1d57cb11a731d334564.jpg)

![images](../iclr_results/4789_Sharp_asymptotic_theory_for_Q-learning_with_texttt_LD2Z_learning_rate_and_its_generalization/images/ca4d178343dd0e99c2764ced60774df56668906f7f05a4c26eb49dbed86501ab.jpg)

![images](../iclr_results/4789_Sharp_asymptotic_theory_for_Q-learning_with_texttt_LD2Z_learning_rate_and_its_generalization/images/ccb4342920782df534221d75f49a22c195e2565efed498a4ca849c14830582e7.jpg)

![images](../iclr_results/4789_Sharp_asymptotic_theory_for_Q-learning_with_texttt_LD2Z_learning_rate_and_its_generalization/images/d1b9b9f338b8963d3d82a6a35890983ea39a79f7cd55a0dca11f0a6f5ef50a3b.jpg)

![images](../iclr_results/4789_Sharp_asymptotic_theory_for_Q-learning_with_texttt_LD2Z_learning_rate_and_its_generalization/images/d4da4ebaf63ad1bcfbdcd24d010636de60c1981adb6643bc8cb2f0b1bb1f0c84.jpg)

![images](../iclr_results/4789_Sharp_asymptotic_theory_for_Q-learning_with_texttt_LD2Z_learning_rate_and_its_generalization/images/f8dadbe80289e98935ac99846fe44ecfd4ad4802beb650b0b6c7f45d8f9a3db1.jpg)

<a id="paper-4790"></a>
## Selective Rotary Position Embedding

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=AQo1SEElNb)

### Images

![images](../iclr_results/4790_Selective_Rotary_Position_Embedding/images/03280b4ba3e3cd80ff891210090fdf9e0119d082bd8738a92dfe61e263b5b5e3.jpg)

![images](../iclr_results/4790_Selective_Rotary_Position_Embedding/images/2b66e4c0e6b54c211521e680b1bd9cd1508cce3672282ce49bd17efcfe6c8398.jpg)

![images](../iclr_results/4790_Selective_Rotary_Position_Embedding/images/459bcdd9e020e45667556c2b3a8de7fca4e8d8c7cf8943f22356d9325587aa15.jpg)

![images](../iclr_results/4790_Selective_Rotary_Position_Embedding/images/8685f8d86f8f9280279c443046b36d899f43dac66045d8c3a925d3e1e820133c.jpg)

![images](../iclr_results/4790_Selective_Rotary_Position_Embedding/images/9429497be9e16260415f064de2e833862e0a5b3c7f12848f237d635746937eed.jpg)

![images](../iclr_results/4790_Selective_Rotary_Position_Embedding/images/c3c75bb5f36c5829ba38191e0486da5c138a9aa5b6f4ff3afde2360fa773e136.jpg)

![images](../iclr_results/4790_Selective_Rotary_Position_Embedding/images/df541787d15a1c1ea4c8e1b603880acc0a5385eabcb06e6191d26022b9b0cba2.jpg)

### Tables

![tables](../iclr_results/4790_Selective_Rotary_Position_Embedding/tables/06d627388f3a1c54b7c08268d45c8c43c56307f3f5a85c9e674ab5dd871a2a69.jpg)

![tables](../iclr_results/4790_Selective_Rotary_Position_Embedding/tables/1920d27b924ea212bde76942692e1aa96eb8dede6a4f5106858a2d35800a8930.jpg)

![tables](../iclr_results/4790_Selective_Rotary_Position_Embedding/tables/5e34a384432a059bc7f3a2526ce4c9fdb9856860af7bdbddf83b031d7ea911f2.jpg)

![tables](../iclr_results/4790_Selective_Rotary_Position_Embedding/tables/8911d86d0b4f1d0cd244aa9f471a2d38028625f803cbbb1a1103f273c6202dae.jpg)

![tables](../iclr_results/4790_Selective_Rotary_Position_Embedding/tables/93b590f0ff29f97f7a20e87487e4d34207fa9347eecb47b419e3d99f4e27b7e1.jpg)

![tables](../iclr_results/4790_Selective_Rotary_Position_Embedding/tables/9e7e8d66692b7061a7f18f65b62443cb41c2da931a8cedcce327cef6d19da63a.jpg)

![tables](../iclr_results/4790_Selective_Rotary_Position_Embedding/tables/f0489828560d1758ff3fedc8c4a642d16017b62e1f44609ed2dabe001e8052e6.jpg)

<a id="paper-4791"></a>
## Canonical Tree Cover Neural Networks for Expressive and Invariant Graph Learning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=yumDmlGCc9)

### Images

![images](../iclr_results/4791_Canonical_Tree_Cover_Neural_Networks_for_Expressive_and_Invariant_Graph_Learning/images/2ee9f436a321d58a494293cb7869b1420cd811fa10fde117074bd26b99e460f7.jpg)

![images](../iclr_results/4791_Canonical_Tree_Cover_Neural_Networks_for_Expressive_and_Invariant_Graph_Learning/images/5e484dd3a5a80bd51ce7ad010725e16c8e4838f9c0b6db80879805af9fea31b4.jpg)

![images](../iclr_results/4791_Canonical_Tree_Cover_Neural_Networks_for_Expressive_and_Invariant_Graph_Learning/images/a46efacb3804237c3d4b47bfb3b70302ecd28d6fba2e7762f7ca22cfe11a3d05.jpg)

![images](../iclr_results/4791_Canonical_Tree_Cover_Neural_Networks_for_Expressive_and_Invariant_Graph_Learning/images/b52c36e202d7f75493dc604c2bfa35571e0dd2f4febcbd35c2ddbe2df723d44d.jpg)

![images](../iclr_results/4791_Canonical_Tree_Cover_Neural_Networks_for_Expressive_and_Invariant_Graph_Learning/images/cc1eb350a3f564bd7808ecd05748d9298c57a7b0feaf1a1fdf281b83961c2f18.jpg)

![images](../iclr_results/4791_Canonical_Tree_Cover_Neural_Networks_for_Expressive_and_Invariant_Graph_Learning/images/d581ae29f37576a910d126802567cbbe6fee2a95e26bfd24f1f21d3bfc7a6b4b.jpg)

![images](../iclr_results/4791_Canonical_Tree_Cover_Neural_Networks_for_Expressive_and_Invariant_Graph_Learning/images/edbdb7099c41c33c3169941c96957ee1915fa8eedcd8759cf7e9f8c2d177c5da.jpg)

### Tables

![tables](../iclr_results/4791_Canonical_Tree_Cover_Neural_Networks_for_Expressive_and_Invariant_Graph_Learning/tables/518f2ca00cb7bf99ad0bdbf2c3820c933d249f4b9892916caab1fc71e96e9fcf.jpg)

![tables](../iclr_results/4791_Canonical_Tree_Cover_Neural_Networks_for_Expressive_and_Invariant_Graph_Learning/tables/64a354ae4466c1438441b6e24e99a2db1aaedc2d2d330544ef38f41beecb04af.jpg)

![tables](../iclr_results/4791_Canonical_Tree_Cover_Neural_Networks_for_Expressive_and_Invariant_Graph_Learning/tables/660a07274545d7851d8083efbed21cb3748f2cc5bdc1ebe7fc8c5299307f295e.jpg)

![tables](../iclr_results/4791_Canonical_Tree_Cover_Neural_Networks_for_Expressive_and_Invariant_Graph_Learning/tables/6ff396e6b7e187049df04463a292a1511c2d1634c7108b147e56f69d32cbd7f8.jpg)

![tables](../iclr_results/4791_Canonical_Tree_Cover_Neural_Networks_for_Expressive_and_Invariant_Graph_Learning/tables/762a3744fa6aa23e596a93fe1bef469189d8a4f58102db554bc23bcc0c19e8c2.jpg)

![tables](../iclr_results/4791_Canonical_Tree_Cover_Neural_Networks_for_Expressive_and_Invariant_Graph_Learning/tables/7cdabe0f00712b9c400eeede9ea45e56edc0d95eee93ce76b0b937c455e521de.jpg)

![tables](../iclr_results/4791_Canonical_Tree_Cover_Neural_Networks_for_Expressive_and_Invariant_Graph_Learning/tables/d10d7bcd47c61e254d5ae656ab2390aca95c1e0aefa8d7e10a351944adad7745.jpg)

![tables](../iclr_results/4791_Canonical_Tree_Cover_Neural_Networks_for_Expressive_and_Invariant_Graph_Learning/tables/db2d4b12f4bd93e91a3282aeb3113de23785f060ee83b79f84f7b8e65b8ee7fe.jpg)

![tables](../iclr_results/4791_Canonical_Tree_Cover_Neural_Networks_for_Expressive_and_Invariant_Graph_Learning/tables/dbafe3aa59c9472ba636af51765ad723243b527439aaeb676a223bcb45128b18.jpg)

<a id="paper-4792"></a>
## A Scalable Constant-Factor Approximation Algorithm for $W_p$ Optimal Transport

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=RPQKJxrEPs)

### Images

![images](../iclr_results/4792_A_Scalable_Constant-Factor_Approximation_Algorithm_for_W_p_Optimal_Transport/images/0eb3ed16137199611c038ba737ba6c377d19910bb06848a8d1d82f1756ce1741.jpg)

![images](../iclr_results/4792_A_Scalable_Constant-Factor_Approximation_Algorithm_for_W_p_Optimal_Transport/images/587b38caedec0419c792118ced36ab06ceae6b819e57d48e4e378a23b80b51ff.jpg)

<a id="paper-4793"></a>
## When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=34V0IZytle)

### Images

![images](../iclr_results/4793_When_Scores_Learn_Geometry_Rate_Separations_under_the_Manifold_Hypothesis/images/23940daff9461ee32ecd6fa3987ef1d52b23071f892655a4f2b4556f437a746c.jpg)

![images](../iclr_results/4793_When_Scores_Learn_Geometry_Rate_Separations_under_the_Manifold_Hypothesis/images/46edb9bdb0b6eacae286c82d46e77580ad5c5f55aa6c370c401a3c2d18f53289.jpg)

![images](../iclr_results/4793_When_Scores_Learn_Geometry_Rate_Separations_under_the_Manifold_Hypothesis/images/9a95ce46e0e9fbb13b759e41be0c2b6d0ffbbdb1977ce0f96facb87ece528ece.jpg)

![images](../iclr_results/4793_When_Scores_Learn_Geometry_Rate_Separations_under_the_Manifold_Hypothesis/images/ad389733cc954eef9a244667fb40d691ee05e2f8f84b4be0eb4f0ba6298f3014.jpg)

### Tables

![tables](../iclr_results/4793_When_Scores_Learn_Geometry_Rate_Separations_under_the_Manifold_Hypothesis/tables/2e21fe03e3cfe358b36c5f1891fd3f4b10655c3e48ee3e25a73086f50ba7b590.jpg)

![tables](../iclr_results/4793_When_Scores_Learn_Geometry_Rate_Separations_under_the_Manifold_Hypothesis/tables/312b3cf225d6daa202495c2e9518f6f33a90e514cc9eff7d701aa9b057a5e50a.jpg)

![tables](../iclr_results/4793_When_Scores_Learn_Geometry_Rate_Separations_under_the_Manifold_Hypothesis/tables/3b418bd432a26728776c96c5cd367278652f6229aa0a65335072dd5e91a4f374.jpg)

![tables](../iclr_results/4793_When_Scores_Learn_Geometry_Rate_Separations_under_the_Manifold_Hypothesis/tables/aaef8e10b83f828c3865d1fc516d8eeb0e1de9f8e63c18f68ef3a438cd0b6c52.jpg)

<a id="paper-4794"></a>
## ProofBridge: Auto-Formalization of Natural Language Proofs in Lean via Joint Embeddings

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=U2jxHXuOX9)

### Images

![images](../iclr_results/4794_ProofBridge_Auto-Formalization_of_Natural_Language_Proofs_in_Lean_via_Joint_Embeddings/images/28a0ec838f42fcadffe02845e44427227e5dd2c5d6d22514e93eeb32d5327c2c.jpg)

![images](../iclr_results/4794_ProofBridge_Auto-Formalization_of_Natural_Language_Proofs_in_Lean_via_Joint_Embeddings/images/4a5a40818ae4b2bddeb5cb91f753a162fb08e9b8a38d70b8a36400b9fad4c94f.jpg)

![images](../iclr_results/4794_ProofBridge_Auto-Formalization_of_Natural_Language_Proofs_in_Lean_via_Joint_Embeddings/images/6c15848f8a54b59b0bd0889a2832f42aee1cb46d812ef9f3c5b64a475fff27b6.jpg)

![images](../iclr_results/4794_ProofBridge_Auto-Formalization_of_Natural_Language_Proofs_in_Lean_via_Joint_Embeddings/images/85b7ae6cf75354bbf2dd7c46647846a0dcca124738b0a3a9cea0dfb501d41376.jpg)

![images](../iclr_results/4794_ProofBridge_Auto-Formalization_of_Natural_Language_Proofs_in_Lean_via_Joint_Embeddings/images/a33912ca23e8b74d47321dc90a56eac69887d33418c8f118b65bb7d1b161d98c.jpg)

![images](../iclr_results/4794_ProofBridge_Auto-Formalization_of_Natural_Language_Proofs_in_Lean_via_Joint_Embeddings/images/a5fee65809071818726ab0d571597c355bd1a6b9ada99395c029d93c7c9ed2d0.jpg)

### Tables

![tables](../iclr_results/4794_ProofBridge_Auto-Formalization_of_Natural_Language_Proofs_in_Lean_via_Joint_Embeddings/tables/1d4a531753be1e7122cda6d6e87f533c02e4f8854f01a6ac9b8208725b8ffa41.jpg)

![tables](../iclr_results/4794_ProofBridge_Auto-Formalization_of_Natural_Language_Proofs_in_Lean_via_Joint_Embeddings/tables/a14013ef506e3e79e55e1b4b9a7fcb87d00a3beb0abf7e6dd5356e99fdf9b0eb.jpg)

<a id="paper-4795"></a>
## In-Context Compositional Q-Learning for Offline Reinforcement Learning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=ZBbKLvH0w4)

### Images

![images](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/images/050aa43ec6b926d85c4b86d28431e4ed2c67a05849f90f478ae6bad79276407c.jpg)

![images](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/images/0f69664031d9c8adcaec3bcc1a91e22b62e99ec71882847ca0b0ff96b467aad0.jpg)

![images](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/images/1a4b0ae30d8aacf216872699e60670ce2a2e3358687156257f1ae41ebe736c15.jpg)

![images](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/images/35e083e4c03fcd9849d67288afeff99e61a89e716201779e71ebb9b32f5c884b.jpg)

![images](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/images/40739aa962e25367f75afc9780531bcaaef9dffcda9120067af4af9fa4a7c592.jpg)

![images](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/images/5c9092e577c48d96869ac685b51d422cc56855a568e9c91c67274c7066ff3683.jpg)

![images](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/images/a3e0a8b824586c76a6f5d14d2882c1d60f07006982d90e38d29aaf61b999bb84.jpg)

![images](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/images/ac175b3e2b881347e05464f1597090fec2b440267f8033148d47f9db99d5621d.jpg)

![images](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/images/c72b6cfb1e2b9ee0fe4629ae1023e521bacdd2d90fa4f91c191eb36490ef73cd.jpg)

![images](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/images/cfff7a1c9728b11a781685c2a437e55a1e42522377a53b2218b73227cf6c8f52.jpg)

### Tables

![tables](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/tables/027d8aa7ca572f693563c28365cf3789edb3de44df4f8659ae37d2e99b068fe8.jpg)

![tables](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/tables/1560ac7eae1039eff1017a2d9013b3bf03f367cc753ce9a6d7c8a58daf58d7f3.jpg)

![tables](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/tables/29539890896ebd29449f365c8ae35a1e9938bc52214271e47b53e82bcd1e7abe.jpg)

![tables](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/tables/447d6a10b064f2e92527d1724cae8ce65bb14dfb0be7bf66b570c429857dc7a4.jpg)

![tables](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/tables/4eec00424b7bdc10b862487658e025c4a9fb8f395cde91b504d447f1aa6b5dab.jpg)

![tables](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/tables/5009e284ec4d495d03b151c6733ffd7b8b1a7eed5b3924012f31dd180645a0c7.jpg)

![tables](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/tables/5131fdd533c7c000bb342a80489c9fa303cc0071a16cd9e0f22313f00a620f5e.jpg)

![tables](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/tables/59b0be1b116af951cc8ff6e2fcca82f3fa8e3d4fafb60d11c31a066fabc7d2a0.jpg)

![tables](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/tables/98a81aade268f96446da389270e2b0e504ee5f0fdd07d952a44e586a429404d3.jpg)

![tables](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/tables/b273c42b9bdb8c411faeb7e34307ee4e669b5d05c24616cb637bb9acfa3971c4.jpg)

![tables](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/tables/bb4818a800c8a13b3c380725539fca293cbe4b5373a2fa5e91cab8dd3f01e15a.jpg)

![tables](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/tables/d9d486fd6903e4c644f0f791b1be4e80801483485e43c9d6fa6341f14a7475f5.jpg)

![tables](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/tables/df49933dd03dc1ee56a66d00a41bc9ba6cdde45987b22efc612e8a7cbdaf6efa.jpg)

![tables](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/tables/f69cc7ae3a2adae366e62d1eab165b63d8d9b62c8661aee94fbab74ff4312b9a.jpg)

![tables](../iclr_results/4795_In-Context_Compositional_Q-Learning_for_Offline_Reinforcement_Learning/tables/ffac684cd1a6cbf5b47f33742ba8864b23484fa9a816c006e1255f22ef4d96bf.jpg)

<a id="paper-4796"></a>
## Improving Classifier-Free Guidance in Masked Diffusion: Low-Dim Theoretical Insights with High-Dim Impact

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=mMK9pvQJxf)

### Images

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/061bb3be63e1c6b77f368550523fb32e34a580083132384a6590c3664b73ee05.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/2562d3d24d0256ba433ab5ccb781552534a2c7ad53c7a9d7ca407de894c05e41.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/263b5adb57877b42d06d60e05898c8c684a701937ae8d4ba10b9f5fdebe695a5.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/33cf64917432b9ada25bb1dfcdb35de9a4c1b6a291ccea67e352039f36af9149.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/41a91c066615aef4458e6119ea6b25e82ebd1984ddf651d84be87afbd0d41ead.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/468b5a369963134de2f731ca0eb8f19e68076ea3e68a0260d932dcec6907b17d.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/4834abfc68e999710162d7ee74149161f3e39cc5426fd8efee5a2e9da4abf5db.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/4b5ecc62b855861424d8c9abc167a2de90fe2ec5d9211395f5aea859068cd8bf.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/5370558a9656ce4f2e404e775f4372e338a97d79b395da259fbde3f42f5e9ce4.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/5492ba5b2029bb84829130eab5cde05641f9edcb39b67e8ac7167b60a65fdd7b.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/579980d666b90a11e57b645e634415b20fe7c3b0dfb387c7a29b76398e46cd1f.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/5e1d8a5bc53f76baa14648e67c1cea22cc64f72cce657a9825afd1a21e496b07.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/60a49dd6f54cd2acc22212796bae6c360024102e704c840fc1065543ffafdf09.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/64cb20686cbbfa7bdab6c4f7f8f18665f629e4fc5defb058850917d81e2b9ced.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/66511f959e2a0108c557468c742c35a47642ac583321a8d713451623f6a0eab8.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/6df133e5e6401dadf2e65a479ad451d61484c16db241f15decda4a20e2655895.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/8f4e4e7ea96892c2990c93f16c4a7404d8e47f1f40a8c2ffa0948631a764fc4e.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/96443562e3dfb91775453902e958109627cbd331fb19309aeafef44771487003.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/a8fc8c51ee84a5794c42bd7bb9f3e94579af2f16f78bbd31b99997582d340343.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/a9a2312ce71db1ba7d65f2d74c896929108fcf44fcf8cb09d3b32dd211cc5e66.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/c0da098234cf5a7fd513ff9f303f6a5a242c79c0a308397a1c94c867291b5f08.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/d3cfc2d5f75f7c546d32bfce9c4e4b21b84ac0fd375f898d2ef2c385b3eca670.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/da784e72991a4c9a7a60576345ea1f680e0e8052487622a45ab59ff59e26b919.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/e4386bce24618cc1bf55544c40cec494c0157f6a3168c631ea874aad4cb7950e.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/ec5cbcaa2356ec2485d782b015b3f222d4231f1bb54bec01502aec046fc38fb3.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/f14defc05eafb9039f2d17e7767d7c513c1f18bc92be6290cb9b09173b84fe54.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/f6d9b85cd4bc1a429790a3ce0af55e48d5fdfff5a171c02f3650ba21418219f2.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/fba52a95102b0b320992dbec889d645d737ccb3d808d08ae40841852aa3a6932.jpg)

![images](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/images/fee1770c0d6981136dcba34917ce675465064b9aad0408d271aa342872b47572.jpg)

### Tables

![tables](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/tables/2cb9c36e5e6a4446f6abd6c9039079bdfd6b1b0e055e7b717c50a1afa4e373aa.jpg)

![tables](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/tables/30ec1db202fb0c368196239656e99562731abccd07fecfaae41e8f9862fa6d49.jpg)

![tables](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/tables/5583c375556cfcbcaf5a285d649eaecd0bd7adcf81d1ee3ebd049bf7a331ab77.jpg)

![tables](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/tables/5912ff84461d69d87268db7b2ff1a13d1b42c07f3dd94172538aa8381e1d734b.jpg)

![tables](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/tables/9b049bafc09e74462a63303315638caf05267473ae9fa10cf88363463ca5e5ca.jpg)

![tables](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/tables/b3698b271fac03ccd734ab9aaaf39893a14d6747daa40832892a9675623b8288.jpg)

![tables](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/tables/d33f01957833bf01a552568b3bc0707f85962d37905e26db5882ff33877497b8.jpg)

![tables](../iclr_results/4796_Improving_Classifier-Free_Guidance_in_Masked_Diffusion_Low-Dim_Theoretical_Insights_with_High-Dim_Im/tables/fe4b48eabb52310bd0ed255cc621f4021659f1bb0500a00feb07f58d86e3b341.jpg)

<a id="paper-4797"></a>
## Flowing Through States: Neural ODE Regularization for Reinforcement Learning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=FHFDCsB9UC)

### Images

![images](../iclr_results/4797_Flowing_Through_States_Neural_ODE_Regularization_for_Reinforcement_Learning/images/0a0239cd587bc336cf0056493ea0a9024bf56c4e49ea7f6034ae3bf2b14fabd1.jpg)

![images](../iclr_results/4797_Flowing_Through_States_Neural_ODE_Regularization_for_Reinforcement_Learning/images/206fad5846e33caa3e7baeee17df4a5afd54abc40260040918cd74c5b93d6b01.jpg)

![images](../iclr_results/4797_Flowing_Through_States_Neural_ODE_Regularization_for_Reinforcement_Learning/images/4b28aa9748df61b1bae08eed9ced930a681b5ea6fd7d14f64b7f6a277f7e89c7.jpg)

![images](../iclr_results/4797_Flowing_Through_States_Neural_ODE_Regularization_for_Reinforcement_Learning/images/4c49f2038c46eec8be6e597ec941cd67562a437bb59c9ed78525eedad2f25fd2.jpg)

![images](../iclr_results/4797_Flowing_Through_States_Neural_ODE_Regularization_for_Reinforcement_Learning/images/5bb1a3945d5c759804b7f560221ec223a852da84d03ed807ee4483a282f0b4a5.jpg)

![images](../iclr_results/4797_Flowing_Through_States_Neural_ODE_Regularization_for_Reinforcement_Learning/images/8960908b225f5ea6b7b23bc0e42ef6ae10b6366c926f37efa3d0e16f6b17da55.jpg)

![images](../iclr_results/4797_Flowing_Through_States_Neural_ODE_Regularization_for_Reinforcement_Learning/images/cad9bc7cdf6f6fe3c9dc7308e64adab302e61e823b46bf11f64e12b857b1cc58.jpg)

### Tables

![tables](../iclr_results/4797_Flowing_Through_States_Neural_ODE_Regularization_for_Reinforcement_Learning/tables/1d9a02ccfea3e073c701d0de248eade6156cf0849c37437ba93f86e56ea788e0.jpg)

![tables](../iclr_results/4797_Flowing_Through_States_Neural_ODE_Regularization_for_Reinforcement_Learning/tables/6ffe040d622ba1cbb128823f5fcc3422d5090af67ac4fa104a47e6377becd2ce.jpg)

![tables](../iclr_results/4797_Flowing_Through_States_Neural_ODE_Regularization_for_Reinforcement_Learning/tables/a018d5b29addb34bd6d3c7858a44bb2897b730d3b02dfe847b4413c840d9b760.jpg)

![tables](../iclr_results/4797_Flowing_Through_States_Neural_ODE_Regularization_for_Reinforcement_Learning/tables/c1bae1c440570c805afead5bd3b9f17570c6ae404a2c7ee1282b4f8bc2d29c33.jpg)

<a id="paper-4798"></a>
## Predictive Differential Training Guided by Training Dynamics

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=zSTgrLkpRi)

### Images

![images](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/images/0299d706cb5160db7c9b60db89ede1505b9d91ba58f808f7bf0caa0a01207208.jpg)

![images](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/images/1d454dd6927c2bdd3c926d11b28b885f6572a57eae08f0cf32b208881b7189a0.jpg)

![images](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/images/2aedbb0036d0825b264987a872d5bbf1ebb2cdc127464843e42508d1e6075975.jpg)

![images](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/images/3de56387757b33a45038ed73523987d2adca9043e10bcb2a18bc4825cc4dbfc7.jpg)

![images](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/images/5c30203e6580b5992081b8f2e26239ec82922fe486c08dbe185add951c1174fc.jpg)

![images](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/images/5d5eb4870e445c839559e54295edfd5e4ffafa443f94d3280694db4e173f1361.jpg)

![images](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/images/644137e7d365fab6fbaf6c7397fc3b185572a12aece7a927728e6da6fbc22707.jpg)

![images](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/images/64950108fdba3049377141ac6a530c16a58fd7a79330e6679617dc55279feb59.jpg)

![images](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/images/8151193e90f2a5292ec754c1f22da6fdd2243a01a4ddda90facabe6626b0fe92.jpg)

![images](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/images/8277b97dc2c9cc15b6858015db4a9d47478c5533bb6513fad593a3adac92ec38.jpg)

![images](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/images/83f22da8ce9adefb152b099ca2467a8e0c4acc6a0a20d31dcb5bfb908416e499.jpg)

![images](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/images/af38a8ef5f22fcee825b9347ba0ea50e234dc0a69dae6d82b9401ff06098bf15.jpg)

![images](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/images/bafa87891de48bd63be7f751de263a87ef7cf395b7645a174eb8f2663d3c4c0d.jpg)

![images](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/images/bb428273974d717bd8ecd8634534c0be7c2b557330f7238c6c3c34e553ce488a.jpg)

![images](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/images/cba55d06d97a2d782c0185558bbd0b76ea55ffe1098b1ed032341c25f1bf6d27.jpg)

![images](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/images/d7b5357239c4efb88c3a0877084b23495c69b1e186e3e1976840861c7daf8319.jpg)

![images](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/images/e4925e3c820a1e601e87381d4720df6c33fa2c9c24d1ed48408812264aa0efe3.jpg)

![images](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/images/ee08fdedba86522a7c3b0aaf9dd11efaffcbd12fc072d4182756869f821fbca5.jpg)

![images](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/images/f6c529ccef565957d7e37da1d3054685ee79f6f5871b3dc8120a33f0a0554a51.jpg)

### Tables

![tables](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/tables/008f3e9238588471f05237a828abb3e74a12323361dd21c0cd917065fb308b0e.jpg)

![tables](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/tables/0933f2dded959581fb848f4297da3999636ff54e65acffaee68b55c2651d8350.jpg)

![tables](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/tables/3ea7b39eddcdcf778cae1b9cba67374af49fdd3176a430042ecfadb949904d85.jpg)

![tables](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/tables/4bf832bed413d903914d84c03b84fa016d9edbcbdde20e42061f8f0e5da5ebfc.jpg)

![tables](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/tables/6208c9f42eacda003cbf139a08bd0ea525b6cb09ac6b41145a25d10a7a50d7a9.jpg)

![tables](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/tables/63cda6ab94f36ae96e694e7f84134b5e63b626e2b6e07857e2c8bd3ebec8d373.jpg)

![tables](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/tables/6c4a12ef17e1d639135d94e2f10390f092093afa0a01e236f21751d20bad133c.jpg)

![tables](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/tables/84df40ac8e1a9eb569d8ae95cb992d458acd8c5510f932118855f0651d520e10.jpg)

![tables](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/tables/99e6cfa6c11564a17c788943dd64826d199a7b088f59c124db9949a746c2dac3.jpg)

![tables](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/tables/c763edd95265af63c96197d67186752a4b4dd7d3330a1f90b6d0c498997a2417.jpg)

![tables](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/tables/e01a0623702b3a4d92db59da08129c17cafd31e3712efdcd368c5573d752a85b.jpg)

![tables](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/tables/e0c71a380b78709b3378773f5765e8a11b97c8734011da8b88870ea9c67181cb.jpg)

![tables](../iclr_results/4798_Predictive_Differential_Training_Guided_by_Training_Dynamics/tables/fb79ab1b6b92ca2336e58ddfaf3e5bed2f066fc440defdfa2c03c5a64dd847dc.jpg)

<a id="paper-4799"></a>
## Improving Reasoning for Diffusion Language Models via Group Diffusion Policy Optimization

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=JaqvespRBP)

### Images

![images](../iclr_results/4799_Improving_Reasoning_for_Diffusion_Language_Models_via_Group_Diffusion_Policy_Optimization/images/131c82e636d4cf5e6f0cf7dfd7365c27f69e08cb506122fe580e27200b18003b.jpg)

![images](../iclr_results/4799_Improving_Reasoning_for_Diffusion_Language_Models_via_Group_Diffusion_Policy_Optimization/images/1ef271e7ea9c6cc62183d7bd016746b313c86d5252fd94fad582d3910e11230b.jpg)

![images](../iclr_results/4799_Improving_Reasoning_for_Diffusion_Language_Models_via_Group_Diffusion_Policy_Optimization/images/8ea2c952f2c2005c0b12ccdab9d028bbe09e30a87ccd003163c94e55d952dfaa.jpg)

![images](../iclr_results/4799_Improving_Reasoning_for_Diffusion_Language_Models_via_Group_Diffusion_Policy_Optimization/images/c4a0e0feb45702510a64f600d03d6a637dfb2541e54debd967e9feb9c0fe6764.jpg)

![images](../iclr_results/4799_Improving_Reasoning_for_Diffusion_Language_Models_via_Group_Diffusion_Policy_Optimization/images/da847070d6028973357ecb6b1caf12a9ecbaaebf32aad51a4d400cfb38ec147f.jpg)

### Tables

![tables](../iclr_results/4799_Improving_Reasoning_for_Diffusion_Language_Models_via_Group_Diffusion_Policy_Optimization/tables/10cf70777183cd4b38e02caeba44fbc9dcc09db3963c854a854ead8d3767e8c5.jpg)

![tables](../iclr_results/4799_Improving_Reasoning_for_Diffusion_Language_Models_via_Group_Diffusion_Policy_Optimization/tables/1cee697f85fc6dd36eb391d4ba4f487efaff9c1b30aa899c304358c7348ad374.jpg)

![tables](../iclr_results/4799_Improving_Reasoning_for_Diffusion_Language_Models_via_Group_Diffusion_Policy_Optimization/tables/6ead896f607e4f177b317dfa5fe38cc123b8c6bbeaad93fc021162e9a0ba0aed.jpg)

![tables](../iclr_results/4799_Improving_Reasoning_for_Diffusion_Language_Models_via_Group_Diffusion_Policy_Optimization/tables/9a0cf2d44d3a99ed4f4b462e98f06ea396e40a678e320aa86fac140d0a4f66b3.jpg)

![tables](../iclr_results/4799_Improving_Reasoning_for_Diffusion_Language_Models_via_Group_Diffusion_Policy_Optimization/tables/b81d009aa03d9d223ab61afb123210d82fff240406f361ddcef3d5984ddd0338.jpg)

![tables](../iclr_results/4799_Improving_Reasoning_for_Diffusion_Language_Models_via_Group_Diffusion_Policy_Optimization/tables/db14ad9e81e25d7c19e06546c2b0dbe4bd71df2414675396e63977f9068202bd.jpg)

![tables](../iclr_results/4799_Improving_Reasoning_for_Diffusion_Language_Models_via_Group_Diffusion_Policy_Optimization/tables/ffd13b531818db77419f07add2719f92a57ca57403d73104d3c12d540448046b.jpg)

<a id="paper-4800"></a>
## Octax: Accelerated CHIP-8 Arcade Environments for Reinforcement Learning in JAX

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=BdeUsYlyIf)

### Images

![images](../iclr_results/4800_Octax_Accelerated_CHIP-8_Arcade_Environments_for_Reinforcement_Learning_in_JAX/images/063646e65447019c6b6df682ea7e741fafec836fb0c5e5ca706f23dc3ee026b5.jpg)

![images](../iclr_results/4800_Octax_Accelerated_CHIP-8_Arcade_Environments_for_Reinforcement_Learning_in_JAX/images/3f9a1a04a595b17f9cf5234ab64d97b73fbbd66ff4afb6af33b32593d505ee29.jpg)

![images](../iclr_results/4800_Octax_Accelerated_CHIP-8_Arcade_Environments_for_Reinforcement_Learning_in_JAX/images/514aafd0243cc10cd5b8abf4d71bff34d9b3b0213de55c86e2ec0f2769ac35af.jpg)

![images](../iclr_results/4800_Octax_Accelerated_CHIP-8_Arcade_Environments_for_Reinforcement_Learning_in_JAX/images/640648b16621e9aacd563c62b2c1a6af606f8d3007cf83b375e69b594581680e.jpg)

![images](../iclr_results/4800_Octax_Accelerated_CHIP-8_Arcade_Environments_for_Reinforcement_Learning_in_JAX/images/be1fd522876dd3f594fd27d4c3f84cc8ad5fe47b80aa4d015efefec8323a65e3.jpg)

![images](../iclr_results/4800_Octax_Accelerated_CHIP-8_Arcade_Environments_for_Reinforcement_Learning_in_JAX/images/bf670097c817ccc54473352ee2dc214758ebffd4315d268663b9733ae9cad926.jpg)

![images](../iclr_results/4800_Octax_Accelerated_CHIP-8_Arcade_Environments_for_Reinforcement_Learning_in_JAX/images/cc2ca609aecb1c7dba64b44e2fb87f6b20eb3b4cbb20d7340afcd6bca29c8be1.jpg)

![images](../iclr_results/4800_Octax_Accelerated_CHIP-8_Arcade_Environments_for_Reinforcement_Learning_in_JAX/images/ce71a39aacb02fcec53ebc06d717199b084dce355904f1e90be9bf8c43e8e1aa.jpg)

![images](../iclr_results/4800_Octax_Accelerated_CHIP-8_Arcade_Environments_for_Reinforcement_Learning_in_JAX/images/d5a96fb89b2838b9904a6eb71c55525f0aca44ab057da73771fab737c785a68e.jpg)

![images](../iclr_results/4800_Octax_Accelerated_CHIP-8_Arcade_Environments_for_Reinforcement_Learning_in_JAX/images/d749218d2a5f25057b08f528a3b13497ac8adf0b2ddfed289754b3a20e54d7ad.jpg)

![images](../iclr_results/4800_Octax_Accelerated_CHIP-8_Arcade_Environments_for_Reinforcement_Learning_in_JAX/images/d920abafc662d2cae53505460c32bf5bb9636cc9957379f2d13a2602205c0707.jpg)

![images](../iclr_results/4800_Octax_Accelerated_CHIP-8_Arcade_Environments_for_Reinforcement_Learning_in_JAX/images/f99bc9f86ca33ce188a3c5ee828bb688f950a1885d2fe6edce4c4e5b3ca77d9a.jpg)

### Tables

![tables](../iclr_results/4800_Octax_Accelerated_CHIP-8_Arcade_Environments_for_Reinforcement_Learning_in_JAX/tables/023c6e8b8f6be000f5f188b3e0ed1c0e8877b9a0ec7e9b66b0d19484d72bc9a9.jpg)

![tables](../iclr_results/4800_Octax_Accelerated_CHIP-8_Arcade_Environments_for_Reinforcement_Learning_in_JAX/tables/bedd750d2347796aca1619534a6e9e9a60f34363e30820aa70e046a9f3884d1f.jpg)

![tables](../iclr_results/4800_Octax_Accelerated_CHIP-8_Arcade_Environments_for_Reinforcement_Learning_in_JAX/tables/d9d7c009302acf748cfaf691e81c1aecad03394fe944a198dd3129b0f693cad3.jpg)

![tables](../iclr_results/4800_Octax_Accelerated_CHIP-8_Arcade_Environments_for_Reinforcement_Learning_in_JAX/tables/e101ce593a06637b6bbac003c6e4140fdab43cc61877378254188c36a5d6f475.jpg)

![tables](../iclr_results/4800_Octax_Accelerated_CHIP-8_Arcade_Environments_for_Reinforcement_Learning_in_JAX/tables/fbaff0075e4aabdba87004c8fc25bd1e3ab433b5afb42199d3512e3f6d26f48b.jpg)

<a id="paper-4801"></a>
## OpenEstimate: Evaluating LLMs on Reasoning Under Uncertainty with Real-World Data

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=sAzUQkP47r)

### Images

![images](../iclr_results/4801_OpenEstimate_Evaluating_LLMs_on_Reasoning_Under_Uncertainty_with_Real-World_Data/images/19c40decbb9ee743b01f09ba399575bfcabfb35071ed975a11277559a4fd2dd7.jpg)

![images](../iclr_results/4801_OpenEstimate_Evaluating_LLMs_on_Reasoning_Under_Uncertainty_with_Real-World_Data/images/3cd957fe16ffc963499824bf0330a321051c5032e947720e9206ea4a1aefca78.jpg)

![images](../iclr_results/4801_OpenEstimate_Evaluating_LLMs_on_Reasoning_Under_Uncertainty_with_Real-World_Data/images/4132ff860ce6db3d04c6b2ffb1de4695ea8de996340708eb29145ef2f25e4704.jpg)

![images](../iclr_results/4801_OpenEstimate_Evaluating_LLMs_on_Reasoning_Under_Uncertainty_with_Real-World_Data/images/6b2f5e3bb9cbe280e33a3c4d8dbdd30b3a53d8f8bf2bf11d6d63b7b3b2a4cfdc.jpg)

![images](../iclr_results/4801_OpenEstimate_Evaluating_LLMs_on_Reasoning_Under_Uncertainty_with_Real-World_Data/images/723e8b67833a752c9336b75c0f173dac2cf5fa4fa61ed97b0fc739970160b02b.jpg)

![images](../iclr_results/4801_OpenEstimate_Evaluating_LLMs_on_Reasoning_Under_Uncertainty_with_Real-World_Data/images/8c6cbe83eefd4c6af73326daed5feaf361980860cbcc3ca2de6a3b0bab1f8c7a.jpg)

![images](../iclr_results/4801_OpenEstimate_Evaluating_LLMs_on_Reasoning_Under_Uncertainty_with_Real-World_Data/images/b314d4fe585cdcd54620b7e1ea196cb27adccd433b55fc4e8a3149340893b0ba.jpg)

![images](../iclr_results/4801_OpenEstimate_Evaluating_LLMs_on_Reasoning_Under_Uncertainty_with_Real-World_Data/images/eeb8b0d66ae24d538f0b8c99f4f90ec6a7f11ef03db7e90a7cadf575c346ad29.jpg)

### Tables

![tables](../iclr_results/4801_OpenEstimate_Evaluating_LLMs_on_Reasoning_Under_Uncertainty_with_Real-World_Data/tables/2e0b4117c478536adbb4738ed4a35db28a96360ceba068471189c7bd478f0282.jpg)

![tables](../iclr_results/4801_OpenEstimate_Evaluating_LLMs_on_Reasoning_Under_Uncertainty_with_Real-World_Data/tables/556f03592b015e2a43df8df217885e73d3ae46838da0684b01021b95ac69d453.jpg)

![tables](../iclr_results/4801_OpenEstimate_Evaluating_LLMs_on_Reasoning_Under_Uncertainty_with_Real-World_Data/tables/73c517cd8658eb774e1c7b40c87a979c22c8e5b46642c081cbc3e728b54c7dde.jpg)

![tables](../iclr_results/4801_OpenEstimate_Evaluating_LLMs_on_Reasoning_Under_Uncertainty_with_Real-World_Data/tables/969e097ec01107452059689b80247ccb02c4b313ee7a923a8bd99e5935042aa0.jpg)

![tables](../iclr_results/4801_OpenEstimate_Evaluating_LLMs_on_Reasoning_Under_Uncertainty_with_Real-World_Data/tables/a5234e522788a0b6958ede83a2dfe1ee4272bdbb3001a87554128cc06ed58c4b.jpg)

<a id="paper-4802"></a>
## STAT: Skill-Targeted Adaptive Training

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=m3jG3GaNIj)

### Images

![images](../iclr_results/4802_STAT_Skill-Targeted_Adaptive_Training/images/108f962d20a5fdbad2e66f76a35a041fa737a9c21213a55aa73c167a4a95d895.jpg)

![images](../iclr_results/4802_STAT_Skill-Targeted_Adaptive_Training/images/1cb6667b41fce03abfe037a37514ef5f12176d14e87b939413c2460abb0ccedd.jpg)

![images](../iclr_results/4802_STAT_Skill-Targeted_Adaptive_Training/images/52dbec0bdd1ec2db00301e217faef0cfcc4c152b5e88df42886fad9099f8ef4c.jpg)

![images](../iclr_results/4802_STAT_Skill-Targeted_Adaptive_Training/images/90db7200e12ba58b05a2ff35e46f600d57b58e231e8c59745a1a83a0761ead4b.jpg)

![images](../iclr_results/4802_STAT_Skill-Targeted_Adaptive_Training/images/9b4bbd55e198e29c2290d03ab128978d9c3cc4e03352804f8c44e1e40cee75ce.jpg)

![images](../iclr_results/4802_STAT_Skill-Targeted_Adaptive_Training/images/f1690f4bdc27b9291d3931ed08d0f7905a6d231c8e4ae7451d639004773645a6.jpg)

### Tables

![tables](../iclr_results/4802_STAT_Skill-Targeted_Adaptive_Training/tables/042fab2d6db9956908051676270224f17d687b78b78a33a35a9667ddb837a36b.jpg)

![tables](../iclr_results/4802_STAT_Skill-Targeted_Adaptive_Training/tables/1fc87b6c6d284d2459e392d3171611c7408647a07f0b947b1b936c9deeec6a06.jpg)

![tables](../iclr_results/4802_STAT_Skill-Targeted_Adaptive_Training/tables/35de3ca99af74e7a57a8ed979fc64e25985e7e1b451ab007e1af27d8875957b3.jpg)

![tables](../iclr_results/4802_STAT_Skill-Targeted_Adaptive_Training/tables/47111680c917afbe52a337dd5ba025c20ba029fffcf2b801d816a385c74e801c.jpg)

![tables](../iclr_results/4802_STAT_Skill-Targeted_Adaptive_Training/tables/4c4cbaa004ec2dad50f45c2772dd70fc0dda7e303023260c0abccc2cb0357860.jpg)

![tables](../iclr_results/4802_STAT_Skill-Targeted_Adaptive_Training/tables/50dd831265097cfabc615f6c23ae8597ba3d50f7024350fded8dc473ddffcf7f.jpg)

![tables](../iclr_results/4802_STAT_Skill-Targeted_Adaptive_Training/tables/744e56542405dd0bdf4864333b780d42010adac6d4a36ae8789c9c9e98a68a45.jpg)

![tables](../iclr_results/4802_STAT_Skill-Targeted_Adaptive_Training/tables/86e5d2568a1263601b99b15198c5529ad3391563dc0a8894fab13a346a949f6b.jpg)

![tables](../iclr_results/4802_STAT_Skill-Targeted_Adaptive_Training/tables/acb09e730fbe87336adfcfe24a1ea65486bc483b120e416ed335e526adf98a24.jpg)

![tables](../iclr_results/4802_STAT_Skill-Targeted_Adaptive_Training/tables/b93f1677acef0f8a7b2f4dbdb3d66986cb2a06bab36701900dca68e505a7b70f.jpg)

![tables](../iclr_results/4802_STAT_Skill-Targeted_Adaptive_Training/tables/c4b7a7bd06910fe5ef2e84cdfc7b35a8e756a66edc5dbb85eaae7009f7dbe157.jpg)

![tables](../iclr_results/4802_STAT_Skill-Targeted_Adaptive_Training/tables/edce8c7c607ba7b9187bb12a7ce8f456f0fd3026168186fadbc4cd845a466489.jpg)

![tables](../iclr_results/4802_STAT_Skill-Targeted_Adaptive_Training/tables/fd8135ac71aa4dbbeea7bdd74793a0a946308f37c564b564e19fe2a2f63cd6b0.jpg)

<a id="paper-4803"></a>
## Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=c1bTcrDmt4)

### Images

![images](../iclr_results/4803_Rubrics_as_Rewards_Reinforcement_Learning_Beyond_Verifiable_Domains/images/31a65a6c041c09adbfa0a9c1cff346eb72a15c0f6443f5eeaa912d4e7555e9c7.jpg)

![images](../iclr_results/4803_Rubrics_as_Rewards_Reinforcement_Learning_Beyond_Verifiable_Domains/images/9e790b8c7caa294e620384a0e5b61ccdb7129b5d70a6d1d2612b4122905ad0eb.jpg)

![images](../iclr_results/4803_Rubrics_as_Rewards_Reinforcement_Learning_Beyond_Verifiable_Domains/images/b552160b96c7af019f836d736a7d0fe86588eb96be041270ecd10d555e3e13e8.jpg)

### Tables

![tables](../iclr_results/4803_Rubrics_as_Rewards_Reinforcement_Learning_Beyond_Verifiable_Domains/tables/0bd311bf6c9c061637eb0f9d5dae80db506b3d3402e0ec0378a829c3287ddf83.jpg)

![tables](../iclr_results/4803_Rubrics_as_Rewards_Reinforcement_Learning_Beyond_Verifiable_Domains/tables/18f437e775f110717592c664b4bc133b025d5a0ebca85d4f030c07d67588f702.jpg)

![tables](../iclr_results/4803_Rubrics_as_Rewards_Reinforcement_Learning_Beyond_Verifiable_Domains/tables/290f655fc8f6d4b85764825bd581f40d48fad52f59ce01762aea86b0a22416c0.jpg)

![tables](../iclr_results/4803_Rubrics_as_Rewards_Reinforcement_Learning_Beyond_Verifiable_Domains/tables/2e90976ec7724a82272435a92c51972de76fd098df35cafca52623a50c542fbd.jpg)

![tables](../iclr_results/4803_Rubrics_as_Rewards_Reinforcement_Learning_Beyond_Verifiable_Domains/tables/363511effb637d8f14f11fdcd595047a95eb9b9484485590f7adc333085c5930.jpg)

![tables](../iclr_results/4803_Rubrics_as_Rewards_Reinforcement_Learning_Beyond_Verifiable_Domains/tables/568c631ccd01e868f038a2e0a6745b38c5072f6bc488b09e213933ca716d80a5.jpg)

![tables](../iclr_results/4803_Rubrics_as_Rewards_Reinforcement_Learning_Beyond_Verifiable_Domains/tables/6310cbdcd5fb258a4db1437356c91346f50d7c4f53c3e10331149d9be394b920.jpg)

![tables](../iclr_results/4803_Rubrics_as_Rewards_Reinforcement_Learning_Beyond_Verifiable_Domains/tables/876296537acc7c0b3897cd086518e526c19ea523b9238725596a87c3f583482c.jpg)

![tables](../iclr_results/4803_Rubrics_as_Rewards_Reinforcement_Learning_Beyond_Verifiable_Domains/tables/8c3b9114d820fdcec91ec394f4fdc3716f2b1a9cc35defbea4c373259362bae1.jpg)

![tables](../iclr_results/4803_Rubrics_as_Rewards_Reinforcement_Learning_Beyond_Verifiable_Domains/tables/b1482b0a15f4ba913b38d91a6e9e785078fa67fafc96c25c5838ecd31fed14f8.jpg)

![tables](../iclr_results/4803_Rubrics_as_Rewards_Reinforcement_Learning_Beyond_Verifiable_Domains/tables/c6c92e727103db1019d124f62d43187f9234bdd05f89235576ecfc4d044753dc.jpg)

![tables](../iclr_results/4803_Rubrics_as_Rewards_Reinforcement_Learning_Beyond_Verifiable_Domains/tables/d1e60ecbf7a044f0b6dd8e1cbe231e310faea0d6ec645e0cc341747b4594f998.jpg)

![tables](../iclr_results/4803_Rubrics_as_Rewards_Reinforcement_Learning_Beyond_Verifiable_Domains/tables/ef74d99bed625488bd5900787bbbcdf6af0c43aabafeb99e6d18773feb064038.jpg)

![tables](../iclr_results/4803_Rubrics_as_Rewards_Reinforcement_Learning_Beyond_Verifiable_Domains/tables/fdddb76b763084664a3dd6e43c7301de1273da660da77ff4002e90736b9f000e.jpg)

<a id="paper-4804"></a>
## Feature segregation by signed weights in artificial vision systems and biological models

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=lnTX3GoeTY)

### Images

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/028035cd62227144582cbd2dc1d49641888a6ce77fdadb77d61eb48b8da906eb.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/12835e86d420392d62414cb88a8253141a669699a2bf5846bc453f2b88247f36.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/137dde3723a9638a589c015ac6eafd44dbaeffb08bbc60ec4539a845e7544ccf.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/13c742555ef4fb16198dbe85d970c952d625f3a31429c882c64de88e9d0a079d.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/182af12121b011113e7852fc0d14228acc5100a7df1fb4be26a5b28162d38423.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/31a9f30cd7cd35ff961736340a434403cc5d39a999233dcd7776fc8525d11e81.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/3f90732843bce709e45eee2e82c9251bdb922027a38425328e05d01863fc602e.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/541bd8343755ca5ddc3b5516e3ed7282971c2144b752126ed8a40b0c5d5967f6.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/558d67f60a1dd508f2ea9ff11ff716aa1cf9511d0fa03d2704379d4e8f5b9fec.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/5c7f392802d052ac37f5b5fca8b3f2d89e6ebdff46711178eeebf1ad1fd26901.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/763827bc4c09a21d15cf341198fe6dd40ab6d9aba4fa10cb414e3bad716428af.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/793645747cd91f1d9e84f00dbeb38ca1806bcc0d4f924273c966d0e10378fbb3.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/84ea0fa5b994131569f13f287bb0677ebeea4b749b5bc89ba110dd49030fdaab.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/88ac9f00addec829baf172343e29a9822657cf31b74eb8fd6acc2e74219e8b7e.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/95b3b9a24dcae8c9fdb216fc9b8efc8c8b43171110c3ba10b7205549461eaa3e.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/9ba6732b47ccee635a9224bb16f286484454e85f39530147355c84f16c750bbe.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/9da9d0662ed79a8f32cb7ad52d7edcc02f98bc8a08d4e81fd050bf7b352822dc.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/a36052c96b34484b143fe4a9cd75d4e86a8f4d802bdae7e6c3717fe7c0b16475.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/a78d0fb4ee50a6ec04d4975c607899b8f65c8670b0cebda388a99ababd64c86c.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/bd2dce293922291d436926d1f43ca7e296b3a0fed89281caef771c8f1d75e9c3.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/c6e5415386ccfe8dfab19923c2d1dfd59781e7ac977e1d8684f277f8820899d7.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/c92d03750d8e5b5aa43f7d19b53b73cc545f6700aca4a73a0982b5f16e4ae6f0.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/cf1911548534b5592eff01ca0cb3ea4b75cc1bc5328725cc9176630718ae6965.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/d16189d587c4eb569afadc755169a0906d2163b3cc779a60f0e2826167e5b0b3.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/ed22d118fdebaf08f1181ad05e584f60ac377d3931e794a09a5ae9853b380d7f.jpg)

![images](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/images/f14c2b85fa1d467d56a0c0da93ddb2aa7b0cb68b21ea6b01288ee585264f1539.jpg)

### Tables

![tables](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/tables/07f80a679caf40570469dc4523ee016cddd50c49032b335bc5c548141748df95.jpg)

![tables](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/tables/8647403a80d530e4c89c46ffac0101718b6ea4232c334cb441c650982a7ceec3.jpg)

![tables](../iclr_results/4804_Feature_segregation_by_signed_weights_in_artificial_vision_systems_and_biological_models/tables/8ef5ff03f387ba446fe1fd69b09ed64887be0e05a3bbeecbae83e73ac7b5f9da.jpg)

<a id="paper-4805"></a>
## SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=zWb7ueH16c)

### Images

![images](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/images/14d4d2f0fa8bc3c5ba3c9cb93764e6c5e2b8a2b219441942eb621892e4b173a8.jpg)

![images](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/images/32d45e8b6a9185f6d9bf345441f6ea5264d8bbc1ad5b7c2f4be8c8a29441299d.jpg)

![images](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/images/56d1c592eb6cf20fe6653eae4dff4fbb671fe2f0e6069303c9a410f4e499c422.jpg)

![images](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/images/7e4f661429880b2b6a1507d41f4009e7b9633198bbd173600cef0bf7396770e4.jpg)

![images](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/images/a666c05cfdc594d2395636732634699533d097e1412c1be44049512ea655ef7c.jpg)

![images](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/images/ae0c58030c1bb69310cc26ade1d0d946b68bb8f02d9838aadea5c841d1b6ccbd.jpg)

![images](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/images/d302cc03dc58ca2b8b1db534a08a23dfc3a559febb6106c4b1bccc03fba63cbb.jpg)

![images](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/images/ecf370979074ae865a1848e07183550c8f20198177dcb718565c9e11e3e98caf.jpg)

### Tables

![tables](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/tables/0ad825610ebf626896ddc16f209fc64fff0081fa0b28a8f88563ee47f472c743.jpg)

![tables](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/tables/18092cd8beeff066f13c0725572249bc4cc303d4b9d37f03a5eff6f982e91db2.jpg)

![tables](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/tables/1c3459450a396858fa1de4ea65dea95adb2cc78c96b00d16eb9b831e02c10e99.jpg)

![tables](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/tables/20f7f3b2e1c5fa83733c9d972b346672aa0c60f187913515f89a7dbf1d002572.jpg)

![tables](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/tables/532b1e13f0bce635e5553961a85aa967ea25ac05193870d239d460b28d077fdb.jpg)

![tables](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/tables/606330f21a92610ab14a3c28ccf43460da0fb992c1ef184cb2f00fd0567be731.jpg)

![tables](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/tables/6098eda46f7c5126469eeea3ca775841135def196f4ab5860ee6731c7ada9d3e.jpg)

![tables](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/tables/60f105fa8a877216d29e1d56a9fc9b1c04413fbd0bfd7376754d798ca85303f6.jpg)

![tables](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/tables/64fa40a788d373d36a9cfbef293914b6265522de12e7c5828aba3e28e0f06ce7.jpg)

![tables](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/tables/82051b49d55064ef4fd5bc8e6710e1ffe686edcad39a523e495cda71f43124d6.jpg)

![tables](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/tables/8631397eac0cf5838a15db82792795423f1e8494d4918ca515a771058afb7216.jpg)

![tables](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/tables/925f8a7241ae251525fa1792aa3c7f7bd5ba4a5c5523053fbe9bce383991a540.jpg)

![tables](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/tables/a607602d25a4dbd9352f99a6b61d3cd507d32b1ce09eaab1f763a1b936fde8b4.jpg)

![tables](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/tables/a6c4c0609a19188d40b8a07dd082d974fec63c02f5edd63452ce72c72b25ffc5.jpg)

![tables](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/tables/b1dca98c4d3357c44ac97eb4ddd7e351dc3e3a600c68e3889b03e636a89087ef.jpg)

![tables](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/tables/dee7a978f4bc87b378574ac9eb2c0ebb6b6210381346ae72033d0f823798b3b5.jpg)

![tables](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/tables/df0dd6d36fb367486faea05157f7016ec014ed0ea3d1fd2d98d11eb23ec196a1.jpg)

![tables](../iclr_results/4805_SealQA_Raising_the_Bar_for_Reasoning_in_Search-Augmented_Language_Models/tables/ead0aa7f58f10067a7eb7156f0a7d3fdc60c88a3e9c310bce815910ceed086b2.jpg)

<a id="paper-4806"></a>
## The Lattice Geometry of Neural Network Quantization: A Short Equivalence Proof of GPTQ and Babai's Algorithm

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=L4SwHIZEaJ)

### Images

![images](../iclr_results/4806_The_Lattice_Geometry_of_Neural_Network_Quantization_A_Short_Equivalence_Proof_of_GPTQ_and_Babai_s_Al/images/54bfa380097a41c55d0b19762b18710c6725cb4c7e2a9b57e71d5e2f367477a5.jpg)

![images](../iclr_results/4806_The_Lattice_Geometry_of_Neural_Network_Quantization_A_Short_Equivalence_Proof_of_GPTQ_and_Babai_s_Al/images/7bf3f34708c8a7da18708087849b19b957db528f9efe695df6bd2d6bc8e4c1b5.jpg)

![images](../iclr_results/4806_The_Lattice_Geometry_of_Neural_Network_Quantization_A_Short_Equivalence_Proof_of_GPTQ_and_Babai_s_Al/images/c69e6377b8212029ec8d08aa721c03c322a85c60d5cc2b80cb80e594b31f89ec.jpg)

<a id="paper-4807"></a>
## Zebra-CoT: A Dataset for Interleaved Vision-Language Reasoning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=c6XIVI3TiQ)

### Images

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/0052bbb51838f56b3f03fa826be47f09c34e1b4f07780a926be40d29d745ba7d.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/0a4c16073931f2678fb0553cd7c3e98c23847170ad859b9a714263e84a28f3f1.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/12e8788a5e54c2d84a74fd73ae1d707e54fd77bf8f62533fe00d38c4e4dc5ccd.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/19bcbb3503fa05ec4117d87af2c792f37a667d40927e4eb39430eaed3c2007c2.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/1c4109862a71393d89dedf7b37468aa1bfd3270cbfcb1ec23ec3e20c87f617d6.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/1e0e9a5c9ac5650e141df9aa1d630fd33df57d06347bfcf1e436be6689c70924.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/2190a807b80c0ec9f434efc9d4817f3a3fe90092cdd1233f2e84d60b787a930f.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/22e378adac6f649eb6432a3d676c77581c1f38771ab3981e57060f1018cae414.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/2882eac49512819efd8a8cd827e29f60667601f85800fa110e0812d264e8e5d1.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/28dc778b2f756724ded878a7d6c4afdb32d64f01f4df0221b23f4ba5d70d2c28.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/3865da45ddcd7406030d8c32a3d77cc7cb9156d6b28b644535b7d9bcec8ea219.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/40b6c283b8f5299643f2c7cf0be27698a865a5d4d3527ee459091a2b7ad312e9.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/42365d8cce50e1c7ed35d459d647611d136e1bc7f79a71efbffb83af6d246497.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/4f6dc13a75c11838eef9d6aa6c48e22e46ff6450812ae8bca5c9c1e44daeb720.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/550b3f5e37aa02c045ec06ef57a2246d775af4d6fc7c6c926a00e33de380ccc2.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/58d076b6a1b079d6f6fdd0474d04e0cf05ff23c1ef626b91dc944806d5fa74af.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/6994aba111815d3c1c6649cc243990978ff806592532e24836207a5137ce4432.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/79f18271bb5918eb12593a634d86f8211b0f54f40598a0c11477b6b80a8828fe.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/7a9c2342f6e3eec96171077be8e71054fc490658985454eeca6df9f7823c82b3.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/7bd3afa49b9c4f5a2c8e042d2844efef8edde9aecfd6a63a16f878993dc61d9f.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/84ec8e04bb3af3127c639f2c1254012584288dae68889479399d956af8b5165f.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/9423302fd6f95b8c4feb60d5a625273a1a8e718bbe03c315ca6bea6afa48efbe.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/98c303307c6f0473a73965c6075729cdc41b5a365648e5d9065ba382b59f6db7.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/9c9c584b6144650aa0e2d3005c5502996aad7971be056768fc2ab4e147afb415.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/af3c1e04311945e31f93bfa5300df84e0e50b6e3c0128fd74e194121d117e7e5.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/c4a9016232c0fa77a88649b94f55f2b8274fc6af0228d5f329e3ae1be184a63d.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/de5b98d41c6c97500bdcf354f35937a34d5d5a0ef0a9f0b8cc3314963ef013e6.jpg)

![images](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/images/f02bc9db7b38fe374ecfd0d2b4c53a90d4356c9aa88eb455ccf6a617afd7592b.jpg)

### Tables

![tables](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/tables/1a940e817b75ebf63d167605df6c936c6f46061378ff75b222e8eba109e11f63.jpg)

![tables](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/tables/3c938624069f7203462a87991090f47a1d7cfc583d72a060490ac33e450fd5a9.jpg)

![tables](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/tables/4b4fc537944c799dd7cce0d3b63968fe11644f2a62f7c0c6ae5484e4bd6ca05d.jpg)

![tables](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/tables/544cbf9b30cae6a77f57a3b411ad239a633d41be5a6ae37d07eac073ca0be4d1.jpg)

![tables](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/tables/72333c7e1e84201955e42c3e4011d02a48b1f45b8813da220f913306f8561a7c.jpg)

![tables](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/tables/9a01fa9527c66d7f12f32f7993368945d5e06ad740ea1de33aeb895122559995.jpg)

![tables](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/tables/c22860a2c19f099295d6ba944fece5783decdfce64aae5226359353b68218233.jpg)

![tables](../iclr_results/4807_Zebra-CoT_A_Dataset_for_Interleaved_Vision-Language_Reasoning/tables/e2c9f3a4f98dd0f89fe02631c313e0744886e4553f7dd3c36214ec7ab0078fc1.jpg)

<a id="paper-4808"></a>
## Improving Code Localization with Repository Memory

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=8yjWLJy2eX)

### Images

![images](../iclr_results/4808_Improving_Code_Localization_with_Repository_Memory/images/3ef4273eae65843f9832fc89d73b64e59cae8232f46af0dafd9672dcb7760fbb.jpg)

![images](../iclr_results/4808_Improving_Code_Localization_with_Repository_Memory/images/5b51b6600668e166b9cbc26bc5f9443233c75708caaf6f042920c246eac72d1d.jpg)

![images](../iclr_results/4808_Improving_Code_Localization_with_Repository_Memory/images/9f19dc0d1b92865cea7b5840dd9b31825af4b298b25702fd2422079b7b95223e.jpg)

![images](../iclr_results/4808_Improving_Code_Localization_with_Repository_Memory/images/a37b0ec7350025a2f0e9fb155e515fbad3f773bd7781a466b8505178b6cf9001.jpg)

![images](../iclr_results/4808_Improving_Code_Localization_with_Repository_Memory/images/ce0389e05dfdf7a7b249aa8939b4a284edaf3b72743eceb10637eb98c7950f7c.jpg)

### Tables

![tables](../iclr_results/4808_Improving_Code_Localization_with_Repository_Memory/tables/0174ed2b58a663cbb1b97777077154260baee1ce75a4ee590dde16a0e6bb194a.jpg)

![tables](../iclr_results/4808_Improving_Code_Localization_with_Repository_Memory/tables/0fecd56f7a0440918704e183162343b95676e0937f76f16ef1067c8137ba40c0.jpg)

![tables](../iclr_results/4808_Improving_Code_Localization_with_Repository_Memory/tables/240d9d10d8b4c4a870e835431f47925e394b946a4af8ba5aad065d650389bb47.jpg)

![tables](../iclr_results/4808_Improving_Code_Localization_with_Repository_Memory/tables/31a12863b9a3a6b3511c46e7d264a05d90f511989a2ee2f16ba48e97848222ad.jpg)

![tables](../iclr_results/4808_Improving_Code_Localization_with_Repository_Memory/tables/6b2c9cc9eb9a072eb1a570e444401fd78e8bcce1d2eb1a850e9351f38f3b2c08.jpg)

![tables](../iclr_results/4808_Improving_Code_Localization_with_Repository_Memory/tables/7f9fcc1aa38c865e269676b381eb32bafa24cb2fb3cd7de3d8499957318758b0.jpg)

![tables](../iclr_results/4808_Improving_Code_Localization_with_Repository_Memory/tables/99091f0fd562d32339ff0d71c31e1e3ca09186ee5174dea1115bb78ae1da7286.jpg)

![tables](../iclr_results/4808_Improving_Code_Localization_with_Repository_Memory/tables/a1b111dd2206166aa88d3a014e4577e56b17aa9a0dcb1046f51b976ede524d5b.jpg)

<a id="paper-4809"></a>
## The Limits of Inference Scaling Through Resampling

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=j8H84v6AZ1)

### Images

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/045ece41c169a54d5b0b0c0987ac3e9cc6af79825fb1f2bf27302849cc317668.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/0e3319163c91e7c07023444f7986be358d291f22b3a748744e43d286b9cb8ac5.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/0e44fc742513d8d81b26ebfd245925f7b01fa4190fbf99cf61622757e450dc89.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/10d8e039f3d3755bbebcb17bd7b835ffd0273629387e0e448d2ccbc3e1877e71.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/1ca5ee62bd130b4bac0e00e327626dc6179d607f299698043489407ca52a56aa.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/288b081e14affbea71f7df73a55c3f59ba7c916702900386c360015e77456708.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/4e618dc7672bafeb84bf40251dfb208d939e08a06b81482835ca55592fce93bb.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/52717edcabb0fa06c87ec1ae5df7e689834199f7e73ce8661fac929579ae5f31.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/580f12b3964e8c4a320996777e08b50386ae3dff0de10ccba8a632137bcf1339.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/594987c326c71641206edfcd45257cf5c3900060593f95f7a2a99b423c8f6b83.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/5af24c52fad866f56d512a94d115cf025c6862897a284360c337f79030d59502.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/5b366f4757c466d4a552e38baf2bf875880049856e44340ae56657795a87a3b4.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/5f10ffb01f37d34b5f1515b90b4977c3591203daa2927b403725098b2c5d5039.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/6301b5228473d19ff1c45e78e89e98cffcac57c9b5c77c00a79369ee8f88e595.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/678030da9c47e1954eb99697438fd462828148933c54ca3aff5ba34b594af923.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/699d27336bf8008eb7207f92ae440202ff8ed570ef24305c9405f13cb45e7607.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/73e23b2f4e819439345942e08d9a4ff76447aeaa6b7919247313c622e2d0905d.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/892c8b76a73ad26f2ef3a4d2991c9b2ff5bd71192cea957e423f0ad139d7aacf.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/a5cb3a1395198b55d77dbbc85579a581ab66e53f969c50e5a1e61ee6b7267b87.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/bb2a3383449fa23ebf40d52949c47a546b9b1867a1a7bedcbf22ea5717a16be1.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/bccaa415c7424ae63ee80217dba6e7fa244ee1c16aecfe9cc057de91a29c241f.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/c7660837498e231d0b42523e0d1dce338cb0d31c029d7126723ea157b7dcef63.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/cd8b131c7fa963f995aa64154f1fed0cd6d0e5b84fc63377aa13ca56c3be13f5.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/e15c807b570ce8e737abacc2300dd032fb073ab8f202709ad3d1729d4d4bc55a.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/edefb9392cb81f24cb133b641496bc5eafeaa15ec2f27f9e4bcf309035cbcace.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/f6b2d5a7edbef77534272e8d9542f5d987a15ac6cccc9d3c3a3ab609c4e8bad8.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/f89ce12021009745f46cce8ffac5ae67472ddf715de1ea90b8d9b7c03dc9cd4d.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/fcc6db2773b1965143a1e8467031c24e2e9dca564d653f4d112cf6299c154b34.jpg)

![images](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/images/fea09d5df6c0839d98b7dfe3ee61b413ec9f87721602a6476369cd74350a4cc7.jpg)

### Tables

![tables](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/tables/2798ea484efef9bedea46a3912f0320810c4bd3fe8a8d7b16e265a119bc545d3.jpg)

![tables](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/tables/654139b05869d59b381ed2c7a62ab18351740fb40f706d01729541c171bbcc4a.jpg)

![tables](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/tables/6ad125aab24155ebd9c5bfe2c77c9d0d00d77ecf8fa00e41da7b862253158502.jpg)

![tables](../iclr_results/4809_The_Limits_of_Inference_Scaling_Through_Resampling/tables/9e540e2b251c6dde69bc4944ed3e9833eb88284144994ec80a3cb615dd69cb2e.jpg)

<a id="paper-4810"></a>
## Don't Shift the Trigger: Robust Gradient Ascent for Backdoor Unlearning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=voqtsqYS6j)

### Images

![images](../iclr_results/4810_Don_t_Shift_the_Trigger_Robust_Gradient_Ascent_for_Backdoor_Unlearning/images/033ceb0cc56331ff4c4e38817678fcd2f2e0572055202aef99e142769d43e943.jpg)

![images](../iclr_results/4810_Don_t_Shift_the_Trigger_Robust_Gradient_Ascent_for_Backdoor_Unlearning/images/053c3330c624aef7064b0854aef1687e98b10f216576a25d29e7a2be499c48fd.jpg)

![images](../iclr_results/4810_Don_t_Shift_the_Trigger_Robust_Gradient_Ascent_for_Backdoor_Unlearning/images/0b26c65337624a74480a2663207387f8bcd9de7290c9f980cfcee8f0afe94a72.jpg)

![images](../iclr_results/4810_Don_t_Shift_the_Trigger_Robust_Gradient_Ascent_for_Backdoor_Unlearning/images/e45205475afbc4f5552883e46583acda64e11966e837b7ca764648036348f814.jpg)

![images](../iclr_results/4810_Don_t_Shift_the_Trigger_Robust_Gradient_Ascent_for_Backdoor_Unlearning/images/e5ce50f6bcc1facb020aa21abb4cb8297f513bf9210384e90fdbdb1bae212520.jpg)

![images](../iclr_results/4810_Don_t_Shift_the_Trigger_Robust_Gradient_Ascent_for_Backdoor_Unlearning/images/fd7ab58a5229c918c9cb4f7e0d2cd6778924ff2f6ae88ae130c4e8228a45edaa.jpg)

### Tables

![tables](../iclr_results/4810_Don_t_Shift_the_Trigger_Robust_Gradient_Ascent_for_Backdoor_Unlearning/tables/2f69badb317c4aa602b208136fc30936ff1d6710ce0be9591757dbfbf88e31c9.jpg)

![tables](../iclr_results/4810_Don_t_Shift_the_Trigger_Robust_Gradient_Ascent_for_Backdoor_Unlearning/tables/5bfe9ee259114432964ab11ee55f0fa4739e27d9672e5b22d30df3ca7070c6f2.jpg)

![tables](../iclr_results/4810_Don_t_Shift_the_Trigger_Robust_Gradient_Ascent_for_Backdoor_Unlearning/tables/671ed1bd094841b6dfadb2b7f099db644aee21e26d14cf041f319a248f1b2534.jpg)

![tables](../iclr_results/4810_Don_t_Shift_the_Trigger_Robust_Gradient_Ascent_for_Backdoor_Unlearning/tables/72345e885b16b03b713f06e21f84fc18159c4f86ce7690b07dffd32abecded4e.jpg)

![tables](../iclr_results/4810_Don_t_Shift_the_Trigger_Robust_Gradient_Ascent_for_Backdoor_Unlearning/tables/77bdb877887fdd4332fe14b3a176d26818702b8af0eca815f88b5300ff29bb5e.jpg)

![tables](../iclr_results/4810_Don_t_Shift_the_Trigger_Robust_Gradient_Ascent_for_Backdoor_Unlearning/tables/7b9266a8342b8d0135f7b88f781784f5fc279352bd04b488e7c88c64bdb7ccd6.jpg)

![tables](../iclr_results/4810_Don_t_Shift_the_Trigger_Robust_Gradient_Ascent_for_Backdoor_Unlearning/tables/9e88437c723d437b8f18fbcbabc1821ea0bb6a370d867e0bdf4915ce6ff3ce8f.jpg)

![tables](../iclr_results/4810_Don_t_Shift_the_Trigger_Robust_Gradient_Ascent_for_Backdoor_Unlearning/tables/a5621db777627624a7289826c009c5a2bb61e40953a7d72cb8eb278ba911b342.jpg)

![tables](../iclr_results/4810_Don_t_Shift_the_Trigger_Robust_Gradient_Ascent_for_Backdoor_Unlearning/tables/e5cbc25aeca64955f72d9bb1a9922b01a2222020615e87ef2916a9c4bf534f1b.jpg)

![tables](../iclr_results/4810_Don_t_Shift_the_Trigger_Robust_Gradient_Ascent_for_Backdoor_Unlearning/tables/eb61a8568fac2824a51bf27b63e6229b0272531cfce45153d3762b22c85d6a7f.jpg)

![tables](../iclr_results/4810_Don_t_Shift_the_Trigger_Robust_Gradient_Ascent_for_Backdoor_Unlearning/tables/fe6580503f0a09e4d6bd3c7909a615feb34a6f0866435f8da8772202ddfca70c.jpg)
