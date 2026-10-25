# ICLR 2026 Main Papers — Part 158 of 199

27 papers.

- [SparseEval: Efficient Evaluation of Large Language Models by Sparse Optimization](#paper-4243)
- [Inference-Cost-Aware Dynamic Tree Construction for Efficient Inference in Large Language Models](#paper-4244)
- [Towards Understanding the Shape of Representations in Protein Language Models](#paper-4245)
- [UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking](#paper-4246)
- [Peng's Q($\lambda$) for Conservative Value Estimation in Offline Reinforcement Learning](#paper-4247)
- [On Measuring Influence in Avoiding Undesired Future](#paper-4248)
- [High-Dimensional Analysis of Single-Layer Attention for Sparse-Token Classification](#paper-4249)
- [On the Impact of the Utility in Semivalue-based Data Valuation](#paper-4250)
- [Escaping Policy Contraction: Contraction-Aware PPO (CaPPO) for Stable Language Model Fine-Tuning](#paper-4251)
- [Streaming Autoregressive Video Generation via Diagonal Distillation](#paper-4252)
- [A Two-Phase Deep Learning Framework for Adaptive Time-Stepping in High-Speed Flow Modeling](#paper-4253)
- [Cross-ControlNet: Training-Free Fusion of Multiple Conditions for Text-to-Image Generation](#paper-4254)
- [Beyond Softmax and Entropy: Convergence Rates of Policy Gradients with $\boldsymbol{f}$-SoftArgmax Parameterization $\&$ Coupled Regularization](#paper-4255)
- [Reasoning on Time-Series for Financial Technical Analysis](#paper-4256)
- [Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets](#paper-4257)
- [AP-OOD: Attention Pooling for Out-of-Distribution Detection](#paper-4258)
- [Multiplicative Diffusion Models: Beyond Gaussian Latents](#paper-4259)
- [Constraint-guided Hardware-aware NAS through Gradient Modification](#paper-4260)
- [What happens when generative AI models train recursively on each others' outputs?](#paper-4261)
- [Composition of Memory Experts for Diffusion World Models](#paper-4262)
- [Dyslexify: A Mechanistic Defense Against Typographic Attacks in CLIP](#paper-4263)
- [Bayesian Parameter Shift Rules in Variational Quantum Eigensolvers](#paper-4264)
- [Enhancing Image-Conditional Coverage in Segmentation: Adaptive Thresholding via Differentiable Miscoverage Loss](#paper-4265)
- [Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding](#paper-4266)
- [Unsupervised Representation Learning for 3D Mesh Parameterization with Semantic and Visibility Objectives](#paper-4267)
- [Stable coresets: Unleashing the power of uniform sampling](#paper-4268)
- [Verifier-Constrained Flow Expansion for Discovery Beyond the Data](#paper-4269)

---

<a id="paper-4243"></a>
## SparseEval: Efficient Evaluation of Large Language Models by Sparse Optimization

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=CZAzAedGSV)

### Images

![images](../iclr_results/4243_SparseEval_Efficient_Evaluation_of_Large_Language_Models_by_Sparse_Optimization/images/030d498639e5fbe612acb667ad638bb8f814208d7a660cc58043201d833f6e31.jpg)

![images](../iclr_results/4243_SparseEval_Efficient_Evaluation_of_Large_Language_Models_by_Sparse_Optimization/images/048588afb5c25c79e742aee0882ed8496ffb791db248d40e3813e9fa051fa01e.jpg)

![images](../iclr_results/4243_SparseEval_Efficient_Evaluation_of_Large_Language_Models_by_Sparse_Optimization/images/3c1ebb433b94762b26996f3388294e5481cc1891ab1ec7a7a877a00af17eae29.jpg)

![images](../iclr_results/4243_SparseEval_Efficient_Evaluation_of_Large_Language_Models_by_Sparse_Optimization/images/67d8b8adee7bfa40bdac87966fa903c258535c5668e90e6377417b99046b4909.jpg)

![images](../iclr_results/4243_SparseEval_Efficient_Evaluation_of_Large_Language_Models_by_Sparse_Optimization/images/729e3f4ccb5ec96d5523caad6ab35c7e7f4c60e5f5bdb73ad923e1898c84851d.jpg)

![images](../iclr_results/4243_SparseEval_Efficient_Evaluation_of_Large_Language_Models_by_Sparse_Optimization/images/b358ce6ab84e74a736d62974d14cbd4d435124fb8ab58f3b6a4264ae6a533d38.jpg)

![images](../iclr_results/4243_SparseEval_Efficient_Evaluation_of_Large_Language_Models_by_Sparse_Optimization/images/e1a6f7c971eb15297de3f9f5a5d248159fbfd4fca1c53c4e25a77fbfc42da766.jpg)

![images](../iclr_results/4243_SparseEval_Efficient_Evaluation_of_Large_Language_Models_by_Sparse_Optimization/images/e717ae49597ef2e2875c58d2d7a44cd298ca590189628a49f1ca5d36baa920cc.jpg)

![images](../iclr_results/4243_SparseEval_Efficient_Evaluation_of_Large_Language_Models_by_Sparse_Optimization/images/e819cae04bb2b2c588488a2136fea12714ab099239272b81dc3ce5b8addc7bef.jpg)

![images](../iclr_results/4243_SparseEval_Efficient_Evaluation_of_Large_Language_Models_by_Sparse_Optimization/images/f31cbc3115236f87bac2f232a1a54a7de75c099bd18a275e2bb4b7aa35e3ab5a.jpg)

![images](../iclr_results/4243_SparseEval_Efficient_Evaluation_of_Large_Language_Models_by_Sparse_Optimization/images/f97ca2ca02728bb2e5be02119f81565c92a3551f7927e6192b2daa05b502d388.jpg)

### Tables

![tables](../iclr_results/4243_SparseEval_Efficient_Evaluation_of_Large_Language_Models_by_Sparse_Optimization/tables/a625a0f095afb6bc2d9890178cd635bf20d39da882d4c971401e6e5d632da2ab.jpg)

![tables](../iclr_results/4243_SparseEval_Efficient_Evaluation_of_Large_Language_Models_by_Sparse_Optimization/tables/e90da3628e853650b04ccbc784de429459ce38182514ffe7299e1e70665d1d89.jpg)

![tables](../iclr_results/4243_SparseEval_Efficient_Evaluation_of_Large_Language_Models_by_Sparse_Optimization/tables/fd467e08752cb6a8b0a2ce4a715e7dd3224fb39d26e567bb85f8031832f7252c.jpg)

<a id="paper-4244"></a>
## Inference-Cost-Aware Dynamic Tree Construction for Efficient Inference in Large Language Models

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=iaWyRYthFf)

### Images

![images](../iclr_results/4244_Inference-Cost-Aware_Dynamic_Tree_Construction_for_Efficient_Inference_in_Large_Language_Models/images/14875edf7975b98fded1f2f676ea03f5df1eaa7e4950337ef9eb7292fd515857.jpg)

![images](../iclr_results/4244_Inference-Cost-Aware_Dynamic_Tree_Construction_for_Efficient_Inference_in_Large_Language_Models/images/29d68d1e51b1534c977bdfa313a66561e1a440e1d90899351cbc8f6ef0ff1f47.jpg)

![images](../iclr_results/4244_Inference-Cost-Aware_Dynamic_Tree_Construction_for_Efficient_Inference_in_Large_Language_Models/images/2a95a28fce7b2c1a6426f788ba46e182349dc01c97afab6ed81d49c2779bde35.jpg)

![images](../iclr_results/4244_Inference-Cost-Aware_Dynamic_Tree_Construction_for_Efficient_Inference_in_Large_Language_Models/images/339a5177c57f7b5c2dcccdabdbe328fb0aec492483290013d8c4cf43cd257188.jpg)

![images](../iclr_results/4244_Inference-Cost-Aware_Dynamic_Tree_Construction_for_Efficient_Inference_in_Large_Language_Models/images/70e11c9f18f64bd6a130f5d1666b4af51c3b9bd511840040664ffcf416f02ab4.jpg)

### Tables

![tables](../iclr_results/4244_Inference-Cost-Aware_Dynamic_Tree_Construction_for_Efficient_Inference_in_Large_Language_Models/tables/5bc7d5d6ab9bcd584baf9fcf3908843dd890fc241ff053054b69daf113e0f520.jpg)

![tables](../iclr_results/4244_Inference-Cost-Aware_Dynamic_Tree_Construction_for_Efficient_Inference_in_Large_Language_Models/tables/9b5fadb9615a1a569808130baf50e0dee90162b520666275ca94599d4d7856fc.jpg)

![tables](../iclr_results/4244_Inference-Cost-Aware_Dynamic_Tree_Construction_for_Efficient_Inference_in_Large_Language_Models/tables/c025d07251f85987fc35dcb487c0f5da1896936de7dd7fac56d127cd7e60c6cb.jpg)

![tables](../iclr_results/4244_Inference-Cost-Aware_Dynamic_Tree_Construction_for_Efficient_Inference_in_Large_Language_Models/tables/e21ce6489e611283b3e75218ff4bd37f29fb8a67e450a47a0611181258b336df.jpg)

![tables](../iclr_results/4244_Inference-Cost-Aware_Dynamic_Tree_Construction_for_Efficient_Inference_in_Large_Language_Models/tables/e9e2af7bd99a1a616f6730e631cab1740994e42c16912e9259dec36a45b771c1.jpg)

<a id="paper-4245"></a>
## Towards Understanding the Shape of Representations in Protein Language Models

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=Dnn8SSBJaY)

### Images

![images](../iclr_results/4245_Towards_Understanding_the_Shape_of_Representations_in_Protein_Language_Models/images/03f6e2c6eea233b6a0b3eb1947572017552ffd39b4c32ded17767ceb96586f30.jpg)

![images](../iclr_results/4245_Towards_Understanding_the_Shape_of_Representations_in_Protein_Language_Models/images/10012e2795c0f7ff8c130d04732b8210bc304ea0301632f12fff189a8cff1d48.jpg)

![images](../iclr_results/4245_Towards_Understanding_the_Shape_of_Representations_in_Protein_Language_Models/images/2ead32e48a6004d6d5cc68612df3e268966542c3cc75153ef5e7a259ed8b91e3.jpg)

![images](../iclr_results/4245_Towards_Understanding_the_Shape_of_Representations_in_Protein_Language_Models/images/3426458aaed81b58bebadb3955506d37bab045e52e48e4967defab2aae4af89a.jpg)

![images](../iclr_results/4245_Towards_Understanding_the_Shape_of_Representations_in_Protein_Language_Models/images/450e8d1d9dcda7bc68dc7b38e268f8d2bda5d25913b60a687fb07c2bcfe0e9b3.jpg)

![images](../iclr_results/4245_Towards_Understanding_the_Shape_of_Representations_in_Protein_Language_Models/images/50966e3512303b64e1212f0f66461f99de3d9104a9f4a960f767324236a0d49b.jpg)

![images](../iclr_results/4245_Towards_Understanding_the_Shape_of_Representations_in_Protein_Language_Models/images/53166ffdea696bae9eb42e80df98d723029b73ed7e2b69c88f75dda257f75e32.jpg)

![images](../iclr_results/4245_Towards_Understanding_the_Shape_of_Representations_in_Protein_Language_Models/images/587a263617debabac29b76967ab63850434bc36d069d0e5ddd835ec29aba724c.jpg)

![images](../iclr_results/4245_Towards_Understanding_the_Shape_of_Representations_in_Protein_Language_Models/images/728ddd6706e0cdd32edcc33cf4dfae3633e6c82b6f64914cebf281ff35ad7066.jpg)

![images](../iclr_results/4245_Towards_Understanding_the_Shape_of_Representations_in_Protein_Language_Models/images/7cb7ccfc06940cc25efebd80bef9356fc4e633b2eef60823b0efa7e2c6b5fe81.jpg)

![images](../iclr_results/4245_Towards_Understanding_the_Shape_of_Representations_in_Protein_Language_Models/images/afae645e9be564ddeab06a8ed53640ac5805d10370b3618b74438b10e5dd643e.jpg)

![images](../iclr_results/4245_Towards_Understanding_the_Shape_of_Representations_in_Protein_Language_Models/images/f172fcc815c5326cb85df0f7c5f6a373ab61e0fd65457d5a10bbc961b5cf66fa.jpg)

<a id="paper-4246"></a>
## UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=aykkGh9TIy)

### Images

![images](../iclr_results/4246_UIS-Digger_Towards_Comprehensive_Research_Agent_Systems_for_Real-world_Unindexed_Information_Seeking/images/1c083e659d0802e03ff9cf953f058c2311100e1bc3741c85bfd62586aa85e5c5.jpg)

![images](../iclr_results/4246_UIS-Digger_Towards_Comprehensive_Research_Agent_Systems_for_Real-world_Unindexed_Information_Seeking/images/447baad6404b6ddc06b0224ab160c769617689cc708b2256ce3db76a733edb71.jpg)

![images](../iclr_results/4246_UIS-Digger_Towards_Comprehensive_Research_Agent_Systems_for_Real-world_Unindexed_Information_Seeking/images/d6ccaeeb8032a93edceff3ed9ebffa588b5fed31e8e14608bb960bd6d27b230c.jpg)

![images](../iclr_results/4246_UIS-Digger_Towards_Comprehensive_Research_Agent_Systems_for_Real-world_Unindexed_Information_Seeking/images/d7b3a3f53941fbc929de9000f9e56965486902ce6d72ff34b8eb8a8caaf342b7.jpg)

![images](../iclr_results/4246_UIS-Digger_Towards_Comprehensive_Research_Agent_Systems_for_Real-world_Unindexed_Information_Seeking/images/da154bec068c5b24ad5e5f74a600ce8388dd8ba9fbc6fe54b5ca10a90c2fce2c.jpg)

### Tables

![tables](../iclr_results/4246_UIS-Digger_Towards_Comprehensive_Research_Agent_Systems_for_Real-world_Unindexed_Information_Seeking/tables/1d8486cb1bb4cc8a3d573c8cf2a724970ae1e8546c4b2d6e30a1c9b6e0140bff.jpg)

![tables](../iclr_results/4246_UIS-Digger_Towards_Comprehensive_Research_Agent_Systems_for_Real-world_Unindexed_Information_Seeking/tables/2e3debc934667dbb33d4cff3558afe71acfe9deb26c76da66b8567b7bf0d4df9.jpg)

![tables](../iclr_results/4246_UIS-Digger_Towards_Comprehensive_Research_Agent_Systems_for_Real-world_Unindexed_Information_Seeking/tables/54d5dae4d24368de9da0a91e8e31c3fd9d41792637d565fce6f1ac5f07f4cc7b.jpg)

![tables](../iclr_results/4246_UIS-Digger_Towards_Comprehensive_Research_Agent_Systems_for_Real-world_Unindexed_Information_Seeking/tables/af91c8b76c0e518eaa41447109bb5a03a03d8bdc119c0901e28c93fe1627aa3c.jpg)

![tables](../iclr_results/4246_UIS-Digger_Towards_Comprehensive_Research_Agent_Systems_for_Real-world_Unindexed_Information_Seeking/tables/be894cf9590811d141e00ac529a2700d64b97a20dacea005ddf968b5888ffd32.jpg)

![tables](../iclr_results/4246_UIS-Digger_Towards_Comprehensive_Research_Agent_Systems_for_Real-world_Unindexed_Information_Seeking/tables/bf31ceb17989f90ca09541f7e98fd49f3ec250a4b29cd8c4e7abe7fc815e2792.jpg)

![tables](../iclr_results/4246_UIS-Digger_Towards_Comprehensive_Research_Agent_Systems_for_Real-world_Unindexed_Information_Seeking/tables/bf8d3055d7b29bd7da6106026ba3a7f25d21e218d6b82c10c7577ac17b1282e4.jpg)

<a id="paper-4247"></a>
## Peng's Q($\lambda$) for Conservative Value Estimation in Offline Reinforcement Learning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=Ml4AtrrfQT)

### Images

![images](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/images/3ba3615d999a1703f18cea175559391146a342c2405159113dcf6d9ca392bf11.jpg)

![images](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/images/666424fb6fc62bcba7370501d57b64f5704e3237e2fd37d3a5db7c53b0c37865.jpg)

![images](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/images/830d5126acdd387758099c80d810d053a3332b4a4060c1ccfb8524bc5ee1038f.jpg)

![images](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/images/9398c958406ff90232909b96192b6c798143f7e73947e67f2087e08855e1d4ac.jpg)

![images](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/images/a65f5a4c1ba8175482797fa6fbf5bd0555b491eec3da525c94c5c748035ebf74.jpg)

![images](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/images/acd252b6b6b77ce847280de1b0736c0997e55e98ed3fe8ed1cd2037a6d6c33c9.jpg)

![images](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/images/c3ae591c2f3f9d6738c71494d6f126c1f1daeb91852812f55d8a54d582406a32.jpg)

![images](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/images/cc1eac6a8d3f6d224649bd91528cfb9a221e2813ad8d926919d90e2a52da7e63.jpg)

![images](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/images/d5910b7c8f91e670a4ed6031e6bcd3f7c97430b4c0b0b30e780fedb2ea493027.jpg)

![images](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/images/ecbb4f3a72cf6f44fcce55c4deb15fa446214e50df8543ef7ef2111c76c42ad2.jpg)

![images](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/images/ee0879958be258fdf5f7d32ca029c8c0777b066c16b666cfb096ff9a37e3fb90.jpg)

### Tables

![tables](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/tables/53ee607742f52052d0a186e3c541c3fad5993c465dca90c35e9d4433f97e7108.jpg)

![tables](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/tables/5a4c57443a0f165ef86ef67dc3f9019da23848e6a9e55c9a38aafcf900586cff.jpg)

![tables](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/tables/5fb1985e1847989a12fd5a42f8514d6f822351458e03d0e9f22949dd88f2de65.jpg)

![tables](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/tables/69fddc54db71b7ddd600a73b285715df9b728ed06952cc3bba77fa81aa081da8.jpg)

![tables](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/tables/6b544f2f543b0bcc30ed82f828786ab48967bbce3f2a846b08b1921c125799f7.jpg)

![tables](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/tables/70c6acebbb29ef188f2d3d26692da03cffb493f1c1e9643df015999d3aa460b9.jpg)

![tables](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/tables/7cab6a1cf1f356162f25ffbb2a6f059e7d20ca11485306eea2c17ef7b49b44a2.jpg)

![tables](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/tables/8575c64bf1230a078d39ccd17d6b7dc9750ba4d9ada3e67fcfe81d90be481980.jpg)

![tables](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/tables/8d152e9a0bf4649615baacc6100dcd607a15778b86eeeec738fdc468bc9a1a63.jpg)

![tables](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/tables/8e8e3b8735fbbb397b353fa23c54e6a167121892de4d1526052a3e2c610b9b56.jpg)

![tables](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/tables/a518ba0e57de98ea205879c310e76c5fae2733ba3810a78495337be0b8ae6e67.jpg)

![tables](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/tables/ac9bda08b165aad5eafa54f9a55dd72e20ca40dff9ceec35b7ce36f0f478736b.jpg)

![tables](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/tables/ca1aec0abf564ed5ba643092f4f4671a524909c466db61c35ae57916056f1668.jpg)

![tables](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/tables/d0372de7293d88ac558ce4387e3dff5a4244dba5a877a2c789d01dfa0b8dd1ea.jpg)

![tables](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/tables/d9848d9e89322ecf0f78098ede5ad249333418f546d2b9e302bf70b0a0eaf94c.jpg)

![tables](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/tables/e75845209e3ac1fa8c380e86235336c73cd0b422880442a54d2d4af0d2dae317.jpg)

![tables](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/tables/e764f793e66312de28ec2e3dc64fd5199c32d47c9af8cdcbe3d8b983f44fdf12.jpg)

![tables](../iclr_results/4247_Peng_s_Q_lambda_for_Conservative_Value_Estimation_in_Offline_Reinforcement_Learning/tables/e8864eac1fb53bfe8b94957f1f697ec56e9f5088c3b9a15b761f9542f4f5c4bb.jpg)

<a id="paper-4248"></a>
## On Measuring Influence in Avoiding Undesired Future

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=VHdF91MvJq)

### Images

![images](../iclr_results/4248_On_Measuring_Influence_in_Avoiding_Undesired_Future/images/3449b2daed94e8c16ae8874f313c193ff45cdf2713794db5e0cc24c11e4bf568.jpg)

![images](../iclr_results/4248_On_Measuring_Influence_in_Avoiding_Undesired_Future/images/3d10bee17dbbf66dd723cfee3edca0dc6061d8e40104cd3868d9f04f9094ea82.jpg)

![images](../iclr_results/4248_On_Measuring_Influence_in_Avoiding_Undesired_Future/images/6f9181235b1325d75081677d0a1ab8d8148006991507e6c702da16e8f92084ec.jpg)

![images](../iclr_results/4248_On_Measuring_Influence_in_Avoiding_Undesired_Future/images/7ef134d25073fd58dfb22975c75a2a03c1fd4a1e186fe81064aaffacc58cbc8b.jpg)

![images](../iclr_results/4248_On_Measuring_Influence_in_Avoiding_Undesired_Future/images/a0cc0e315d1fe3b697609aad8c4b10218017f90683d40d79070d3f435be6301c.jpg)

![images](../iclr_results/4248_On_Measuring_Influence_in_Avoiding_Undesired_Future/images/e1a8b2540ae82438ed5e84db58973a909eb2c788517a46d8249927198d06d465.jpg)

![images](../iclr_results/4248_On_Measuring_Influence_in_Avoiding_Undesired_Future/images/ed4ecac7e6ae96cf2f28cc0bfe629c12fdadb28b1190232e86a17a0fed10fbb7.jpg)

![images](../iclr_results/4248_On_Measuring_Influence_in_Avoiding_Undesired_Future/images/f847e9dd6ecb6ee3dbef27af89a4d147a183472f9f62656eb45d169a0bf70d15.jpg)

### Tables

![tables](../iclr_results/4248_On_Measuring_Influence_in_Avoiding_Undesired_Future/tables/15160f5625c817bcb523036a3d2d27798357d16447796fcd453cde419c181365.jpg)

![tables](../iclr_results/4248_On_Measuring_Influence_in_Avoiding_Undesired_Future/tables/2e6b80fe2c8a1fd88d96053344b6ffe84ca2c8f82e7183b8b37468fe75bc3917.jpg)

![tables](../iclr_results/4248_On_Measuring_Influence_in_Avoiding_Undesired_Future/tables/c969e51bf669f5203e34cb29fd942d302480fbfd95ce1262eb4d001f5610f11b.jpg)

<a id="paper-4249"></a>
## High-Dimensional Analysis of Single-Layer Attention for Sparse-Token Classification

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=Ae7VWAEIAW)

### Images

![images](../iclr_results/4249_High-Dimensional_Analysis_of_Single-Layer_Attention_for_Sparse-Token_Classification/images/2b672039770a0b3ef127eae8551c88712e3edd92d29a2051d53551e8200e6861.jpg)

![images](../iclr_results/4249_High-Dimensional_Analysis_of_Single-Layer_Attention_for_Sparse-Token_Classification/images/6a13df7c1192d599de5994bd7fbbfac88b87d38d508ef902b58305395c92081d.jpg)

![images](../iclr_results/4249_High-Dimensional_Analysis_of_Single-Layer_Attention_for_Sparse-Token_Classification/images/91a95c08cc5aab5e70734454da0accff5869f0f6c23b38888fb78db355c78dfb.jpg)

![images](../iclr_results/4249_High-Dimensional_Analysis_of_Single-Layer_Attention_for_Sparse-Token_Classification/images/a9a74b924bdbad5e6c5b0abd75a369266122595aab25778134e8e3e0a7e70c38.jpg)

![images](../iclr_results/4249_High-Dimensional_Analysis_of_Single-Layer_Attention_for_Sparse-Token_Classification/images/c90f46a085eb079bf778dbd7cbd3e525a5ffed91549238ca23b8bebbe72f2c27.jpg)

![images](../iclr_results/4249_High-Dimensional_Analysis_of_Single-Layer_Attention_for_Sparse-Token_Classification/images/cac4478bf609c81c135952d9cefe620ec8d71ade125785950c27b43955b74c2b.jpg)

![images](../iclr_results/4249_High-Dimensional_Analysis_of_Single-Layer_Attention_for_Sparse-Token_Classification/images/d72e9a4323c8d2b0474431fce07744a44687396848efa05ff4914952e949658d.jpg)

<a id="paper-4250"></a>
## On the Impact of the Utility in Semivalue-based Data Valuation

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=n5RKFmC8Zl)

### Images

![images](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/images/03ad2246b740a6aa8a0fd42c91fc8abc2267b78f34218936b93e1d10f6032a11.jpg)

![images](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/images/053f332ca22c187b8809cc5a4bb52097e715e538dbd91faced57d59a0fd4ce9e.jpg)

![images](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/images/12a0d1e8d06804c000a286c6f9df7cbe028bb5907cefe571833475a1203c5886.jpg)

![images](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/images/2150255863534b29b8c1a0ebceb3831233fdc4cd1a4b6fde920e580628e00c14.jpg)

![images](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/images/42f935f2781ef87e99c70a4946d420d6c434e7ec26b59366785f4dcf6861eb46.jpg)

![images](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/images/777d8687f986a669eba3b891090b41d077c3b727624abc5c78a13825f3fb5d8c.jpg)

![images](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/images/7bec5272f0c2266f77b10d39d212fb56cb58678d57998dd94a5c7aedd18e915f.jpg)

![images](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/images/8ae01d593d63e222c1f191c0663f9fd496083f20e1b76a167ec8c6cc3764004b.jpg)

![images](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/images/962309b90711e332e591ea3c1b0150e239f39acd8e0e2bc49b94c5ebedce4a51.jpg)

![images](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/images/afdb0de9925ed145328129ac66e6b2c4958473ee5fa72aa88d224a1c4c6b135c.jpg)

![images](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/images/c001429cf79e37147846bc496b6200c0ee2ccd92899a5536ee499f2ba2e50067.jpg)

![images](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/images/cda60fa1a2074fff4404eeb7a7e5bf997e22abc4318e11303279ea8063e3bfbc.jpg)

![images](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/images/d6274afd2d88b420671ff5c2157bea32445291606faf8c5ca08156396c000f96.jpg)

![images](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/images/d909189a5a4e19f942f614c344bd89bc3e58b8e7a576047ff841d0755dba3a41.jpg)

![images](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/images/db70aa5b9dd38aa10469c66be86c9147684e02f18ead724eb8c067471ffdc860.jpg)

![images](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/images/ebb73270e55edfa5ced78889b1c0b30a392be2323268e3be0e3af60fc58fd689.jpg)

![images](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/images/ee4144d74fcee2cc362db8f4dd9e60c80fe2ad477552f1ee697fda8bb72846ac.jpg)

![images](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/images/f1bdff0818e1415c44e14697fd0ef094a43c13a4b9d2f3cd99c35ac452275c65.jpg)

![images](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/images/f9b42ced5ff5faa941895aa331dc8a7e383cc26931659a1afdfb06a3d14ce566.jpg)

### Tables

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/0304bfa6819418febd4514fecb23b5657651069b89ef55478f91eb7727de2c04.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/22bb7fd37ffe31c80ecb87590f165e4d90750a11bcc0a3396b5f8a38014a94e8.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/2b5937112ebc43604dcae259ef81ad3c5ae1fe4c25f25a0a6b3cf569b3129277.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/41fe6ded0190245e26b6d38cb1b3660df2c7ea5cfa30693c50d754c2b66ced78.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/5478007a7237ed51400a688448dce2faa463ee36b3b844996c5c0bfdc4b1d462.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/54ee46d4feba212771f9affe3a2fc768a22a0c045ed05560dc81be830d9d2107.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/5e512be941655fe4e85e6ab416c85c775182c5b8688f89720045ead9bf7ec3c1.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/6d33f64be07175ba6daad8e62c2142148f8601461b5f84f22c161f9cb79df76e.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/9fbbdb48eebb61dc4088ee157fe38f1722fc8049d220d2b0f68239aadc0063a3.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/a053d58ac0ab30fb1ee85b27023966d99a4fde3b5fa6545e75ecb7e5c42c74ea.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/a6f62b548e38cb5ae9320469f4b865d4b66396dc567a25409bd925a6c1fd5133.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/a78ca76dffeb79092cb9e4cc89997882114cfddb3d319a900a3b8c2c976898a7.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/a7b1cfe81de4d08d38f384e962ba8da2edce2b5fa2bfdd7d27de05bd15bdc895.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/a85fc7d488e0b95d1c957fd63d8cc8f213f4dbc132af82603fccd1ec944b0701.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/a9f7b2283e2b9ac8e871ca814e78fb8ea4026a97edab8d5f30fa267e7e12187b.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/b1e3162791fe59d48649cc8f6a49ebd6901a53e0b666c394e93f073c3b14acb6.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/b37457ee5761f296963920d3cc8d9a8b770bcade0c933d05c63929e751db514d.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/b5426362f7740544544966a82a092deb950e03b8fd552bdc285abbf0bb624972.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/bb6594250baf6ece5884295d9f6e1e2aa7508d09a544799e215e725f357cb9ce.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/d50558f1d095af6b2bcdce28c68b79bb620f43d55955625bb2df36ec6aacd2c6.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/e855ec4103f1c51cf87dbc85d4e29548719e1d9e70ba9094b8f6f0065d3a4efb.jpg)

![tables](../iclr_results/4250_On_the_Impact_of_the_Utility_in_Semivalue-based_Data_Valuation/tables/f18b3a0f648d333fb8413fd67b4e88f8ad880b896355d3111dfebbdf1392fc88.jpg)

<a id="paper-4251"></a>
## Escaping Policy Contraction: Contraction-Aware PPO (CaPPO) for Stable Language Model Fine-Tuning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=vDlkJewkDu)

### Images

![images](../iclr_results/4251_Escaping_Policy_Contraction_Contraction-Aware_PPO_CaPPO_for_Stable_Language_Model_Fine-Tuning/images/4bff3d8409a741f2ce1c34b7e91089753635fff388875ee642da760e42332b8b.jpg)

![images](../iclr_results/4251_Escaping_Policy_Contraction_Contraction-Aware_PPO_CaPPO_for_Stable_Language_Model_Fine-Tuning/images/9f700d06376fdc3a7457e7fca2dc01003127af04e4ff0e20f207d3628f769315.jpg)

![images](../iclr_results/4251_Escaping_Policy_Contraction_Contraction-Aware_PPO_CaPPO_for_Stable_Language_Model_Fine-Tuning/images/b26d52489db8dc92df1e0397ef31403aca4042e510945dd981c23eb481960dec.jpg)

### Tables

![tables](../iclr_results/4251_Escaping_Policy_Contraction_Contraction-Aware_PPO_CaPPO_for_Stable_Language_Model_Fine-Tuning/tables/0b7a448e57a73177404ebd4e1ee2d85e351cc12339e7d7ebd13d6eee4b796bd9.jpg)

![tables](../iclr_results/4251_Escaping_Policy_Contraction_Contraction-Aware_PPO_CaPPO_for_Stable_Language_Model_Fine-Tuning/tables/130f46e3a16c678dbedd25ec3c69634c1014f045e243b4dd97ff22a648fb7eb0.jpg)

![tables](../iclr_results/4251_Escaping_Policy_Contraction_Contraction-Aware_PPO_CaPPO_for_Stable_Language_Model_Fine-Tuning/tables/29904dff3a6f991c41e13518b9c1583657c9998417588c529226cee62b856f1e.jpg)

![tables](../iclr_results/4251_Escaping_Policy_Contraction_Contraction-Aware_PPO_CaPPO_for_Stable_Language_Model_Fine-Tuning/tables/38667f7b23ca5d22704198d51aedbb201c75ae812eca38da42bb565ab312c859.jpg)

![tables](../iclr_results/4251_Escaping_Policy_Contraction_Contraction-Aware_PPO_CaPPO_for_Stable_Language_Model_Fine-Tuning/tables/4252718ceba9a5a944f3b4628bc17a7c3568b2483725cc019f2498a492194c26.jpg)

![tables](../iclr_results/4251_Escaping_Policy_Contraction_Contraction-Aware_PPO_CaPPO_for_Stable_Language_Model_Fine-Tuning/tables/43b634724cdf269ea999cf30493faed52e79e0daf72006e32cdcd9aab8c2e686.jpg)

![tables](../iclr_results/4251_Escaping_Policy_Contraction_Contraction-Aware_PPO_CaPPO_for_Stable_Language_Model_Fine-Tuning/tables/48de25830b775c09634b0f76a1c53f03e410574f7a6d56b5ed69deb9bb3c04fd.jpg)

![tables](../iclr_results/4251_Escaping_Policy_Contraction_Contraction-Aware_PPO_CaPPO_for_Stable_Language_Model_Fine-Tuning/tables/6849c2eb4f9a8871e8560b0f215a711e26f93346dafea03a2c0c42ed4d6d3fca.jpg)

![tables](../iclr_results/4251_Escaping_Policy_Contraction_Contraction-Aware_PPO_CaPPO_for_Stable_Language_Model_Fine-Tuning/tables/6cd096087b03a711682f1c580b8412bbe835f2cc6ec9eafcfb5e48c6e5d2ee42.jpg)

![tables](../iclr_results/4251_Escaping_Policy_Contraction_Contraction-Aware_PPO_CaPPO_for_Stable_Language_Model_Fine-Tuning/tables/93d75f630563f75cb27f7f3604dea88b3df45ad6dae6cca213c8bdfebed0417c.jpg)

![tables](../iclr_results/4251_Escaping_Policy_Contraction_Contraction-Aware_PPO_CaPPO_for_Stable_Language_Model_Fine-Tuning/tables/9dfd2e9467a6177191471c2119e57c9109c34606375b8893cbed97c465c99216.jpg)

![tables](../iclr_results/4251_Escaping_Policy_Contraction_Contraction-Aware_PPO_CaPPO_for_Stable_Language_Model_Fine-Tuning/tables/f5a9a40b1f3fb17c45c0e06f0c583ec300921961aaa903e0e0f5565a5b921bc9.jpg)

![tables](../iclr_results/4251_Escaping_Policy_Contraction_Contraction-Aware_PPO_CaPPO_for_Stable_Language_Model_Fine-Tuning/tables/fd464cca41649fa0875ccfd8de48ef425317f7d95fad765d0eb7ba6dd8c5ea59.jpg)

<a id="paper-4252"></a>
## Streaming Autoregressive Video Generation via Diagonal Distillation

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=X7YW6STzeL)

### Images

![images](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/images/1993236a37088314cffba83881b6e618c8eac8d1cc5f4823e134c8608168f969.jpg)

![images](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/images/1c3fdcce55a9fd591323d511e2dc7c762a834c4114855229a86396e595e00ca1.jpg)

![images](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/images/275a824cf8370ac4b4d3f4f2e75ce698ca1c7fe5859d2b84f8eea37be8a99650.jpg)

![images](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/images/401199e115018a2a5701cf5724c165c8bb7929f58ebd2696236843ec39f72f27.jpg)

![images](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/images/4d89f2ed671b38c68bf4d6be0629eae121b20d44aae0855fa9106612580843c6.jpg)

![images](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/images/5dfc9e44bd334a635adea7fbeb966a453413a6b4e6992037f0dce5efb19d5a6a.jpg)

![images](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/images/95fc7de1dc12d9e36c7727dd0017216503aeeef68d461101eab5e88d98ef65ff.jpg)

![images](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/images/c2959ee6f4d4ee4cb1309c579acbcbb5d500872cca96427e6bbfed8ab2243029.jpg)

![images](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/images/ca9b227e4c355caeeac515b74e76d847bb0b281b19ee50577e56696b7ca99e7b.jpg)

![images](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/images/d06aef70344cc9afd9f1ebccf43688bb30c6e215f3255344307df85417d1d645.jpg)

![images](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/images/dd7d8e375d4223fa4c36c8421eefabdc879334f3a43c01a654d566b57be82a7f.jpg)

### Tables

![tables](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/tables/1d0d57b3ecb62189391869ebbb8bb4e232fc68375bce66018840fae41a94b519.jpg)

![tables](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/tables/27783e12aaa4f48121e33cc7da8ced43f2e3837f51f04968b90a352f92d66ed6.jpg)

![tables](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/tables/3a0a33c154dfb9aee254ae7f6bec489af3df64e3bcf81cf39852f927120c9187.jpg)

![tables](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/tables/3cb6f7797cf492f9a3d3a00eacbf0411470507df7b2fd31045d9bdb1667cea48.jpg)

![tables](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/tables/446dc26008da04d69b206a73cfb874f5f906b957cd006cd56bb289151dfd4c25.jpg)

![tables](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/tables/4b91b2b8f966f07787a1cfe42b38db1ece5e4bca813802194af6ba7883e75735.jpg)

![tables](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/tables/5d3d579cc8359a1383b1989538c50a9019341844a69d0bcd3d616d2565e18e9d.jpg)

![tables](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/tables/68c8968f1a4660c1134808e8922149f44156315058160539070949dc73e568ae.jpg)

![tables](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/tables/97df6ec65c0cb2831a8136847df1ed91a2cbfc2584219f9ff798faf9a419cc7a.jpg)

![tables](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/tables/c4faf00a1d40151c7a7bf3b5f768a48d0b285505496f36cbcf8d5b406807ba45.jpg)

![tables](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/tables/cde8c0d9fce5476cdf2b8df8fb03f099008e2a7b9209db7bfdd3b9b73ed01a65.jpg)

![tables](../iclr_results/4252_Streaming_Autoregressive_Video_Generation_via_Diagonal_Distillation/tables/ea74a016ec916137ff4f8238f3a78a1201df47b40bc4f687c1c6856dfc928b53.jpg)

<a id="paper-4253"></a>
## A Two-Phase Deep Learning Framework for Adaptive Time-Stepping in High-Speed Flow Modeling

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=d4gzLgGl7I)

### Images

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/0e8bf7a76dac3866571dc12ba1426601abb703321ccdf30420b887cb2d2ca125.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/12e0d4f9c1ffb601f5117e42e63078d04bf7f8f9442cc392462786df43112524.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/18c07ee6973ee7d9baec399a13f4e05daf9f5717aa2e2bf7f1e92799834a93aa.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/2326b08781c2b4f1602aa317fd7dc5e4d47296e628d882bd54a7bc72d622226e.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/232a273afb0f7fdcacfa69516f917c7ea332ad605b5d1cdb6175cc044d418d83.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/2adcd5e1e53d3e677e347f8b955b19d00b69844a24a5ca90f38fcbf6294ed62f.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/2b640fe4a11f10f5440946de6e91928f8e01ec48334c236c95ade5ae21c7b47b.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/2bb80e5b5b468122343e3c40264102c3ff2a5baa43f0ee3ee7563e43a483327e.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/2da7820bc07118b93e842dd6e2d973f2bf2cd895264d11b3857da7b20b642530.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/42d8100716b9e4d5a078d1c9a57648930e96d6648397dfba2ebf9520bd4306e3.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/542ba792cb0a16c170b7f890e53e8d8cd8e820505dfabc328e03f860a22211ac.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/59bd4cd7dbf58d5491942fcea09a91c8dcd91db6188b19cb90df1e4fe7faacc9.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/5b53d25b01cf61511f73bc0fe0e98bbec4920decc778aa54f4310694f396e959.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/5c8b8231ec531720a8ed15f2281fa2cd810eded97d23e9cc596d9ca5fe1dc66e.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/60c289d2b4be1f5b112388a4424460d3f1a8396378efbd5f0e8089e28dad4a50.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/61444b1535827552d8e467923725892d1d9c5e2d2f35cc2d480a8c1f5fec5947.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/68d8d65725431c67368d8e932fe2c5f1470e3e0f4f3fab5e799008c6fdd42cb8.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/68fc49e3b2a202152d3978e7e4b417ae39c633e31e6f1498d791ef255b6d505d.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/773be2d5b611766b5cccfa75d160101dca83a1ef08858a9755583d12667eaed2.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/78dd6c9093129e7c1930c9dfda4d4902400fe6b442c66a490c791ffe9e03cd10.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/7b46716d72ff81f65461c2f05600d73caafba3ce27ffce05e930eada33655941.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/7bd96d8d4ab50fa28a31d2ea49d0a5360f3c6042536ed5f383634e3a8abc85df.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/830f369625a7fdb1f6aa40a67e0ab2be9f6f6a8528dca9cbfbf222dbaac06a52.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/89df01afbd98226c1e14580db5f84cf09736fc041f05d569b7e1feb5d2016335.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/96be12d5b6297b3a19b58faa561d57cb5fd60cf85f3bfb1175c2207e20bd0947.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/9b86fd0011b4d9489d3a3af8a7e83f4ef1465a6c8cd55cd8d495f5c0008caa5d.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/a007f56d7bae8330fbf6fe0cb6e396f53c88c92cf56d101d198042dec10d1ba3.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/a25fe98fca3e50a9d1ce1a7461cedcdc0d5e418cbdda349bb8e5d9de04c4857e.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/a46f33d3db04b93864b93e7c4ab8e3bf5a8d2ddc698e1f04b61d00c04b2739b9.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/a9611694fb28ca8de016b185af4bfcba133c2f5b21b29a714f03309b602548ca.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/aff1121de272440470e01809b32ec3e970ee3b8023dc70c88f4e80f565862022.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/b014ee03404532dcddc2e36d16b1e2135564865cf9c46730442d544cfd2cf046.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/b31383492ab1fc44f4e89ec142b52494b36ce6ea38285dafaf895e67f0e8b525.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/b31daed1fb5f913431cf835977d2fe6ddd10a01bdcb95b0a5144716819d9b396.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/ba051831e35c20d677687a8bdc4e1016bfafe8de992630f4197ac7246f4b3b6c.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/ba95d1dca2d9e51d0770c92629fd9cc030b536aadd7578b70478d528f9672784.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/bbcb649765e340b56f10cd8b2d3d81ed61671f20fb20a18ce5c73aed4b25b7c9.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/c3e86895de907d4c96ce7d0df6c84d0a77e114c0bc08676e9c833715789aee60.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/cd42c57bf82f41975586bab8019fe6f778b3d4d652d17f8c420dd00f19102e76.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/cdce7d1ce8ba1ce4741b329a82cf3d58f760c93f54efe8cfb09fc3f239442dfa.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/d5b7777bb8e574bc9e8b29dcb3540ad01cba79a96c30f784e97abc99b9b90b39.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/d711ad9ce16d583b35245098ec41e0238b03500f3094b33bcff93c1bb7157f7a.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/db954a11339a63af15bc1617c911c3a64a96d8997ce5571c84cf6e58ff84cc3b.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/dfb78827ec3742c4fa77725e25b37593616364d977a44a2f740814f8f0a17e9f.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/e4cb0269848b4bb7eb1a9934d78a61ec651c29aaa22a00633c4457d6022bac05.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/e9c913e8e49ad5efc18116681e9582c93b6b5d91724abb5f5f0aeac1955fe853.jpg)

![images](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/images/f2e2f640f3a3ce4761c1f6afeb120e672860fd5d7e0ed86d2ea41b5cf50a7c6a.jpg)

### Tables

![tables](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/tables/137dcef2c254303632330a622bc34cc899a43a078384ca2109373926e32b71e3.jpg)

![tables](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/tables/1407885c039375de1f55cbfd36070c9e08bac0dac79275d06dc188d5f909ba50.jpg)

![tables](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/tables/2a0da03a4173196755993a2ae474cde84b7057442d7c9c3dae0a02534c0e47c1.jpg)

![tables](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/tables/2baf46eaa681349a246d20b490250cb1cd9f0bac726c2a7bf70762702f045eb1.jpg)

![tables](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/tables/30be14e299aa8b44361afd40406436c843fdeeb36c031c3a561c2f86db5b0082.jpg)

![tables](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/tables/38fea96c007c98202bdc56294453050c25b0448ad1877f8fd21fb5c623f7251a.jpg)

![tables](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/tables/4a4ce92da13b2434935e240fde7804a7955d6ff0fd9d38d6d08aeecb72e1f76d.jpg)

![tables](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/tables/51566da4a7188a5fe5f520f3e6f985ca0a03ee6bca9bafa1a1c4a62defb45c12.jpg)

![tables](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/tables/5cfb2634242775812cabd0e9aeb025ce9bac9458edda9fee20f5cf4578f0cc80.jpg)

![tables](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/tables/5f4196bd2aa8e1497d355557e5a874915243b864e5ded2fef5c1b39efd9ac3ef.jpg)

![tables](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/tables/6fb7ebe962064b3034da242cde11c7e90ab567f44a0a08dd8ffb6d99534bfc49.jpg)

![tables](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/tables/7835bde1f3142bcc53397936c004f0b9a1eac2e74c64c17a60b5eae23c1bf013.jpg)

![tables](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/tables/8cb6e4fb1aa994803bb36de972419670bacfdd5744bc2dd070974a9a5b9ad980.jpg)

![tables](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/tables/a120d89a867643e7e56cd776108a01082443f8cf81ae1f146c9083d83aeb25cb.jpg)

![tables](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/tables/a136527c91dff58ec990db0aa8126fed0bd54fb25ec9c5b5eb796b611820a131.jpg)

![tables](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/tables/cf61161335ed82270a0565ca594c5d7c8f55783b07fb36d6fedd75aa3982301a.jpg)

![tables](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/tables/d5543b61978edbbadc75ad5ff5cfc56e608db01fc015c48bd56b27e64ca58def.jpg)

![tables](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/tables/d9355be07ddfc0c4108517b29e4e0f4c5b1d3eb8c3e15024c2a85257ed877d28.jpg)

![tables](../iclr_results/4253_A_Two-Phase_Deep_Learning_Framework_for_Adaptive_Time-Stepping_in_High-Speed_Flow_Modeling/tables/eea43c95fb4059189724928c0fc9e457dfdf9d16f9f34f69697b063608896549.jpg)

<a id="paper-4254"></a>
## Cross-ControlNet: Training-Free Fusion of Multiple Conditions for Text-to-Image Generation

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=89j1hUxOiF)

### Images

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/0025406fd7e18391e4e1103961475a101e8a87152290b74f6eb4adbde59055e3.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/004a1e00d8f6d3e246fe4ab35697c17958e8c6f4e3cd89941731ef1a97fb70a2.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/07948b14adc8d0c5f2f27cae958c793b98d163b8fd1a2e88a9ea0957bba342c2.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/13066ef3f7a1c1a7cc1118c1f1ed248bb96f925bf208693df0ea882ca5bab2de.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/185772dc2c3991e9390d6059064e53b03f9c59ab3042675955f8329d665c973d.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/19bb021a94986daabe63bcdad2d6e350c294bb2caf2c72a78791843e1b268698.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/1a9890ff56aa02856c59eefca57acb553ea9ae1ff90abc233f6142e6eaa191f8.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/1e117c484f03c10370da7ba36279e7f980dbef7b4b5a38e9d01aac8262ab42d0.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/28bd971bf22eb0926ec9e3a316376f730acbe2fbcad3c3206257751959d9f1bb.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/2ecea6b072acc4ea209282eb6c39420d6fe5c014bd4cb771e31cb6655e8f8fdf.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/3103da3809b87e88b5acc2ba9cb17a9fdb84abdae071cabc9a271854afe0e38c.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/314aaf16f701d581e92ac9a359a9a77cdd015f14f23208ffc5b3e00ea0da7d71.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/3bdc06b3d7391d661266c2f52ff6d4b279186922af13c008dde43f402361c773.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/423942d60bde6de731ade5a5677404d2154660619311b984e24202c5d60eac60.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/4781c188337cd44f3d9592f580e07f199b961ee1a1bf792397064cd263d47b8e.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/4c62dc3bae108126ba12d41fb9af3e43ae47de80ecf4e8b87ad177f44b93de6e.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/4f275429c4ada2827716ca9c3c5e70c04507c6949f5a79393643460180a47b32.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/5125085bed464a52a5b860925ea3ba105410a7f63572b5664d87258c15739522.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/584b651c99eabdd858c4e8fceeb462c1d5c1dfe4b0db2470b48fd7a1f8b43ddc.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/60dc57249a1511fb6a6a4779ce93a2e387cdad3f4bce29ce6fc1b387abbfa184.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/6194e9e7b2b8455125170e0969c0c0216dad3d8c26d4f48df91839acab74a1c6.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/62cfc2217dde17ce3d137d55d31cffd3fd25314940cf977071d3cb925dbf17bf.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/643f40715c756879e48581164b00bbfc10369cb2654b610141cd92019802e028.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/6747fba2592c1346af3e6eb7a9c670ff512939f7e985c922d70436e1d70b9e4c.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/68ab1ab115d242527d2fe1c5ae6525c2277ac5cad8efedec9a014e9cfd4e9428.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/6d8dbc115d54aded5e742e05986c0924fa191eeb4485aaf405d4fbc49f591bdb.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/79090e2f4f93de4728a5e04ce1312ad81d6d042d85edbeafa114dff0976e79d1.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/7aa22c04f0188cad596b71ed51d4e70c8b6b632321bd10d4bf092d965bbfae59.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/7c4a41ba393468cb72023997adb2a56800a460f02062ca54cf58d8f4cfeca975.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/805b859f0077c98c409fb8094bf76a5dda76db57c921aac11e4478c6f6abc544.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/82a0a726dc88f2aeec7d50c6f1f5154803986977ca41d942c776383d93b7f259.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/852a0275c913536f12d0315043a1e0fb1003087143c57fdc570c7327ad374fed.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/88c45e0213023cfb03b5dccd004d98c5f6c998585407611d97e67bcc080d79ba.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/8a4e8d40f6ccd586e6e50e346ec62cd158cfc7d9200d78fe4c2c580657264d8a.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/8fecd51eee316a7403788bece560c5ee9411688c6d5bbe07abdbff4cd5516d21.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/90d83e4eaabc25e679febacb70821fdfaa0cf02dba8ad97e5e41f775b32d65c0.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/91a5a690f234a971572ecf5437b5be109552bcadc617fff1a27125bbe52d3317.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/93105b25dd8213d41c4c8cd468f929d097e003e0f722a3c374a582250102b182.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/9686b7a3bc49d11127492b6fbe90b5a3d02b8cb22267cf5495075605437ff8b7.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/a11910c400f4b4599efbf89d59c8d36d731ec0ae295185a5a2ce4affbec31f76.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/a86f130a61a185778b807d6a972a100e7ce89bee68f14f0eacbc8325a0c9299d.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/b2481589969d7d0fa21415bf6b0570a97f2bfe0509bd2a22aeca11ca9a22bafc.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/b54a9a1f3b18549e1e1a507e6af386476c6fd0eb447de53467d48de66f6ca7f6.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/c1f28fe229c982051ed54459d5620940255abd36d83b8c833b4ef4a433406fa2.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/caceccce34bbfe663a619c9ace6f37f61266341ac4facb35cd6d2aec23f53b49.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/cb20634de136638f6b4449ee685ecbdfab950204edfb4d2a932a3a8686fc9ae7.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/cb8ffc2df9cc607b5f9b6cf15382aee22bdb34fcf0a53358940e92c423b37eda.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/d7a98dd3fb8fc3950e9ee21907202d615c77d05b99b1f2bed94bbcae1d3bfa11.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/de0e2183760dec261a0996187578ce3ecffb17dbc43302daf18e2674bf8075b9.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/e10461c74f2fe335982b0ac562a598c5c7af47f709bb4299ce7194fe3a61f6e0.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/f684b465f3dca06733e4b5c970a30856b4aeb5b504f9153dcb7049e97c6b1a60.jpg)

![images](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/images/f9fe6e1dfd82e5976de4aee9806bf4be5a88cd3374a5f6e9a7ced896d64439ff.jpg)

### Tables

![tables](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/tables/0a573d6c93ad77889e2a7fa1e0e6ddab13b0240c6b62a10d939e90f54326f2e7.jpg)

![tables](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/tables/37968154cfd79c1069a4f74c118b5a3dba865226dbd65e21b866a71ee90de1b6.jpg)

![tables](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/tables/6b020b3d577e8ab10d01410470d5e25a3a89d6871cea28ae9fe0c12c196836dd.jpg)

![tables](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/tables/7692a4fba779c09c83cedbb7816922813a26360e91ef440cb96a479b71f85b7c.jpg)

![tables](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/tables/8816c3b44fe768074592109cc3728b1a7b4c611a62e8fc0d27ed427d1ce08732.jpg)

![tables](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/tables/c56f5d2ccdebb7b52ee1457eb091ea99c8f00528b3c25f02f7d79d75c010506a.jpg)

![tables](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/tables/d67552c40c42907acc00de89b38cb30e6d0ac4cceaf2c780b80e97500b4c9243.jpg)

![tables](../iclr_results/4254_Cross-ControlNet_Training-Free_Fusion_of_Multiple_Conditions_for_Text-to-Image_Generation/tables/f223593c12898ea4f647917a1e584c5fa6e0df649657ed92a300f33323d4a5d9.jpg)

<a id="paper-4255"></a>
## Beyond Softmax and Entropy: Convergence Rates of Policy Gradients with $\boldsymbol{f}$-SoftArgmax Parameterization $\&$ Coupled Regularization

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=O93c9H4SXc)

### Images

![images](../iclr_results/4255_Beyond_Softmax_and_Entropy_Convergence_Rates_of_Policy_Gradients_with_boldsymbol_f_-SoftArgmax_Param/images/5616dcbe6fb7273134109adde61fa82cf9685d37eb3c3adf4b11b9ba6797d829.jpg)

![images](../iclr_results/4255_Beyond_Softmax_and_Entropy_Convergence_Rates_of_Policy_Gradients_with_boldsymbol_f_-SoftArgmax_Param/images/5b81b24a171f82177caf13e3e780dc5ad75f295117043f7b92b6f91fadf96e40.jpg)

![images](../iclr_results/4255_Beyond_Softmax_and_Entropy_Convergence_Rates_of_Policy_Gradients_with_boldsymbol_f_-SoftArgmax_Param/images/e051530f4242d905881c1c10eeb9421fa4dc30bd08babaecb0e1103a6695533c.jpg)

![images](../iclr_results/4255_Beyond_Softmax_and_Entropy_Convergence_Rates_of_Policy_Gradients_with_boldsymbol_f_-SoftArgmax_Param/images/fd69d39edf8ec97e434c15edcfcb3070ed59503b4ecdf55b248c54a41eaae813.jpg)

### Tables

![tables](../iclr_results/4255_Beyond_Softmax_and_Entropy_Convergence_Rates_of_Policy_Gradients_with_boldsymbol_f_-SoftArgmax_Param/tables/04b12cefe5608743839a2f4353890b47359015ef8860682f182b4d299f4d18c3.jpg)

![tables](../iclr_results/4255_Beyond_Softmax_and_Entropy_Convergence_Rates_of_Policy_Gradients_with_boldsymbol_f_-SoftArgmax_Param/tables/8ddecf322f2be7a1dd2554303f6db7033ce68790db0eac2763b31279545e6ee6.jpg)

![tables](../iclr_results/4255_Beyond_Softmax_and_Entropy_Convergence_Rates_of_Policy_Gradients_with_boldsymbol_f_-SoftArgmax_Param/tables/96d3d7dd49dfbc1830ec71f37e582c1a0f011615c7416520725fb5857e777a3e.jpg)

![tables](../iclr_results/4255_Beyond_Softmax_and_Entropy_Convergence_Rates_of_Policy_Gradients_with_boldsymbol_f_-SoftArgmax_Param/tables/eed640ab8245fdbfce55710593713ac333b3a9315943c98319d929ac9e8c3fe6.jpg)

<a id="paper-4256"></a>
## Reasoning on Time-Series for Financial Technical Analysis

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=PcjIe5xNaY)

### Images

![images](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/images/16d11e23b6f9fe19f3db96bb707e0e8209f924f9b67012048e19bd01f7f3b854.jpg)

![images](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/images/3155ef71b78ff4a7c87a2c58c2afa6969d581ea2f9e9e29bab8497a4185661da.jpg)

![images](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/images/3e3f76269f736c2ae7128c9c1f288dca299d13698e493f9b90193972bdcfbcb3.jpg)

![images](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/images/421b19e6bde5dc5a1a244b649cc5382a20474a51a6d4afdfe4376b177b7c549d.jpg)

![images](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/images/6170d7932a4fa3f9152b5584b7eaa85be50d735a2b4c55b4f4ab060425712765.jpg)

![images](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/images/85da4b8e6987ac2c20390fd9235ad137c2718a08b4bf6d9a53bb92fbc6d20ca2.jpg)

![images](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/images/aac8b71d34be3d07788ab658b7a3cace4019f45d7a0def23bd5b37de550d530b.jpg)

![images](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/images/b2d30b660ce4a1b27de72c70a1eeed50a9c58fd18935a2753675fc4a396f66f1.jpg)

![images](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/images/dac4facdb86a13cb53703c5c2db370e5dd1742caa5d30a3ce27ded8bc584f7d9.jpg)

![images](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/images/f433f55f66699514c396edba90bc0677f607eef5edab1e75e3e2935f97028e6e.jpg)

![images](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/images/fc1309c7c3100cb35259c2b1cc5efe70bc87a65af9a36406cb83124715a8e905.jpg)

![images](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/images/feec51163af93f2e817e4aa15abf74a9af3fe4f348dfbd13a08d522e7db5a58e.jpg)

![images](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/images/ff56c3ddf611925507d278533ce5e4904aaa5dafff947d604a567e01eab71a00.jpg)

### Tables

![tables](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/tables/060e0b2f1e48f51d967fad4d6a10f85f912429b58a390a84521b130dbbdbb989.jpg)

![tables](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/tables/194438ef6849310183b104627ee9667f59a000aedbb1785cbc76543ece4dfc16.jpg)

![tables](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/tables/1cd92f0bbc210246863e379a6d6ba95e53a7d145262d7c5e43ceeb641a1a0994.jpg)

![tables](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/tables/2472a552fb1adaae948405f9605d1c18d2d72845c7c8f033deea08c34d76fec6.jpg)

![tables](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/tables/391bc2c1d43e0fecb245df6d98e7d75b0b0fc7a1e947b10401722f180bcdd0ff.jpg)

![tables](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/tables/41e44b7253ff5a3cefb235730a8bd1498f5058c1b9a85c0848c725d933a0b8a3.jpg)

![tables](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/tables/5e20de65f5be11b30ccb253be5d860ec75fade0416b98c3370a76b84670ce107.jpg)

![tables](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/tables/685b0731728532475ecb35ce0e227cc3927650d1ecaba041def92b50cb5bbb3b.jpg)

![tables](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/tables/bdcadcd9dd99b5e42f1106f93c3a42ef23d15979ab9a52c3dfe733ae5b8ed56f.jpg)

![tables](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/tables/c31da9e6ad6d5fa33dfb4586a8e753ae8d9a5ce97cd83c78b39e50189171ac89.jpg)

![tables](../iclr_results/4256_Reasoning_on_Time-Series_for_Financial_Technical_Analysis/tables/ca96ffbdf662faeb53292f72b5b3e8d8725d74853dae4720accc6bf552b91797.jpg)

<a id="paper-4257"></a>
## Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=mruL1LDjzV)

### Images

![images](../iclr_results/4257_Mitigating_Spurious_Correlation_via_Distributionally_Robust_Learning_with_Hierarchical_Ambiguity_Set/images/172cebfb0e6538aa036452eefd0ae2a9a138790be3959cfa7be4b1fa7700762d.jpg)

![images](../iclr_results/4257_Mitigating_Spurious_Correlation_via_Distributionally_Robust_Learning_with_Hierarchical_Ambiguity_Set/images/27769ad573fb15d01f81d1d6fbacc45e52afec11b296f866b1a1672ad9e07490.jpg)

![images](../iclr_results/4257_Mitigating_Spurious_Correlation_via_Distributionally_Robust_Learning_with_Hierarchical_Ambiguity_Set/images/312f82686dfbaec77e9b731826c9701e0542f7ce526ce68ab1186db6dfaa1145.jpg)

![images](../iclr_results/4257_Mitigating_Spurious_Correlation_via_Distributionally_Robust_Learning_with_Hierarchical_Ambiguity_Set/images/3806287178522d08e3b8e1f51949cede74c107a9b571db884f45878a6650e55a.jpg)

![images](../iclr_results/4257_Mitigating_Spurious_Correlation_via_Distributionally_Robust_Learning_with_Hierarchical_Ambiguity_Set/images/4221b2e01c59b0eb18a5eeee844ed535891266a6dbaa01be8bbecd91cd3f09ed.jpg)

![images](../iclr_results/4257_Mitigating_Spurious_Correlation_via_Distributionally_Robust_Learning_with_Hierarchical_Ambiguity_Set/images/45c6141eb5ae87790fc815f242523a9a140160f2c5633889ae817ce490b7cc71.jpg)

![images](../iclr_results/4257_Mitigating_Spurious_Correlation_via_Distributionally_Robust_Learning_with_Hierarchical_Ambiguity_Set/images/548c9841678683b69a59f24fa73a207b298538bbb7dd37bc026cc5e3639993ac.jpg)

![images](../iclr_results/4257_Mitigating_Spurious_Correlation_via_Distributionally_Robust_Learning_with_Hierarchical_Ambiguity_Set/images/729e43e0353737d88f9a6e103e760689d2e903788c8cf5d0197b99973cd41d1c.jpg)

![images](../iclr_results/4257_Mitigating_Spurious_Correlation_via_Distributionally_Robust_Learning_with_Hierarchical_Ambiguity_Set/images/73625eabb13c760761b66abee415722d9773617f019a9ec2e071765321c1d325.jpg)

![images](../iclr_results/4257_Mitigating_Spurious_Correlation_via_Distributionally_Robust_Learning_with_Hierarchical_Ambiguity_Set/images/7bed85716023fc36fec19310ef7ba96c364cf9600b7640de1d107393145548c5.jpg)

![images](../iclr_results/4257_Mitigating_Spurious_Correlation_via_Distributionally_Robust_Learning_with_Hierarchical_Ambiguity_Set/images/a65ca30c3b4754df12c74f1dcda375ad8e1539685a8fd9c26fe53c318ffd7ef1.jpg)

![images](../iclr_results/4257_Mitigating_Spurious_Correlation_via_Distributionally_Robust_Learning_with_Hierarchical_Ambiguity_Set/images/bb070617d6a984a4c6febf8355f7ee6569d167e8f6318c99b0db91b6077be72b.jpg)

![images](../iclr_results/4257_Mitigating_Spurious_Correlation_via_Distributionally_Robust_Learning_with_Hierarchical_Ambiguity_Set/images/bc6a77ef5e259569cee473c3a0b69b7f79df9c9008dc757e5d3f0bc682bbd0bd.jpg)

![images](../iclr_results/4257_Mitigating_Spurious_Correlation_via_Distributionally_Robust_Learning_with_Hierarchical_Ambiguity_Set/images/d86e0fea069ede73d0048492d3aa213d896bae9522ace666c64678d0c7c17f61.jpg)

![images](../iclr_results/4257_Mitigating_Spurious_Correlation_via_Distributionally_Robust_Learning_with_Hierarchical_Ambiguity_Set/images/ebac5e13ac0d85bbbb8459e48a39414a87e4a3d9074d9eabd9f533a3a7026424.jpg)

![images](../iclr_results/4257_Mitigating_Spurious_Correlation_via_Distributionally_Robust_Learning_with_Hierarchical_Ambiguity_Set/images/ed44644985515dcf6b89b62ebf1bc1b0e53844a0ac12ea7c603b8d2330ec8ecc.jpg)

### Tables

![tables](../iclr_results/4257_Mitigating_Spurious_Correlation_via_Distributionally_Robust_Learning_with_Hierarchical_Ambiguity_Set/tables/341b3c2a46d9179844b172372aaec9cbb29169b624c6187eca8c94c5c55ab0b4.jpg)

![tables](../iclr_results/4257_Mitigating_Spurious_Correlation_via_Distributionally_Robust_Learning_with_Hierarchical_Ambiguity_Set/tables/3a61f4587d773eb8e56932afd0b2439e28ea0981bd6bfb3a7ac760068ed6688a.jpg)

![tables](../iclr_results/4257_Mitigating_Spurious_Correlation_via_Distributionally_Robust_Learning_with_Hierarchical_Ambiguity_Set/tables/4c8b2e0f8f67888fb783b29f55deb9306704469f174d69536c438c93905693ff.jpg)

<a id="paper-4258"></a>
## AP-OOD: Attention Pooling for Out-of-Distribution Detection

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=fEYonozhKk)

### Images

![images](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/images/05dde63fa5e201014689b823feca6939c6d1afbff2447b3250c10230bcff3f65.jpg)

![images](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/images/1de185f05f1d48179e2bb0369587a9d8ecabe3d794ace1762e9e02c79708e026.jpg)

![images](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/images/25ece8fc61c2cda3900a0cf47abb155ae62fe033ef96c2790ee4df3e44ca00a5.jpg)

![images](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/images/3ed2811887d65414afb1df3fff4da163118e5377f8dcd739aaea130e9d009e88.jpg)

![images](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/images/4e2c3f9a9cea252afc6f3d84b71e7238ba94212853acac24bf09fb1013ff2ec8.jpg)

![images](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/images/55da0d5dd0e4f0a4d9effff5f38bcf00e9d5ac6e7fe5035ebd9e8b0abfffb363.jpg)

![images](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/images/6849db09d432219f3ac0ac9a06171a0c22ce4bae425f4cb70ef11d264d7b9cdd.jpg)

![images](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/images/703c9873187341b36323c9c4245845fd4ba13e48634939f1cb5b68094f3cdfa8.jpg)

![images](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/images/a2294ec70004f5e5c3faf98776ede44b8d4e853cf1bd4426e09d254ec9faf26f.jpg)

![images](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/images/a4eae7952294e9abfbf9654ee756d7158de9c528fe4820ea62f0c716209e8191.jpg)

![images](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/images/cd128b09f534a7a3bcb53f93f134c7c428deabce1ab69e8b1d7175e2d6e7822b.jpg)

### Tables

![tables](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/tables/02715e5843371b08b1baddb03113c1b6e6a6f985399d96811265bedbc9ccf54f.jpg)

![tables](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/tables/0ab5c742d90b06516db2a2ad7745e91e297e9298f60043830e3f53b16475b66c.jpg)

![tables](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/tables/3f09a21da533016ef0142776aaf0dbfc558afcd86a38d4c4a1b250bcc4433e6a.jpg)

![tables](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/tables/7e1fbf7676f2c301ed9dfa51198daf9c6fb715bce050f35e05e1f67379d3d490.jpg)

![tables](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/tables/8a80c3a15bd003debdc54c369c2d140a94a86fbc71eed63663067553bd91bbc8.jpg)

![tables](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/tables/8a98b9e25d061f1bef9f3b69ae8583ae404079281d7e5bdff6e08aaf1978613d.jpg)

![tables](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/tables/a20c4d448a45bbf39abeb651663526835d51733afa9df11c6d3301c37e3a07d3.jpg)

![tables](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/tables/a7558fe18d1e463082f61e2e5f3e1c48bfe0bbdc98ca0510911861b99124bf18.jpg)

![tables](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/tables/c0a8d02d05d0edd9045f69d08d164e6c1bfc1b4521deb38e1a8adb1bc4921c3b.jpg)

![tables](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/tables/e24d442dd629aa3841a6fa84343ad8fe8989d078e9a1a6f3e806fa670e206fb6.jpg)

![tables](../iclr_results/4258_AP-OOD_Attention_Pooling_for_Out-of-Distribution_Detection/tables/f16148a07442c22cf2dfb10b8cea946db6d79435f6e162e6634e5c3bd0481c01.jpg)

<a id="paper-4259"></a>
## Multiplicative Diffusion Models: Beyond Gaussian Latents

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=F6w8LcJJFA)

### Images

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/09531022c2c09440dc8e1b2435eec8ed142a409c279a8242971e56f7a581b2a6.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/1239970d90eff035282899386d4e4b6cf6a3ed73ca8b770a6bfd1557c6ff6642.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/1e653c7a56dc542b2c3f4fb2f18732a23473d4ed2541880dede9d536c48291fa.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/1e74a4214eb6eeb7fd0f830908636419d01b7d0be5d5338860a92ca0047e0f0a.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/1ffc2bcef17bb8702745d22737023e0b1717f20ff632d28d0d98e9f0f5dcddc6.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/285b660da8e86d60f59716d1a503e3cea68e7609f1e7e593d27158929e9205cc.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/2cb16cdc314c4557d476efe568d9d1e08f76ba991f9aef4fd416533ff25f48d0.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/2e3db619a4353facdf94b83a6208e41028666e7a92a96b3dfdce22260fa06f4a.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/32b6924629793c94cb7bcf147d717e605e9b064029cb4220a8a905246aa99070.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/3e65cba1bdcbec1d915372446ff1d745773fab2b25bc64ba1ce3af1345ae41ed.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/42719d5e8aed55d9bd7ef2a3d750ae4f470755cfa4f8a2bf37c950ce8f3b390b.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/4b7a0f563f76b68ca36e5dc3ccf3e08284c92a493b4dfaf25eed23affae0ed03.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/54ae3db175c6f5f0aa7b402ca28bf6e47ac4d8887b08d4b78ae66951b68b0f21.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/5653efaceda7f67edcad19aa6de92b97598c3df57ebee5ed0e0e0958dba0fc58.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/60de38e2bb42e80129186a2dea38dda896f79885ef2d80ef40200305c071c8d2.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/61633f4f295477bd7320bf6bcd9e3710bda6bd6b04164b3006e767ede5552205.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/618e7c4f4c3c3c0914ff1981a573dd990ac7706fbba3f2ba540182fa7a2ffa31.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/6e2b5d6319d77986bc0553f039608dc11af54e11b5c3788d466acede2a3e33a2.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/6ee1ed659a0662f47239ec17af1475c0f5b953abacf6a89ed78baef2f69dae8d.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/70a3d4204f26099e94b69b485066167464a0f5b8935575ae668c343ffb1451f4.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/75ee06bec61b2615c1a9d8ee0206f6e3c22d1fc8adeff9aea184b5d2fc006bd9.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/772e266ededfa92f1380e81b209943dc0bc18888a752ec9e668e4a68651cbdc0.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/83cab6f868cb0cbe53a65917741cec30a090e5499b335bd385c7ef3cec0094f9.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/8adf76e5720f8616652accabd20858d0ebecff0da42f31f7abbbf50b0ba44ebc.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/8d716bad6f64bb993d0d651a1c6a970b3b0254c34774870f38a9492c865c5492.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/8f4e7349db26a3df290fdf785966c398324c181379da7004bbeb3ac26b76f10c.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/9380c504f37b93dd3d1195df2908e6aaa4293f139aabde3efa9a6d6e0f82ecb7.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/9d0dd187be8fc116bef3892236519cbfa6e2caf98106028ce3f4d13c5e217393.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/a1e6a2b1b8deb9eb9fd5e312cb78461fa74e59f2ffd2db0125ee1fb38acdee2e.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/a5e758fac3b16a5b2c3f41a84683fd4ccacf05c38ee90522dfa9db033dfbfa78.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/a6f4f6139c4ada1f1eaa6608b93b47fda1fa7ebdc55d38d166308b53b4623213.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/a7e4fdeabd06b16f8fcad501a64a53bfab87045181e9283d732c41f3d4ca27e1.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/ac5c64b4c3158fb598899d24eccdc6b001fa0eb1785f43b061093b815a61a52d.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/ac84cff970614b7fe641420654c5e32ff8c18cf94d923e9278a94bc5d0f6fc93.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/b481c88ebdf6f82a77ff448834049e639bb66f6377705cee65fe8d87547f35fe.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/be0b0cc81894abe381184a621d73fdaf8c56900d08d0096191718bfe605ac556.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/c3b331a533a4d0e5b39eddd1b82faa4b1b9d09713ac630b6802b92548ea6ee2a.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/c556c7c48f598108b9ab6f18322bbc9a5709418a43f3b8a6a865f8576996b75f.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/c83956f7392ed3c163a10564c90f0d5578c1a6ba34dd8d60b8ef45ef6bfda619.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/cc9b7d50669d6c33843567347ca8a6d49351c48d34cfdf4f0dfe2885f898202f.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/d81840d7605b60aafdca548ae87f03399c0726dd94c4dc99e8d650503ae018a1.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/dbeec4dc292b096b285ef429fc711b7a7571451a27f1b5abc0da92c57c0c3f31.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/dcddf62ce03deb3408b3c8469b9a32a37fa38cbd21685931eecce2f97489c624.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/de7dfbfac204031cf52127a1a35f3b34e911c487e683747c73e67fa9ba93f5d2.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/e32eda11869c5b0ccc5c84ee4dfc6187de2855133a9170da1dc791d1e572b332.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/e6447fa6a2aed0a82d00d8818801eff0764864f14f52062cca89225b6eae2b5f.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/e70e44958c07aaad16e014ff5b57134066a96ba7db45ded4acecfbf47d84bf46.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/ef45cf5a8d74d3c3268aceddc180da39ef458bfc3e4736c902d434003fe52b43.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/f0bc39d4e7a0dd48fe924ee838cb9bda614a85b4b1b38a40e0707fb1852b35a0.jpg)

![images](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/images/fe61491c972598702b24838cb5848f85499669fdd50916ea0b994b442635e2c3.jpg)

### Tables

![tables](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/tables/033e9fe16de71d6305a6e45ba54cb9563e3a7bb5a4306e3eaa7ae52c820dd675.jpg)

![tables](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/tables/2a303c0d658be263707d25926dd871d0f6c0f96ef1bb7d99635eb9d435a5f4f3.jpg)

![tables](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/tables/370949c1149030c32128a9ca679e36fe5605293d7ed4aa2cc60b4d99cfcb1476.jpg)

![tables](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/tables/50a8b16a50f7a68caf960ef8aca574d6582e738f1a4f1847dbb010438f2937b6.jpg)

![tables](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/tables/63b7b2c84d9be0db3ee0790f9577e9b15df6749a6148538c9c704aa0c063e137.jpg)

![tables](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/tables/91b443f02fc65063c0ccaf4f825d4662247db6a2c708d22ace6d03332556327a.jpg)

![tables](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/tables/e04f1058bd7017a1ab3beba75aa7b33719bc87121b366224a95db8f92d3ce90d.jpg)

![tables](../iclr_results/4259_Multiplicative_Diffusion_Models_Beyond_Gaussian_Latents/tables/ef2a4d8418b3e91edc2c3007ec753a2e4b714ad938b52fc21cf66f0045428064.jpg)

<a id="paper-4260"></a>
## Constraint-guided Hardware-aware NAS through Gradient Modification

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=fcKMFIiNFQ)

### Images

![images](../iclr_results/4260_Constraint-guided_Hardware-aware_NAS_through_Gradient_Modification/images/18a56fa5e08afe25be320828f83c73ad3456f502000a29644e680be97e8d6217.jpg)

![images](../iclr_results/4260_Constraint-guided_Hardware-aware_NAS_through_Gradient_Modification/images/1f91c4a662c1466be917f19eae389e872860b1bf38f93f0ff55a9e8aed545c4b.jpg)

![images](../iclr_results/4260_Constraint-guided_Hardware-aware_NAS_through_Gradient_Modification/images/bd472805066737d4db119a8e58dab6bfaa61893c3225d3f96f37d0528f699988.jpg)

### Tables

![tables](../iclr_results/4260_Constraint-guided_Hardware-aware_NAS_through_Gradient_Modification/tables/0fe649e96a76282fe608b5b20fbbd712a2008b7210d22f5dea1e7ce07db6a3b2.jpg)

![tables](../iclr_results/4260_Constraint-guided_Hardware-aware_NAS_through_Gradient_Modification/tables/3fcaa93992fd395f391a1b1b3087def3592763272931367cf462b28c0766e2e8.jpg)

![tables](../iclr_results/4260_Constraint-guided_Hardware-aware_NAS_through_Gradient_Modification/tables/50bc20acd88ceec8a60a52b45dbae28f8c5c1b3a0a7af2c63bd4edbaa6260809.jpg)

![tables](../iclr_results/4260_Constraint-guided_Hardware-aware_NAS_through_Gradient_Modification/tables/71ea644afe701f8453b3f5dfa5cfac1d13da0aad3501f307f6b6ddceb9c2bb45.jpg)

![tables](../iclr_results/4260_Constraint-guided_Hardware-aware_NAS_through_Gradient_Modification/tables/73d27d8d1f9d834b3d40cf1d09dd60a018cd205163458ffdd3e75ce4ef14b5bf.jpg)

![tables](../iclr_results/4260_Constraint-guided_Hardware-aware_NAS_through_Gradient_Modification/tables/9f2a1487ef12a0484dfb520fbc3470fde3f5a582f40ba7e6d444810dbc07fd52.jpg)

![tables](../iclr_results/4260_Constraint-guided_Hardware-aware_NAS_through_Gradient_Modification/tables/a629f9a18e1af88210dab30fac9c388ad5a3670e7e14da4ecd0bb334cc3c1083.jpg)

![tables](../iclr_results/4260_Constraint-guided_Hardware-aware_NAS_through_Gradient_Modification/tables/ba03323bf3bef46fea8dbfbaeddf426837c308e28066ba54d4d6a9087e56cda1.jpg)

![tables](../iclr_results/4260_Constraint-guided_Hardware-aware_NAS_through_Gradient_Modification/tables/c35e98cbcc2a696a2761d9112b0cab579b2ffdea9bd2aa6db8e093e9cb74668d.jpg)

![tables](../iclr_results/4260_Constraint-guided_Hardware-aware_NAS_through_Gradient_Modification/tables/d7906f8a6a909c523c3fbd0fad4f96b6ed5363881c1b754b10997b1c1cb8537f.jpg)

![tables](../iclr_results/4260_Constraint-guided_Hardware-aware_NAS_through_Gradient_Modification/tables/e8b6f656fa61179958ac77b2da3cb8fd5a6e59c298f56b18b347005fd2830080.jpg)

![tables](../iclr_results/4260_Constraint-guided_Hardware-aware_NAS_through_Gradient_Modification/tables/f46f4416d52c6efb1b0ecd694c45217504c56a19fa3bc4dc2529da694c1836f0.jpg)

<a id="paper-4261"></a>
## What happens when generative AI models train recursively on each others' outputs?

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=JEU4PBaX85)

### Images

![images](../iclr_results/4261_What_happens_when_generative_AI_models_train_recursively_on_each_others_outputs/images/0b779823bc76cc0c9c0cb8440652d7fe97bf422e07a55e886b92e5e2bb698f49.jpg)

![images](../iclr_results/4261_What_happens_when_generative_AI_models_train_recursively_on_each_others_outputs/images/10448bcc93034d06fff00fa08b1e39bf26839e9dee215e633a1b8db85a8c1b65.jpg)

![images](../iclr_results/4261_What_happens_when_generative_AI_models_train_recursively_on_each_others_outputs/images/380a23b7fe5ada17ddd2aa811cdbd89b29dc5067443775374df04cbd99ef50f2.jpg)

![images](../iclr_results/4261_What_happens_when_generative_AI_models_train_recursively_on_each_others_outputs/images/43406a2c986c0958c230d353d2dfe8753eb6769d1e870d6c998955b85ab0c9ed.jpg)

![images](../iclr_results/4261_What_happens_when_generative_AI_models_train_recursively_on_each_others_outputs/images/85182c8c34d15adbd6b1aa894d7e99ee7b4b8ecd25ea2ad5c7f33589e69fa764.jpg)

![images](../iclr_results/4261_What_happens_when_generative_AI_models_train_recursively_on_each_others_outputs/images/caba496d5bec953bec31f9eb8b294c014f63910228047380e4711f873721d8e9.jpg)

![images](../iclr_results/4261_What_happens_when_generative_AI_models_train_recursively_on_each_others_outputs/images/fc60686f907177b5d466c338ffa325625cbc309638e8b717f04ffd5cfe7f4eed.jpg)

### Tables

![tables](../iclr_results/4261_What_happens_when_generative_AI_models_train_recursively_on_each_others_outputs/tables/4376b5610138a67cacaf63acace43e1a49823431bc7023c3095a10219dc3a350.jpg)

![tables](../iclr_results/4261_What_happens_when_generative_AI_models_train_recursively_on_each_others_outputs/tables/4a6f06e5d53d6e508e665b0b8a144217ba0ef22c42ff50c77f844a0cb47aa0eb.jpg)

![tables](../iclr_results/4261_What_happens_when_generative_AI_models_train_recursively_on_each_others_outputs/tables/4e32ebd61a86758b0049de5b147ddb928a42a146a3a14e0b09618ec249d77024.jpg)

![tables](../iclr_results/4261_What_happens_when_generative_AI_models_train_recursively_on_each_others_outputs/tables/6c249e03381f57946e09a4cda5ef29c15d029415b944fdb62bf1e06d384bb62d.jpg)

![tables](../iclr_results/4261_What_happens_when_generative_AI_models_train_recursively_on_each_others_outputs/tables/95d3dc58ca87cda9a5d958260172b056ef6c476f2c9941f43194948c7f82e112.jpg)

![tables](../iclr_results/4261_What_happens_when_generative_AI_models_train_recursively_on_each_others_outputs/tables/9cd3e14ae484aa2bb7a82563658c15f8229b256beaa5ed930031cc23edaf287c.jpg)

![tables](../iclr_results/4261_What_happens_when_generative_AI_models_train_recursively_on_each_others_outputs/tables/a3344455bb3321a77e42e4072dbf5d517b320d788aef4620dea3a632f0320af7.jpg)

![tables](../iclr_results/4261_What_happens_when_generative_AI_models_train_recursively_on_each_others_outputs/tables/bedfeee13b04de882a216da87d039a1218705ec9219fbc22bd827fb8310f0b89.jpg)

![tables](../iclr_results/4261_What_happens_when_generative_AI_models_train_recursively_on_each_others_outputs/tables/ce615c14205c9468da4d7a4d4cc771a20faab07bd65cbf041c3f6f41dda1171c.jpg)

![tables](../iclr_results/4261_What_happens_when_generative_AI_models_train_recursively_on_each_others_outputs/tables/e3c1cfe1688bdf8c9b8f39555682de42ca7179dacec6537c4ce60447ef34d3ac.jpg)

![tables](../iclr_results/4261_What_happens_when_generative_AI_models_train_recursively_on_each_others_outputs/tables/e99c3f0d2aa31d8602bea087f53f1bed6c7f451c79fb3d940aae3a9a4fb221d0.jpg)

![tables](../iclr_results/4261_What_happens_when_generative_AI_models_train_recursively_on_each_others_outputs/tables/f1fff9b500034aaaa1b5fbaf83a381362e62a6b210d101fd7b2c6b4518a21099.jpg)

<a id="paper-4262"></a>
## Composition of Memory Experts for Diffusion World Models

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=sUEdpZCHdp)

### Images

![images](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/images/187a4ce6a42b5d778eb40bcabe5874f1a80f07f39178d58bff529ad5ef4e0ce2.jpg)

![images](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/images/1aaa79af506ecf0cd205cc94c9b3a64241b9cac54464b2e7ed921bb6a1065712.jpg)

![images](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/images/2375556fd2c88e86b0224d09af48f1143694f3c096889699061497cce49267ca.jpg)

![images](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/images/27cb3960b0e6c2a54aa6db1f46393fe140a02575cbf45badc1ed8652363685e4.jpg)

![images](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/images/2925faa8ad9ae93db57d6c0f9edce2c60f9a1e15fe27a9e0122f03517ac9b066.jpg)

![images](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/images/4b3aac644cb8409558ea02bf26925625b4a8c8b447f13191ff09955270dadbb9.jpg)

![images](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/images/6ea00d81d2e89d216a3bba160a702d904ccfc7a2e6d6adf13cd5cd21f7c01beb.jpg)

![images](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/images/826de0ebe8834eb9f6225aa8e258bdca967b1dfeaf6ef91a3dfff2f4dba055fc.jpg)

![images](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/images/834389a5cdbf53e3ad80ee4f7c88286589016732c21793677cf5c7207ac6bc2a.jpg)

![images](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/images/87c4c12d78e7a316cbe7bf9bbcdf964de3ccc40818f627740efe3a9543dc8272.jpg)

### Tables

![tables](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/tables/064c52684df354d7a8f334387f99f5ff39f5d48663dd85f5b3ab1bdd5aaf049a.jpg)

![tables](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/tables/383d5b792b79b763bdf5c13e26bdad36278555cc9a96a1c2dda9e8eafefe14a7.jpg)

![tables](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/tables/a2aaac899c71bffecadcb97435ffe072dcf41c4be16ef6d69902f5ec8d43ce3e.jpg)

![tables](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/tables/b47d3318834103515145cdb810d56a6c73747359e06ffdabbde6ff0f30abfdcc.jpg)

![tables](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/tables/b93efd180905b4c2c874254a3ea6822f51d697333024322dd1915a37d677a455.jpg)

![tables](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/tables/ce7eb36a3f2210cedfe7929c81bed71433e47b3f228dbfbd6a5142cd879e5bf6.jpg)

![tables](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/tables/cf5b2a431bfacb01d776fa9edff8c1a233e22b630df2371903aac592e3bd0441.jpg)

![tables](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/tables/da88b9624319c5ae0f3627949ddbc5e56a181d76407b125607b13d97303fa42a.jpg)

![tables](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/tables/db3292f89e6053913b5dafc47f7de5ad5b84ae903e2f0f3884be187c2481ab7b.jpg)

![tables](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/tables/e60e937e444b25ef624c8fe6043d973d1da07c7d14b892f2bf384fb87c05be0e.jpg)

![tables](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/tables/e632923e97b882f19d02fcf9cfaff8ef8f8afd93b0b5c0b5e5630936d41c2c3d.jpg)

![tables](../iclr_results/4262_Composition_of_Memory_Experts_for_Diffusion_World_Models/tables/fed3e38a7fe63b75058eb643ad2a1f3cd6917237020a4146e8ec82adcae995f6.jpg)

<a id="paper-4263"></a>
## Dyslexify: A Mechanistic Defense Against Typographic Attacks in CLIP

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=UI7mbsIZeN)

### Images

![images](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/images/19e735835e5a934bf05109e4ea2a41c68f4f4f1afc996eb5f0f16ec0a1d2e861.jpg)

![images](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/images/233377e6365fae74c55f50223a92637120cfd53d30da7ca4bc41925bf65a769a.jpg)

![images](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/images/3755654dde9da6328339a3d7cfe8eb0e7ba80901fed878d904ebdaf468ad47e6.jpg)

![images](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/images/386ac3df777ca08fd2d287e2c827b5dfb5cb791093da678bc0df3adb34ad3e71.jpg)

![images](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/images/47c649d9451d5527cf10b19ba6bd25916667d27bd9ae911d958bc247c7d9c0d6.jpg)

![images](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/images/772e0552d882f49c96d322e605ea4e66a02e76037de1aa545eaad41ea10242bb.jpg)

![images](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/images/814aa673d69d915d600c893496c29585125281efe50411d0532ba15dcb2ce1b8.jpg)

![images](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/images/82a901da3420da07ca73b5da0bd46caa74c1a0abd65ddeb5773c1f32fbce2bc9.jpg)

![images](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/images/8a7dfcc5f48e7b1d3518a6195cfb2614af37bf4e4e5d1b263cb967b11ff7019e.jpg)

![images](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/images/a94437e7e16204279a9c669d3756900e67abcd4cb517963ccf1f7e37bfba576d.jpg)

![images](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/images/c79929c0eeba0dada5d0dc12078d4b6536b179aaa02dfeec9e7eb95ec8dcdb30.jpg)

![images](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/images/d6a240c81cf9a8acd5adfa04ea65908c00c553350d34a04fc6b523f8dcc9c8d0.jpg)

![images](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/images/d6b20b2e792ee97de1d8f0212a85109c5c125e07adcb59d1ec37170a16b71424.jpg)

![images](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/images/e3de904f79a8bd31389a2e42c97c79f34e1ecba71148fa4cf7f66ea8189ebb94.jpg)

![images](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/images/fd8ee5ba0078b0cf38330981dadcf3d2ee71178e9fce0b196ca8d0d97b18c43d.jpg)

### Tables

![tables](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/tables/174bc60ae491daa6f8d232760591d3c1ed7a97f84499b45820a23142649b7b89.jpg)

![tables](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/tables/4c6bb1cddaba4eb2e4d3d19e204f2d1fc8ce26f06c8a379c77d2983f2e7aa45e.jpg)

![tables](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/tables/57ce4b524fc22a7e673cdb0f2e234234f5671184f20814f72e54c7d1140ab3b2.jpg)

![tables](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/tables/63fb8ec72f0c11132a4888ec76939daaa8174f347fd8d5814e338cf10c6ddcf0.jpg)

![tables](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/tables/8ad089c81247703e587d3493e46a3fe07a907a7a2bd3a81a71e1f7d9050e4707.jpg)

![tables](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/tables/92c10c0f80a26988610fd6f1cdf66cdf1654d56395fc07011fc4cc8ce6f6e6d2.jpg)

![tables](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/tables/a7e91226d2108e776040524aa4248af5ba7f333cd95980828fe9ecb53305ff88.jpg)

![tables](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/tables/f20ed0a71b46511d0abf62446edfba48e91188ff92c60b1016ae68a4ded3f750.jpg)

![tables](../iclr_results/4263_Dyslexify_A_Mechanistic_Defense_Against_Typographic_Attacks_in_CLIP/tables/f84b9e60d18084c777f3b7097c572b3dd285358d6a045c4addfb00de59c42004.jpg)

<a id="paper-4264"></a>
## Bayesian Parameter Shift Rules in Variational Quantum Eigensolvers

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=cS0L2kj0lj)

### Images

![images](../iclr_results/4264_Bayesian_Parameter_Shift_Rules_in_Variational_Quantum_Eigensolvers/images/1f68b5fe8f3082d3e23cf98faac122067bc6ec23f25cab66d8773f4e2c0fcba1.jpg)

![images](../iclr_results/4264_Bayesian_Parameter_Shift_Rules_in_Variational_Quantum_Eigensolvers/images/4a81148eed07c5de4dec2c2b1ff6bf876541e91577950039a489713af64ee87e.jpg)

![images](../iclr_results/4264_Bayesian_Parameter_Shift_Rules_in_Variational_Quantum_Eigensolvers/images/4d08b18da0a89954aa8bdf5d010c7d3a11f404f3ba7f77fa2653226fb4d7326d.jpg)

![images](../iclr_results/4264_Bayesian_Parameter_Shift_Rules_in_Variational_Quantum_Eigensolvers/images/5e3d2677a240b5ff16449e72956f3581b4fe17171848b2972c6c435fc1fb7a0c.jpg)

![images](../iclr_results/4264_Bayesian_Parameter_Shift_Rules_in_Variational_Quantum_Eigensolvers/images/7409d3aa11be2d6a842d5c2544bf4ba590e97bf3276a9a988fbc0a7c823b10a7.jpg)

![images](../iclr_results/4264_Bayesian_Parameter_Shift_Rules_in_Variational_Quantum_Eigensolvers/images/8ff530e25345ed66a7bc095ad90e1e68f7bcbfc804bfd1ee8593dba0c7872cb0.jpg)

![images](../iclr_results/4264_Bayesian_Parameter_Shift_Rules_in_Variational_Quantum_Eigensolvers/images/adc92419eebaee707efbd9e7339d4210509648341755f7da5b507704bcb8bad5.jpg)

![images](../iclr_results/4264_Bayesian_Parameter_Shift_Rules_in_Variational_Quantum_Eigensolvers/images/b9f5de10dfb02c595fc305088e67ce02a6157c53d839b81f132f61f8a71e2bdb.jpg)

![images](../iclr_results/4264_Bayesian_Parameter_Shift_Rules_in_Variational_Quantum_Eigensolvers/images/c0d9b0bbdd8cc5b05476984951adec56596be6a6e1a17522a10bfafbfee90458.jpg)

![images](../iclr_results/4264_Bayesian_Parameter_Shift_Rules_in_Variational_Quantum_Eigensolvers/images/c1386d9667ad2c4f7aebe1eb5ad231031a94ff8a96de0980101bfa16291b2b0c.jpg)

![images](../iclr_results/4264_Bayesian_Parameter_Shift_Rules_in_Variational_Quantum_Eigensolvers/images/d58fac38bca9f853eba8900932841d4bd2eb69b56aa3e2a47f990b403422bae5.jpg)

![images](../iclr_results/4264_Bayesian_Parameter_Shift_Rules_in_Variational_Quantum_Eigensolvers/images/eb2c4b863f6c7bbf40ea1c45c110d822a5c116b8cc578c4f58ab4a709cd3ec96.jpg)

![images](../iclr_results/4264_Bayesian_Parameter_Shift_Rules_in_Variational_Quantum_Eigensolvers/images/f57120b24a42660966eeacc59cafa9bc237c721634b403058e42aec4e1e3ca09.jpg)

### Tables

![tables](../iclr_results/4264_Bayesian_Parameter_Shift_Rules_in_Variational_Quantum_Eigensolvers/tables/3cb59588686588e2376c73450613058d0c9bd9d37529b5921e0a1ac5018adbb3.jpg)

![tables](../iclr_results/4264_Bayesian_Parameter_Shift_Rules_in_Variational_Quantum_Eigensolvers/tables/6201ae450e58257363974b6c6e6c2c4d175ba973cc0f2487a14ff7f5b4c2c099.jpg)

![tables](../iclr_results/4264_Bayesian_Parameter_Shift_Rules_in_Variational_Quantum_Eigensolvers/tables/73b113b0830571efb1d538a4b9365b343fda85b23717c52ad5c9af3afb456353.jpg)

![tables](../iclr_results/4264_Bayesian_Parameter_Shift_Rules_in_Variational_Quantum_Eigensolvers/tables/7d40a35cb2e51ca04ddea6d6597ee2b85ebdd5d7cb8df3e39ef780d335f96c50.jpg)

![tables](../iclr_results/4264_Bayesian_Parameter_Shift_Rules_in_Variational_Quantum_Eigensolvers/tables/7f3e86b3b9e4cc6865978482abc966c26cdfe1acd0499e7fb6ce7d73a67c9e5a.jpg)

<a id="paper-4265"></a>
## Enhancing Image-Conditional Coverage in Segmentation: Adaptive Thresholding via Differentiable Miscoverage Loss

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=Gd2AiWes1J)

### Images

![images](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/images/05610a73b221c092ae67db26cec1574ba51809d4c42c0085d5b1a7472361efaa.jpg)

![images](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/images/2879ef6d2d93912a128ef38449a035e2c2dbd269d6e7a514683f676694f1c6d8.jpg)

![images](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/images/2ae1eb10a407674b8cf0ac3c423db85c85e19aa0dea728c4cd13e38b033b9eec.jpg)

![images](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/images/42e11d53a3e320d993e409100033e6146545710badfd26337aff4609f7eb8e69.jpg)

![images](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/images/5868c949875f0898f9231eff17851993ef9b3ce5b03c4758b4b9efe3150c4141.jpg)

![images](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/images/5af41b49ef0d4564f497048342f1315cee75e36c1aaf69e789c1e1adda5b1657.jpg)

![images](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/images/6820cd9f3f25bb67ae097a41dd356eb55af7c4df55fcfb6434e8381aa92314e2.jpg)

![images](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/images/6f4348ccee5ee5caa077e4e9430f67abb4a4226d5c3d1cb019120a37d44cdd18.jpg)

![images](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/images/75b4cfeeb34f45475a3264b85823cd3e713e9eda789bce67cf28645b0c8f63be.jpg)

![images](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/images/8d00f020e909f730391ef2ab7ae420e8fe63b0672acfb5b095c63d4a9326bbd2.jpg)

![images](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/images/b5333fe5a4f9f35c12e843174be15e5a60073de7ad9a3ef0085f8e9329167a92.jpg)

![images](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/images/baeba3e3c44ee368b487e879c4689a4926496848aec05d1df459c481d436e4c0.jpg)

![images](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/images/bbb6ab97abe9cd9a990db8ea2095b822d0516669862da658e2905e21372bcd43.jpg)

![images](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/images/c440f53c98a7414c5022f35c7e1a27ed71ce338c778fa5f526ee6bac8e376195.jpg)

![images](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/images/c80c206114eb168d45dfd4ca4c30b57987854d2dfd510a1e49c0f33a0026a5e9.jpg)

![images](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/images/d91274b051642a0f2403accc4618f5bb594681d44dd8771434286215794cb448.jpg)

![images](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/images/ff3152acadde31a222ddb4306d2ff76ad2088a56f69e39cf5ff58a4f83c91d00.jpg)

### Tables

![tables](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/tables/27ef4d3aa65cc9a23c129d42231deab0fb3848f22868b1c9b3eae4c9a254a0f3.jpg)

![tables](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/tables/468bf9f84a0ff8dfe2700f358447147b6c0263427d4315392735fdcc139ca6d6.jpg)

![tables](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/tables/82c1da3b9f32584f10093a09932a54217a41e8a6af0ae8cf770b636152c432e2.jpg)

![tables](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/tables/97dde5aa6eded2ae97a225fcf0f8f94191ea0fe62f5548442df004cbd48b1c4b.jpg)

![tables](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/tables/984f573ea934cba84d59e4642c2379a652da9dc63726e143925d8014da0ccfd0.jpg)

![tables](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/tables/d8c3ffe92fd77cad9eed6b8180b89cf1cc05345dc9ee4e41818144f440713349.jpg)

![tables](../iclr_results/4265_Enhancing_Image-Conditional_Coverage_in_Segmentation_Adaptive_Thresholding_via_Differentiable_Miscov/tables/ff6b02889250d7e59016152cc13b4d26d300629cfc13cf904d3894757d38fd44.jpg)

<a id="paper-4266"></a>
## Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=bFJ8Sdr224)

### Images

![images](../iclr_results/4266_Learning_to_Parallel_Accelerating_Diffusion_Large_Language_Models_via_Learnable_Parallel_Decoding/images/17d165dddaffbc8ba4bf0ccd621f0225b3b46f638bbe53fac651665a936785df.jpg)

![images](../iclr_results/4266_Learning_to_Parallel_Accelerating_Diffusion_Large_Language_Models_via_Learnable_Parallel_Decoding/images/2289bb9620ab1cdc7f91d5e9430e09e3ac924ac88d3532a7a8c7f495f8b0cb6f.jpg)

![images](../iclr_results/4266_Learning_to_Parallel_Accelerating_Diffusion_Large_Language_Models_via_Learnable_Parallel_Decoding/images/3c594802cd92fbfc3cff0d9b28b44fdfa86c02a6781d1cb26d24d00c40272395.jpg)

![images](../iclr_results/4266_Learning_to_Parallel_Accelerating_Diffusion_Large_Language_Models_via_Learnable_Parallel_Decoding/images/5d1a8e50c9d46dc23a2984fe0b90cf354b04f962c2b7325c3f43a6325ee881aa.jpg)

![images](../iclr_results/4266_Learning_to_Parallel_Accelerating_Diffusion_Large_Language_Models_via_Learnable_Parallel_Decoding/images/6e9c7f8f13e54e434b1e87e25aa443d64bf95adaf60f613656f8964e7807ea64.jpg)

![images](../iclr_results/4266_Learning_to_Parallel_Accelerating_Diffusion_Large_Language_Models_via_Learnable_Parallel_Decoding/images/8f71fe6bcc74831b92df15412fc797229afbc723c7a387f6a607c52d0d8f2c4e.jpg)

![images](../iclr_results/4266_Learning_to_Parallel_Accelerating_Diffusion_Large_Language_Models_via_Learnable_Parallel_Decoding/images/bab1af319c0bbfc80fdc6003082af40df0b794c30b2587eb187e7d40e767841b.jpg)

![images](../iclr_results/4266_Learning_to_Parallel_Accelerating_Diffusion_Large_Language_Models_via_Learnable_Parallel_Decoding/images/c0bd1ebbb3cdc9ca3617e0e7cdbb2ecc0663def2235fd740cb563e854c46eec7.jpg)

### Tables

![tables](../iclr_results/4266_Learning_to_Parallel_Accelerating_Diffusion_Large_Language_Models_via_Learnable_Parallel_Decoding/tables/30c1576ec3a32a3b2ab4444c62b2881c4706801d3ef1adac0dcd366f819dbdfd.jpg)

![tables](../iclr_results/4266_Learning_to_Parallel_Accelerating_Diffusion_Large_Language_Models_via_Learnable_Parallel_Decoding/tables/5a0dbb0f59f10dc721473daf554c08a03a1537e95e44f5307c79cf95563d6054.jpg)

![tables](../iclr_results/4266_Learning_to_Parallel_Accelerating_Diffusion_Large_Language_Models_via_Learnable_Parallel_Decoding/tables/97a9a5914df71590986dbbe214168aba0650dc8db28ccf9a9f6a077d08f03ade.jpg)

![tables](../iclr_results/4266_Learning_to_Parallel_Accelerating_Diffusion_Large_Language_Models_via_Learnable_Parallel_Decoding/tables/d39d92dbc29eb081e22ba74f264a094f102f9f72051c0c7e7a96928e096e81db.jpg)

![tables](../iclr_results/4266_Learning_to_Parallel_Accelerating_Diffusion_Large_Language_Models_via_Learnable_Parallel_Decoding/tables/df19db54017a5f54b372b79fdc5956d141275ec72b784fcf03649d526ffa70be.jpg)

![tables](../iclr_results/4266_Learning_to_Parallel_Accelerating_Diffusion_Large_Language_Models_via_Learnable_Parallel_Decoding/tables/e3a0bd762a282c4495c60d66acbea8f4e0dd8104c29852d9ff4ca7bff13bf5fe.jpg)

<a id="paper-4267"></a>
## Unsupervised Representation Learning for 3D Mesh Parameterization with Semantic and Visibility Objectives

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=9LYsvna4Sk)

### Images

![images](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/images/002d7d0a839c10b610b7ec629a3ac1e559b5f1f6065ddef1e8c0052d47ea4700.jpg)

![images](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/images/0884a705d1da8d773e7af102dc448892283c2ba0753114234d44c3341307d4b1.jpg)

![images](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/images/41a47a6fe76b475fa56a0a582a095f38bc562330e069f96733820838ba6bb1f2.jpg)

![images](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/images/4e9bee48510ad643e807f8cf41cae149dce44756cc62b19b8874ee060eba4a85.jpg)

![images](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/images/5bc35b6337d5a6d15f11993eae6ee1604d602ba20c60700097d24c71178f8f77.jpg)

![images](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/images/753fa460732b1f4ff6046385adbdc68cb028969fe4d42868d239a648ef865477.jpg)

![images](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/images/8d870432f9851271c8dd301e290758907f7fbf9718b032f60c5deb8ce55b501e.jpg)

![images](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/images/8dbccad2a1019ae7883b53eca2ba76e13f9659e605efba0bf79ef8158c161099.jpg)

![images](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/images/b15a055bee95d62eed05bcf93aa7cc0cd74ecc23dad53fb9795f2bcadc345ff8.jpg)

![images](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/images/ebff1c6dd9abb4b4d1f1f9a7d750f9a5d9b222b2c2f64c20423f3def1d19cae6.jpg)

![images](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/images/eeeca76f46ba6a1dd1e41f1ab4edba3988a8c5ca4805452aecefba76c0faba37.jpg)

![images](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/images/f4f08adaf813895018233d792c5ab5635632e402357a7693bff18f0aa9a9fe9d.jpg)

![images](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/images/f65aeeb192f6858b7e8fb0a44f01d306859488f3cd2564f5edf7b4fd48dbb9d6.jpg)

![images](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/images/f93f2e7cc7244efe0c226a056a1c8b6e682ab11521fd828b5f787f677e646ea3.jpg)

![images](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/images/fd7843decdf08d6686ae776bdb583944a7aae5591ca822a11e6e5496a3706b20.jpg)

### Tables

![tables](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/tables/07b0639f031ba9aca196cb7a87cd97fd98d5ee463de779e1ac80a66766952882.jpg)

![tables](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/tables/3be55d43a354367aa997a0c566556455ed5d4e199f08532e067e73c087229723.jpg)

![tables](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/tables/64dbdd551152e1aaa60dfc49f05a026ea9b4037061cdf827ec9ea2d684f43476.jpg)

![tables](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/tables/66ab758c354cc034ee3a14cb584d774305dc4364368888f66ee26b665b4b88ba.jpg)

![tables](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/tables/71f7cae83ee73c0d440ed137cc762f2a0291e9ab9e8ee422ef70ce44e0b2ba9f.jpg)

![tables](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/tables/7a8786e5c73346052494573672ccbe58e50d270bd606fda1ed3baade01e9a0f8.jpg)

![tables](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/tables/a1332db84a409aabb2b083e8d67029e73c69d847797cece05e7d4e01adae5691.jpg)

![tables](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/tables/d28a4ff89da92e99f8d1aa706e1d09d4d4216c5796a6d96024b1b632744a7224.jpg)

![tables](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/tables/d3eee6d50898643f03ad5572f0cc8307abdca3f414d710adc526add5bc0ef88c.jpg)

![tables](../iclr_results/4267_Unsupervised_Representation_Learning_for_3D_Mesh_Parameterization_with_Semantic_and_Visibility_Objec/tables/dfc71c2ac967c0c979f95b2d1d512c561622ccf7403b955420a2671841b17aed.jpg)

<a id="paper-4268"></a>
## Stable coresets: Unleashing the power of uniform sampling

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=sOpAa8iR0A)

### Images

![images](../iclr_results/4268_Stable_coresets_Unleashing_the_power_of_uniform_sampling/images/239e65eee055f54ba1f490546d9d2fad2fbbe8503f7f584807fe03c1f382e583.jpg)

![images](../iclr_results/4268_Stable_coresets_Unleashing_the_power_of_uniform_sampling/images/67f781070a90642cb6339f8ab6bce6763963c96d1a8607879e1f2f3a42d5505a.jpg)

![images](../iclr_results/4268_Stable_coresets_Unleashing_the_power_of_uniform_sampling/images/895244d2ef9394103e1d35091547b463e491e5bd545d3942d4f8d6aac1b56c51.jpg)

![images](../iclr_results/4268_Stable_coresets_Unleashing_the_power_of_uniform_sampling/images/97e4acdb3d953e1831090474f99ed049607d8160f1cff2d8ce986196d04a6772.jpg)

<a id="paper-4269"></a>
## Verifier-Constrained Flow Expansion for Discovery Beyond the Data

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=IfDYQbsWf4)

### Images

![images](../iclr_results/4269_Verifier-Constrained_Flow_Expansion_for_Discovery_Beyond_the_Data/images/052ba0d2c0c4bb55417ffa927173ec0d68bac4210e0e26276932f4708d86edde.jpg)

![images](../iclr_results/4269_Verifier-Constrained_Flow_Expansion_for_Discovery_Beyond_the_Data/images/3240f293a71444d096e64783f3b19659b9ad9e49395bd746710fa9e0ece83a61.jpg)

![images](../iclr_results/4269_Verifier-Constrained_Flow_Expansion_for_Discovery_Beyond_the_Data/images/430711958b055c6ef5cf03a49eb00bdf8becece78b735aff7c1fe84dd0d5aba1.jpg)

![images](../iclr_results/4269_Verifier-Constrained_Flow_Expansion_for_Discovery_Beyond_the_Data/images/475ef6d25a2f30e98e2b7e18745c18dd828748e355f4b936a977060d68eabcea.jpg)

![images](../iclr_results/4269_Verifier-Constrained_Flow_Expansion_for_Discovery_Beyond_the_Data/images/5a9041690c2e46523f095f3756c7471d657b16cb2db9346e72b72a57bcc1fd91.jpg)

![images](../iclr_results/4269_Verifier-Constrained_Flow_Expansion_for_Discovery_Beyond_the_Data/images/7c7be2de7dd1f60ef0702baa826ef2f95b679cac6352e3c1379060c80c1aa431.jpg)

![images](../iclr_results/4269_Verifier-Constrained_Flow_Expansion_for_Discovery_Beyond_the_Data/images/addceb1a40ea36d27112a9af85541981f866ad6009a42e363b26badab6e3234a.jpg)

![images](../iclr_results/4269_Verifier-Constrained_Flow_Expansion_for_Discovery_Beyond_the_Data/images/df1cf7a1eab3872fdaa3e574145a6178d259cc6dd48a1fd8fa9e14f30630c482.jpg)

![images](../iclr_results/4269_Verifier-Constrained_Flow_Expansion_for_Discovery_Beyond_the_Data/images/ee7368139b81076032df100952dff6e17cf9fbb26d6a7aa3c5eb045c5a2ce2e6.jpg)

### Tables

![tables](../iclr_results/4269_Verifier-Constrained_Flow_Expansion_for_Discovery_Beyond_the_Data/tables/230df58bf27ad442e6d0e57a38eb6cdb43f0a291175f40b33ef8c4682c32b906.jpg)

![tables](../iclr_results/4269_Verifier-Constrained_Flow_Expansion_for_Discovery_Beyond_the_Data/tables/6fb5482cd7ac53a1540054a84539ad0a74a667275475b45433b790a144c26123.jpg)

![tables](../iclr_results/4269_Verifier-Constrained_Flow_Expansion_for_Discovery_Beyond_the_Data/tables/8d93547fac0c1f92df2086004c27cad59548bea5613c2c4e6d06ac7ef9bef529.jpg)

![tables](../iclr_results/4269_Verifier-Constrained_Flow_Expansion_for_Discovery_Beyond_the_Data/tables/f0249983ccd92ac3c45881b3c7f7c7a49c5119b4f10b7abdd348b8a5b5a885e2.jpg)
