# ICLR 2026 Main Papers — Part 152 of 199

27 papers.

- [CLoD-GS: Continuous Level-of-Detail via 3D Gaussian Splatting](#paper-4081)
- [Benchmarking Open-ended Segmentation](#paper-4082)
- [gLSTM: Mitigating Over-Squashing by Increasing Storage Capacity](#paper-4083)
- [Pitfalls in Evaluating Language Model Forecasters](#paper-4084)
- [Matching without Group Barrier for Heterogeneous Treatment Effect Estimation](#paper-4085)
- [Reasoning Boosts Opinion Alignment in LLMs](#paper-4086)
- [Poisson Midpoint Method for Log Concave Sampling: Beyond the Strong Error Lower Bounds](#paper-4087)
- [MoDr: Mixture-of-Depth-Recurrent Transformers for Test-Time Reasoning](#paper-4088)
- [Delay Flow Matching](#paper-4089)
- [T1: One-to-One Channel-Head Binding for Multivariate Time-Series Imputation](#paper-4090)
- [Flow Actor-Critic for Offline Reinforcement Learning](#paper-4091)
- [Lookahead Tree-Based Rollouts for Enhanced Trajectory-Level Exploration in Reinforcement Learning with Verifiable Rewards](#paper-4092)
- [Neural+Symbolic Approaches for Interpretable Actor-Critic Reinforcement Learning](#paper-4093)
- [Pisces: Cryptography-based Private Retrieval-Augmented Generation with Dual-Path Retrieval](#paper-4094)
- [EUBRL: Epistemic Uncertainty Directed Bayesian Reinforcement Learning](#paper-4095)
- [Adjusting Prediction Model Through Wasserstein Geodesic for Causal Inference](#paper-4096)
- [Concept-based Adversarial Attack: a Probabilistic Perspective](#paper-4097)
- [Alignment-Enhanced Integration of Connectivity and Spectral Sparsity in Dynamic Sparse Training of LLM](#paper-4098)
- [A Stitch in Time Saves Nine: Proactive Self-Refinement for Language Models](#paper-4099)
- [Rainbow Padding: Mitigating Early Termination in Instruction-Tuned Diffusion LLMs](#paper-4100)
- [JointAVBench: A Benchmark for Joint Audio-Visual Reasoning Evaluation](#paper-4101)
- [Robust Adversarial Attacks Against Unknown Disturbance via Inverse Gradient Sample](#paper-4102)
- [Local Entropy Search over Descent Sequences for Bayesian Optimization](#paper-4103)
- [From Sequential to Parallel: Reformulating Dynamic Programming as GPU Kernels for Large-Scale Stochastic Combinatorial Optimization](#paper-4104)
- [Selective Expert Guidance for Effective and Diverse Exploration in Reinforcement Learning of LLMs](#paper-4105)
- [TABLET: A Large-Scale Dataset for Robust Visual Table Understanding](#paper-4106)
- [Fused-Planes: Why Train a Thousand Tri-Planes When You Can Share?](#paper-4107)

---

<a id="paper-4081"></a>
## CLoD-GS: Continuous Level-of-Detail via 3D Gaussian Splatting

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=zgs0L72R4c)

### Images

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/04efe1eaad68426b995e1114bebc9479152bccf30f8e901cc4ff385a9bc44a1d.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/0b0b7a54492dfcb04429b19fbab374d5cd306ab4e7e2c7a3cb2112200a022252.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/0e1f18013d28ebd273cc4799e191653214601eb6e258a381781edee5f858c481.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/1f1bf6a240580aa410b3540a775a1350d3e1f0d6a9be9c00527eb39245a18d29.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/23b013c0f3e22030bb3f75b9650ba7b288bd420dc1b6c17c2336888ea9f24531.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/262907ceddd18641d9020eebf5d005041d8135c99bd91a55dfef0ff239c77326.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/325ba826f51a94b29c99208c4fe81a0f0fa886255c89c620c11e559f7f18e67a.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/45605dd1609cd61869387dab2fcd15242b46d03cb8fc9a20bb719fc889f4aaad.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/5089361a9a9957a3903d9eb687a7a6ac4042ccc506ca484be51a97d0f30d2d95.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/65309250ccd824b11b1784802edfefc004b2c0abba58f73429d0eb5825d51e87.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/791433b3a799d3fec12c4f3d0ad933a7d40277dad6ec0f465b955ecbf767059a.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/7eff33b7c062aa986595d2666be108dfee790e8e24ebd3bd432194f164e60175.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/873de35be609abf8acb8d24aeb8c9f80bac13b8449b0c423d830861379691651.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/8aea3c7225f3d938841eccc0f83311115b05f2f4acd3d61329c71a2ccbd48b54.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/8cfb698e8be8755e611ef53c21351a905c2fd489eff3787e3a094be9be2002fa.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/96bd14e779572a8abb202ff42a0c46471be685666c1acc814d4a5ae86fc1d114.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/9a9b3a6288d243cfda8cd24824ed7788d3ca6d5138138083f076fc63824212ca.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/b44ce2345383804ab2bd023bc6c6e3ddc5a8910b98f64fb4df2d5a999a6b6186.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/b7aaf076481fa247168f325e895532ec4ed4fab90ab5750972e5399a9b857d89.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/bda078aee023c85beec41f57fc2ace4503f4bfe74dfed053a477d4074eaa9942.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/c8232c8ee5b4a3421aa34a9b74f64f45e13ea01da238135b23cfbe8b43de54e7.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/cac025128f2a545b596ea17c7eeaaa8a878ea483e71ea12ba4e0dc31f9e890fe.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/d14726ce0725c368cdef39fc70ec8deb3c8cba4888f1f09974d359189c8efbdc.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/d2643871785c6eba323f83e39ac5a4e1a479925fa394e10147e8dbb0edd6f505.jpg)

![images](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/images/e68e17d2caab699d238a1ba6b4a9430ec268d65a60a1679fb9d41b7d93e4b831.jpg)

### Tables

![tables](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/tables/0e64340164c0976485da928a6ac9831d4236ab69fd631f68ec5ed6f3a966f2ef.jpg)

![tables](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/tables/1dc7f4ffb58ff2c3be1da975381f27ef01b4ea974f967df2740a77d612c598bd.jpg)

![tables](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/tables/581b819a2d2c8fcb6059153eca4b36a73e160a9a9036fd653bbb0e99ff5cd9ec.jpg)

![tables](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/tables/5aa27e8d291b62a837382f52f33a61a84139e8a8867636379dd44906ab5daf97.jpg)

![tables](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/tables/6095594b1daa8e5153a8789a037a12031ecd519d29a76b5c12118cea1a5daa88.jpg)

![tables](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/tables/65b91a630f90c42d4e1933569e4f694dc35ad79214392ff3ed58fea54d2eff0a.jpg)

![tables](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/tables/6ff449a56db2b216ece6b10c8e9c2c9cef7c27424452f4ecfaf580a864cd1e13.jpg)

![tables](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/tables/88e3f8c3e24e00cef2ae617cee3598ed0284bc6142c83da029b305417ec7b9d1.jpg)

![tables](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/tables/c9b397d6a032b2330f65f992ebd67fb4102920dc932d0fa8c1e6ff9abe7eae42.jpg)

![tables](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/tables/d935f5d5073278bd8418ace2707d55cf2a2a1a22215211d86cb48b9cb4d582d2.jpg)

![tables](../iclr_results/4081_CLoD-GS_Continuous_Level-of-Detail_via_3D_Gaussian_Splatting/tables/e72e453ddf476d80d90df0ad0127395481034f618c8f73ce4f183bf0c8ed9cfb.jpg)

<a id="paper-4082"></a>
## Benchmarking Open-ended Segmentation

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=QSmwRnw8GP)

### Images

![images](../iclr_results/4082_Benchmarking_Open-ended_Segmentation/images/2a6885e5d2fae47f07fa8ca1bbcf01f9a48d46adaae30e1978d859ff0612e8f3.jpg)

![images](../iclr_results/4082_Benchmarking_Open-ended_Segmentation/images/3e5665f2304d3d376a7a63cc1f665645c4d92774bffb60fbdfd7925ef71d1f3a.jpg)

![images](../iclr_results/4082_Benchmarking_Open-ended_Segmentation/images/49c792af42552bbf2cd4c9fdd54ee9c2ed95675f0d40375ef9747ed81bdd9cb5.jpg)

![images](../iclr_results/4082_Benchmarking_Open-ended_Segmentation/images/51d47601f4ace0d8e5537bb1b18d14a48997a10c79763dd96b038f7910727d40.jpg)

![images](../iclr_results/4082_Benchmarking_Open-ended_Segmentation/images/68a8d48a65dabc3ee794a9859ff0acd37681f8e9b4202c73e32e976a93d20bcb.jpg)

![images](../iclr_results/4082_Benchmarking_Open-ended_Segmentation/images/6ac12397b0f539ed61485c14e1465104e3b354fb16e2cb44a86441b928dc0998.jpg)

![images](../iclr_results/4082_Benchmarking_Open-ended_Segmentation/images/7c87ba32b348654a53b6e05f0bde1e96b68683e1c1e21d0366e62520d86f6033.jpg)

![images](../iclr_results/4082_Benchmarking_Open-ended_Segmentation/images/a4c6f0e3507b48ac053704ac4ac58414b90419f6d1522bb8c97ea7aef2871d2a.jpg)

![images](../iclr_results/4082_Benchmarking_Open-ended_Segmentation/images/be72b769963abfc668a722aafd3016aec1031152c20587de87315c27c8e87dec.jpg)

![images](../iclr_results/4082_Benchmarking_Open-ended_Segmentation/images/c8c628ec5d79000cda6ba5e7bff013fa9102a238e43974840515ac47336a1ee9.jpg)

![images](../iclr_results/4082_Benchmarking_Open-ended_Segmentation/images/cd5da08d77ce8ab91d3bc323871316f674116988b45a3b92bd220fd8da7ea650.jpg)

![images](../iclr_results/4082_Benchmarking_Open-ended_Segmentation/images/cfb6c4e34d3a257db62535e7d8ed3c22825ffdd4e3a75f2cff11cb6527f73e79.jpg)

![images](../iclr_results/4082_Benchmarking_Open-ended_Segmentation/images/d1283b39decde846e35d82a4501b4cceca71c7d5135f5343ff28eaa0d8cd5633.jpg)

### Tables

![tables](../iclr_results/4082_Benchmarking_Open-ended_Segmentation/tables/7d2043ce08d20f38aca7aab6ad743a3ba0562d9623efc41c08f61a16fc5fd65b.jpg)

![tables](../iclr_results/4082_Benchmarking_Open-ended_Segmentation/tables/d5b585ae090676b57be7718dc68ffb28f15d193301fe850175560cfc625a7dc1.jpg)

<a id="paper-4083"></a>
## gLSTM: Mitigating Over-Squashing by Increasing Storage Capacity

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=c4mXEOXlAL)

### Images

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/05aa74756cbdca7318441f1f9ab7b070444a72432abf805bd5465ba918090df7.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/0cc8e67af198ea3458697ef8053ba60f3dbad2e468bcfd79aabca8e0d0e897d0.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/1bc30fe86377a374c99be276b4c7258c78e8e99a3733910fc139bb0e132ef323.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/21f8926571aa4a028655535ae4d4810fc9d2dcddfd619c61eaadcab3205c2205.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/3bddf07948d0a34cb053e07dcd21ef9a3e214aab94a0254334d2ded30e501e3a.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/3c7400c7e081fdb0473e3d504d29fca891e22a3b0965f6feac026d3b87179c68.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/3d5aab43580e6b3aaea222d8d1349b4b0fcbbf7b35f5c56726eb4e53f4a43e67.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/3f5ee46c9fcef2ec5fe615366296f60dea41123bfe3cc55155f20a96b9290c1e.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/4eec9db52537fc714938ff40d67158c408efb5d254aaa918bfb495de625a6239.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/5ed145a89e9e60ab2baa51c4dcd5704cfdf1620ed8723a52f282da78f557e2d1.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/67d6fbab927671cf994e6632addd54a39937140b3475fd49ef614d676864b74f.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/7636d5529a360cf1faf3d1d5fd95e49349aca9bad4bf0cedd726f58558e1f9ff.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/81314223b7792fc03648bc43e11113b3f44a40739b17dedc0890f161ba90005e.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/8588cac01cf1a9f09d5bc5d5530051be80a909b691004e6ffed8860ebde5e0c9.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/8f430a7eeaf865c3fb13e4d1485c1bf11ce553e8ea501a205bc3e42dbd8859df.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/9afcbe5c85b8247b5711a26b623775580a595305168a3bb8bd14e6d1839a99e2.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/b89b3ac7c3ab675da4408ba43d543152b83e0938e710fc6a324272c88fdc2fd5.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/b959841ad8136b7d4b0acbf9bb1cd5cb53362e782d4e168a27c502ad9fa1c4ac.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/cbc5dbc5ad1efb14b6f364697b7b1f60e0a02f46a3f3859753018e12ea62d4a1.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/ddf2c17643b6500aa8d15db21f8fb80ab2aa72c3e4eead8e16ae66e5e61310b7.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/e561e1eaf656203110af22111d8795fc335852a67490f367b11ff18e2e34bfb7.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/e95b4a200e1c33adcf0bde8ac9ed0cb16fd9734352c886040ff1d90ff1d36e7a.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/eb75b957de47eb73a9e4ef3f78412be792aa4cf0b6d0d78f31552302d30b0651.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/f3487c634161158c8fc1ae7144e5bc3b2d839617e11a78788e5033c2d606ac12.jpg)

![images](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/images/f8f5773941547084bda22082a3e6c9f8a986597325919fc0678ffbe65f5e62fa.jpg)

### Tables

![tables](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/tables/0be5d73933651148fbb7f498dd8d969db7893022db3fa02f4f6788247eda0523.jpg)

![tables](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/tables/4a34c7e905955f9874d3b7583f0736c67043dc447c3cbf2b338f891838cf9db7.jpg)

![tables](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/tables/4c9e1971ed78efb5ea03568c977495882eb6412cffbd6145fb1239e0cd543d7a.jpg)

![tables](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/tables/4f9099b867590ccdbe00366b6641001fca7f295b71007798acb447d57ca98d28.jpg)

![tables](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/tables/5108027f7317dca74303f9191d7a2f2ec7fab6e9dec5e376a11e72ee15e2c5f2.jpg)

![tables](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/tables/7950f288f7a1d4fe8c809748d1bb8aac594fdc4415ef679aa28108e3c279bf81.jpg)

![tables](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/tables/8d546455462d4c86336410da9e6760e5191468569a623d3c7f64d757493de862.jpg)

![tables](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/tables/b20dd2ce8ada94cc55b6c77ce18f7e0de0b47b18a03f9cbfa13198b375e54ed3.jpg)

![tables](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/tables/b2aaebb29c179bb4d2065ce768ee42ac78e5735669cb114374219ab67c3c1e43.jpg)

![tables](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/tables/b50f17ff3f5b5cb7d988b22476d948b3e129c3847c50dd845d3ad548fe7f38b2.jpg)

![tables](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/tables/b9b70a6d3a90a365e10029109891eca7075b47b16abcbeeff5c6514965342402.jpg)

![tables](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/tables/cff6d11d6fbddbbc8622d445a8c483d55aa27b5c232fc183838ceb8c66933df8.jpg)

![tables](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/tables/d7c8295d2b37816b59913f9b4baca5e95010ce2fe6cb031b8db270f61ed02b82.jpg)

![tables](../iclr_results/4083_gLSTM_Mitigating_Over-Squashing_by_Increasing_Storage_Capacity/tables/f4523527d1aabd76cbad990a19ded158133841dde29ca347d0c8958e1516e893.jpg)

<a id="paper-4084"></a>
## Pitfalls in Evaluating Language Model Forecasters

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=z85kARAoyD)

### Images

![images](../iclr_results/4084_Pitfalls_in_Evaluating_Language_Model_Forecasters/images/02c63953674959d19cd6c2aa2a4cb47a0a144fd39b9e4f186f164b709ef34cdc.jpg)

![images](../iclr_results/4084_Pitfalls_in_Evaluating_Language_Model_Forecasters/images/33711985114a77dd241a28966532753212df3f5a78f7323ead3824c8454b1ee4.jpg)

![images](../iclr_results/4084_Pitfalls_in_Evaluating_Language_Model_Forecasters/images/49efef38b273857697de05094562bb6c1712534422e8143a06c1234effd858a4.jpg)

![images](../iclr_results/4084_Pitfalls_in_Evaluating_Language_Model_Forecasters/images/65afba97186aad11f0f43c07d2013a0ca23a3e4f08f5a7a3c2797c59e676bb3a.jpg)

![images](../iclr_results/4084_Pitfalls_in_Evaluating_Language_Model_Forecasters/images/6a4095ad92b2ef599bd2fe274aa3ad9c148f5d1633dd1db681b77f75bf6e1507.jpg)

![images](../iclr_results/4084_Pitfalls_in_Evaluating_Language_Model_Forecasters/images/6a91e5b8663c9acee2ef66336497a11e3efb8fabde1ada19393f0045437fb43a.jpg)

![images](../iclr_results/4084_Pitfalls_in_Evaluating_Language_Model_Forecasters/images/763dfb225f629fdabf8da81b4626153553f36511ee826a9c1dcd3149edf4c0c8.jpg)

![images](../iclr_results/4084_Pitfalls_in_Evaluating_Language_Model_Forecasters/images/a96433a497a1697528ab19f7ea9e07a036233ca3d12ff410a5cbd45308b4648f.jpg)

![images](../iclr_results/4084_Pitfalls_in_Evaluating_Language_Model_Forecasters/images/f991173f9d32651f3a9eb6cd6c102e43657a9322335d97aba5bc383956325e6c.jpg)

### Tables

![tables](../iclr_results/4084_Pitfalls_in_Evaluating_Language_Model_Forecasters/tables/5ed42edd99009ad1673573632c05abe94c6349d5c5dc8f61fe4ecfb2e71822f6.jpg)

![tables](../iclr_results/4084_Pitfalls_in_Evaluating_Language_Model_Forecasters/tables/6d618b57fa2d4bd76afb5c538d7ec6291e78d4c64ba6cdf912523fe53946ac98.jpg)

<a id="paper-4085"></a>
## Matching without Group Barrier for Heterogeneous Treatment Effect Estimation

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=lGaZimFbss)

### Images

![images](../iclr_results/4085_Matching_without_Group_Barrier_for_Heterogeneous_Treatment_Effect_Estimation/images/98e11eaa7854afdfcd6c75089f4eea8b1ef25a5e574c2911b3f78565e91cae67.jpg)

![images](../iclr_results/4085_Matching_without_Group_Barrier_for_Heterogeneous_Treatment_Effect_Estimation/images/b020b482d650bf3abbe983e8620bb8bf4b3246523bbe9db3a329be6eafd78578.jpg)

![images](../iclr_results/4085_Matching_without_Group_Barrier_for_Heterogeneous_Treatment_Effect_Estimation/images/fd52781399abf6900a550c5fcef914a759483b11d1d12f5dee83efae9d3923a3.jpg)

### Tables

![tables](../iclr_results/4085_Matching_without_Group_Barrier_for_Heterogeneous_Treatment_Effect_Estimation/tables/07318cf2ab42ab0a02e9dd34a5cb390f222745db384e1e721b63917c91e51bca.jpg)

![tables](../iclr_results/4085_Matching_without_Group_Barrier_for_Heterogeneous_Treatment_Effect_Estimation/tables/2c611dd8b37f54386ccacd49ba807466861d40f027e26f0945241bf3df73fb2f.jpg)

![tables](../iclr_results/4085_Matching_without_Group_Barrier_for_Heterogeneous_Treatment_Effect_Estimation/tables/2c6ffefbe8f59f23d5b53e3e9b71b9d87a62e3c9e43ca0c642a129fd2e609234.jpg)

![tables](../iclr_results/4085_Matching_without_Group_Barrier_for_Heterogeneous_Treatment_Effect_Estimation/tables/5af2b9e9caefabb7e60980ab14c1530786ff753a349b797753236dbf181b278d.jpg)

![tables](../iclr_results/4085_Matching_without_Group_Barrier_for_Heterogeneous_Treatment_Effect_Estimation/tables/64ed67b02e11da19279d390c1d1c3ec1967161367ec5d6d586294db5daf44d56.jpg)

![tables](../iclr_results/4085_Matching_without_Group_Barrier_for_Heterogeneous_Treatment_Effect_Estimation/tables/7a9324e519643430928dd668e1e7a1f2c01e6bf52ce32f9c9a443e90cb171409.jpg)

![tables](../iclr_results/4085_Matching_without_Group_Barrier_for_Heterogeneous_Treatment_Effect_Estimation/tables/8ee24e4ade41cdfa2414407e367a1b23f9bb440a4546dc6ae15ceeb5e797c5c7.jpg)

![tables](../iclr_results/4085_Matching_without_Group_Barrier_for_Heterogeneous_Treatment_Effect_Estimation/tables/aaf2f917562c2a454f4a2518d892f92b9803acdff62d6687ded7f679c4c56271.jpg)

![tables](../iclr_results/4085_Matching_without_Group_Barrier_for_Heterogeneous_Treatment_Effect_Estimation/tables/f1081b23d1f9bcc67c7f7a7e5c50db55c7cd3037b5e6be12133c229d5b311008.jpg)

<a id="paper-4086"></a>
## Reasoning Boosts Opinion Alignment in LLMs

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=mdunHhVDPz)

### Images

![images](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/images/1be4a957554ec40f867a3b9154135166942314fb9c0da55171279fbc560da4c7.jpg)

![images](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/images/2a7f25b7f4258d4884f6b24a508b5db7d57e2df19776dc9edde81d628d5a11b5.jpg)

![images](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/images/41be33fbc27b4070d9fc0d33c4df0a6a157caf50799e94dbaac0dafcc39d1653.jpg)

![images](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/images/4b01db1f999bd381a92ce9337e2df74267f4dadb2679b448a99328eec0bb4313.jpg)

![images](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/images/6ba085dfe69e830a748ebae77ac0d2c805ae2436d3c01356e7ad46143dd7ff22.jpg)

![images](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/images/6f9c4a645e588c5398f117e045ae91bafe9fd0dae29ee017345aad9ba2bd9494.jpg)

![images](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/images/711aaf8a75deca909d9451782f9cb5cd84957d85fbf84dc15d81ef7b29a629ac.jpg)

![images](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/images/8bec3afa7cadab943f2478e8d4a90e5c5d231ec61d4c8ffff758bf948b5597d1.jpg)

![images](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/images/ceb350fe537f6d4fc26ca4d1047e6939c61dd06e22336e752c45e429549522e5.jpg)

![images](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/images/d9ae8b5a19b1fa179c453f1ae72de33686d540788a3b88df94d0721f9e82d06f.jpg)

![images](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/images/e052a44076c0cd9b3e2a2fb387b9bff1ffb9bbae252f7edde49be7f1e444e188.jpg)

![images](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/images/efe2dd18db866d4f22dee38b3516893940715c6274432957de9f4f96043e7c1a.jpg)

### Tables

![tables](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/tables/3102b097ae008a927a3bea2e8ad7e9062eaadf5cb9488fecb6998c483fed6d35.jpg)

![tables](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/tables/35b11c44ce54d2fd0b06bc57257cf34621e32d0e6bbc5c74f1b5260b6d48dc6e.jpg)

![tables](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/tables/53ae58f89dc906c0bda81fba1f144b06c9088e928af4fb1a1d9ca3b65b41901c.jpg)

![tables](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/tables/81ad3c8e9595483c0e4f1ac8c159434180e41f1146a47476e9e68b1d13be363f.jpg)

![tables](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/tables/a1c760134ffc51bea27bdd069dd16d3ee9ab745186a54afa7185b38d9b1dff87.jpg)

![tables](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/tables/b9f2f482ca1969dd3cf908a6b146d10393378ac72a9f4fb02e676479aedfd784.jpg)

![tables](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/tables/be3c6540024a34ce868d0191c62bc828fdd979bffc8feca2dbbbe6d1b2a4f963.jpg)

![tables](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/tables/c213ccdd375c7c95e0741b2740b9bd6092a2847e19c93efbb8c708beae8746f6.jpg)

![tables](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/tables/d05a9c60d799dcfe690a54c3ac48ed0625a3894a20845a46c451e52f6de2c2fe.jpg)

![tables](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/tables/d387b73fc85a06c6f2ef60c1129ae068f98ec76f8f4d327ac896cfa53f7d6a46.jpg)

![tables](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/tables/da46ed9b5d7c5e74a265a97d75b2988dd54c3648e9ee18f0d8940ac21d5ca0cd.jpg)

![tables](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/tables/ee3030b9c0ab2ddaf9a0d764c8e01fae46babadcf03a9e69ed305c77e5b3307d.jpg)

![tables](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/tables/f6b5dfa160eb43dc4813639cfab3a384d7a4a62b1210011d7e942729e8037fd7.jpg)

![tables](../iclr_results/4086_Reasoning_Boosts_Opinion_Alignment_in_LLMs/tables/fddae06db048bdb40e9f53cb713afaf27150b69ab689546ee1c6bcc7df2fc35a.jpg)

<a id="paper-4087"></a>
## Poisson Midpoint Method for Log Concave Sampling: Beyond the Strong Error Lower Bounds

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=j2wEu2ycTg)

### Tables

![tables](../iclr_results/4087_Poisson_Midpoint_Method_for_Log_Concave_Sampling_Beyond_the_Strong_Error_Lower_Bounds/tables/c1f91d5e1a47f71d651c811f69189c5722570a325bcac54ab8933ee2f4dd375c.jpg)

![tables](../iclr_results/4087_Poisson_Midpoint_Method_for_Log_Concave_Sampling_Beyond_the_Strong_Error_Lower_Bounds/tables/e21f41388980205c5288027941b5b8c3c3e0e13581bcc67e6dd3c7c500dd23c1.jpg)

<a id="paper-4088"></a>
## MoDr: Mixture-of-Depth-Recurrent Transformers for Test-Time Reasoning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=9Pba4rcQbE)

### Images

![images](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/images/03f67c356769a4910a6a8466aa283969171b0d1cf2ff5dc6bfce7db90106dc71.jpg)

![images](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/images/10bdb383a4c94891f5e36eab1f5a7b5a2398a3980c31eb9cce7e3a5b54dc2646.jpg)

![images](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/images/25007a9e0f4edac37fe7fa16f347ae13232e451ca7a46ffe2aa220de362cad56.jpg)

![images](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/images/2d70d4f59d888ecea9f9c6f96f22b27dcaabf73c2d6073a419042d67a88d5a30.jpg)

![images](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/images/6cbb628308c5e0f9682fe55970ef512a442b6a207334702ecaef548df8c49e3a.jpg)

![images](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/images/7359ce25b1584501a6c8293b4d5c3d4a8f2cc65b037b752f81b49b8635718cbb.jpg)

![images](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/images/9a4e8ad605892e64227a8032d4fb8316e4157b75c8f5c03c09a4b01dd78fb2d5.jpg)

![images](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/images/c0f9bc847a37a65ab6974cb2d014d36aaba698dcc8a68b50321f3264f87330df.jpg)

![images](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/images/cbe22c81bd3ebcbd79ad90ab271b2c868735355f22363bc94092baeb83b89b8f.jpg)

![images](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/images/cc37dbf4d39403e97189f16fb9eb6053dbaf1db3247669653f43e1a847c8c1de.jpg)

![images](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/images/d002fe6dc3e0b58768f79da0ad8326ec405b56a0c7e3a82d249d48d6bad940f7.jpg)

![images](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/images/f8dc474dab25ef54a2c41178e8b9f619d9983708f051b1d9741c3785690fff68.jpg)

### Tables

![tables](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/tables/1b87c6992b4d273a722b41de4b66671dca019f609119b1ffc15466a4bb2ec4ef.jpg)

![tables](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/tables/3a8d465f11e6a97b0d5f029c8f9ff8ca369bf69fcdc0923853c380cca62e0b83.jpg)

![tables](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/tables/61bbac658f146f6d8852c3da62b264199d3f0950c2c851f2bc6826a27877245a.jpg)

![tables](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/tables/71ba5fe4fd10f9d7cd4f0f7509e102324337d18c2c6cc1c7efe5c9c0e20433d3.jpg)

![tables](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/tables/96acb5550b9f9707d2fbfe69d6012af729b8512975d1911470eb126b499063fc.jpg)

![tables](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/tables/9f0be7b23f4803c343672811f7fba643e489081c0389a5fca3bb72584dfd7b99.jpg)

![tables](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/tables/cfc3cfc5786a203b20d29b4618f814799d52fdd569a57bde86f0e46ca2aa3517.jpg)

![tables](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/tables/d2c8cfe47a2cdfaa16daf6c0a1d5c6b5abc88e57bbcfc5f8fd9f14e7ff21be18.jpg)

![tables](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/tables/ee1808dcc6f3d1d06f24dede91c3eb9e3409de8d77d44f9775d5109c5a709312.jpg)

![tables](../iclr_results/4088_MoDr_Mixture-of-Depth-Recurrent_Transformers_for_Test-Time_Reasoning/tables/fdc422ae72bb875bbc282b01e7684ca6f6df7aa40de62931065bf329e65cd96c.jpg)

<a id="paper-4089"></a>
## Delay Flow Matching

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=6lH1XblLpo)

### Images

![images](../iclr_results/4089_Delay_Flow_Matching/images/49f8bf3da55f5607c0e8631aa65baa18bb203a6d66431a1ccaa8fb6b8cb9e298.jpg)

![images](../iclr_results/4089_Delay_Flow_Matching/images/74956faa0fba1c5f6d79ea7bfbb8a22411f18def3bde5d181071c03cfcb16f96.jpg)

![images](../iclr_results/4089_Delay_Flow_Matching/images/b8b52666438b051ef27fbf17f8a0f302047ac9d7564cc89182745cc785d9548c.jpg)

![images](../iclr_results/4089_Delay_Flow_Matching/images/cbd8bd0d46ecdc257e60c721f0b6687accaf9ff179ecaa4df2f57588a72dd6b1.jpg)

![images](../iclr_results/4089_Delay_Flow_Matching/images/d36ca320c86452e47c6369dd765251aad9abbe206dee8ce5b25a744851767ef8.jpg)

![images](../iclr_results/4089_Delay_Flow_Matching/images/daa6c387f9bc1acd1fb504e5c465951493517351f16d60ac91547b9c2309c151.jpg)

![images](../iclr_results/4089_Delay_Flow_Matching/images/de618df4731317ea2448ed16247823a281a3b488a18bbcb15d61a7bbc5a14486.jpg)

![images](../iclr_results/4089_Delay_Flow_Matching/images/eaa0529d1704e6dd4d40e5e86d598532e40480ae9956e3ac8a5f6dff1c3aa95d.jpg)

### Tables

![tables](../iclr_results/4089_Delay_Flow_Matching/tables/2486746ea46f3b7b1175821456566d1ba073bfd833c7f7a5e9fa2623504ece6b.jpg)

![tables](../iclr_results/4089_Delay_Flow_Matching/tables/3b5bdfd7284aa1eb455d856974d0b081e68781eb158866df834662d851cbf5b0.jpg)

![tables](../iclr_results/4089_Delay_Flow_Matching/tables/4b8f2d5bd63a9b054799048ae5ad67b513044e6b1d503b0c8149a44039452f85.jpg)

![tables](../iclr_results/4089_Delay_Flow_Matching/tables/72acdc49d26e9b84b3af43adff62eed22932b34f497388a12ec8271bb3e883b8.jpg)

![tables](../iclr_results/4089_Delay_Flow_Matching/tables/79fa38e152b4d1da31fcca4b7e101f4bcdc87eef7bc435626ddc5acf61a3927f.jpg)

![tables](../iclr_results/4089_Delay_Flow_Matching/tables/80581070a1ee668a0044bc147c91e11f7bff24ac120f85f8ba9d19d6c429799c.jpg)

![tables](../iclr_results/4089_Delay_Flow_Matching/tables/90d2cf853231b7dd1f9d08705c1986aacd071a38aa1a5b804293308452de5464.jpg)

![tables](../iclr_results/4089_Delay_Flow_Matching/tables/953321b526c17023f7de726d8a8f2026b3da3a738c5f446720596f4481a54446.jpg)

![tables](../iclr_results/4089_Delay_Flow_Matching/tables/9814bae539324666ffb7a4f6594d865e07c80635de342cd5e66d9b1649c704c4.jpg)

![tables](../iclr_results/4089_Delay_Flow_Matching/tables/ab669e1d584dc40f7a828de4bb0a2cae268d3e14decabf6066830fe4bec7431c.jpg)

![tables](../iclr_results/4089_Delay_Flow_Matching/tables/cab1c70c32b941f520fe5d74a76045377a968062ba6e16d4d9be5c7622078447.jpg)

![tables](../iclr_results/4089_Delay_Flow_Matching/tables/ce9876de5ec5c430337a868c0178c31d19096ab83fd615d0e2dfd54c3407c2a3.jpg)

![tables](../iclr_results/4089_Delay_Flow_Matching/tables/e4135f960a8a8a5e54177c75fcd5ceeef08e05df7f900fe7819a153624b9f1b0.jpg)

![tables](../iclr_results/4089_Delay_Flow_Matching/tables/e9230bac892caca1ff648ae4609f231bddebd564c49e7408df6af78fd3b51650.jpg)

![tables](../iclr_results/4089_Delay_Flow_Matching/tables/ecf089ca56e7fbc66b90aff58279cf02adb5bbc259b0dbfc7af3f417fde24196.jpg)

<a id="paper-4090"></a>
## T1: One-to-One Channel-Head Binding for Multivariate Time-Series Imputation

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=IAnIlFsPEW)

### Images

![images](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/images/07399e71a2cad167856bd8fb7489da572c7725d9f61c9cbb74d1166c11216790.jpg)

![images](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/images/1952403197e27eab40d52456b66135d28f4c45c18ee9fe817451ecda8b40cd97.jpg)

![images](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/images/4d0a34afb1748c1ab2eed17d31268886490f6623c6e5f4519228b4c2f5189fad.jpg)

![images](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/images/59cf858deef4eba296fb17c34b38195071541658306ddd5e140b1aa603a1de82.jpg)

![images](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/images/6546781c2e6c1473c357b79a4719cabcf9b6c574c695eb1302b056f7738eddaa.jpg)

![images](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/images/c3f996291b7e2a8234f40053ff9f3437aa556c2af4cea3c18273fe59050a5644.jpg)

![images](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/images/fa150e6eba1dc002364ef22d86af4f08085092b1089a66a75bbeaa7242b51640.jpg)

![images](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/images/fe431e59a6bff5f1630797ca9c1cccaade508180e266245250795d6c324f6e04.jpg)

### Tables

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/035113c9234cad33b3ab1d5ea3cbd95a7eb11f4735b1267ac784f97d5f180a63.jpg)

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/0b2fd503b118f52a57a18c1bd692b29cf99b62c4856f3f83376ed90da017f367.jpg)

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/0b7bdfcf15bc6381c7e935fa6aaa8e226e14b39b033a299d882d7679966c50ce.jpg)

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/0ca9e12fdec2547417c283e02414203116bfaf453e9f163f4fff73206e77159e.jpg)

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/18ec132fed191dccd4e40ee6ad3a60bf20c8041a6c9ff2c4a2704cf50371fa0a.jpg)

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/205f1e3f5965b86e66e4c1c68d797fdace6b88e249209687c0b494e20b9dbfe7.jpg)

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/314ef77b5f9d016af3247079e7d83d6827c24814447c73c9fb7a2ef9dcf9457f.jpg)

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/373d4078b9b3f2f8ffea8122d7aa944dbacd202811c115de7a2957bba94d4684.jpg)

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/49977327da7defd63c23c417d93b75a5c8f7db6fda4da61cbf52a00ec44733ef.jpg)

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/689162a9b635973697996beb9fe1cc111e53459615d1eac0db72574b5f3d1081.jpg)

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/6d5a264ec7524f5e130884be92c22317d9c7fb103cebf39ec5131a72444960ef.jpg)

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/72cf623e36c9031729588fffb85f711e8e35d642ecb980ffa9bd3b2d84331649.jpg)

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/829c1531d8c96a6d6bad52eaa4703ed09fdc10bda9ccf33d0fbc31d995cc76e4.jpg)

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/8e096797b4d4188a2366da8eeee74195557d506ced425d723ee7f967f40e1895.jpg)

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/96acc401768b05e0b04133ea872050349e95210ed1ac9119691f16575843d148.jpg)

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/a3138d6fd23e76544750c35ece0248a5bb069901aee39ef7e8299db16bfb211d.jpg)

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/bd50ecf58f8992a957f940b04d1adf2aadfb41459bf488682fe9dbf8c328ec7f.jpg)

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/c9e966cda5fb954bbd91bf1aeb429e800c154eaaa044acb2bc9c3a79498dc4fb.jpg)

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/d1926d56212d897018ae0ecde9ff3c17cd6deb698ce78557b88c59bf2220a881.jpg)

![tables](../iclr_results/4090_T1_One-to-One_Channel-Head_Binding_for_Multivariate_Time-Series_Imputation/tables/ea319f9f74fb6dbfea29200322d784f26f06a8b555d56b970b1bf091a9a85e40.jpg)

<a id="paper-4091"></a>
## Flow Actor-Critic for Offline Reinforcement Learning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=wuncwN7iZN)

### Images

![images](../iclr_results/4091_Flow_Actor-Critic_for_Offline_Reinforcement_Learning/images/29762eacad06ed50af6e750d399dff57073629c51e751707120a1a7beb62e716.jpg)

![images](../iclr_results/4091_Flow_Actor-Critic_for_Offline_Reinforcement_Learning/images/400aabe582501df693ffa968ec9ed79b162fc4cbf210aaa9acc4e8a29ab6eb56.jpg)

![images](../iclr_results/4091_Flow_Actor-Critic_for_Offline_Reinforcement_Learning/images/477e1f2675a7f09896a9d8ad3b1cc6aa17809d24559fee8e1dbeff2fd23a06a4.jpg)

![images](../iclr_results/4091_Flow_Actor-Critic_for_Offline_Reinforcement_Learning/images/4803653f1860a0632963d264d20e6fd0854f6484261d214b911aa70f554e6f5c.jpg)

![images](../iclr_results/4091_Flow_Actor-Critic_for_Offline_Reinforcement_Learning/images/8974bed0eeef6355aec90050e0b0cf28bb0b6bdbda12ed4cb4d4633d1a9618a9.jpg)

![images](../iclr_results/4091_Flow_Actor-Critic_for_Offline_Reinforcement_Learning/images/8d6a7a777109a9130331e22fcffa63ebf1461bcf86eba57b135eaccfad55d713.jpg)

![images](../iclr_results/4091_Flow_Actor-Critic_for_Offline_Reinforcement_Learning/images/c0e0a3cdb79435d305f6b32e272ad22963adc0669df06dc4b33865f8551d7a80.jpg)

![images](../iclr_results/4091_Flow_Actor-Critic_for_Offline_Reinforcement_Learning/images/db89390c1895c0cd73dd971983aed0e0886fb4ad2acb899bf1a7469b2fddbc73.jpg)

### Tables

![tables](../iclr_results/4091_Flow_Actor-Critic_for_Offline_Reinforcement_Learning/tables/0a4fdcc6840c48ebfcd8799f105b3d24fe58e9e95ea23bdbaf175403b58819be.jpg)

![tables](../iclr_results/4091_Flow_Actor-Critic_for_Offline_Reinforcement_Learning/tables/29294d29c8dd1b500c4a5d653677250443a4fd44edc4e56f5acc9d292bf277d1.jpg)

![tables](../iclr_results/4091_Flow_Actor-Critic_for_Offline_Reinforcement_Learning/tables/298e727617202b899201727b179551519235ef237e3fcdfd073833e2673fd030.jpg)

![tables](../iclr_results/4091_Flow_Actor-Critic_for_Offline_Reinforcement_Learning/tables/2edfa885402b81a0974b8f9f5ea0922e03231df906eb10c4a6e43e2eb152b224.jpg)

![tables](../iclr_results/4091_Flow_Actor-Critic_for_Offline_Reinforcement_Learning/tables/5074a59bdc7c28e44b51141e88c51dceb461c5ed909afbf649518deab01dfeb0.jpg)

![tables](../iclr_results/4091_Flow_Actor-Critic_for_Offline_Reinforcement_Learning/tables/8efd7dea0138f9e7d746c29271c4f1967043e707dd6eb58d0a2343cabb4f354c.jpg)

![tables](../iclr_results/4091_Flow_Actor-Critic_for_Offline_Reinforcement_Learning/tables/ba82970902dc1501a03e5be83b18ff858ebca9680001014f3deeb16f179f50fd.jpg)

![tables](../iclr_results/4091_Flow_Actor-Critic_for_Offline_Reinforcement_Learning/tables/c2113db3e6e58b1abe39519cc78e0deb7c1b7cc6566adfcd27758124eccb5110.jpg)

![tables](../iclr_results/4091_Flow_Actor-Critic_for_Offline_Reinforcement_Learning/tables/c484abed9046208c487511536bd0122342c3b787caf81aef096a88cc5d0f06b1.jpg)

![tables](../iclr_results/4091_Flow_Actor-Critic_for_Offline_Reinforcement_Learning/tables/d4b82b8ecc989ed37faf59514a94a3b09f5065ea98780c68a1335bcf3c51bc62.jpg)

![tables](../iclr_results/4091_Flow_Actor-Critic_for_Offline_Reinforcement_Learning/tables/f91adbf4d686813ae969731995b4463ab7a4c7b0f99c330795e94cefdadaa4dc.jpg)

<a id="paper-4092"></a>
## Lookahead Tree-Based Rollouts for Enhanced Trajectory-Level Exploration in Reinforcement Learning with Verifiable Rewards

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=4nLvUk8edu)

### Images

![images](../iclr_results/4092_Lookahead_Tree-Based_Rollouts_for_Enhanced_Trajectory-Level_Exploration_in_Reinforcement_Learning_wi/images/57bb3d5359a2536892128e9b4f70a1d723383d90c183319dbef8795fe8f7ca45.jpg)

![images](../iclr_results/4092_Lookahead_Tree-Based_Rollouts_for_Enhanced_Trajectory-Level_Exploration_in_Reinforcement_Learning_wi/images/7990228c6ae9bc01563bcd0e35a61b0dee12e055d37b198357f7b47913ea1b2b.jpg)

![images](../iclr_results/4092_Lookahead_Tree-Based_Rollouts_for_Enhanced_Trajectory-Level_Exploration_in_Reinforcement_Learning_wi/images/a73f2d9e59546291b1899f182bb099eebba1cd408a060542771a497b8b7ba7ed.jpg)

![images](../iclr_results/4092_Lookahead_Tree-Based_Rollouts_for_Enhanced_Trajectory-Level_Exploration_in_Reinforcement_Learning_wi/images/b24162175b84d593141035b6b03a21c409936e6eee1852954b9897bb20afdc18.jpg)

![images](../iclr_results/4092_Lookahead_Tree-Based_Rollouts_for_Enhanced_Trajectory-Level_Exploration_in_Reinforcement_Learning_wi/images/b77addc496b69a37a3829e50d5e236df71cd594042888fe541a4f6ba2893cbe2.jpg)

### Tables

![tables](../iclr_results/4092_Lookahead_Tree-Based_Rollouts_for_Enhanced_Trajectory-Level_Exploration_in_Reinforcement_Learning_wi/tables/3657527aff5cc052b524434cd99061927a082ee3e74378733c46b2c4bdd59d12.jpg)

![tables](../iclr_results/4092_Lookahead_Tree-Based_Rollouts_for_Enhanced_Trajectory-Level_Exploration_in_Reinforcement_Learning_wi/tables/381035e9a87b629ea6920bed6e8c770ee04c96cdf3c271565d78507d33545f87.jpg)

![tables](../iclr_results/4092_Lookahead_Tree-Based_Rollouts_for_Enhanced_Trajectory-Level_Exploration_in_Reinforcement_Learning_wi/tables/458c95ca7fccb27c9b3e6d064fb7b969fa48983f60583582831298dc4bba1af7.jpg)

![tables](../iclr_results/4092_Lookahead_Tree-Based_Rollouts_for_Enhanced_Trajectory-Level_Exploration_in_Reinforcement_Learning_wi/tables/4776719f43bd045a64b8176c0f1359f96748fe469f057f36df940671d73a09ae.jpg)

![tables](../iclr_results/4092_Lookahead_Tree-Based_Rollouts_for_Enhanced_Trajectory-Level_Exploration_in_Reinforcement_Learning_wi/tables/5e833829d5aa10f3f1de7a17690357d6431bfb5953bf5349b720de633532787a.jpg)

![tables](../iclr_results/4092_Lookahead_Tree-Based_Rollouts_for_Enhanced_Trajectory-Level_Exploration_in_Reinforcement_Learning_wi/tables/646d5ca3823a25568780362c6464e260fc0be7842d6159affc906a82e8bed43c.jpg)

![tables](../iclr_results/4092_Lookahead_Tree-Based_Rollouts_for_Enhanced_Trajectory-Level_Exploration_in_Reinforcement_Learning_wi/tables/ae4955d6a62b4e1b39513bf2ffd3bc6f1dd367346dbead943615461e3aa4a884.jpg)

![tables](../iclr_results/4092_Lookahead_Tree-Based_Rollouts_for_Enhanced_Trajectory-Level_Exploration_in_Reinforcement_Learning_wi/tables/c2dd008e319b7c848695e710c4ba8cf9641cd9ce985c238028d313362801775e.jpg)

![tables](../iclr_results/4092_Lookahead_Tree-Based_Rollouts_for_Enhanced_Trajectory-Level_Exploration_in_Reinforcement_Learning_wi/tables/e39a33d4b7a2c60c326b43c5bfffc7af8cd4fab562b37d2efff56f7755fc47c4.jpg)

![tables](../iclr_results/4092_Lookahead_Tree-Based_Rollouts_for_Enhanced_Trajectory-Level_Exploration_in_Reinforcement_Learning_wi/tables/fcf981b1a557d14a19d5f5ed9e6dbe533b32b796b46333edc70931b9dff8d504.jpg)

<a id="paper-4093"></a>
## Neural+Symbolic Approaches for Interpretable Actor-Critic Reinforcement Learning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=0XIsA0PxJM)

### Images

![images](../iclr_results/4093_Neural_Symbolic_Approaches_for_Interpretable_Actor-Critic_Reinforcement_Learning/images/052a7ccf95da1daea80df6b6b53797ee66fc5682b03459380a10fce08abbc778.jpg)

![images](../iclr_results/4093_Neural_Symbolic_Approaches_for_Interpretable_Actor-Critic_Reinforcement_Learning/images/0adf16c1ff066a45b5485dc041b0860512dcedd2eac4eb4ac245ae472e104e53.jpg)

![images](../iclr_results/4093_Neural_Symbolic_Approaches_for_Interpretable_Actor-Critic_Reinforcement_Learning/images/54ffc186eb3478ceb88117061a6b9925069c2da1ae36e65aea07262f58c699b4.jpg)

![images](../iclr_results/4093_Neural_Symbolic_Approaches_for_Interpretable_Actor-Critic_Reinforcement_Learning/images/62f621036c8182e3cffbebc36bdd8cd8ad2b9b32db1efa0b5f4af7a7335c172d.jpg)

![images](../iclr_results/4093_Neural_Symbolic_Approaches_for_Interpretable_Actor-Critic_Reinforcement_Learning/images/63dcc968557877c9f998863dfaad53f2a1b9cb43e5d6d92079194c349c58e5df.jpg)

![images](../iclr_results/4093_Neural_Symbolic_Approaches_for_Interpretable_Actor-Critic_Reinforcement_Learning/images/6404a6373b6c03c2729c1d50f23aec2013d163f0ce7ee7538e2ad408b9aa32b1.jpg)

![images](../iclr_results/4093_Neural_Symbolic_Approaches_for_Interpretable_Actor-Critic_Reinforcement_Learning/images/838d31033ad49cfedbec988595b22f5f3ff2c35173d06ea00fdef8b085a6b05c.jpg)

![images](../iclr_results/4093_Neural_Symbolic_Approaches_for_Interpretable_Actor-Critic_Reinforcement_Learning/images/a03b910e40b44d4a5ba61a287ef156afa21116df31d5a91170921424baec1b70.jpg)

![images](../iclr_results/4093_Neural_Symbolic_Approaches_for_Interpretable_Actor-Critic_Reinforcement_Learning/images/ccd1dce7d9d41a836f57a46c6fa9d348199602fface985bda9f275ae41061d6a.jpg)

![images](../iclr_results/4093_Neural_Symbolic_Approaches_for_Interpretable_Actor-Critic_Reinforcement_Learning/images/dc81959aae14502984605037ed748e47fe02cee365e0b4a526e7a83a8c32fb4a.jpg)

![images](../iclr_results/4093_Neural_Symbolic_Approaches_for_Interpretable_Actor-Critic_Reinforcement_Learning/images/dd9868d0dac9191e8e42b5b25010fc687639e4eb6080a39324904a6701365d47.jpg)

![images](../iclr_results/4093_Neural_Symbolic_Approaches_for_Interpretable_Actor-Critic_Reinforcement_Learning/images/e5c5017219543dc0e91267f1609f074fafbd5d6c7fc19f213ff737ed3a3baaee.jpg)

![images](../iclr_results/4093_Neural_Symbolic_Approaches_for_Interpretable_Actor-Critic_Reinforcement_Learning/images/e5ee509593afd43739e6a362f5899f83e2d2c3a460767a5c2e47764db20b4c96.jpg)

![images](../iclr_results/4093_Neural_Symbolic_Approaches_for_Interpretable_Actor-Critic_Reinforcement_Learning/images/e66b6989c1a7ee3fa62b8718145b7e67413b38c0b6a98c54c22f058859c95514.jpg)

![images](../iclr_results/4093_Neural_Symbolic_Approaches_for_Interpretable_Actor-Critic_Reinforcement_Learning/images/fa9725470be7aad801131a8a96d4d6652c80c956edf90f38de948b178f9cccb5.jpg)

### Tables

![tables](../iclr_results/4093_Neural_Symbolic_Approaches_for_Interpretable_Actor-Critic_Reinforcement_Learning/tables/b645ab60c9e4db05091e341296da0b2bb168bbdfbb9ca8ca009a1541eed764b8.jpg)

![tables](../iclr_results/4093_Neural_Symbolic_Approaches_for_Interpretable_Actor-Critic_Reinforcement_Learning/tables/c23c960653784dc6b3612587442e9e3f8881d618dc25d9b864fe9f0ca44c4fac.jpg)

![tables](../iclr_results/4093_Neural_Symbolic_Approaches_for_Interpretable_Actor-Critic_Reinforcement_Learning/tables/c80d4b7c7f34d97aa1e908b584d88984d878167502855fec0c4f205d861a9a16.jpg)

![tables](../iclr_results/4093_Neural_Symbolic_Approaches_for_Interpretable_Actor-Critic_Reinforcement_Learning/tables/f3ae310da561e8902fc93e439fd45423e2404b4ce0eec72b0fc1591b065dc572.jpg)

<a id="paper-4094"></a>
## Pisces: Cryptography-based Private Retrieval-Augmented Generation with Dual-Path Retrieval

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=Re3A6vzCTC)

### Images

![images](../iclr_results/4094_Pisces_Cryptography-based_Private_Retrieval-Augmented_Generation_with_Dual-Path_Retrieval/images/0a4f0b574b361c433fdc970eab178da90e79c40d0177fd30d2bcb09481207c89.jpg)

![images](../iclr_results/4094_Pisces_Cryptography-based_Private_Retrieval-Augmented_Generation_with_Dual-Path_Retrieval/images/b08ff04c4aec6a6851d832e80e9d48734f2d82dd6a4be30fc347d2b0e4d5af35.jpg)

![images](../iclr_results/4094_Pisces_Cryptography-based_Private_Retrieval-Augmented_Generation_with_Dual-Path_Retrieval/images/c13c0438981e5144caabb6d897e6cb08ddbb963efd9bd0be3c54911de95e4575.jpg)

![images](../iclr_results/4094_Pisces_Cryptography-based_Private_Retrieval-Augmented_Generation_with_Dual-Path_Retrieval/images/d235ecc2736fcfbbdfb8786b417455b9d6102fe52f6f6d97af7671682d4839a7.jpg)

### Tables

![tables](../iclr_results/4094_Pisces_Cryptography-based_Private_Retrieval-Augmented_Generation_with_Dual-Path_Retrieval/tables/101561efd72fbcd16516b56535ecbfdeb3e1c564e6540484c5f29ea184df5d2c.jpg)

![tables](../iclr_results/4094_Pisces_Cryptography-based_Private_Retrieval-Augmented_Generation_with_Dual-Path_Retrieval/tables/4304a20fd7a94223469faf635cfd35608a4290d8e37b2ebd07fbf9ead27d6107.jpg)

![tables](../iclr_results/4094_Pisces_Cryptography-based_Private_Retrieval-Augmented_Generation_with_Dual-Path_Retrieval/tables/528ced799e0e4ac65bc60654cd9cf9f57c4a6a204fd9859e97eb5e19ebec9558.jpg)

![tables](../iclr_results/4094_Pisces_Cryptography-based_Private_Retrieval-Augmented_Generation_with_Dual-Path_Retrieval/tables/5c4146abe6106bc1c97566224dbd6ec2e49d2bd0ef386329c20bb16057d2884c.jpg)

![tables](../iclr_results/4094_Pisces_Cryptography-based_Private_Retrieval-Augmented_Generation_with_Dual-Path_Retrieval/tables/753281fab965e69990ca1e97f807f5198f9720eca9f3f3e19ae23244987d5af0.jpg)

![tables](../iclr_results/4094_Pisces_Cryptography-based_Private_Retrieval-Augmented_Generation_with_Dual-Path_Retrieval/tables/85bef22302cfd2a342ce984595a564d3136eb1583cbea5786fdee70cf2e718c2.jpg)

![tables](../iclr_results/4094_Pisces_Cryptography-based_Private_Retrieval-Augmented_Generation_with_Dual-Path_Retrieval/tables/9b22f1518ce17ceb986f4bb63f96f0f923c83095b1cf0f366c749fe81e7480bb.jpg)

![tables](../iclr_results/4094_Pisces_Cryptography-based_Private_Retrieval-Augmented_Generation_with_Dual-Path_Retrieval/tables/b455fe56130f18a2a75a29ce687b657d5ab560f714311b49dc5c8393fa0ca684.jpg)

![tables](../iclr_results/4094_Pisces_Cryptography-based_Private_Retrieval-Augmented_Generation_with_Dual-Path_Retrieval/tables/e88ec75837b1499ea5560286836e1ef9ac5e3f78c4c53e88ababe9f5cc7a478b.jpg)

<a id="paper-4095"></a>
## EUBRL: Epistemic Uncertainty Directed Bayesian Reinforcement Learning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=KASqlcI6Nm)

### Images

![images](../iclr_results/4095_EUBRL_Epistemic_Uncertainty_Directed_Bayesian_Reinforcement_Learning/images/433c8ac918f7f56be3796de626d001ec7c579b5c9c727d8dc83d89dd1c30dd0d.jpg)

![images](../iclr_results/4095_EUBRL_Epistemic_Uncertainty_Directed_Bayesian_Reinforcement_Learning/images/4643a3a4cb07fee2616668458abf9a1e49a5f60ff17c04ea331bf191f81fed39.jpg)

![images](../iclr_results/4095_EUBRL_Epistemic_Uncertainty_Directed_Bayesian_Reinforcement_Learning/images/af0a3b43efc07eba260f0f37d5f55939629a947092d1b9a4f5f8f1f75743b3d8.jpg)

![images](../iclr_results/4095_EUBRL_Epistemic_Uncertainty_Directed_Bayesian_Reinforcement_Learning/images/dd328094509fa8979ab2ac53c1359eb9c4c178cc978e95853deae9eba0870643.jpg)

![images](../iclr_results/4095_EUBRL_Epistemic_Uncertainty_Directed_Bayesian_Reinforcement_Learning/images/f623758f482d36f59985a383cfb474bd71274f291ce344c2ec88e0bfe96275a0.jpg)

![images](../iclr_results/4095_EUBRL_Epistemic_Uncertainty_Directed_Bayesian_Reinforcement_Learning/images/fe26204c33eddba833da07730acc9c5911fa78337efc51f25a5099bf4916a993.jpg)

### Tables

![tables](../iclr_results/4095_EUBRL_Epistemic_Uncertainty_Directed_Bayesian_Reinforcement_Learning/tables/043afea1bb839ff1743d8a1002b090475646f60002454cb53668cbd1e0448f15.jpg)

![tables](../iclr_results/4095_EUBRL_Epistemic_Uncertainty_Directed_Bayesian_Reinforcement_Learning/tables/0c54dc245b7463e46744caabc3bc1d7a44a2db25685527e8d87480127ae33079.jpg)

![tables](../iclr_results/4095_EUBRL_Epistemic_Uncertainty_Directed_Bayesian_Reinforcement_Learning/tables/96702082214d5defec0719624c2639583f21c02fee2e8a4726769c3d5f09d3e6.jpg)

![tables](../iclr_results/4095_EUBRL_Epistemic_Uncertainty_Directed_Bayesian_Reinforcement_Learning/tables/ca1a82236fa0c5878e36075f92f13532e8a509a663b5ceb1d8aecef376b0fbb8.jpg)

<a id="paper-4096"></a>
## Adjusting Prediction Model Through Wasserstein Geodesic for Causal Inference

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=pYLoHuLV45)

### Images

![images](../iclr_results/4096_Adjusting_Prediction_Model_Through_Wasserstein_Geodesic_for_Causal_Inference/images/3b0a561a85211405f4d41e54b0926f6845f1f1cbb14e8922547b4e34e845b8d0.jpg)

![images](../iclr_results/4096_Adjusting_Prediction_Model_Through_Wasserstein_Geodesic_for_Causal_Inference/images/50508c4787c76fd788f35b958d721d71631188b2dace21ed07794cdb2231f3a5.jpg)

![images](../iclr_results/4096_Adjusting_Prediction_Model_Through_Wasserstein_Geodesic_for_Causal_Inference/images/5969590d0ec76a068838d611cea4e149e6957fbeda1e3b20b8ff6f6514135e3f.jpg)

![images](../iclr_results/4096_Adjusting_Prediction_Model_Through_Wasserstein_Geodesic_for_Causal_Inference/images/ad7476107ffca304da44d5089675a79ee84d745c5f771b4a8616d632c7d485fb.jpg)

![images](../iclr_results/4096_Adjusting_Prediction_Model_Through_Wasserstein_Geodesic_for_Causal_Inference/images/ffd8943f78e59be523f6213c282e16f325b0229ef69e269503d86343295cfbf2.jpg)

### Tables

![tables](../iclr_results/4096_Adjusting_Prediction_Model_Through_Wasserstein_Geodesic_for_Causal_Inference/tables/130327c5a535094d2e29e03916aa5cd2ab0e61b60fcff0265ca8744a5fd8165d.jpg)

![tables](../iclr_results/4096_Adjusting_Prediction_Model_Through_Wasserstein_Geodesic_for_Causal_Inference/tables/1648d7559a1744f7fd664072753ae2a287a648d560273715474e2b4c76471355.jpg)

![tables](../iclr_results/4096_Adjusting_Prediction_Model_Through_Wasserstein_Geodesic_for_Causal_Inference/tables/7ad3be077f886f9fa810e55f15f1ce055f4637bf945aedc4b3ed16c1be6ab10b.jpg)

![tables](../iclr_results/4096_Adjusting_Prediction_Model_Through_Wasserstein_Geodesic_for_Causal_Inference/tables/842e6d9fbece71e7dee2ec508c9696950ac9f0b135b360a52aa5a0f08469f596.jpg)

![tables](../iclr_results/4096_Adjusting_Prediction_Model_Through_Wasserstein_Geodesic_for_Causal_Inference/tables/84d5dd8c195a9a3031aa24d3ee3fc703a5d36b954643cbf6a4b8e4eee60b512a.jpg)

![tables](../iclr_results/4096_Adjusting_Prediction_Model_Through_Wasserstein_Geodesic_for_Causal_Inference/tables/93471ccad54f46ec17a5441f45e04c8a5fa34dc30bcaae577f40615e0eccf45f.jpg)

![tables](../iclr_results/4096_Adjusting_Prediction_Model_Through_Wasserstein_Geodesic_for_Causal_Inference/tables/e001947790501cf1d93058f12609c109ecfbbbc566ab8d060ba7799c2cd7bb04.jpg)

![tables](../iclr_results/4096_Adjusting_Prediction_Model_Through_Wasserstein_Geodesic_for_Causal_Inference/tables/e012d31df99aa8cfe31a96408f1d81c1aa982f6dca2ce2b22bf4314f0d2f7c2e.jpg)

![tables](../iclr_results/4096_Adjusting_Prediction_Model_Through_Wasserstein_Geodesic_for_Causal_Inference/tables/e02745c84a00ac8fe0f35cf0b79f2a5dcfdce1ce5d721fb138702737258c3191.jpg)

<a id="paper-4097"></a>
## Concept-based Adversarial Attack: a Probabilistic Perspective

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=SoVgrFEgWt)

### Images

![images](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/images/5421c1796ba79e25997a573560bc0d387179c213388d2721354471189d1eaef3.jpg)

![images](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/images/63c59487a94b37230d003d0233112950c02b395cbd55252934ec9f5a4193c7d9.jpg)

![images](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/images/889d364e2a173a3c5c71cd90e505c46e2e0da013be4828363fb5a7e3b9e1706c.jpg)

![images](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/images/88a0d49d0e9569555928756f78324a6a4da3320211f3b9bfa8baeef7f6d9bdef.jpg)

![images](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/images/981668d3b1e4b8a041a017bfb306830794f9093e9ccb908ab17c16c728061ed8.jpg)

![images](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/images/a53471f7746e4796e5463bb3f9e7a4ba57bb3980c0ee4d5fe24d5d839760c6a3.jpg)

![images](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/images/e154ece41b4181a15663759634f580e02783d8090c8bb39e410a90ddf009f356.jpg)

### Tables

![tables](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/tables/1007f07749533d375c821cebb3e39394c2e2cda3d554112ad428a935f090c686.jpg)

![tables](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/tables/1e28bf51cffd1a2ab6dc50ccb3ad663b0d8a3e87f7e234fa9489e45a24e2f980.jpg)

![tables](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/tables/222be51d61641d6643491edf50aa8b352f6118170467d3326eac46f3cadc2059.jpg)

![tables](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/tables/3b894585c725b66121066f916384d40a258d63aee5fe59d7099ea25d2fc59bd9.jpg)

![tables](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/tables/408797d3d46fc4d80931e34b59179d01addfc88d27ee46a7325736d9bfa1396b.jpg)

![tables](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/tables/6838e7cea3cf8224ad4a49725f5260ff9dbe023243c38b7a0080b7618610523c.jpg)

![tables](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/tables/6e6d6241bf3167d72f3a22b9ce5c7130bcab32ac2c8011adb4dfd3db8f62d3e4.jpg)

![tables](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/tables/778a6cd8124afb01a441199a95d883fb86ab9c07ad0ebb7cae9838e648a56dd5.jpg)

![tables](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/tables/84edd79135f3049f899a02f3cc29de4ecd736cc5049d7adbbda9a070c245e866.jpg)

![tables](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/tables/8a1fff9b41fe50da27d5e8330432d533240d1da40eab08ea940bfa33d58e2560.jpg)

![tables](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/tables/d2d907505a859f09b241b2ffd170ddbd1101a6f3228c9dc5d6f60b3706c810f8.jpg)

![tables](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/tables/d70ee032c2e36dc0148d1a1640c9e22df1e4a3a6d82ae2a037286168f60c7dad.jpg)

![tables](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/tables/dead6e657d86c70c275e364c7e2da020913fddd1257b9f0301ab91bdf3d4f7c2.jpg)

![tables](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/tables/e6e02f1bc2ba9d5f4408fdba305251b211aa0d175a8e0cb54ed6d0682ab04e04.jpg)

![tables](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/tables/e8a422400bef19cb523be2b4829439f4d91ad1bdf9dd6183a16affe1fd11feb1.jpg)

![tables](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/tables/f28eb9b47fb29663bddf5057ce12c50400ee0cad58727da41f629d5ec11d9e12.jpg)

![tables](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/tables/f5d9d3fc1ba89567f3211b9fd85b327ccdcc9cb96a0d6999ec60b3bcf004b667.jpg)

![tables](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/tables/fe172df4f03a5018cb63421da7bfcd5ad2e74c1d8c03e835b9eb97a359835e87.jpg)

![tables](../iclr_results/4097_Concept-based_Adversarial_Attack_a_Probabilistic_Perspective/tables/ff9af6d271f8c730ec31fd6feaec35edb7b0034ec74a344eaa98587c1c7928a9.jpg)

<a id="paper-4098"></a>
## Alignment-Enhanced Integration of Connectivity and Spectral Sparsity in Dynamic Sparse Training of LLM

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=jZplmg7Ad9)

### Images

![images](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/images/038cee05b0e8fdecaeb74229710df82a301346362dbb6f691a506ad113ce50bf.jpg)

![images](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/images/2426e082c52a023f0eb72c8ce4cd139577a29efbba87c0037eb31adf7129b4a1.jpg)

![images](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/images/452bc9d56e5fd95e85c7a0ad4dd9a3523561933a9ed9018e7a48c289e2b65d24.jpg)

![images](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/images/47b77c2645ac86045085f2e1fa442c071eb6d17fdcf9925c7b896fe9c2cfa1f2.jpg)

![images](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/images/513d8556658fbe4d0fde567a7195c83f37a44c07a9c854212cfb451213c524d7.jpg)

![images](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/images/72d1a31ffa12d62ff2ed20095507a880c0fb87a7454216e21cf48683f2170b84.jpg)

![images](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/images/86b9a28b81811048c9f7cb3ee78d8cf62c9fb2057049b2e77fc1e2f99bba7130.jpg)

![images](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/images/8dfb200194577bd6d6d6402672934398b246e09a1fe290153ba71052f71c5820.jpg)

![images](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/images/bb04d2882a7ece79c2ddbd35b633f991091e7b462430014a8dd45acf1cc57c6b.jpg)

![images](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/images/f863e386b0da50d42cf0c7c547bc3fc721afdfdc2c30f0dd445c947a1d92a0c4.jpg)

### Tables

![tables](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/tables/0736e502831848e1b2c7162a20ccde4f687af4c0efa7bf29cb0f9d496d8e25bb.jpg)

![tables](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/tables/0e93cd62ef7b3c2018b4bf27f9fa228ab676367612005e91c2d39f7a4868c696.jpg)

![tables](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/tables/28382d05577c4df840d4b5d9dc60c7ede04c8624f91fcd8b103ac135f1b43130.jpg)

![tables](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/tables/293556046468af6a6db22b21fbde19f71aa66cd525db4a4c40dbfce6589cd539.jpg)

![tables](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/tables/53564c0a37b39c2d8194b3904012c76f81392bed10496222ead9f5b9b2a7d105.jpg)

![tables](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/tables/8aa44cd065b41750368e7710ca006f71a3315b649496528bdb60ed6e012cabe5.jpg)

![tables](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/tables/9dd01a2e072277935eee57485319a590587f1928be51fb4b856a3cd5200c1521.jpg)

![tables](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/tables/b7cb982a6ad5136761c52f7d6f1edb0a446aa618a243235a16686ce74ea7c2e9.jpg)

![tables](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/tables/c666d9834c9c195936115d3334f44ff9ba5fba73018e34df5cdd0af92317f143.jpg)

![tables](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/tables/c6678dccd170c4f08a0f8e9c2e2bac0095ed300870adf430d5fa3d1d3f194c49.jpg)

![tables](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/tables/e4d742e4e9e05fb69fb23e5a2a2888cebf439a6d30574011cf326647ec2db6be.jpg)

![tables](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/tables/f1336d4385b5f471704feeb5d45f717647182c1da8e90b447f547a93d10cbc53.jpg)

![tables](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/tables/f384b082ef6fba85f092b3de8ea4041204f011d95ffaaa7fc1eb900aff3df3ca.jpg)

![tables](../iclr_results/4098_Alignment-Enhanced_Integration_of_Connectivity_and_Spectral_Sparsity_in_Dynamic_Sparse_Training_of_L/tables/fadbd434d5d2376419976f4579c3fc6e1b29d09c09d862ef5eff3e811482f733.jpg)

<a id="paper-4099"></a>
## A Stitch in Time Saves Nine: Proactive Self-Refinement for Language Models

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=0GaCfBRFnf)

### Images

![images](../iclr_results/4099_A_Stitch_in_Time_Saves_Nine_Proactive_Self-Refinement_for_Language_Models/images/0350c23f382d61880560221ca588ea18e8cded64574e8010735b57ed54676652.jpg)

![images](../iclr_results/4099_A_Stitch_in_Time_Saves_Nine_Proactive_Self-Refinement_for_Language_Models/images/35425aa4fce755f13116047de04edc09d813bd7b21223178542fbb284bad5280.jpg)

![images](../iclr_results/4099_A_Stitch_in_Time_Saves_Nine_Proactive_Self-Refinement_for_Language_Models/images/71dcfa3c3cdda72c4b413352719fcbd317391bb9ac71b4e6ae894b062864bb41.jpg)

![images](../iclr_results/4099_A_Stitch_in_Time_Saves_Nine_Proactive_Self-Refinement_for_Language_Models/images/829435a643fc85e913f1efdf2b7e53949a59c24b13830ec36f813b890b6ab4e7.jpg)

![images](../iclr_results/4099_A_Stitch_in_Time_Saves_Nine_Proactive_Self-Refinement_for_Language_Models/images/ddf4d9da049be3d776909ff8a09ea0a5c5ea955e088b60d04f2b4e2f44952919.jpg)

### Tables

![tables](../iclr_results/4099_A_Stitch_in_Time_Saves_Nine_Proactive_Self-Refinement_for_Language_Models/tables/80cc65df92764587fd9be9519f1c58da9a93d2baa3ca200b1ca6d3bd70a7fba9.jpg)

![tables](../iclr_results/4099_A_Stitch_in_Time_Saves_Nine_Proactive_Self-Refinement_for_Language_Models/tables/aa6e4393b8d97553c78801d2eaeb6075f4b661f323444acf89f632653f859690.jpg)

![tables](../iclr_results/4099_A_Stitch_in_Time_Saves_Nine_Proactive_Self-Refinement_for_Language_Models/tables/ca246ea90505a7fc8f5320235ede54034b8d62aca93f9abf1566c3e3e93829c2.jpg)

![tables](../iclr_results/4099_A_Stitch_in_Time_Saves_Nine_Proactive_Self-Refinement_for_Language_Models/tables/d5dbe4615acd15812a9a288f378b437f4fa8248a6d5413dfcbad0e1b137cd91a.jpg)

<a id="paper-4100"></a>
## Rainbow Padding: Mitigating Early Termination in Instruction-Tuned Diffusion LLMs

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=cznTlh7Msz)

### Images

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/04329d4dce8b66accaf81e93570d09fcc1943da579a5aaf2444df7081cb2377d.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/0697ebe0bccd67ed56a06bf26f2ed35b8fac9c6ff254732d3757c31e62409de2.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/090d7f9d4358a349d0e53a0faf6387617407dbf8a8fc28dc8f705ae317719c60.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/1bef8cc037de2b9560507c746b599f429fbca52eece25a929268e650ccd82d52.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/2580493a828b69ea60047d5a77d6d10ebc526bf9de1f1bc77ab6ee6187416db6.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/2bba639951aed4137b08a5ceecd7d78ca312cb979ae8eb89cfde8a711c435f88.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/35b5dcedfe8ecd707707bc224b1e82a48dddc7d10996f301f005aef416049e95.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/4ae20d76e66fa87f248a335608cccd7df50c58dc464a5428f229df4b7a73d2e3.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/5bef1ef6bbd52f9c1c26eb17bcfd5168fd7408b079d4577e7602a6ea9093e5aa.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/6a5654ae2b7338233239ad7a985e3e618b3c622063bfdc71d74d8b3dda3c9e04.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/91423247cf78ae7155eab6c1d0a89a15aa11a3575284caf7993efa3209e687a1.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/993f410a9b87183ebc2cf16e98f87343cf788d04f06fc71da989a6162c6836c3.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/9c57925a48a6a66acd5e102dcba6ce31591c113d21b0dc1aabd55f4ffaa8d366.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/b0969a4cfb63f05e2f8c1ed411f139c04741f273a095d408b1052b9405a2be4a.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/b30304fadacafccc5eef5f1264637c3b5e9c00182e9e10ea0d4f4c435f128cac.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/b7fd284c4a0361326e30eca4dbea06dcd1ff73752304ab0e20c302a10981c2d2.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/bc70d59bc080da4be5de070581ebec6ecf0c87f3895bb98df6b4a19f912f0024.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/caa2559718b1e24113dabcd5c77da04e9679997a53ef32479b6bc1c8eeb21c1b.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/ce0f60391bfc08055722df6d791ab933641d93e5192fb80ed0a60d816d14aa23.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/dca78941a77526df2346c172e23fa131f68b1620f1baad8df976675f49f2166b.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/e69c278746daf654cc44a20922a42695db4866bdbaabf1a1a49357ed6512de5d.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/f61bf086df5107ad0075dad4436813d9e5b35e9d8c5caa2324b4f9c5e3107b55.jpg)

![images](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/images/fe9e15d62864a1af6ca5893c2404480f5037002955be1882f70464aba263ad6f.jpg)

### Tables

![tables](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/tables/03622604840ed9f9921601e83e4b2a7866ef610c3a8c2d95d4e9e38f4240381a.jpg)

![tables](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/tables/4f62947cded64ab2b9df420064289fff0c0821709d28cd5f9c0820532335e17e.jpg)

![tables](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/tables/b1dea1726b1b3adbc9b29ad0988074ac662338f3f00e922d059221747d59695f.jpg)

![tables](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/tables/bb3128d8c77ad56be81cbdc7b6c414f5dac2bfa4b711fd9ca8cce0c36445e9e2.jpg)

![tables](../iclr_results/4100_Rainbow_Padding_Mitigating_Early_Termination_in_Instruction-Tuned_Diffusion_LLMs/tables/c23d39b284c45d55212b6f876fc57cf5e5be81144aa64e5d6a4e356293dc4077.jpg)

<a id="paper-4101"></a>
## JointAVBench: A Benchmark for Joint Audio-Visual Reasoning Evaluation

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=Zg1YH8R5GG)

### Images

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/09edaeadd0169fed9622f8a8fa6e9e578c7c37073efa9c2d219352a7079efae9.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/29ca21358fe9a8a8a742005600be141d514d8588768d6badcd7f763e17f55797.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/2e529bef74e078b8c370aa2890fc92beb126a3c0d5783d8993947b6fb97bf497.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/30fda1a7a3fcc22f30b948c1eeb9fd55809f2a94ec45eade1d8185c8597d8596.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/3850be385c4f46e825a241ca19b045413c5c995c250122acebc0f346d062d736.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/3f3545db51dd816944a059ae42c447d8f1b1d6685c2672a16eec1e82e96c01e3.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/4c41117a9592bf22e01808ca0dc3cd1b9a7d0c4bdd6f4fbbbd0a5f7b739465f7.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/4d84a1fa23d1e7f8bfd1fd2477c7b1bc460c7c8ad3923dbbd4a5c12b5c3c14b8.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/4f6218cdb8a8ab09e582d5b998ea79da870cba7a64bc5e9ad1456d60121c35fe.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/631049a3b820b6ed751eb88dd7ade9a0ce983c248b1943ed9e56f08c5ab0b115.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/73fca40791c9336b4fdac386678ee36911d7ecf03512218f72183f7698645cde.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/7d119ec9dffe359eaf20cf43dc3a3636ff151e88d687cfe99cfd4fc1a551e123.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/83c03203dc96e513846776fa2fd9cff554d303e55504c63baed0ea18a7c78494.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/995bff0ac9492874cb14ab92a17805513147324b6971f3c1b5132727ac9cec57.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/9b1326d283976643edda805a1328362a02c98a231607bd4af48929c4663aca08.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/bc37fb862775c5f79e5e3d25f4c98e2da638e6efc65bf1876f842deb102c736f.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/c4e5c24a3385be6b7388b483caeea672a9397eba54dc2695707656906aa57f1c.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/cd2fb3d496a79bb082a3f9c92e489acc69c0290dfda5b7f6aa0dd3db54c8b87f.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/d82d94a4fcdac0ea9085d846611ac48466cc5e3c22f1d1d571971e0a6150a061.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/e7b8d1287f95f64c28bb03cd358b0d39b4412601976e350b81dc3c326cc55838.jpg)

![images](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/images/eda0f05265258233455f6a5a00e3490d4852dd95bf399f1f5e3f003f746cc288.jpg)

### Tables

![tables](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/tables/7e366be2c800741bef6cd1971b7948f369dbd0465d0a0fb25a882ecff8b1a6e1.jpg)

![tables](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/tables/9872f0643f31dd3c0cbff814c928910799a8f1d455fc6b2e9aefbfbbdc6be7e9.jpg)

![tables](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/tables/b91c01f8af420ad856656bebb1d2abdd750444a63613be8377b8c46ba85d51d0.jpg)

![tables](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/tables/bc9505c1acca2949b1d6cca6ee0cdb345069ee44b024f20b310c2ce9c1daa567.jpg)

![tables](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/tables/e1e05430c3689153e31aad7f53abf14aeb52778e977e153173d201cf45b0117d.jpg)

![tables](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/tables/e21b197a63143a8b893f65ec282e098b86e4dac0a4615fa004be9e84a5d182fa.jpg)

![tables](../iclr_results/4101_JointAVBench_A_Benchmark_for_Joint_Audio-Visual_Reasoning_Evaluation/tables/e4469187dfb39679ed4c2ce3cdd351d5541b255e0fae80578be3e293c3d092f5.jpg)

<a id="paper-4102"></a>
## Robust Adversarial Attacks Against Unknown Disturbance via Inverse Gradient Sample

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=WhFS8mxWJh)

### Images

![images](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/images/2cec89e81b6673ce90cc805d8fc52b5176a35afecc7457b7b925d61a6bb74cfb.jpg)

![images](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/images/59efa54dd44aed612e00b3f2832bec4973a3e1c57c0cd8f42f5afe03614bf857.jpg)

![images](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/images/854586e264babf4ec344f4e0393fb982e1ca4405faa9355d2384055b02a8160d.jpg)

![images](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/images/8fbdefde7bd289870abbdd55edbb51681d249c8cecc2aeebb1f7085b301780de.jpg)

![images](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/images/b001c45211ad0fa33876a015dc00c080ab9c07c01201e539316cebc944fe7ef4.jpg)

![images](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/images/ce8352d3431221a7b8bc1f1a68348a531e94526c9f67c9fd8632c2426ad5b530.jpg)

### Tables

![tables](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/tables/0ced64deeab21cd284ade48a2905cbcfb81d349a6cd4187b5f44432590672c9f.jpg)

![tables](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/tables/137a3bea30f4eb438a1c100845db6b4319eb6a7bbe79d0b5014193fb8ee95257.jpg)

![tables](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/tables/1dbb0722e60986f5ffaf5f2f641424390b563215bcc79ca17df709e572f9f002.jpg)

![tables](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/tables/2720977da45fd053bec72a2fb6b34a9e88949c5024bb20c65617b90426b1f217.jpg)

![tables](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/tables/30d6026ef38fb7faa57549c4da55a86f711c52f38fd0904b491a77998ef84c69.jpg)

![tables](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/tables/33da7344723b60515d94a0c8bd3dc83902c37a4a64a6c1183e69ba21b3a37fb6.jpg)

![tables](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/tables/37acc09fcdc8a1bde715d443d6995bad27de319807276a85fa903cb058715044.jpg)

![tables](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/tables/4499360ccd90470434c88ca19253ea59824935f77aa3cae016d59bc0233bd91f.jpg)

![tables](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/tables/45fb0096d9a3e098fc8f147868bf96d6634ece84c7592c1dd4407d13daca2374.jpg)

![tables](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/tables/5d74f971bdd7973dca71bf31cd62d7819b22fe741cf0488b86b0291d83929882.jpg)

![tables](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/tables/79b090f3a338a1fb719f5706549751d8c8e4891c76d7f974f4ba1e9f90dfc481.jpg)

![tables](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/tables/82f59f221c391c883606a78c7ccc5646c79ba37f44a368ce86aa3a981a91d973.jpg)

![tables](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/tables/8fb9ddfb4ac03480897f3be9a581e97c92a97b585a6b4228ced5ff1723768c0c.jpg)

![tables](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/tables/958e29e252642a8066976bfa190d3af197c6b5fc1bd92040d89be980ece2fdaa.jpg)

![tables](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/tables/9f00c09cf55dfa988501ccaa7fd204f6719886addbc7caa2d5babe1ea1e086b8.jpg)

![tables](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/tables/a49bcf53ae61ba14fd2eb5cf30802e401cd081321871c8d9c2062233e05e128e.jpg)

![tables](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/tables/b4cab121159224abb1aec4535e879c723279e5b0341a0b5270afd07a3b38ff35.jpg)

![tables](../iclr_results/4102_Robust_Adversarial_Attacks_Against_Unknown_Disturbance_via_Inverse_Gradient_Sample/tables/d6a5bf1f256bc9989257a69e02326ffbd454276580e8a42ce5101013faa122d4.jpg)

<a id="paper-4103"></a>
## Local Entropy Search over Descent Sequences for Bayesian Optimization

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=cPxmLZmFa7)

### Images

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/04992119993b9d87954dd2c45d08dc9d4b5302a300df97062fd815e7e0c77c4d.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/0a13511dce3087797d19528a2fd64e42070def3c4216570629866c8478ceefe8.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/14f78c7c1c8a5dde2dc7fefa341b5f30476b68e54f5b477d1984b4fca2a2104b.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/21676b39a0da52238d65ad2f9f1f97dc04193138e0162a1f12bdb4b7515937e8.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/2c0307af13f12bf378980becab1cc65f3eff6f81894af0b588fbb926c6ae435f.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/303837ac4a9ae6446d36b9f9fb8c9fd9acc5b6b3f79ad425676210383d76ff03.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/3320840aa901cf742e1fb761e9155c93d10ac8888010fbb60022d9b4ebf1fd3e.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/4109c4530f544a9a5de224abf4cbbce9515ca64e99bc81c9787e44313ad6bd0f.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/4bcd262bb317499f959cb7e42fb0372bb1afa603911d494a4daff852179f578a.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/5474374fce6dacaf66bff5103af4aaa9d8cb9fb97f52ae6db656e5179a3b1ebd.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/559c5642108f7f9592dbe0afa934d1a2b4cb2ae651978d10c47878af84ed42fa.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/57471c05d00b994bb323d58beefe0171eb4f4a266dc4d442a4c61a19bbc23468.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/5c55d307a32e7144e71be6eae5f0fcfe6c37df17a7a824b44fb3cd910710d66c.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/7babfb077f0dab2ba0bd2dffb09ae79f6cb9e36719868f35aaa36ebec8105869.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/8042a7787100c0e7e3e16650402a3d29592c7608b5c60b6203f662c01673037e.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/80ccd5d25dd75849cdba2ccc42ff21fd47f9b78f1ea754f9406cca5e9aab6ce5.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/811254cde7ff38c63c3ef2e2134e6e603a946adf8fcf790cd0dddb3b451f2dde.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/833c86dbbd38db1a43f698ed25132cc65fc996490a17bdbfc949261a9d378465.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/8c92ce0ff513542326c78aee8466b98586eced06a5de0c0a7263e186662e1ffd.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/8d02ef8ad6bce50d0ead1e6f960df11ac9c62c2f6e7f6b8b4e5fa913b803934f.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/95ffb617a0a4adcff61435e92ee81f9177004b81cec334fc23c12e40823aab96.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/b3aa808cd3b0bd57b2200814b3a4a0bbb5af457e7c5ecb33bc0605afb88938a8.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/c80c4674857602cf8c908f50db2cde1ff2db97ca4d5f8221db0ff5d7c325fe21.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/c9eb2e9e95512ca1d2ee381a8fc57c1deba49dd2c876c16e6b29d8aa83eafad7.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/cb4163f5e21ab45958c2215f5b762ff68cc4bcb2b7292857a99b6c25fd1e2507.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/cc42d7bea82e098d6bc89906764640b84c5707d3068d421548d53c15dfd1a5c5.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/dc3d715ee8081cf2ac7858b10199426dfeebbadf92f113b0b2bac64ef8153adb.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/e1bd2376b559ff1d24b6d0356a429587cd0cef9c135f10376ecb883ae0d4374e.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/e1ed9c564fcd67b89efe1c84189e7360dca014f518e6e051a6303bf5d659fbb3.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/eb4e863d5bd6d9f13b8975d5d48edfffa78e3b9392aad769472c12cd87e707b0.jpg)

![images](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/images/eb724520d4772313ceedc322531f95a567213728adecf1ee86f6af4e90992e42.jpg)

### Tables

![tables](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/tables/0fc3579ac8dcc959bdcee5e51c0077cec0faf8fa3f543d996bf1be3dfa29aa5f.jpg)

![tables](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/tables/251d1e20af27612b4cad05920f4cca05b7c8e12de73fd68386b71062facfc1e1.jpg)

![tables](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/tables/46c20b42736562af4accb108d61d8684f8eda28d79c28522e51895524c7da164.jpg)

![tables](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/tables/63471a512ef239bab81b9f0e88ea429ab7fe133a801c0fb3319e03651dfc5db9.jpg)

![tables](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/tables/88e15b1c0b7e4dcad75d861248805a9133492f424f93de24e0a081150494f4f5.jpg)

![tables](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/tables/8959d63405e23d7183d40ba10abb0ed0a1aa7c48205235f548d11b62ef6827fa.jpg)

![tables](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/tables/9ae75365159b94c77d275eb50050c94a7a8edf49618d049f5c72fe81bd82da93.jpg)

![tables](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/tables/9b8791ee18a3b0c11c270945ec7276556a22a2c44636a8ae039ff08e39500161.jpg)

![tables](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/tables/a56a253047ce4cdf955cdc38bcf0f871e057dca624b4a1f4253e6c3b652b2325.jpg)

![tables](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/tables/d421af575a7dd5eed1a2e88e9ee2944b948b42764148d21c01f5e186016da3bf.jpg)

![tables](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/tables/d94bf380ba74d5b68c1a373579819dfd5f9332818a99dbeca1d5b452063f2baf.jpg)

![tables](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/tables/dd4e9f0e9442161e71c337c9454a0ba7202fe172be1b3be1d0b7dc00ba1ce4d3.jpg)

![tables](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/tables/f474c7f379cf78c02d2c4162949907b2d704dce510f28227359327fdae4ae3dc.jpg)

![tables](../iclr_results/4103_Local_Entropy_Search_over_Descent_Sequences_for_Bayesian_Optimization/tables/fc256fd3a183fcdaaa2a48cb929f769c0b42981850532aebaeafa346e771bb2b.jpg)

<a id="paper-4104"></a>
## From Sequential to Parallel: Reformulating Dynamic Programming as GPU Kernels for Large-Scale Stochastic Combinatorial Optimization

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=T9FDmsmoBj)

### Images

![images](../iclr_results/4104_From_Sequential_to_Parallel_Reformulating_Dynamic_Programming_as_GPU_Kernels_for_Large-Scale_Stochas/images/08e7a33b4475c13993fb28995b3a31d122f0e12b2b513acf0762852d4f1fbb3a.jpg)

![images](../iclr_results/4104_From_Sequential_to_Parallel_Reformulating_Dynamic_Programming_as_GPU_Kernels_for_Large-Scale_Stochas/images/11d8c21b861b7b90968cfe87926664ec949b0a6926eabca6da41f04d7755255c.jpg)

![images](../iclr_results/4104_From_Sequential_to_Parallel_Reformulating_Dynamic_Programming_as_GPU_Kernels_for_Large-Scale_Stochas/images/213f05f5303ee41194df8eed87b5e2eedf2270f4c087c60cbfdd3c8528cd7b0b.jpg)

![images](../iclr_results/4104_From_Sequential_to_Parallel_Reformulating_Dynamic_Programming_as_GPU_Kernels_for_Large-Scale_Stochas/images/3036c44a4bfae3801abc734e2c66239d0a89603f56db2ae978c15398fba229ed.jpg)

![images](../iclr_results/4104_From_Sequential_to_Parallel_Reformulating_Dynamic_Programming_as_GPU_Kernels_for_Large-Scale_Stochas/images/4d05f296244c82b5db6ed5572ec152606106e863355467d9c3c1c9a2c4508b53.jpg)

![images](../iclr_results/4104_From_Sequential_to_Parallel_Reformulating_Dynamic_Programming_as_GPU_Kernels_for_Large-Scale_Stochas/images/71deaae0009c9b6da435972553e6c8535be0c32e1d22880e14c4d0e4338ea702.jpg)

![images](../iclr_results/4104_From_Sequential_to_Parallel_Reformulating_Dynamic_Programming_as_GPU_Kernels_for_Large-Scale_Stochas/images/7afd4bd351745c2f05fd332cd3176e49c8d63ca2e0bed3e51e3f2c0968e795da.jpg)

![images](../iclr_results/4104_From_Sequential_to_Parallel_Reformulating_Dynamic_Programming_as_GPU_Kernels_for_Large-Scale_Stochas/images/ae0cb3c1054f79be75c5741576bac038ae9ef1a9ba90a97fb5930ea60314dae5.jpg)

![images](../iclr_results/4104_From_Sequential_to_Parallel_Reformulating_Dynamic_Programming_as_GPU_Kernels_for_Large-Scale_Stochas/images/e59da0d23b53950babaa5340df9cbf97affcc53151ce413f504223e737b4e0c4.jpg)

### Tables

![tables](../iclr_results/4104_From_Sequential_to_Parallel_Reformulating_Dynamic_Programming_as_GPU_Kernels_for_Large-Scale_Stochas/tables/23b054f1270086e8a8e187451a7c8df4a0416ac1fdcb3eba91db471efbc249b6.jpg)

![tables](../iclr_results/4104_From_Sequential_to_Parallel_Reformulating_Dynamic_Programming_as_GPU_Kernels_for_Large-Scale_Stochas/tables/2e04564895aee65a9f5c522586103d9717dc52091fca8842889930a80aaac173.jpg)

![tables](../iclr_results/4104_From_Sequential_to_Parallel_Reformulating_Dynamic_Programming_as_GPU_Kernels_for_Large-Scale_Stochas/tables/544dae5afdca1d58b93111176f281a14a4f87eeeac276c585a3c777dac4704f3.jpg)

![tables](../iclr_results/4104_From_Sequential_to_Parallel_Reformulating_Dynamic_Programming_as_GPU_Kernels_for_Large-Scale_Stochas/tables/9c22681da644add9adb8b2a38775d6f899b1ee205e983a28f5d7c1af09653154.jpg)

<a id="paper-4105"></a>
## Selective Expert Guidance for Effective and Diverse Exploration in Reinforcement Learning of LLMs

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=axlFycAkoL)

### Images

![images](../iclr_results/4105_Selective_Expert_Guidance_for_Effective_and_Diverse_Exploration_in_Reinforcement_Learning_of_LLMs/images/4483a769e680f0d4b0cd8ce085c12102d6ddcc3ab0d46bd2704769e690bfc97b.jpg)

![images](../iclr_results/4105_Selective_Expert_Guidance_for_Effective_and_Diverse_Exploration_in_Reinforcement_Learning_of_LLMs/images/926af00f874100db5dc39762db6a636299c8d4b57e50b626b371eb39d2fc5453.jpg)

![images](../iclr_results/4105_Selective_Expert_Guidance_for_Effective_and_Diverse_Exploration_in_Reinforcement_Learning_of_LLMs/images/e866ca4b3d80eef31cae2eb4366a803acd6db46fb02679082fdbb06b465fed3d.jpg)

![images](../iclr_results/4105_Selective_Expert_Guidance_for_Effective_and_Diverse_Exploration_in_Reinforcement_Learning_of_LLMs/images/ec9e0711634e80ba19f6e6981b7318ff5be2e9c5ea3310a8383dec75a81fd22b.jpg)

![images](../iclr_results/4105_Selective_Expert_Guidance_for_Effective_and_Diverse_Exploration_in_Reinforcement_Learning_of_LLMs/images/f39254a343477d70dc04771aada6e158ca4aab32617eb6a8f9d80f3335029892.jpg)

### Tables

![tables](../iclr_results/4105_Selective_Expert_Guidance_for_Effective_and_Diverse_Exploration_in_Reinforcement_Learning_of_LLMs/tables/1145633d9c7778f7e62af3d20641fc60d49f849b31b0612bff81e996e40b2009.jpg)

![tables](../iclr_results/4105_Selective_Expert_Guidance_for_Effective_and_Diverse_Exploration_in_Reinforcement_Learning_of_LLMs/tables/6b2769d61f1f2fdcd0bb3360a13a55be4f9c23cbcca98acd2d0a4a4bb95efbe8.jpg)

![tables](../iclr_results/4105_Selective_Expert_Guidance_for_Effective_and_Diverse_Exploration_in_Reinforcement_Learning_of_LLMs/tables/6b9fe729eccaf5f56194966c5a5ce3125e61437f6e36cc7175496dd40975a683.jpg)

![tables](../iclr_results/4105_Selective_Expert_Guidance_for_Effective_and_Diverse_Exploration_in_Reinforcement_Learning_of_LLMs/tables/6eddcbbe9fe0ed0bb91b112449a8dfe3e962eda06915bc3cd917c7e910c920a7.jpg)

![tables](../iclr_results/4105_Selective_Expert_Guidance_for_Effective_and_Diverse_Exploration_in_Reinforcement_Learning_of_LLMs/tables/adfac4e2cae9f32de30ab8599c205bb413f792bbd8a1f0d121ab7788620802b1.jpg)

![tables](../iclr_results/4105_Selective_Expert_Guidance_for_Effective_and_Diverse_Exploration_in_Reinforcement_Learning_of_LLMs/tables/c8d26c6f1ad409b531c444f4517f811a587653de9d7928bea294cd2b0c53ddbd.jpg)

![tables](../iclr_results/4105_Selective_Expert_Guidance_for_Effective_and_Diverse_Exploration_in_Reinforcement_Learning_of_LLMs/tables/d130df4839e014de1f67a8621bdbde0fc48d6f1731ecca6bff5c2df9aaff6322.jpg)

![tables](../iclr_results/4105_Selective_Expert_Guidance_for_Effective_and_Diverse_Exploration_in_Reinforcement_Learning_of_LLMs/tables/dd4dcbc947368d93f7d0a420bd7b569a82b61b2340c4b7795861c68645d0dcad.jpg)

![tables](../iclr_results/4105_Selective_Expert_Guidance_for_Effective_and_Diverse_Exploration_in_Reinforcement_Learning_of_LLMs/tables/f6e7fc2d7c6814bba72b24f20987b5187bb9e2e328b77c6c63988e3bcf20dbb4.jpg)

<a id="paper-4106"></a>
## TABLET: A Large-Scale Dataset for Robust Visual Table Understanding

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=5UbeQDlYDj)

### Images

![images](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/images/036d628023b80ccec10679e26510096dc2813895f3acc99d4786a90fc1973b0a.jpg)

![images](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/images/1395842e8b4efbc986f16d5db05373f14c4b67eaf137bfa2b3fbce8b0a8674ac.jpg)

![images](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/images/2232ca428ea8f3a99b0fd01d9b5890ffc6580bba205f049a8585dfc40fb91199.jpg)

![images](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/images/2329c065a3434b035831215eeed08650868f028b90e1c7620c52081a70822ae5.jpg)

![images](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/images/2c79e5df29934f23880787030ee3526f74afe338e43057084e8972918bedc17c.jpg)

![images](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/images/3d002d27fe08b4a77281b57a383de6a5741ceb589e07d0141ca0607dfb6d9a4e.jpg)

![images](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/images/3e11e03a74d5e464e9206d27d012e7cd1dcf7a4e1cee4a4565abbdd3a3903474.jpg)

![images](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/images/59cd186c0c890922eaa2cf6e24ccd728de5a4601ae7c0ff2bcc668cd15c2fc33.jpg)

![images](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/images/6d420bb5f192e9a1689f1aa3effe85a4a8999250ef3b0696de6ecf70c937f8ac.jpg)

![images](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/images/77c08447568afdb2af745f11eb3c8abb514971c27a7075accc96bc61924a7f7d.jpg)

![images](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/images/7d62a8beb423458c5cf3ca7012f4dca705c5399cebbd31d9b80f493dcc87e482.jpg)

![images](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/images/b4da0f796ebbcfabf3c6e00c5a51429a97d5ce52d440681f1e7df919aca94f49.jpg)

![images](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/images/c0735d6c9614473f6559672e2a5c67c8db621773fcf78b3457e7be7939239bed.jpg)

![images](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/images/dbc5f1f0175af64c5b56c3697d515343409da2cd8269a66a0d868d4ef00360d5.jpg)

![images](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/images/f0d73f2a2e386f1227683d0a32ec3fea45ec456357f51873fbe94242c5a2ad86.jpg)

![images](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/images/f1648830611986478be4364354297397fe9bca6f82065ab119eeb1815a911089.jpg)

![images](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/images/f8c7f9a619bcfaec90a0a0a201f8138226c52207469a0f0fff5859060ce3de46.jpg)

### Tables

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/0b95596b3bac57b787b78744a4acc0b73818db3faa734c73aba8afc3d4d1b023.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/0ce1420ca9d9a9e035dafc3a4867b326a514cda11dabe820cae7fe72a94f4c04.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/1588d23f474389fde9ddcb45fb9b9d47efb33bbccdcd0b78931bd09a89a8aea2.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/24494b28e1789e812c5b88de2758516d33ed432bcc238966b6c585ae2a31a703.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/2869911dd6b3cbfa3d269f06ebd2742c79622a7ff9c4327a8d4bdea07f998c45.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/2f8f35c9bbade972a5ed77f7402d300442fbe0edba706a9725ab8189cce98f12.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/33c3f48e473496a6a38e52228cddaa491675ceb43e73d18d8c1296b4b7fe4b4c.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/4f057bd32940cad3b573a05ac0abc3ef17c709060decffd88b7587892889954c.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/51200b9284aeb3bb92f200795f2d42b441b311edd76d306dd85f2c93c19629c1.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/66e884e397113429243f26e262e5e8bf9d99f937f980196ef9221b25bdf72375.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/67e8a5685daca9f64691e688b274a547e4e4edae1338ebbb636c9a16f707e640.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/681312d3854dbd360ff4550eacc56b6e6622e2480c93a2a799df71ba71f53553.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/682e574784efb03091ee0e3419a249581aafee5d0bdaca7d40d187a7a0762c5b.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/7456603767ea0008a937903a7b41bd2f79e7343fffa13ad31a3e563f7db754aa.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/7c64bf7383a34b4a9fb36eb1b5f4132cf42fed207d138cc0cdcf7ac46fd4f85a.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/830cac086a00608b703d40780f19ba853d23a61b175d5873d2a2d0bceed6758a.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/8690d3405722825afec9c436bb10727cfb8c865a958fedae90d10745a27507c8.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/86c28c69e9cd9e8ee9210421aa6ee497332159fa3ed6449433df87ba41088acb.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/8c6a96a191206b57d4232b9a8f885ad8a9ccfde4c428701b1e17b1519a314493.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/97eb255afef27658ffe25d53391e15293b29da50a94eb45ff1670289b41576e9.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/992ac69d28f1036857ed7d32ad502db2b4b6519ddece049ac07c508b5ed029d6.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/9b4ae368b05cdc68b401ee8bd724fc1dce5e1f0129ee0c09b0a47e1376d5d006.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/9ce44d0bed7697e7fd9f02f8766ef04669db6370e0e5261eb0177babe7538e45.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/a4656f313d94d7f3f60432e2290e1d61e270cb61964d38aeb130fb390d490f04.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/ab6dcfdb52b7e8fc30c284080d8ed0e3889ce513febfc1f8cb0fba5e747fc481.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/ca4bc91ad978753443a925d01d5aec52d9feaf9eb725ac3e18d0a97be1a33ed3.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/dab3abddc334ba068e51ad6b1394b39e229473b6a83a690bf8f33a2bcd72afc3.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/dd1bd4dea2a242b63ec8f191c7d288a46cfb49731ab26cf042c458f7f9795327.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/e9e1a14c29d96be5d54009adfea86bd704e10349a2dfb07e18091f3c5e14c70d.jpg)

![tables](../iclr_results/4106_TABLET_A_Large-Scale_Dataset_for_Robust_Visual_Table_Understanding/tables/ee4a05224fddfe65a2e020ff71f913af67b3066dc481f2b8ee3a5617fd3a9cf0.jpg)

<a id="paper-4107"></a>
## Fused-Planes: Why Train a Thousand Tri-Planes When You Can Share?

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=bAG7lS1AUL)

### Images

![images](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/images/08ae3a1f75381a0ac752da136a9c3e669f762cadd2dba95d1e00ca1e31f02e1f.jpg)

![images](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/images/0d7cba20c23c9716f845baed928ef60107368efa68bbb6d3dd0c88e659087122.jpg)

![images](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/images/185a4349f7a7f02475431e5df1794a1359343e9610723d5dd4b1237af39e9965.jpg)

![images](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/images/1efe8e970108b3b5c88c8d6a0f64ae0c2a6263ee57c38db2d84fb92865b4bf8c.jpg)

![images](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/images/2883748ad4a489e77a2af40c30d6c1ab29dbcda0960f768bb32ef367bb0ffaf7.jpg)

![images](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/images/3e097d2078a4de905de1a184b76e0539e85c0567178b984b6d01d4ce432a4bd0.jpg)

![images](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/images/720b45b4743800dd331ed7557240cc912989737632d6252cbd2f2ffbd8e84f85.jpg)

![images](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/images/7ec920764c422c251c9e2b7bf2813b59b49761d2c73a5322e4dfa4c495abd29a.jpg)

![images](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/images/84f6de0bfc4ce97c45ad3b546c1e53502e8a8601780d08c90ab9bc9a566318df.jpg)

![images](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/images/a87d7b6943712f47aa0ff59b1cb67911b369f0cf1331cf5d9fc73526a1bc5220.jpg)

![images](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/images/c423b9f0f2e887d2bbe6d49e02864926a1be01273ef2bce86c3a849a54fd7b69.jpg)

![images](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/images/d533d8e9339664c0bae70391f2ecf3897761d85cedfe8b875ac866e96e1c038a.jpg)

![images](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/images/dc0e08783fc2d868b5f0ab433a60523c51b5c545b8d6945743b25103e6118142.jpg)

![images](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/images/e024985012493e901086fff746712f10d44b44a575264d8efdb375b2e3e2d5bb.jpg)

![images](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/images/f8641aa3737737c89cca6cb846550338d2bc86b03857777507f40aff3a69c68d.jpg)

### Tables

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/00ac7c844090eda33d684971c72687b6ead1819ab3c38ec7f8fbc0cea1295a59.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/08cb9545e31519ba8799b4311d4e753aec59283962dab3d951bd1192cdb083a1.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/1514c1166dae90bb44a1fe7c714d06729251b2b23bc0f26c3d1bde0562f16ab8.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/25b9522cbc2c24d7b89bac2ee891a1ae6ea31b29cdb89ca293a34fddbc33ccc2.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/3e457f87726dc041c33d79cc5877e2da124e14d8d39754dbb7c6568008b38ae0.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/42874e69c9c3330a44eb878f92bc2ffbbe1ab1a5dfa6eccbb7aefc4a20ea3dc8.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/49b630326516535c936af6b7a2d71ebf3df0728fc5743db23a9a74290253debd.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/4fb28f7fb36b9e24de5500c449c03743b9b5a0651719c98288309876025e5441.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/54ca42406d1a854ae190e442e169e376f4d386d7b98d48e5765641dfa4920e40.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/6bb17140ecb5e03975d6d11b6cc42c2e44e34cf1e443e7ac2e059933dd27d0eb.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/8630bc104cfc8ae1511586f9f29c952f6bcb11da8da59696cad99c0a932af3fa.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/8ccb08b849440e7481081b13abdce429f3ed7c2d1a4260ade503ff9313fcdd0c.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/96ff8ee9263bb423af4349a36aca35698d2260d0390890e431fb26395a13de2a.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/9cecb44b12d9d9ef6159fbb43118939f0b17e764a38766fa49e022994f17a10c.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/be9cd76b70ab54d634cf5d0f787255fecc8039e17c4b0298dab68cbaef1d62a1.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/c8a735a2492458755c0b20197eb409d2213c0574f78c64aa2c8aba9925fc0818.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/dadbee26c25deb0b355542c808be1c3354f39b2d48d50cd3265f17d9a68ee77b.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/df56dcbccd8006c036ae8efbdefc78676c04b47a170d1100f900bccb9346e61c.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/e17ac4e1239f2bcbc076214f1179bc7417c256d321c970aba7363319d51633ba.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/e2faaca42e74188dfc52ce676a8df5d14a5dfb539aa799e39d11847eee60656c.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/f26ef5f370ca71e06070539a6ced4152e155d3916b4d21784383a1467648fa96.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/f471676d5edd2cc8190acf61d573a18f352ebf123552d0a55c664eb88b1d010d.jpg)

![tables](../iclr_results/4107_Fused-Planes_Why_Train_a_Thousand_Tri-Planes_When_You_Can_Share/tables/fa65fbda8f7f669a77d3713a6c61c51de7f3f882a54c9c2fbaaab0e59ea43120.jpg)
