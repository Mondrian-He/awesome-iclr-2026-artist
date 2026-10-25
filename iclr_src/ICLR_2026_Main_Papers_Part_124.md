# ICLR 2026 Main Papers — Part 124 of 199

27 papers.

- [TUMIX: Multi-Agent Test-Time Scaling with Tool-Use Mixture](#paper-3325)
- [Seq vs Seq: An Open Suite of Paired Encoders and Decoders](#paper-3326)
- [$\nabla$-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space](#paper-3327)
- [Log-Augmented Generation: Scaling Test-Time Reasoning with Reusable Computation](#paper-3328)
- [On the Theoretical Limitations of Embedding-Based Retrieval](#paper-3329)
- [OPRIDE: Efficient Offline Preference-based Reinforcement Learning via In-Dataset Exploration](#paper-3330)
- [Emotions Where Art Thou: Understanding and Characterizing the Emotional Latent Space of Large Language Models](#paper-3331)
- [Learned Meta-Tokens for Language Modeling](#paper-3332)
- [Improving Online-to-Nonconvex Conversion for Smooth Optimization via Double Optimism](#paper-3333)
- [A Unifying View of Coverage in Linear Off-policy Evaluation](#paper-3334)
- [Efficient Multimodal Spatial Reasoning via Dynamic and Asymmetric Routing](#paper-3335)
- [CONCUR: A Framework for Continual Constrained and Unconstrained Routing](#paper-3336)
- [R1-Code-Interpreter: LLMs Reason with Code via Supervised and Multi-stage Reinforcement Learning](#paper-3337)
- [Scalable Chain of Thoughts via Elastic Reasoning](#paper-3338)
- [Not-a-Bandit: Provably No-Regret Drafter Selection in Speculative Decoding for LLMs](#paper-3339)
- [Antithetic Noise in Diffusion Models](#paper-3340)
- [Bradley-Terry and Multi-Objective Reward Modeling Are Complementary](#paper-3341)
- [Controllable diffusion-based generation for multi-channel biological data](#paper-3342)
- [LLMS ON TRIAL: Evaluating Judicial Fairness For Large Language Models](#paper-3343)
- [Go-Browse: Training Web Agents with Structured Exploration](#paper-3344)
- [Can Vision-Language Models Answer Face to Face Questions in the Real-World?](#paper-3345)
- [On the Spectral Differences Between NTK and CNTK and Their Implications for Point Cloud Recognition](#paper-3346)
- [From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning](#paper-3347)
- [Parallel Token Prediction for Language Models](#paper-3348)
- [Self-Jailbreaking: Language Models Can Reason Themselves Out of Safety Alignment After Benign Reasoning Training](#paper-3349)
- [QUEST: A robust attention formulation using query-modulated spherical attention](#paper-3350)
- [Learning to Be Uncertain: Pre-training World Models with Horizon-Calibrated Uncertainty](#paper-3351)

---

<a id="paper-3325"></a>
## TUMIX: Multi-Agent Test-Time Scaling with Tool-Use Mixture

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=HBm3MFtszH)

### Images

![images](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/images/29ec38ef260be618a524f1cb8cba063be12e7ed5a0c5649e16927f49c1fff6a5.jpg)

![images](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/images/3c862ddf0adf534b55eaa62b19d8351c50c40fc97b146704d831add439be1c43.jpg)

![images](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/images/549188a4e00b3bbd6a05edc78de82206d467d5451dfc6c282495ae321bf710de.jpg)

![images](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/images/5bf477efb539c07b8acc7351bb11ccc5f5ffb1c34c23f33fe73e2e4fe3d883c9.jpg)

![images](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/images/5fb4627d01a8e4fbc01af53ba4538a1617e55cd681d338499e3a707f91a0e231.jpg)

![images](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/images/6d4ce436a056854c29b8452d39c0b2cec3646c646f5aa8db27e31aff604d63d5.jpg)

![images](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/images/9631dd9fb88a7c5fe759f81279b2e2aa8c95ac01275a82e972debfdb33511c32.jpg)

![images](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/images/98da58208d403e0a49463f5d78c2c8dd70d1444bd211f05ecb011ecb31021ee8.jpg)

![images](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/images/b4f25ae552498b3f6f2444e6ebd646e8561268137986277d5458f33cad32618c.jpg)

![images](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/images/c377d7450f45fc29066ebeecd1539f26459a5b21f60b6c38f5b4271155139eab.jpg)

![images](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/images/dc97b2919cb1e37e8485428377aef1774adb28fb8ac96479c15622631cffceb9.jpg)

![images](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/images/e75515049bf97ddbc15c0f0399a0625119474cf4bda93831296661449f9a246f.jpg)

![images](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/images/fd34155a60cb7cbfe4d22c99c53fb3042eb2afceff51e2c7cdc62d4fc2a033da.jpg)

### Tables

![tables](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/tables/075a7b65122baa5b4137a04faa5ca65d53192fac86e787d8f0afd0cbe9b1bed2.jpg)

![tables](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/tables/149f8eb48dc9dc83c58e6c813da17a2edf2a2253867dd66d2f707ae0adfe59f3.jpg)

![tables](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/tables/1a5fa57f7c76dc400fef36df3c4c862e633d10f85c4c8c7937753142bd2bfda7.jpg)

![tables](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/tables/25a38b1529a0fdef2bebfc1042df008f55f55eb80898d5e76997ae1e73f7c5f9.jpg)

![tables](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/tables/26b2e8be75b9a29483bea0dde4918287f387eaae9e74a8477531b95d222a0668.jpg)

![tables](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/tables/29b0b1eddf08d00c5446b44fc4cc2f47060e5dd07cf6ae7e456bb5e423104cd3.jpg)

![tables](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/tables/4d541a75a5483318162b582927bc3af67328d37944dc15dfc6e22b8217764cc5.jpg)

![tables](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/tables/6ed5b0cc242345d6831b549761a434f52715e58c769cad270f87451dd112bc1e.jpg)

![tables](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/tables/9f664a270ac5e998c74da0dcef77d54e0560c34764b95851315a58c85fef9c56.jpg)

![tables](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/tables/a58f20be49e292eb9be1cddfce7c605a105b43f1740e7f37193386284feb56a3.jpg)

![tables](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/tables/d72288b0e1746eda3b704930823c7d0b13ba36c145801a7b6eda335c5e80e0d2.jpg)

![tables](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/tables/dd0140f78b09c4173acad53a9a608474654b34ea490231c658798bcf736737e8.jpg)

![tables](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/tables/e43562cd80f7eb113a33939e2cbcad938a7b80f31fdfa1e8972b3e891ac3da48.jpg)

![tables](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/tables/efa84646fd4b4736bbab4c8aba2484c4c3e6c8587df10b70267852bf6ab27f34.jpg)

![tables](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/tables/f3c5bcdb369a58fcc77dc7bbab2ce03387e1f29eeab7ae92c156c636d8f09de9.jpg)

![tables](../iclr_results/3325_TUMIX_Multi-Agent_Test-Time_Scaling_with_Tool-Use_Mixture/tables/f60a17199a99e96bcdefad4358753bee1f6bdf7dc9995b1f0fff51994482de95.jpg)

<a id="paper-3326"></a>
## Seq vs Seq: An Open Suite of Paired Encoders and Decoders

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=z5Mn8Rxi3l)

### Images

![images](../iclr_results/3326_Seq_vs_Seq_An_Open_Suite_of_Paired_Encoders_and_Decoders/images/08787f84d60064c606782918b234b69ccb5b4757038e601994d249f5d99004a5.jpg)

![images](../iclr_results/3326_Seq_vs_Seq_An_Open_Suite_of_Paired_Encoders_and_Decoders/images/85cb4731f43faaabc2b647412dd308fdb3af59126e370b78e4d129debade9073.jpg)

![images](../iclr_results/3326_Seq_vs_Seq_An_Open_Suite_of_Paired_Encoders_and_Decoders/images/fae32f263d523cd514f1ebc41e1d89aa9ae31e6eae865ad01687a9d3e41bab62.jpg)

### Tables

![tables](../iclr_results/3326_Seq_vs_Seq_An_Open_Suite_of_Paired_Encoders_and_Decoders/tables/01ba734e71fcac3f4056619e66bdeddf993d7b0b213f0fd488e071c2ae087e41.jpg)

![tables](../iclr_results/3326_Seq_vs_Seq_An_Open_Suite_of_Paired_Encoders_and_Decoders/tables/0e6a29c2fe38c64c3194cf1ff6c0ec2eab37ca357145bf3db820465f393341a2.jpg)

![tables](../iclr_results/3326_Seq_vs_Seq_An_Open_Suite_of_Paired_Encoders_and_Decoders/tables/126fbd4e708cb45947da54c7e8d98ce26bdb5a9be97dda1fe7b2501e6ddb9a82.jpg)

![tables](../iclr_results/3326_Seq_vs_Seq_An_Open_Suite_of_Paired_Encoders_and_Decoders/tables/266fc26488420a5b9acedafd7abe0fbd4d1dd3a39882bff6b990c5970577ba73.jpg)

![tables](../iclr_results/3326_Seq_vs_Seq_An_Open_Suite_of_Paired_Encoders_and_Decoders/tables/3829a35d16473b59a2b2496713ff71d1d79d1c4d0f405c1fbb674cd2b5bb662e.jpg)

![tables](../iclr_results/3326_Seq_vs_Seq_An_Open_Suite_of_Paired_Encoders_and_Decoders/tables/6d06fb892779b5644bcf0a51b41305c1bdf099c51ad251dc229b4fa9516a96cf.jpg)

![tables](../iclr_results/3326_Seq_vs_Seq_An_Open_Suite_of_Paired_Encoders_and_Decoders/tables/7a425fdebd6cd0f9337b50ecb2acbd2f73647082c14c2d24e2d16f1cb156a791.jpg)

![tables](../iclr_results/3326_Seq_vs_Seq_An_Open_Suite_of_Paired_Encoders_and_Decoders/tables/8786eda799d08880495ce3888f156e7aeaad370be09bd7a6f730024caf059242.jpg)

![tables](../iclr_results/3326_Seq_vs_Seq_An_Open_Suite_of_Paired_Encoders_and_Decoders/tables/9362fa368f593faee9a6de278fc71fa98e630704a7f084ef6b53a3c43ac7a0bf.jpg)

![tables](../iclr_results/3326_Seq_vs_Seq_An_Open_Suite_of_Paired_Encoders_and_Decoders/tables/a501916f2175418a1bd802271102803c8f41b449b8c0831e9ddd32bb8a3584c9.jpg)

![tables](../iclr_results/3326_Seq_vs_Seq_An_Open_Suite_of_Paired_Encoders_and_Decoders/tables/b6c996080a83aeaaeb221ec0e0a22f50f4b2f08b545f5880a133e425b1edf14e.jpg)

![tables](../iclr_results/3326_Seq_vs_Seq_An_Open_Suite_of_Paired_Encoders_and_Decoders/tables/cc1837a78091c1ce0a7b9970339e22fb65c696618bd6735d73381c317b3cb52d.jpg)

<a id="paper-3327"></a>
## $\nabla$-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=pEJAja73dk)

### Images

![images](../iclr_results/3327_nabla_-Reasoner_LLM_Reasoning_via_Test-Time_Gradient_Descent_in_Latent_Space/images/45f74d79ed36d86355542a7819b31298d0eadba6a2c9312726e0381f13a42489.jpg)

![images](../iclr_results/3327_nabla_-Reasoner_LLM_Reasoning_via_Test-Time_Gradient_Descent_in_Latent_Space/images/7b6368788acb0298d59ab9fc37962926aea36ab4ecf2dc5049665184c3613f61.jpg)

![images](../iclr_results/3327_nabla_-Reasoner_LLM_Reasoning_via_Test-Time_Gradient_Descent_in_Latent_Space/images/bc666a69a0fe9a83a8341944222fa6f83e8468e695c16b2ae876ead59757163a.jpg)

### Tables

![tables](../iclr_results/3327_nabla_-Reasoner_LLM_Reasoning_via_Test-Time_Gradient_Descent_in_Latent_Space/tables/43756a595e1673ec71060a50f45d81921fee6cc8f110e48a65fd150a09701094.jpg)

![tables](../iclr_results/3327_nabla_-Reasoner_LLM_Reasoning_via_Test-Time_Gradient_Descent_in_Latent_Space/tables/4b98834834bc46999d1c801f1097d9834bbbf02e094f0e374b1e1f122af5c363.jpg)

![tables](../iclr_results/3327_nabla_-Reasoner_LLM_Reasoning_via_Test-Time_Gradient_Descent_in_Latent_Space/tables/5f985143252f0e4365f8f4931b2c167473dd7a510c04cc653762994b9d82f076.jpg)

![tables](../iclr_results/3327_nabla_-Reasoner_LLM_Reasoning_via_Test-Time_Gradient_Descent_in_Latent_Space/tables/71efec5a57f6ad16ebdbee28b5116072b1b5fcddd71e0e6a506c8190f3aef3ae.jpg)

![tables](../iclr_results/3327_nabla_-Reasoner_LLM_Reasoning_via_Test-Time_Gradient_Descent_in_Latent_Space/tables/858c08c04ba05f8795dcb96ba015213f4f03947ccf15a59b91599facb67782ff.jpg)

![tables](../iclr_results/3327_nabla_-Reasoner_LLM_Reasoning_via_Test-Time_Gradient_Descent_in_Latent_Space/tables/cc91c9a1644b198f14b6ec68258406396e06ffe3c56bc52a349a94ab2f0ab2eb.jpg)

<a id="paper-3328"></a>
## Log-Augmented Generation: Scaling Test-Time Reasoning with Reusable Computation

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=oQ3d4O8BaX)

### Images

![images](../iclr_results/3328_Log-Augmented_Generation_Scaling_Test-Time_Reasoning_with_Reusable_Computation/images/0c2b75eaa3f1a704f5a5b116e85106206f190ca7813b602d3db204b4c8b28343.jpg)

![images](../iclr_results/3328_Log-Augmented_Generation_Scaling_Test-Time_Reasoning_with_Reusable_Computation/images/9784988c4994f856509a0cd444d5a2ee7c554804cf6eb54f130919b521ab7645.jpg)

![images](../iclr_results/3328_Log-Augmented_Generation_Scaling_Test-Time_Reasoning_with_Reusable_Computation/images/dc0df7af8da9464e466c837e30ee1b17f9c4c8ddc0a77793b71b1e9ae62680d0.jpg)

### Tables

![tables](../iclr_results/3328_Log-Augmented_Generation_Scaling_Test-Time_Reasoning_with_Reusable_Computation/tables/20ec9f87ab4dea1a693d524611244cac28f07b04e13d80995f5a1fa51d1bf8b7.jpg)

![tables](../iclr_results/3328_Log-Augmented_Generation_Scaling_Test-Time_Reasoning_with_Reusable_Computation/tables/40901b23cd118efb91e468110415e7183c2d25843f9d548847acfd519abe2eb9.jpg)

![tables](../iclr_results/3328_Log-Augmented_Generation_Scaling_Test-Time_Reasoning_with_Reusable_Computation/tables/42b7032382be7c24bb1d3da039b16cedf6502005320cd47a8aae6603d26eb5dd.jpg)

![tables](../iclr_results/3328_Log-Augmented_Generation_Scaling_Test-Time_Reasoning_with_Reusable_Computation/tables/53e773a4357b30b1bc7acf7251baff4f9462fcc3d361ed42d63f936f7c8231b4.jpg)

![tables](../iclr_results/3328_Log-Augmented_Generation_Scaling_Test-Time_Reasoning_with_Reusable_Computation/tables/6530a8ccca4b4817d05e96d3da24d87f088399913b57e21d45aec0c217f0fc80.jpg)

![tables](../iclr_results/3328_Log-Augmented_Generation_Scaling_Test-Time_Reasoning_with_Reusable_Computation/tables/9f2713d27948155e50e9b25a65bcf60bb33a5c6a312d1012ef38a6ce42e983d6.jpg)

![tables](../iclr_results/3328_Log-Augmented_Generation_Scaling_Test-Time_Reasoning_with_Reusable_Computation/tables/a2e336b85804283727cb7962d469e47720b2368800cb646571724fd96f81eb52.jpg)

![tables](../iclr_results/3328_Log-Augmented_Generation_Scaling_Test-Time_Reasoning_with_Reusable_Computation/tables/c678d2c2047553ee07d0d33467e57fef85aa7b531bedc1c113c3a548233b6439.jpg)

![tables](../iclr_results/3328_Log-Augmented_Generation_Scaling_Test-Time_Reasoning_with_Reusable_Computation/tables/ccf2cb5862ff198320ffecf59e33286ea23b2808bf1018afee1de110a487f524.jpg)

![tables](../iclr_results/3328_Log-Augmented_Generation_Scaling_Test-Time_Reasoning_with_Reusable_Computation/tables/d9b65483eaab88230b32f61d3f4e6e55e58138d888933aceb91ebf0de26de918.jpg)

<a id="paper-3329"></a>
## On the Theoretical Limitations of Embedding-Based Retrieval

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=k9CzIvzfaA)

### Images

![images](../iclr_results/3329_On_the_Theoretical_Limitations_of_Embedding-Based_Retrieval/images/00090367a7865b8e218386bf39e2fa9a4ad9ff77b2e206641a201d04519b34f6.jpg)

![images](../iclr_results/3329_On_the_Theoretical_Limitations_of_Embedding-Based_Retrieval/images/0f8f962b74a07baa935a1c13c03e7169faa73e0dc5ab2ce69c77dcb7fe1787ae.jpg)

![images](../iclr_results/3329_On_the_Theoretical_Limitations_of_Embedding-Based_Retrieval/images/3595e6f35b2d7ce7f215d1b4baf895223c05081b4e061d7cb0abdc422e796ff9.jpg)

![images](../iclr_results/3329_On_the_Theoretical_Limitations_of_Embedding-Based_Retrieval/images/44fb6222bc1f059f5a60dd0240d083a2434986fae52f63c934ddf57dc91f7be0.jpg)

![images](../iclr_results/3329_On_the_Theoretical_Limitations_of_Embedding-Based_Retrieval/images/f38df37f7b5ed0c79b455fcdd95685ff13c72805f4bae0479f9163c08c7c9cb3.jpg)

![images](../iclr_results/3329_On_the_Theoretical_Limitations_of_Embedding-Based_Retrieval/images/fc3118536ba99fc597fd6c5d156c377f47403dba385514b51f2fdb4abe3b2d60.jpg)

### Tables

![tables](../iclr_results/3329_On_the_Theoretical_Limitations_of_Embedding-Based_Retrieval/tables/187be5b67b1af94e12e8d23160465e577e6e820b5cef78f592d2787a758a980d.jpg)

<a id="paper-3330"></a>
## OPRIDE: Efficient Offline Preference-based Reinforcement Learning via In-Dataset Exploration

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=QLDHukpozh)

### Images

![images](../iclr_results/3330_OPRIDE_Efficient_Offline_Preference-based_Reinforcement_Learning_via_In-Dataset_Exploration/images/3f90169b94901c32535e493dc53eb9cd30e6d285502d6840bddc298544be91d9.jpg)

![images](../iclr_results/3330_OPRIDE_Efficient_Offline_Preference-based_Reinforcement_Learning_via_In-Dataset_Exploration/images/904bc7e9587c3450905069565b5e4471e24e31cae120b1520e02c53ee32b19b5.jpg)

### Tables

![tables](../iclr_results/3330_OPRIDE_Efficient_Offline_Preference-based_Reinforcement_Learning_via_In-Dataset_Exploration/tables/178d57a3a5c5ec272bb573a91e9d0ce93fd85171abce86a6abaed3bf81a4dea5.jpg)

![tables](../iclr_results/3330_OPRIDE_Efficient_Offline_Preference-based_Reinforcement_Learning_via_In-Dataset_Exploration/tables/17be91ad1b5fb3f65d739ea0405f1168d5d0dfc8bdabe5a591def20f8be28d3a.jpg)

![tables](../iclr_results/3330_OPRIDE_Efficient_Offline_Preference-based_Reinforcement_Learning_via_In-Dataset_Exploration/tables/355ffc4fc245827de627262032ae05d102e401dedf8b1275a4f8966793bc6659.jpg)

![tables](../iclr_results/3330_OPRIDE_Efficient_Offline_Preference-based_Reinforcement_Learning_via_In-Dataset_Exploration/tables/4b4b5029883d03b055b06863482f80bda02bc55098ea46e1fe44916b83241fd9.jpg)

![tables](../iclr_results/3330_OPRIDE_Efficient_Offline_Preference-based_Reinforcement_Learning_via_In-Dataset_Exploration/tables/62acb761e175e46bfdfafb3da04383f8f66af221bb33bc54e3522781599b98c9.jpg)

![tables](../iclr_results/3330_OPRIDE_Efficient_Offline_Preference-based_Reinforcement_Learning_via_In-Dataset_Exploration/tables/79c9c11244e5e34e4d8944e0679ca482901c4117cf50626bdcea24999c4ba173.jpg)

![tables](../iclr_results/3330_OPRIDE_Efficient_Offline_Preference-based_Reinforcement_Learning_via_In-Dataset_Exploration/tables/7d97182d3c913971d818540e0171be135ccaffcad9ddaf316391156d408fd3f2.jpg)

![tables](../iclr_results/3330_OPRIDE_Efficient_Offline_Preference-based_Reinforcement_Learning_via_In-Dataset_Exploration/tables/830ae956cda584a3babd29bbf83ed2bb424cf6ed5fb02d1cb512995ef8ce7e98.jpg)

![tables](../iclr_results/3330_OPRIDE_Efficient_Offline_Preference-based_Reinforcement_Learning_via_In-Dataset_Exploration/tables/9f08d949ce4dc8922c0857bb1f37fe5fbd570e426b66ed1a25f3d07c5a5e3a57.jpg)

![tables](../iclr_results/3330_OPRIDE_Efficient_Offline_Preference-based_Reinforcement_Learning_via_In-Dataset_Exploration/tables/a24b156347d24a792966042aeb85bcfa95d632a5a03579633175c213e8d287d4.jpg)

![tables](../iclr_results/3330_OPRIDE_Efficient_Offline_Preference-based_Reinforcement_Learning_via_In-Dataset_Exploration/tables/b1d164d379b35a90a6b8dd3896ad5434aa8456522eec3139340b827b97b51d7b.jpg)

![tables](../iclr_results/3330_OPRIDE_Efficient_Offline_Preference-based_Reinforcement_Learning_via_In-Dataset_Exploration/tables/ba8a3dc121508b8ae33e80982dd5f92a7a887af21178d195155c669b0d384ac2.jpg)

![tables](../iclr_results/3330_OPRIDE_Efficient_Offline_Preference-based_Reinforcement_Learning_via_In-Dataset_Exploration/tables/c351b699428e1864e8d249e200d4642753bda6c4cfcff5c08463f721eafaaad6.jpg)

![tables](../iclr_results/3330_OPRIDE_Efficient_Offline_Preference-based_Reinforcement_Learning_via_In-Dataset_Exploration/tables/c60307e503b178cb07fe82381440101b6733ac10953b1a77aa05ca4c4bac9e17.jpg)

![tables](../iclr_results/3330_OPRIDE_Efficient_Offline_Preference-based_Reinforcement_Learning_via_In-Dataset_Exploration/tables/d01e3c07803d52792014971127df0ba0e2d62e013142190339161eaebff06985.jpg)

![tables](../iclr_results/3330_OPRIDE_Efficient_Offline_Preference-based_Reinforcement_Learning_via_In-Dataset_Exploration/tables/dac8938b5b5afcc286d86525fd9836e665f0c84c166d30b4fb691ac1619313e6.jpg)

<a id="paper-3331"></a>
## Emotions Where Art Thou: Understanding and Characterizing the Emotional Latent Space of Large Language Models

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=72TN9UAtNI)

### Images

![images](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/images/0912f002e25ecedc00b9d6227ecb9a0b6e7516eda0d6851452cf5259ac09e3a4.jpg)

![images](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/images/09d4f7606b8ba01b06d4ba017c49f696103dd7c744d4ba84682e8575878f0438.jpg)

![images](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/images/113a2713f0839b8c55b2a4054b0e8d8231e4bd1ce245b46b82bd433cc2029f0a.jpg)

![images](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/images/375c747167630b8acfe3434fb0c146110072233e76484db1bb4bd18004fa8132.jpg)

![images](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/images/43e620190a2b1a7816431851c6e32140d342c7a99d0129c21da8dc02a1051e6f.jpg)

![images](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/images/492c725312883dbb865df45789bb18244897bc2e3a038109df5e2559255c3cc1.jpg)

![images](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/images/66f2ac428199bf9afa6f97101c4f1730275aa558fab29c96758720a6b41ddb49.jpg)

![images](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/images/90c5afcd7b754b0d29dd6ffcaeec2532e245913e685d9517af493d2ec90a5b6c.jpg)

![images](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/images/963f9399a4eb64452407730c97d25f73180dc74fbdf4f703efd573de8cc36636.jpg)

![images](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/images/9c97bfb3b509ec504ec4b2db90c411a63ebadae6bf1ab61f0afedff363e17d8e.jpg)

![images](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/images/ab1e2bf36ccb2d842e16a99f66d66257eb1a8a84870fa0201369804bd280a3ca.jpg)

![images](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/images/cfea4c89451f3fd980ff29ac11d250f9136fba3f2bc984e811a92979ed69f90c.jpg)

![images](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/images/d553bde3defadf1116cda80caeb70b900cd9a42343c2c52febbdf08834df4d6d.jpg)

![images](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/images/f98c4d3b25353f27989a57bbc4d7a1fd9712661b540a6b40933b98067b2f4075.jpg)

### Tables

![tables](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/tables/035ab113e49f4240c69e1690362d59f2a84242ef47bcce5a7fdd9988e1dddfc3.jpg)

![tables](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/tables/2f4b0dc09978d64a05cfd15eeeb73e86c6d3bc4bd4e855597c627bc348847a30.jpg)

![tables](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/tables/3792780aeab43d0bfc99287ef27cdbd2391db596203bae5f00f4ee183849865e.jpg)

![tables](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/tables/4e28efa7dcc4bd2a439b0198f7b6fda4567e1f054b7f3c4d1957bacb581af840.jpg)

![tables](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/tables/6258cc4d6b628d39b42e0a20478352fdf93f70f0381ad9f5276a18c1b5e3c777.jpg)

![tables](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/tables/78492ddcfeb23f033205e2d001401aed6656cfc39035138add1b195641d9e177.jpg)

![tables](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/tables/7acbc8b1ce1a2ae107fd4828517e6e33b7caec6e4043e12b4eda833ae0fc252a.jpg)

![tables](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/tables/a9cc93a563b373030880498319028f52011cfb02831c7f39003deaebcb8801ad.jpg)

![tables](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/tables/b9c75f89f2be5d4c8f6a51d1db2456b60e408c3407d23e7746169e49e0b06aa8.jpg)

![tables](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/tables/cbeb46284f763e0af7ab3c71f0371c32f11cd30e5b9eb41cb1d09aed5fd8b24a.jpg)

![tables](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/tables/cea407e4b2c2982f0912203b5b12ae28cb67542c17f8dce3046542a516eae09f.jpg)

![tables](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/tables/d1f197a4726115e95028e8fdc83466b1bc37f1d2d84c9f3d2770163eabfc8601.jpg)

![tables](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/tables/d7c7022a52f5839598d618467078775878e91ec2dbc8ecb60bbe16edfdd1465d.jpg)

![tables](../iclr_results/3331_Emotions_Where_Art_Thou_Understanding_and_Characterizing_the_Emotional_Latent_Space_of_Large_Languag/tables/e394d3d9e5d4f9bb702813eb4f7c0e3f11e9f1be7532225f27cbeff25c878526.jpg)

<a id="paper-3332"></a>
## Learned Meta-Tokens for Language Modeling

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=eZ5jtFuk3e)

### Images

![images](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/images/047e518b755230f0ceafbfc928cc09357325a4188d57d506200ad5821045069d.jpg)

![images](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/images/5dcdfd2f0a3a32d83be37172796579275c2f824237c1bac77022b89ae6e49a2a.jpg)

![images](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/images/ed18e1240d3b4cb70dda09fb04e7bede953bb515c85641166168e87f833b2b64.jpg)

### Tables

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/0f1df1e26a584ac1662078353beeba6a55f6cf2317504432f7877a524356ac08.jpg)

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/15d73637c92afa938aad7fb19317cfe1cef4d8abdc562591c00bc338c923f3c5.jpg)

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/2d04225ee31ae8e2690bfe4955ddd7535ee77b58969b44cc1cb8ea4b03bd7fcd.jpg)

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/2f8f9d36924ba77b1779553a7743883ae0cc1fb9edd6fa31a7a3bf6bedd7bcf9.jpg)

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/2fd86475c5af34af6e9da95b3ec683ab40a7e63bf305504da077be9dc3efe2c8.jpg)

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/40baaa41676078a9f558a724d1ab97c1dfd7abb3b4f9a47a7f202ed213ae2bd5.jpg)

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/590fb4269e53d869b9033bb0aea229a7ef63e0c5838118b98f0ec061cd151f06.jpg)

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/663b4eb43555da78cd6d6341b94426a06c21b3f2896d73049abccb91e8696ea7.jpg)

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/9f9530dcdf3b286c1dfe524f6c1e97e0497e5a64ef42ffeb5f6f130d0060ea12.jpg)

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/b1da658c07c1822dc03769915ad33be20e412835a7f0a59a41a08e2b7f614173.jpg)

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/b86d528242f0678be433e8a22b5828a189591bca8a8627effe0f84e286c7142e.jpg)

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/bfdf83f584137cc68e2849f8ac2d97402e48a8e48cbcc414fe1d09960c10e583.jpg)

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/c47c53f326296d92fbee6e31a56f9afb19e1d5aec414bbbe50ede657186ad84d.jpg)

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/d3b6aac16610fe36c0f07b60a8a667e8259f5a178f1c311e8ff7c61ee1d28120.jpg)

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/e1423ac4659894cfed240fd1d7c07e7ab3542406f313977033abf13922ad0bd5.jpg)

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/e675a2320df16f78d668773923988fbb5a8f3a175a2be4c9e889e96317e18014.jpg)

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/e79c3d55b16523dfb3adf44da751593bdad7aa4bb3e46d616f5cf3a31fb5ee23.jpg)

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/e8379d19e9544a93f30ffbacbb7591c55100cb1b16d2db6b6cef76690648ab88.jpg)

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/f3627f8865906382f1e1fc893e977be51e3eccc24c1310256e323b25b9b5e73d.jpg)

![tables](../iclr_results/3332_Learned_Meta-Tokens_for_Language_Modeling/tables/f5baf062d903555029446befb2860fe2720cf58bca53c028439ddf44a98d987a.jpg)

<a id="paper-3333"></a>
## Improving Online-to-Nonconvex Conversion for Smooth Optimization via Double Optimism

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=ZAflv4dxQ9)

### Images

![images](../iclr_results/3333_Improving_Online-to-Nonconvex_Conversion_for_Smooth_Optimization_via_Double_Optimism/images/fb05190f72d1c7fdfb2676ca4e73bd9f2da85c4a24b0a70280ffcb71d0454712.jpg)

### Tables

![tables](../iclr_results/3333_Improving_Online-to-Nonconvex_Conversion_for_Smooth_Optimization_via_Double_Optimism/tables/6888833e913a656fc1588d0aa6399502501b0294093c4df8071d1d2d416017d7.jpg)

<a id="paper-3334"></a>
## A Unifying View of Coverage in Linear Off-policy Evaluation

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=FjTYIRufJp)

### Images

![images](../iclr_results/3334_A_Unifying_View_of_Coverage_in_Linear_Off-policy_Evaluation/images/b7e7b83733230a772bea7f550de7c534bee6b1556eb61e969e7e13f6396d065f.jpg)

<a id="paper-3335"></a>
## Efficient Multimodal Spatial Reasoning via Dynamic and Asymmetric Routing

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=BQASoLmREU)

### Images

![images](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/images/19b79c4aef243b44debc8e2bcceeca0429bb3f3683841f5021509b850fbee725.jpg)

![images](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/images/23eb6f1efd8e864705bdf1bc24e3438f1cd823f04de280a0c0a8132ac15834de.jpg)

![images](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/images/4639b801f831055c00cf70d4bb6731d059b14a43fe0cd25af011a7dfc0ab9d1b.jpg)

![images](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/images/499dca9444b1a4a61e2abe3f28b63c449b7282f036635d8f38b4f42ccac653bf.jpg)

![images](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/images/a402a42c5d4c29c77bd78d7c2b69a8b370901c4b599ad4bf55cb82270d0a803a.jpg)

![images](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/images/c5eb1014a91cd6403fdb878a5a161bda2b714d53db3a3cca4627966e698522b5.jpg)

### Tables

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/050b704c6001a1bb0b1e94243978ea0b480892b58bc4ef00e96b26d145da2801.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/08a0357a4de4fa71c927378a90fc106dd0ea1f5007928fadd52cff6ac4d25e57.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/09706990db8217dd88f627a9c6b5146fdc8a80e082a5c1e248de5ee0d59925d1.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/12e2b76cc2419909ce102946ec78f8bd48a8d62b2ebc417e84c9aae9dd3dfac8.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/1623bb5fca066aacbd4d25f3cb212832d575f752df0bc28210f2208221b3c640.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/1d728d872267f03c3a4959566c0b3598299bb2dcf577fe6250e040be9337fac6.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/1de65ed793c1464552c2f66517fd22a38487646598e5361503fc338965ac76e9.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/31cdd419b13d074a5f284cf64ba25f7f5a668da175b8a57a1220f7844a4e92fc.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/360e5087bdc0073916d1caea351f8e8e927c410bae4f6c89e2af683f8d05732e.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/3e1a960c615c5557dc04eba0736afed6fb36899e549556618d1ab95c223a0774.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/4c4ba213348eb6528209b08908d41f9c70e354a6acaec28fa4faad382f5f6ada.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/5c6662b266f91c71741eb4898d31d5f760bfcc3bdcab80b32c7e5445ac4ebf24.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/7d86f5cd52a8c6985d0f65a8319323ee8ce00fa33e774014c1a9d24ae87f37b8.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/88b01204528f05805c3cf60b81af1a0826ed1c88a4866d2470988886e09e6cb6.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/8e6f437f918254d36cb3cabfd6446e4aac3ab6512330d7d3110cf472b4dce489.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/adbcbcc2d98228ca422d63daa8766d79fa853530daf2fe55c8f94756ae8b5953.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/afd26e8a6a9d991f59bc84551ad5a22f52acb94a9059d8cd7377361176a0f5e7.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/bba0e9e3c38519a6ace7bd5c6bceaa0ad341ed9693c8247e0697d77edb3beebc.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/c3d7992e76ccec7d16a665a6ce2cb5236c6c4223ec347d11c40eaf4228750173.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/d16f9fb06f2584cc779d7a449dda5cb00afca063f2b1e31bd53ad1d45bc2939e.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/dadf08b092410cb62e0e02345386eaf7959c3e76456b2a8d0ac41130f1889569.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/dcafe8484f3c47c28819dd4df3a5c815e74a49337f919b0f3746a80a60f5a89e.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/e582312a58971a729a20f2f3368a0710b2ca330d9e4f75b651f1cd42e3758690.jpg)

![tables](../iclr_results/3335_Efficient_Multimodal_Spatial_Reasoning_via_Dynamic_and_Asymmetric_Routing/tables/fe4d5416340f15d6f361591fc93f533f8a0f0106aabd05829428d4bf33d7caa1.jpg)

<a id="paper-3336"></a>
## CONCUR: A Framework for Continual Constrained and Unconstrained Routing

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=gCUY6QIv8r)

### Images

![images](../iclr_results/3336_CONCUR_A_Framework_for_Continual_Constrained_and_Unconstrained_Routing/images/139348a3da4c0a7519c3fb14fda681f66d16c7ab5e5693ef68b4cdeebb6b76db.jpg)

![images](../iclr_results/3336_CONCUR_A_Framework_for_Continual_Constrained_and_Unconstrained_Routing/images/1f544e8b8f9e12b26c87155baca5e81e9e8de9cb9412e0ab367c088bd911908d.jpg)

![images](../iclr_results/3336_CONCUR_A_Framework_for_Continual_Constrained_and_Unconstrained_Routing/images/a4683b43cbb9d16314af769462ac51f8d48c714a3c14f480cc79c5ab9a289646.jpg)

![images](../iclr_results/3336_CONCUR_A_Framework_for_Continual_Constrained_and_Unconstrained_Routing/images/b7c648e4e5e4a07e2dc08bf4f93f4d7973c14b2a34e67bc0390081a2242a2a42.jpg)

![images](../iclr_results/3336_CONCUR_A_Framework_for_Continual_Constrained_and_Unconstrained_Routing/images/c783d6ec868f11831b52b9ddd3fd44e41b1be09347b2ecf47f6ac8c046ebda5f.jpg)

![images](../iclr_results/3336_CONCUR_A_Framework_for_Continual_Constrained_and_Unconstrained_Routing/images/d5765cbcba3fa465604ec6ae9d50a2cdad0aa1a124f4194fc961e66ecb945d1f.jpg)

### Tables

![tables](../iclr_results/3336_CONCUR_A_Framework_for_Continual_Constrained_and_Unconstrained_Routing/tables/047d490bbaccccf3822d2083e20f6ae2eb748eac7246b00b87f34d8cc6d3b974.jpg)

![tables](../iclr_results/3336_CONCUR_A_Framework_for_Continual_Constrained_and_Unconstrained_Routing/tables/0e784732aa988f60c2b7fdb304201c2efc727090b2ec2f89cf70bf755be8c68b.jpg)

![tables](../iclr_results/3336_CONCUR_A_Framework_for_Continual_Constrained_and_Unconstrained_Routing/tables/3533b4f521c1f92dbc676f80ef624e7f1a094fb94436d2a6c0ab1a75070ef752.jpg)

![tables](../iclr_results/3336_CONCUR_A_Framework_for_Continual_Constrained_and_Unconstrained_Routing/tables/37c9059cbe295f0607516d6a49bfbd548c02b0b79f87a405e6c135176ea23137.jpg)

![tables](../iclr_results/3336_CONCUR_A_Framework_for_Continual_Constrained_and_Unconstrained_Routing/tables/6792c0c9faeb52fb4fe13b9e4f2b68c79fb5ad28668d6bb73c794bbc317c4e3e.jpg)

![tables](../iclr_results/3336_CONCUR_A_Framework_for_Continual_Constrained_and_Unconstrained_Routing/tables/a2cb4f5b9cbed7a31e50ef2d3a1c1ebe1c3456fc909a37576edd19be891792a6.jpg)

![tables](../iclr_results/3336_CONCUR_A_Framework_for_Continual_Constrained_and_Unconstrained_Routing/tables/a3e69b20b48daa1e3cbbea9e96be1331db1d5faab9273b7f9d98cb4886172cb5.jpg)

![tables](../iclr_results/3336_CONCUR_A_Framework_for_Continual_Constrained_and_Unconstrained_Routing/tables/ae6324aa023bde6e525908d4ed28824699cc57a542b8ff637b405214d4fb0099.jpg)

![tables](../iclr_results/3336_CONCUR_A_Framework_for_Continual_Constrained_and_Unconstrained_Routing/tables/fea6e68f531913dcec48674afbc4a450e275357942554e7da003586cf13fc8c7.jpg)

<a id="paper-3337"></a>
## R1-Code-Interpreter: LLMs Reason with Code via Supervised and Multi-stage Reinforcement Learning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=FNlNH0iFOx)

### Images

![images](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/images/2a0a439330e5500a8d125b0a43304600ddb18cb45342f90cf5108cf86ac631db.jpg)

![images](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/images/2f499a808f4675f1aa4830487a55508df2f464a7b53840bd10379e56a0100124.jpg)

![images](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/images/3254b18fd222ceb2e6502889d5a110dc155f2d20c5343243163fedbf73ab2352.jpg)

![images](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/images/45f0394daa0a2dc1f042ebf5b0c5b7ea1bbdda89f6b2ae48c819832fe56bba7f.jpg)

![images](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/images/58df9aafca2dc165bfa89aedf946ea0a81da4c14df2a474ce20e8c11ecb7202d.jpg)

![images](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/images/5b2fb8e88eca551a03957bb0c7bbf9575e667bf96728be0b27f22bfcaf199db0.jpg)

![images](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/images/753ebd0b83087b98989c2000b6523b3b504125f2838a20594505fde760744021.jpg)

![images](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/images/84b27c4522f56925f248d01efb7013b4b06ce2e7b35c25652cf8dd9f62646464.jpg)

![images](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/images/8ecf5c994066a6c12dfb00060b00be7ad987934502834fa3e139536c44010583.jpg)

![images](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/images/949824bcf4c38ee31b86bb9a748ef57356b06421cdacdf47dfa0ed1d33e58d7a.jpg)

![images](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/images/98b1b8a16b4631a040bbac194ae4d4756634d7faf4cd9504862dec857dfe9876.jpg)

![images](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/images/d4728ebe80dfce070ed68821261a6cafeffb712b07b25661d017d3710b02daf8.jpg)

![images](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/images/fbf742e603d491187cbcc82a7ceef3c18a07b1ca12b2ee811153cd6535b428cd.jpg)

![images](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/images/fdf026de5f412cd6759902f279c9a2505cff13a345802bc0b91fd12fffde5673.jpg)

![images](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/images/fe27d731b475f532c07adab4d6dfea8d20c6c7f6eddb84b71e0af481b1f5a787.jpg)

### Tables

![tables](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/tables/250beb253d94c18e759c0f2023c7c63f71f4aaf1f90547007cdcc522b9f557a9.jpg)

![tables](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/tables/392fba440fa1e65e3a5e01bdf54190381a66a2d771da96bf4381263809739a34.jpg)

![tables](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/tables/62d04eef704c6d77d34b4c8c65da41b215074c800864545eb51957b3b5d2ffaa.jpg)

![tables](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/tables/803e2483b3ce38d8131c35ca333278f9a522df2f2cd6d642d17c3a2cc93dbaf9.jpg)

![tables](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/tables/888d09d973dda6360f6a53719410463650bb5392cc05cb958d4e9f5b69ecb6bf.jpg)

![tables](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/tables/c811cb0dfb10eb3dd8de2d82b6ae0c6043f6cf97c718dce1d6f4fa3c9706d1d2.jpg)

![tables](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/tables/dbea21bb2cbf13394895b579b4119de49398f0adf9ccdbafed49ec01edef2aa5.jpg)

![tables](../iclr_results/3337_R1-Code-Interpreter_LLMs_Reason_with_Code_via_Supervised_and_Multi-stage_Reinforcement_Learning/tables/de912bc3e8a1a86b4b1c032b0d3f61fe349ac16435a391c85b358c51a707986f.jpg)

<a id="paper-3338"></a>
## Scalable Chain of Thoughts via Elastic Reasoning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=E0Qfhma53J)

### Images

![images](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/images/4b720cf0dfe8768c6cd0b7c9980c3efac330c85481bc181019d45d2e9d931c25.jpg)

![images](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/images/741d089a20d2cd2f316c5b6d62bdf7851d2685ff18cf870f8b1498d2b4f2eed1.jpg)

![images](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/images/7b4b7286274d162a08ed91465818687828170bc379b2bf8bef18c95b4fc074e6.jpg)

![images](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/images/7ebbcbe313662d90c96a87d8ebe17e3ae794b6022ec9764a238e107a7672e11f.jpg)

![images](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/images/8b4a25899ebf9c138e4e6bdb356a4e0465754fb4c3d4d7910918a4421a215e43.jpg)

![images](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/images/ccc54d3b97a8894f192080a7080a07962e8e214a7e5c90f2023dcc7ed233c250.jpg)

![images](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/images/d5ab6032231ed8463640b781a3e2fff3d6ac29489e36ba6a1dc5dd1d966c5bc3.jpg)

![images](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/images/e39c889e3e8391571f4a89edd1801761c7e4dbcc5fc3fce21ae9e43967812e66.jpg)

![images](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/images/e3c763a17002fa2f765819e4fbd029ec5272025be3bffb4d5d384034de981287.jpg)

### Tables

![tables](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/tables/0d5cce219cf9b93d09583abe1e8142279448754b30636fd002590b0f529e5b42.jpg)

![tables](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/tables/2d9d6a9b90e47d85c067e76397c3788d1df73a03aa3591d625a11429040cbd72.jpg)

![tables](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/tables/3b22117a90ad4a9a30af11e54f9b3fdbedf13374fda5018622670f9bb311a35c.jpg)

![tables](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/tables/3d07be2bc4a3d761c395bff71259600c0036befc0f2955198b07460f4905b36c.jpg)

![tables](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/tables/624591f14dcbea5ecf62b5f32ed4c0ceecca06b2398ee409d1351be1b2994764.jpg)

![tables](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/tables/6ac427a60f764f05b059b51a5cfe13b98cb9617c39fc42b9f2e8e062f0dde310.jpg)

![tables](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/tables/6d483e6703db422f2e2b5091760ad8268e7a8525902f6079739635f5e665a612.jpg)

![tables](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/tables/a725d0f6f95c9ec20b7bf9b03fcc9e19994dfa7c490979d46223087fdf4231c9.jpg)

![tables](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/tables/a7870dc739b6b8cacb1a3ab9531bf703fc3d040e89a7e571d5925a214c0d3ee2.jpg)

![tables](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/tables/b1968059071ee68e7dbeff042b1e6600aa44e30356a4eaa79eaa84ef04068b5b.jpg)

![tables](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/tables/c154caaa1b7ce01f3dd7c8ec159dd981bcea3d992ede56dd69702d076b7cf249.jpg)

![tables](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/tables/ed35417ce2653aaf64dc3d4759e0648d6bcd09c17e12522cf2afab9c9d833cac.jpg)

![tables](../iclr_results/3338_Scalable_Chain_of_Thoughts_via_Elastic_Reasoning/tables/fc414cb4e4ca2ef1719554276173a92de247b12f39de834f2565d8e96cc97c9f.jpg)

<a id="paper-3339"></a>
## Not-a-Bandit: Provably No-Regret Drafter Selection in Speculative Decoding for LLMs

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=JMmljf895g)

### Images

![images](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/images/364c74daae7f32de50071ed65acdf5b532e7937f40a5fd9d3e30bfa279a7ad65.jpg)

![images](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/images/5d200bacf11c1c8c2b252f06cc0c108da3b5afcfb3be3ec008e1ed74e28d444a.jpg)

![images](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/images/880cf3aa78bca5148ec242efe4e8c8ffb3fcdb492ca67e2a05e5bcc5deeb4021.jpg)

![images](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/images/9a26b18f54aeaa3e733e1caafb955fe9c8670f16e35cbef6adfd5a182ea78fb4.jpg)

![images](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/images/a257afcbfc5bf7e499cdb3fa73f025248a41780557d9c15855d09b8808d1d686.jpg)

![images](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/images/b8acb4b2d491455f3555f9f3ae85ec116d63d7fff5a847dcbb260e6713d80c59.jpg)

![images](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/images/ba6898b1c6a2239fc0f0be19beda2d90a4fa80ae44a5cb4ec6f70b109f80be83.jpg)

![images](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/images/fef5b52052f816e67f7c7586efebe9ee0c70afea5ede3a315d0ef3acc6448558.jpg)

### Tables

![tables](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/tables/0020e3075577e39f80975a36e3e50cb05a072166c97d5eccbe64a5ed7b52c871.jpg)

![tables](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/tables/1c48490b0627be4bad5729f39f1db7a4c291e0e59e1aff2b056db90d040c2af0.jpg)

![tables](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/tables/409274c5b82ce2200873bbbb87774fd40262742b4c9404ae3bbcd43e64f79f0c.jpg)

![tables](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/tables/475dc72ea2c59d8c385818dffc862faf16410d256e27370d76cb3541c244f46d.jpg)

![tables](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/tables/48399d0e2eb3adb07634e3ed3c8688605e27021cd11b73e3b03ab1d4ec53ac95.jpg)

![tables](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/tables/4e7f1418e67d477c48c4938ffe55c9bc47dbe8ef67db58566f3fec777a985188.jpg)

![tables](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/tables/598254bd3817010c68dd151ec21c2268209f372a5bd656c2cbf803447d696464.jpg)

![tables](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/tables/8fba17433d39e1eff369936cb27521944d6f373fbff6dd3398ff755c8913bb55.jpg)

![tables](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/tables/9cb6941f3f7a4ade9a9ba6ea67194072b727270438c0b5192cbc9a57865ef672.jpg)

![tables](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/tables/b322ed29cbbb92dc094c47ab4491107144e467a96239e0829e85fa66ce660301.jpg)

![tables](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/tables/b37f16f3dc5b517abcb3689c2b4464543f142eb342e6c69181a944ca96c80fc9.jpg)

![tables](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/tables/c831fd26dfc3a551d058cc60e4dc95de93589ab397439e32e4ee2624262b9397.jpg)

![tables](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/tables/e1bc481b85251fcae4a786f33e86f3ecf70fb85242029f54b4e58c6c95cbae01.jpg)

![tables](../iclr_results/3339_Not-a-Bandit_Provably_No-Regret_Drafter_Selection_in_Speculative_Decoding_for_LLMs/tables/e79577a0138548728e960c81bf1c23e79099e01062447d73d845ffe672d9e3a0.jpg)

<a id="paper-3340"></a>
## Antithetic Noise in Diffusion Models

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=9yFORC1tu3)

### Images

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/00c4bfd9f8ec5359b731f286dfd83dc05d6acfc781e4efbfc0117129aacb0223.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/0272ae94b5c3196b43fea53c83e647536955f3213ead28a61777f2d5aa65080d.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/0402c8fa593d6e39c0d0da986432db6b0b1c48e4ac4a186425da594ed131603c.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/069567810c4b766ad9e606ea3a7b1e1493b327ce01d4e2b15f48a52dfb6b280d.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/0a5cb485a890e84ee6ba23351fd9b8629f1bf647ef105698f952d277a36fd62d.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/191bc2f109b123817e818658cb5fcb4df9b712a0a6b0972a3f973a0f66c7ddb0.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/20fa8caaf46b7d8f29216975694fdcbf7672494976f2c27bbb4f385498f61073.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/31e36b0dd0b5e905357d8e1ea85ef1648509e348a3947c5f369d59cde6dad2e6.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/3bdbdbd9cc387eb9ac134ee59999a60ed2cd15fa444aee96dc498d6dd4e91d90.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/41a20dbb7e64dd4aee4ba3abe16a19363802718fdbc02d1340048fcbb2093cb4.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/41b55afc9d6822df78b7ba3b49e8416454b134114c73f14ebe446ee25f1c37ae.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/4632012a1f1794161381f2a4289898f416dee728a9a71924fbf3c1d23077b011.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/46cce499d97a56be7ee95397d1f6f22dc76186bce2274b8ba66a08541a294249.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/505f329af4bfbfb056e132258ae21b754cd616e0aeddfef166323f4f8475bcce.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/54c0b656a4a6ba2f89cb3a7ee5615a501ef0cd8e8136070bb7110930eceb95a3.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/59a31d4a2640337bef7865b2566db65846973865f17fc772d88771df55fd427e.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/6bbfa847d04df472e8390e4ea0e8c8dc540dc440f8eddf2755f3018876064b4e.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/6c1a5c418cb2a6fc0574c4e9bc19ed71fc888904330cc074ad0f80c2c8858e69.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/72721abf2f29bfc90693619a099b277f9eef2037e97f7194118c997012ce14d5.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/88fe93cbe48c70c99d0c78aa6f497644fccc63278a950fadbae9d5b9dbc9d11e.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/89d93a5ba800bd0846f5bbff3b4ee1c0c84cff1a6ae76ecd4e91df1f592f8a16.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/9ed202d451953d45a075ccd82b6cd17c560d95afda31ad2bfed98d1b5720d224.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/b0f5c8297232f678c60424f266b11edb198ea271a5ed94210d028e9c056d2553.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/b253709a9b2626bf38a85d80428fa97b14046d8ec87d6a4a70b479c27e00a8d7.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/b735117b11613c0c42b4e4e19ec371b4789c85c43367b120d10275469a1c4358.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/c910d0b4f8194e07c3f16c3a25ce415cb3421b447b8bc4122f4432b3e972ef91.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/d2e94fd356b077f111be25025c918f7847db952b082bcf23ef0ed92f7ed3cdc0.jpg)

![images](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/images/d624483efcf2256cdd89b2858f2197c35963161d7aa1bc79587bc4751b525ee2.jpg)

### Tables

![tables](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/tables/6721eb3aaf96693dfa3895ec7477830991d9cad929d9d3fe8363bcea1180c917.jpg)

![tables](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/tables/7fd9be82b8103a629b026db8c18b0ccba477e61e45809fa7959562939e10b32c.jpg)

![tables](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/tables/8eb203e0cd7b686bc216b482d1d3f4c4cde6b2ed4158c27973c79683d4477bf1.jpg)

![tables](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/tables/8f8ef1789174853f55b6424aaaf71477ec83170b129f507ccb1cbe68218bc38d.jpg)

![tables](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/tables/92559bacfe41212fa8922a69c4996999337363993f8ed455417217c1151a63b5.jpg)

![tables](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/tables/94b755df9ce1bf9431862577c58161d32f43ef2cc1d6eda1006ab63aa4b83b69.jpg)

![tables](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/tables/9af097dc567542cef57bbb34e619ed1bcb94a6dec4a37443110fa393c639184c.jpg)

![tables](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/tables/b5ed1ccf5d331b2c9d85e68bd0c2836f127af76a3ea02015e1fe3268ec73d152.jpg)

![tables](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/tables/cdaf9c93f50c133947ed69fae90336f7781a0a1cd229c613817df4845b051190.jpg)

![tables](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/tables/e5cb91d59c1dfa19eb72247c7ab8c3cbe202997de4b87afe40b4be14fe14f0f6.jpg)

![tables](../iclr_results/3340_Antithetic_Noise_in_Diffusion_Models/tables/f2e4cea73415ef75a423e8a1dacfceac55cc5753fb417de10d3e3a93bb0d56a5.jpg)

<a id="paper-3341"></a>
## Bradley-Terry and Multi-Objective Reward Modeling Are Complementary

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=3QHKJcwnpb)

### Images

![images](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/images/07a439ed86b4c371ac67b4a3bed7af081d051332383616aca034d9902dcd6fa4.jpg)

![images](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/images/082714bc0d315d7e324ecf994f0201d6f444f74456b5738a5489db1406fbb16f.jpg)

![images](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/images/289e4be164e8219fe87201151eac303e503e180f95cc1e453bc7a5e9967f6ed2.jpg)

![images](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/images/30ad4186aee866cb986d0597338ccb9ca9d9bb3d7231e78110f62b0cfe833a87.jpg)

![images](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/images/3411d8be51adaea8fec64436dd27a2adde64dd0d2dbb61c10e364b9f0b46514e.jpg)

![images](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/images/456d59aeeeefb7d19e4895e1a38dc9635528244a7cb126f55b5009ed432345cb.jpg)

![images](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/images/55e021ef91c2e5d42bbace71973d5a2d42265f7dc7de1b008c421f07f85b95f7.jpg)

![images](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/images/56a761fe2889e378bff6242f7bac312957973bdc952d1a685c2c9d819cc95e38.jpg)

![images](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/images/6e51d4ac8a22f89fb88a0a352d328965ee03208254d68a76d07b9222bb7836d1.jpg)

![images](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/images/71295ad39f2703b1b3d3357e02da8c184a436e50c24f66c9a9d6ce3a3be7f1a6.jpg)

![images](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/images/98ef30d0a676c5b6eafac781e4abf748765142940f18180ebe062be911278de3.jpg)

![images](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/images/ca1b707ec35f7c7f1c054289a4884995741225e2775d19c711e50dd173518004.jpg)

![images](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/images/ccc75ca42cdf68421daef5a776454fa14f706521f628526299e6c57633f71782.jpg)

![images](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/images/dd3700a5969ca66730026093a825ebdbf0f7846f2e2636ffd0c50c7085ac24a9.jpg)

![images](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/images/e234caf952ec0bfc599e413827b97855ec865f430fbf83621af7d35dfd2ad1df.jpg)

![images](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/images/f94add41762478da95f332ae9923144b5d51cfb86faf9685e37264f7db71b943.jpg)

### Tables

![tables](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/tables/02d7ea068ec3ba169c4cf4f1550425c4b24400a12bfe657521983eaac4cab2aa.jpg)

![tables](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/tables/25226f0319adb074c7ce3bcf9f387ce482ab7aa030d142b966b36f0bba5c1c99.jpg)

![tables](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/tables/256d354464d40c2ecf9187278b6dd74ff0ed7c00395bf8b549397932c9d52dbe.jpg)

![tables](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/tables/263681300a5d8d488f2a77386f43b94a06a0ac3ee7787e0f89d88e21e71c6fa1.jpg)

![tables](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/tables/264ff629f833758d05c73a17d81bbec069852a293a74cd9579a18d5d88ae033c.jpg)

![tables](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/tables/410c85932d3d3f488e86c6eb6fc899d354c804f69c6371b1dd50bf5090040b7d.jpg)

![tables](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/tables/467af94de98088c500229f44d299944b93ff8cac4faf2768cb797a24a9add0f7.jpg)

![tables](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/tables/715d36448d038a60b879d3cd180b44ec2b2bcb65f9371067a09e7cd6f8cf4d62.jpg)

![tables](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/tables/81087157aabf189544a5c3ada963c5eca3940e59a5c43c392f3739dd352d4681.jpg)

![tables](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/tables/99ae48648eb15c0d12e310cccba30c352c6a7ea3bdadf587593a23e995a22f36.jpg)

![tables](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/tables/9ce8966abf8c75369509e9708c006ea78e7f47cd25c617f24d49eeb7981ca3d2.jpg)

![tables](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/tables/b4b484b52054f81e8dc87e32ec2dbc157c07df0874a799bb13198f619a51a371.jpg)

![tables](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/tables/d0c98ff034aa9c67e5c3594563dde14f34a6dd7aa6cbadc4bf189ca1f43fb62d.jpg)

![tables](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/tables/d88ff13ffd4eeadba7b4b8ecd6fc984778cc13623c508f32f4f92151580e0753.jpg)

![tables](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/tables/e5644ad98f457226dc75188183fd6a649a628f53e7f863823c3769a1b3084b71.jpg)

![tables](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/tables/f462fb01f13bdd5b9764396b66a3008ff3f907c22224977ef3a96e4d78967b30.jpg)

![tables](../iclr_results/3341_Bradley-Terry_and_Multi-Objective_Reward_Modeling_Are_Complementary/tables/f7216f10549088747cec5d3488e7eee727e569a514ef6175c9c0ffc3d9013ae5.jpg)

<a id="paper-3342"></a>
## Controllable diffusion-based generation for multi-channel biological data

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=t7wIerUT2E)

### Images

![images](../iclr_results/3342_Controllable_diffusion-based_generation_for_multi-channel_biological_data/images/538e0a96a6c51e3e4aa775d398687e0cacdcdee72cdb99bfd046463b6be72ee8.jpg)

![images](../iclr_results/3342_Controllable_diffusion-based_generation_for_multi-channel_biological_data/images/76faf5593e8e29e17125d45f4a26bd6bc53a5cad538b7c2fbd55215ddd96ee90.jpg)

![images](../iclr_results/3342_Controllable_diffusion-based_generation_for_multi-channel_biological_data/images/796987c944f3168966e55c6cf190f4e6a13872e36b75610dd7a13585d42f8d52.jpg)

![images](../iclr_results/3342_Controllable_diffusion-based_generation_for_multi-channel_biological_data/images/99e9549ac6381b84f89f8d1eb90e8177f353cc572582a873ed0748e3f58f8e69.jpg)

![images](../iclr_results/3342_Controllable_diffusion-based_generation_for_multi-channel_biological_data/images/9e189f8683424a7cda4084a6c73974920e726b75e7fd48d4646b337f1a4a4da5.jpg)

![images](../iclr_results/3342_Controllable_diffusion-based_generation_for_multi-channel_biological_data/images/dc7519d25f5ecfa908c101664902263e79e940320cca5616f4dc8e8510ff8e33.jpg)

![images](../iclr_results/3342_Controllable_diffusion-based_generation_for_multi-channel_biological_data/images/fe4a17f2f7f7a942d3df5048dffdda267bcfb81b714bbac1cc2ff51d31a5f2b1.jpg)

### Tables

![tables](../iclr_results/3342_Controllable_diffusion-based_generation_for_multi-channel_biological_data/tables/04922a43b6d0a8d681f261ea65679a66b18f8ff176461cef9b22d4304339e267.jpg)

![tables](../iclr_results/3342_Controllable_diffusion-based_generation_for_multi-channel_biological_data/tables/18118ac1a673868299aab8ba31f51ddad401ab002c1088a4c00ca97f150cc2a2.jpg)

![tables](../iclr_results/3342_Controllable_diffusion-based_generation_for_multi-channel_biological_data/tables/6fa57725e3c3b8e7d909e1ad55f0c0cd5328c93133fd9e28976a55a86256b090.jpg)

![tables](../iclr_results/3342_Controllable_diffusion-based_generation_for_multi-channel_biological_data/tables/77c916f4d976d8a379c297a049146eb884b3f36f8eb06ce1227fab6df0afed36.jpg)

![tables](../iclr_results/3342_Controllable_diffusion-based_generation_for_multi-channel_biological_data/tables/a47ff4900264a2406fc0d0fd59e664e64df6c9e6f9e6ae967b55b57595f52210.jpg)

![tables](../iclr_results/3342_Controllable_diffusion-based_generation_for_multi-channel_biological_data/tables/eb466e0d42ba83906899082bc63a4390abeed061dcd5eaaa681caef8ceb5097a.jpg)

<a id="paper-3343"></a>
## LLMS ON TRIAL: Evaluating Judicial Fairness For Large Language Models

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=C5Ihi4bVQt)

### Images

![images](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/images/16da56ed3fdb503a86a4df1f8d85d5c15b328a671edaaee6220821ef5d04f185.jpg)

![images](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/images/1f31befc82c666bb3feeaf5b178d7ee5dfafb80b75490c5d5e77eca02de49873.jpg)

![images](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/images/33a591ee462c877844c22a50e471536c484181ea28170312a06c15e80c5a0962.jpg)

![images](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/images/364cb9d7dbfb30e66054d9eb8052107f2a2b3c922fac7ed5aee66c8a129c6313.jpg)

![images](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/images/391a62eabce2578d996d6bd58ad30b7de7e3abbd14461b2a0245c2264799bc94.jpg)

![images](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/images/4c72df9063ce16407da1656ba7d0ff0a3c3993ca87865e413e4a29a4121823f9.jpg)

![images](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/images/5355c33ebf0fcc324557b9a20f8337b2743ff282c793adfff70b03bbcd683444.jpg)

![images](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/images/5595cca84a0d27a255085ed1cd90339e5efd8db13c07ed03961b7061465c8159.jpg)

![images](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/images/5b89210a89806af0375ddfaa6c162783522b4b540abb84f33c224c1c68ec1979.jpg)

![images](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/images/5c1e21d30801b711eac0c24f790bc31f18bb2067694fceb9335674c5df939654.jpg)

![images](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/images/5ef7af7ec8586be07e54335c2795bdbe771b69a330411e0111fa17e75132d519.jpg)

![images](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/images/6767f691d2b6ac9f2a53b4bc28777b2d7392383d9d9fe8ebcc3233a47f3e8450.jpg)

![images](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/images/6cf3d21f6376b7bc5c0eeafde0dda7d76cd3876a251e52b32743c39f606c9171.jpg)

![images](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/images/7ff50e4c36d38182580c7e981e7b1555994fdb7e884cf71dcc9d878fe8be6d64.jpg)

![images](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/images/81314cfefbd6b0acc82e91a3d209713ea565a974dd6cbca33fc90a8be8870a53.jpg)

![images](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/images/8470eca27c7ab40bd22851514f4f3fef5ba8d5ad71fa6f0bacf00971303d6c5b.jpg)

![images](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/images/9549ffd0c83215b8594374be234efa5187b2d68a49db75e37625c1977d061e28.jpg)

![images](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/images/f862083d01beed7e55e500adea0a08a965d630b9a52db20bf75af33e4bd8926c.jpg)

### Tables

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/03ce1268dc4138ae4a42249f8f63d0f1fad387de09e267b567afa760f926d85c.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/0b96aa4883bf6a0924aa8832ddb9e9e0dba48ef7ca2abdc0e404e226ddf6afc4.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/0ef1fe9e3878663b3f38828c311926d317fc19f446a376c22efa91dad467f245.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/16bfd5d8d5206eba58b24f9542e391a0f196350f661eb8cb026813e4a0e0d8d2.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/1c054d5859af7bae5e4cebe50f64cd14736d4f5431d98d823d0150259c0614f9.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/20bd6ba616abaf69a68bcfbbf1e96324a0bf4f710c73b8e7aa5b466af2a59d91.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/22d9f350ebe67b54742498901dbba7a0105428ea6ef755f2d3f3ad90a2a7441a.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/250335aa81ce402fe62b85df93075f7ad5e36deebbc6a1dbc05145f2657ce5cc.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/2c9ec66643a81843f4b53c4d23a8d7bc4859e00fbe55935445bce7786cb1cab6.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/2fe794a705e50da04d8b037730642362113e133f6d1eb1c772b926b1ea66ee89.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/2feb8c991746cdcc6ec81121522a5808fe54ff09c3e0184ef325047fe01252d8.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/30b7c5bfc6d4317deb8b48b76400ab64a0d8fbcbb55d3fd7427c8d7d30094a8f.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/319b9e5e41d3fbc98c7ac82b02ef81981b35a249937eca83ed14eec2511a8058.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/33dae9ff49878d683cd6ac3c76e044c2e473f846f03354019ffd35a9d7a62616.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/48a7ce123a7564e4659a6f28482ad620c7008d2f69397baa0904e89e61b5cbd9.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/53f141dfb243d2e24b0a3cc693f0d4754e350219f0d7035294d5afd8e44efba5.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/53f2f4a9077724ad7a5eb69e38b6e7e0e22cec0011daa36db79a4d07ffa84828.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/56a717ab5b1f38f39f5c2d993e3c1530e5dd56b87c4a4d1339776e3a2a3d9ff0.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/58f012c1e508f81f6bc2491edcd280e86f7c289771819ad2d0be859a620bfded.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/5cb117b11d413f5e6d03c847139852068a6ce6571bb6629c38b0c88b501d9590.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/601f9a5b3ea7ba32682e5316a926737d994b1db660d1ec13e9be58c5f8a89628.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/63ad8e813bce8bc1928064bf2fd3e1cd9b2d4c74a5f1b14b5a3e927ecfb72489.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/68986c7b60b0755f506c41f856dc098a4cf9c84d5ce0cffcb3004660d2a12c0b.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/6c9cd6ff2190c78d8df66a798e4f94eacd852480196d8073e7f035d01de171a0.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/6e5ad43b37f107d297eb1245d771f400bfba679f79f5490fc26c69568e96f3e3.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/70c1b4998d830c4ef1735062e073ca3a70ea3948c259b35b61d31e53e70a1e4f.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/74f11428bb17af86504486eceac35d183ede69f56f4438bbe54dcec390a8b988.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/7cc18187a6297d72b6d5d2127a894bb04e201a854a276a32fedb18aa7f6dbba4.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/7d1f56b3d122b5723f74588904ea4e19ad6d829bc8674af97f05eedddbe3313e.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/8007667570089786315c71955c55fea39a69ce9f5a75c2ad2fd1f3d29da0f4a4.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/87b623ea3bacc0ee033a1a158d2b5f630429c437527e6b5f23db2a2e5c64ef89.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/914682a61b4b3fa51a128c5381c8b5174a9b2e0c82e25f2d1774755b639aaef0.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/9628ab00491ed5f7c3b9c1f361732c9660efddd975e490fc21b98368a3be25c8.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/9b4627c98dcc66af184b1f6df88b8901d4de98299434d07044942f30aea6dd02.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/9fd31ce49fa6a45a7a4a4766d138064e18a6ec9e836e231ce91a5668619660b9.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/9fde7b17cfdc89f616e4d2a81343e21d9700a900d310f7a209617dc2c60a7625.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/a5c4876706980efba468894b1476550e330490219548b1506b52536fdad35bc4.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/aee3c16ca243fd01bb3e34e8182bc4dbf3707a9b8fe5dc21d34e78b7e9d01006.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/b97d96329d6c32308acf5ce022562f9585162f7aebff2344bde46031aa9ec93f.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/b9f204d6cd797e1515ef1353e969473c5e5fc40d3a96c076b73004c3ede30134.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/be7e7f788a190b1b4b5070e289fd530d03da0c5bac7672e0031e49f1900c3edf.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/cd703461c2041a70db86a80c3df48bd81694f709675cee67ae48e37830e14579.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/d4e8c497ce30c389defb69808ec41570f37cbb3cb5317a1c650a76b2a8e1032c.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/d783ab0393acc33240852bcc600d9437a2f64c591167a77e9addb533977b6dbc.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/da5cc148b9b5efcdaa730235661014bbfd427ec6047f1e03a93bb6274a1fbc6c.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/e68a1f49d9353a47d0ade839091576a26801577c76d08183624c59bc3e7ce17a.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/f98fac4158185cbfcd7b3e922f0cfba80fe02bd91942980a497a1c48c3340fca.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/fd2840fd21c133995a9e9d751ca6d482a628bb2465f6444e10d852328d0a7a07.jpg)

![tables](../iclr_results/3343_LLMS_ON_TRIAL_Evaluating_Judicial_Fairness_For_Large_Language_Models/tables/fe6d33f117a91aa07c710599e23a1498869d7bd914559b4c182bb16aec7576fa.jpg)

<a id="paper-3344"></a>
## Go-Browse: Training Web Agents with Structured Exploration

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=IpzRWE52yw)

### Images

![images](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/images/06f10c6f80c9f4a3309e884bf9cb2dd18511ce77123924f334e5a550f96bf18d.jpg)

![images](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/images/139441d4515f559234a789b83affc6d0cb7343d9d078770d239d9fdfb18f9305.jpg)

![images](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/images/193e976f34897a4609a7ab807d8864597c84dea8ec9c9b387468336f55df811a.jpg)

![images](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/images/1b81070324fc0daf08370b915d57d594465126e09edb7e34fb8d3575e67886c7.jpg)

![images](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/images/263e466ac081b7b46d638a4f406ab6bccde1c7d426e0b1bd2410e89480fa59a2.jpg)

![images](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/images/2c397e74cee4ad73f5176838da7126d1fb9dde5df7646e7eda04c66092814935.jpg)

![images](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/images/bc717a6e72539100a037948c226dc0f9c6128622a671fdb7d6f248b1c1609c0b.jpg)

![images](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/images/c0821b27ac81ff5739ecbc7572d1e19222b0182cbef365cc6fd924917fb63f8e.jpg)

![images](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/images/d4dc9fe8cf4e5b10ff3577060a021886a9d9c6774fbe74752e00a674f4e25747.jpg)

### Tables

![tables](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/tables/1fec99c463f019717df4f65af1e1bf5d67a15d1070140ee0876f4f6c2155e449.jpg)

![tables](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/tables/227dd20c578567c81d32aeb056e6933d2441c9977d5b9db818d27337f1680f02.jpg)

![tables](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/tables/6b65fe4c6c677785553d37f660e5ef2135edeba56298093598995b47d427461e.jpg)

![tables](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/tables/7473f5ce81738dbb120e974f1070840aeb866130909216508e38e5f61e7ed942.jpg)

![tables](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/tables/8d4556b2c2a13d477ce5d5a7aea91bda474e23c1b593a4800387092b682d26c1.jpg)

![tables](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/tables/a59f45ef33d8e0cf03b73efff845db3c6210fc8ff684dbb149285a3a79035cd8.jpg)

![tables](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/tables/aed7a9f8e233ecbea536e9e1241c920fdef53fa79afd0341465bb5548492349f.jpg)

![tables](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/tables/b94628b86c9e9243dee1c7ad7ee6eef654cd41ac2f58ec942f621c60d4f1eb9f.jpg)

![tables](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/tables/bfd08fa2e8c8024aae672c6f3557b6695898afaee6297585a650ee15e17b9484.jpg)

![tables](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/tables/d772ca6c625bac07d583ff4f7bc9cd7a5bdfb4a0f484b396e4c31f90c6248779.jpg)

![tables](../iclr_results/3344_Go-Browse_Training_Web_Agents_with_Structured_Exploration/tables/e2f5b919e0e3442abb6d1cfad8d2d278984a57d9fdad0b6954d75eafb69b704a.jpg)

<a id="paper-3345"></a>
## Can Vision-Language Models Answer Face to Face Questions in the Real-World?

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=I3dPEvbp8o)

### Images

![images](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/images/18d1d79e84eab1b30ef1fe29548c910cbe0ce4facef59b76c466b2ac692a883e.jpg)

![images](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/images/1cd3b975eeaea6bc9458f78ee06f277bd2f6b6386cef544afa9eda445af695c7.jpg)

![images](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/images/2c3a2f1be3f7b1f3108a67215ce0950150caf8df9f1c9e29e5a99bb2aebdfe32.jpg)

![images](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/images/2cb8d843f61bcdd95adb448619f751cbc4d21d28b9e2735d5c1b6bedf6f81d34.jpg)

![images](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/images/2d5430d301af2612eda391b5cc047189a9b63466b24059be34ff71fbd7cf41de.jpg)

![images](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/images/3fc4e61cf4ea49ab6680ba2fd61a91e370975a80fcf58696003fbeb0558903c5.jpg)

![images](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/images/5acf49d6ac8f5104c0f86d79c161366ff8f217687fceec5abbaa72c15fdbc5eb.jpg)

![images](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/images/7a4b875a8637aae12ee9a17db9e0c95fd5f163cb37b59a6b1270e26de3ea009e.jpg)

![images](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/images/9dd2212a69e077457432d2ba84707ce0accb775b3ac62867a83e13f380948f00.jpg)

![images](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/images/bc4765b38a1f458014c45d74dcf6d83269403abfb0938eea58a2d249be6d1812.jpg)

![images](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/images/c5a0823cd062332c49f6334845e03af6abbab4449dfb18c3b14b473183a9d543.jpg)

![images](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/images/cd9c0f9c97839f19fea1465b546de388e729748940ee638c4274c86afbb0bf6e.jpg)

### Tables

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/12ea71105501407e1b6c29eb070f9eea3b609b11f15bc2dce158601f1fa270ba.jpg)

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/14f65dc03bd5ca773605cdbb35ff3072c05e590d769e07ef10d9825771cc559f.jpg)

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/192c297ef29c651aeabebaa190a7d295c57d3e2a26f5dda8587833ff2879c62e.jpg)

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/1f751e27c1b73dfb0c5860702f09798c66e2a5c4874b9d415a9bf75eaf47d1e4.jpg)

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/29e1fee4e092021fb49d9227c2d1dd7b984791a07c370c2ca5b6adfa2bed8b7e.jpg)

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/2a3f5e1e543f2dd0a08cf8cb34d7596a65114609bc985f7b5c98df8e4af288e7.jpg)

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/3c6832e80b601786a99150ff9c1d1e8a00979d50cb061553a27a8f9299dcd841.jpg)

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/3d0941889c670aabfed28af16713596a3d0602fc5b8445670e2f743613622614.jpg)

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/4c5c8723607d5ddc520b276dcf2d4a48f91188121a1113750653bc3a08bc0534.jpg)

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/7df6aa25cfc05782cb8111889d898843c5460d3e7f7a0ca79b95f98fce9a540e.jpg)

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/7e625035dd987d7c886e83814b9efa9a761473b8a2fb3ffe7b7c9f428d99879d.jpg)

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/7ec4c9872213f2f87793157d5b12df259373b87c093e7c280de5007612c9879a.jpg)

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/808fede24991e5c9b35ec04c840a524ce8fc1a701ec06b0e9fa5b4269f33e415.jpg)

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/a7c9905fc0e23954240a117d5a6a465de3acd4265ca4c01c8fade6c72bd406d6.jpg)

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/d1e6560604bd632d75074ebc95f21934beff729165a3bbae1fa891d70e4d11ce.jpg)

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/d83c7816c09f4b0138b29839a131ac876c988946f9ea4a41751cc27063ab0f0e.jpg)

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/e141aa729da27b9a67fbcd2c4d9c1145af2e325d7ee9dc219e057a426e26c4aa.jpg)

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/e2d1055b197576979d50866e02942a50e855433704b33d796295b1aa4d1eda53.jpg)

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/e8bf649055cd746848b57be251be9a8027e443bb8d22107535079f1142dd08c1.jpg)

![tables](../iclr_results/3345_Can_Vision-Language_Models_Answer_Face_to_Face_Questions_in_the_Real-World/tables/fc89051df90f535064193f14df2e52b9c826a0f51142e5ce4b268a8ae7d3a571.jpg)

<a id="paper-3346"></a>
## On the Spectral Differences Between NTK and CNTK and Their Implications for Point Cloud Recognition

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=U6SnDgI3gG)

### Images

![images](../iclr_results/3346_On_the_Spectral_Differences_Between_NTK_and_CNTK_and_Their_Implications_for_Point_Cloud_Recognition/images/4776a86451be362a47ad9aecb607ffe6c98204adbad28e748cc42576ff9370de.jpg)

![images](../iclr_results/3346_On_the_Spectral_Differences_Between_NTK_and_CNTK_and_Their_Implications_for_Point_Cloud_Recognition/images/7c3ed30a2da27d3ec6102db77efb9f8f4a20d52295ef249a8f503fb987c96aea.jpg)

![images](../iclr_results/3346_On_the_Spectral_Differences_Between_NTK_and_CNTK_and_Their_Implications_for_Point_Cloud_Recognition/images/b6300491fea96b637917955b16847f8a24c141a3f3afbfebb54d4d0f7efdd768.jpg)

![images](../iclr_results/3346_On_the_Spectral_Differences_Between_NTK_and_CNTK_and_Their_Implications_for_Point_Cloud_Recognition/images/c06e35fc982e5080d0dcae7eb551f863e0a8dbc82cea3511712cf12e73b6e401.jpg)

### Tables

![tables](../iclr_results/3346_On_the_Spectral_Differences_Between_NTK_and_CNTK_and_Their_Implications_for_Point_Cloud_Recognition/tables/2f25a7eceb3c101369b7ca742a0907aa085eefc63c876113fbb0bb577b7b86c0.jpg)

![tables](../iclr_results/3346_On_the_Spectral_Differences_Between_NTK_and_CNTK_and_Their_Implications_for_Point_Cloud_Recognition/tables/ac306bb577e40abe617b819508c08e7e7c9429c2b4a5c2928b2316fff77291c0.jpg)

![tables](../iclr_results/3346_On_the_Spectral_Differences_Between_NTK_and_CNTK_and_Their_Implications_for_Point_Cloud_Recognition/tables/e15fe5041116ad96328caa0d31c3a900b593bba074cd5bec662d73065ea05795.jpg)

<a id="paper-3347"></a>
## From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=rkthPeHvAX)

### Images

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/1348c7aa33d4ed8a1a7272c3964531feb2793c2d18de0542cbbec898c43583b1.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/2043fb2bce20e5f3b5738816cf6d918541b651194580accb499bd4af28525ca6.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/337824d66ba37914e37d9cea31065f35e22c04af7bae9933ec55cc2610a874e9.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/33935ffe451df0357270a45b70cb20ab889a49c56d8eecf028e89c5a9a6574c6.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/3e0a3a581a7a9910b3309304da8036fcc57bdb80c0ff1c72e064caab9eefcd7d.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/50823992e7723bb1927ab6715d158dbb47d296d20c50fd9000fe4a520f9ad544.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/556e470e1a68aa2b063beacab812aedb1222f83b88b4544e5b22100d88542233.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/563342a71228d14e932ee52e0c6fc53e07ebdbb17ac0d38ad3e07f2598015457.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/57838a7df51a64548e4ce1411fc138c97bb6e83e3bfd569287a851b2156b20e3.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/5ab7d90665da4963078364923bf1ca347145e9a7ef60903cb19373bf32d683f8.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/5d1bf65415da4c17bc9bd29c807eafe34a07ec256254aa6784c728d51eeaa76b.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/5fa5073042060e2c1343f168e7dd5a0ceafcd79f95858c17f224e2d1efa95106.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/60e9c98f57333b9de216cf7f8a76870f4619e8194174d14e46061d6fcd386e32.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/6ac753390cc9ac2a78397b4bd3213463df4d76e396d718a33d92b6bb383a7f0e.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/6f05856abb4aa9317546d4f25a39e4ffa4deba995e1be52973636b6642347a4c.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/7fc1e51320e45cd8c650b03e9ed014e6052d36ea88c06017b2bf82535990c795.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/a158febb7a268fddd2f7f3f3e01dd725a1c8ced7a1880e6b908bc4acfae37ab1.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/abbceb2595db4dde76428db8c6203efd1ea05b2be0f737465b4af34f390a2255.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/afba2a47c5f3c6c7f599c92e7a5674ec15770aa5b0f9b0fc6330a33375574c43.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/b0ffcf0bae1407e0f0b08101ace791412d4493b725e914062a0dbcfbd82bfb3b.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/b701090a6132d45d7ec9c542a4d2a20e53fcee27e168ff47b0d4f503a5e34743.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/c13a2b5f4eddf1a55babb6ba10e75c6c22d6b86cb65053a02222a66143c6bb5b.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/d77a334e35af56141804495f877afe6acc4cb1f4fbba2692e4e2bb16080dbefc.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/dd1a7a79cef00f8ebfcb36e8e9f0045d3856d42760f67bd9f19db966f53c4f80.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/e4cfb75fc385d341393ecd1cf902ec0c267bbb617e1d489aefc8f230be24d2eb.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/e5a027a1965ed840498913c39fb04ad97b9452d0b90a5c9ed68caaedfb473e55.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/e889adcba04b07262cc62a2696d635490f53e26c7b5d60b2d69523f799022304.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/f34fe43db6a5d0a927293ace89f2cb5ed780ff20d27a2cf51b7a0615c731b5a0.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/f66e0cce8d52ade5636cdd49f679e2255e3104c51aa5dc8933b3d3c586793f38.jpg)

![images](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/images/fbe0ba1fa0c0945aba13de3dd067679444dbcdd9f0b06b26fb3e34d6890dd251.jpg)

### Tables

![tables](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/tables/03807ac9079f4abe5de6a5eeea33461750b2b58f61a550ca2cd80c3720758a47.jpg)

![tables](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/tables/1cca7541ca109601e71d2d32b711c369fdefb97143f0fd82625cdbd07c061dfa.jpg)

![tables](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/tables/2b1a15026c22419931cac045ce77d631831ed503f63d803c5c87dd7064b0a593.jpg)

![tables](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/tables/5f8ef664024e5122307442b9bc0ea6e5a8839651e254542eb60cb3c671e12482.jpg)

![tables](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/tables/6c640c5330ac965545b24c390bd2349522c21d9be2fa5d5063a07aff5c4d8658.jpg)

![tables](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/tables/9d5dd085ea6b538441c3d7752204cb57808d5192d3befed3f628ef3f5dcf6d0e.jpg)

![tables](../iclr_results/3347_From_Tokens_to_Thoughts_How_LLMs_and_Humans_Trade_Compression_for_Meaning/tables/b32af00feaf8188cc998ac9461563a8bb01073b72385005f0985b14830c73ee0.jpg)

<a id="paper-3348"></a>
## Parallel Token Prediction for Language Models

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=AGJomYSrUG)

### Images

![images](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/images/197f42cf45957bde022e6cdd807aef66005ce6f373dffcd612e30f43890b1569.jpg)

![images](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/images/2fe869cdd967aea023832e93b32bcc404d6cea159fa4541e00fa3c26c08e3374.jpg)

![images](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/images/628b58314b4a01246b8ceda8026d2ff595cfcf299c43c74415d4e1eb5012d37e.jpg)

![images](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/images/6ca9898bca423ab6d97c1750f4c6329bd7925a16edc972fa456173efc1a969a2.jpg)

![images](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/images/7f61765e23ed6247c83e087abdd568ebf04fc596552ccc993aab16e02d299169.jpg)

![images](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/images/8cadcb8c24ef1dfb2d93593289e44dc257dcd49b3b8fc549d494b2b85603bdf7.jpg)

![images](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/images/be0c5f5e589dc913e961b360d635b8e21921861d53ed228a87f895b7dcf66d65.jpg)

![images](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/images/cb2bc8a689119c1b5c77c5b5734f4afe722f95b792298462c31c4f03a5641ef4.jpg)

![images](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/images/d8c5bafb77ab11a5839e5e71199465c9efaae28facf5600632320332536aee7d.jpg)

![images](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/images/f7cc64be549eb802cc0fc570dce86eddeed26d1a16a188f3d5cbc0721596139a.jpg)

### Tables

![tables](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/tables/0df01d0f525d1efcef90e0e5ba4ef22438fc50f1b71662ada84a47c7432b7ba8.jpg)

![tables](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/tables/445d2cd1cc072f6f1e0410c8329041637bc3cbb999443d8ab59cc3d0b245919f.jpg)

![tables](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/tables/602fb8147d42fdfa59c935b040245f9ff88120859f780ccfa2c8bf4b151ed587.jpg)

![tables](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/tables/60c7d046f6f2e700a4df565a56b6894e543c311e5c4b9307975c51aec81cd72c.jpg)

![tables](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/tables/61069bb76df57dbb4c94e6c156377fd4cf17564d67364f0b95a8d2ef07b2678a.jpg)

![tables](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/tables/6f8f7da20acecce2182e025883cc00ed712523b2ee887c614017232e927d755f.jpg)

![tables](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/tables/6fbf6571af72c462b8ef96ea3a7579b009486bc56544d9778fd4943467d68e16.jpg)

![tables](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/tables/7034b9482544599faa4c4c62e3f82c8eb64215e72f452c4ef3508900f2f3e21d.jpg)

![tables](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/tables/9e4d0a013ff9ed9944a1cc08605604b7d5b3f0aeca10997f1c1a6cedd0df15e8.jpg)

![tables](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/tables/be44453043b81f4b0ac9c3299c1909f4645225450dd2d7188c2008e2572b6fd0.jpg)

![tables](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/tables/bf5ef6de8c96bcb61ee9d1cf834843c80883891ef4a4e9d4269b0c1b25f2d38b.jpg)

![tables](../iclr_results/3348_Parallel_Token_Prediction_for_Language_Models/tables/cc9f7b218e4b1b9ca60309234ed14b571214e4de63cd689c920bcae596ef42ca.jpg)

<a id="paper-3349"></a>
## Self-Jailbreaking: Language Models Can Reason Themselves Out of Safety Alignment After Benign Reasoning Training

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=akbtPEZnDZ)

### Images

![images](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/images/1113e64b337f18513b521f47a8df25d994cd8b0d4dfcca244fde01c61a717556.jpg)

![images](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/images/1fcf065472835ad300046a7e13534e6944b607e8daadba22b3ce2e16a0e2355f.jpg)

![images](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/images/2b93150dc64ea7aa9211d3db9c084a48e18852c0184a8b1c1052d95227f39860.jpg)

![images](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/images/2d73ea3f2f8aeba71efb0d0bfad1e1ace262333fba1f70ee5ef7013f19f88c0f.jpg)

![images](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/images/3a6ce8c20d3bb1d962e994e761df27807bec28c877efe5023a5c3eb33278c35d.jpg)

![images](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/images/3e3939bdd0008087657331e0e52ec6c40c8d21acbfac8071b4e10cad52274d9a.jpg)

![images](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/images/432920b7500fc6d242019ce6d3469673855dbc6dfb7f3e682c5302d4a9256580.jpg)

![images](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/images/553f5d46fdfecf67f628201691ff71af63f2250cfe30a651c43ef2a14f68c1c2.jpg)

![images](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/images/7440e92d52872e58ebd896d5dbd93c6a8bae71827156be6dadc9ea8e16660dea.jpg)

![images](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/images/79e437d877cdaebe12a7f51f773531aef718dd6cb6a04f9a40ba13d3de19d5f8.jpg)

![images](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/images/7c5bbda1624733f20d07d4f389ea449740d2e521a1f47897f1739c5fe9d1cf8b.jpg)

![images](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/images/8803244f8aafcd433f076e35680f6e7b60925d13bcd9a02d9a613deadbd7fe67.jpg)

![images](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/images/a2152681ee5dedf7bb2ca894e731a318d72c0687b2a54710ef28007e2a31277d.jpg)

![images](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/images/aee21e3a03cc6ea5db5790dc6a410ce5e7770f90340bfc8a80a3730bdef6f6ca.jpg)

![images](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/images/b9bb45a62e33870164a66a7e3cabf11bc10c7cc663a523effddcb452f8703a7c.jpg)

![images](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/images/f5d75edf7eb453570bea823c2f0b277d000dfb94d2d2e212eecd000a36e8c14a.jpg)

![images](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/images/f77ce6249c1a57e9521c431a88aa92a39275f95658ce0c8cfa702f9f03de1bfe.jpg)

![images](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/images/fd3d32c74a99254d6f0d4315b1937d1772c81299803db42592aff823c43e7caf.jpg)

### Tables

![tables](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/tables/052ec086dd9d6b67422c802236a8c099f195eda05350e8d052c26e86d42ed992.jpg)

![tables](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/tables/1be10f161def798d48dedd100ddcad61fffac5fa571b3a28f1d2658f8236ec5c.jpg)

![tables](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/tables/25a8ee23caa77d7b09a6ebc7021495e5b417581b3519a986044a8261413f1d94.jpg)

![tables](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/tables/2cc1ec520abb1a4ee208027601e35a8d28ef76f03f65a802da964f2a5a9adf76.jpg)

![tables](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/tables/7b14d15b6f7a66b1631127407c29eaf722c881117d5f671384685ef9fc25177b.jpg)

![tables](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/tables/a2d97e85a10afc6ad20632bd9d6e9c955c93b70a974b9cde5d1f145fb454eff0.jpg)

![tables](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/tables/bce3776ff4dd6d550b693ecbdce4a435a812b7c8e5aeda8bc091920076b5472c.jpg)

![tables](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/tables/d3237ada2e6feecc8e55c2930cfcbd45a2cc59fe52de7f71277a1525aa69d754.jpg)

![tables](../iclr_results/3349_Self-Jailbreaking_Language_Models_Can_Reason_Themselves_Out_of_Safety_Alignment_After_Benign_Reasoni/tables/fe6c2c22378a470840d4605627b8eac47ebc5bb877d3e372b8783b98e887644f.jpg)

<a id="paper-3350"></a>
## QUEST: A robust attention formulation using query-modulated spherical attention

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=HkztQWZfl2)

### Images

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/012f1cf88f19adbfbcfd14e96ec30a10d128143e764b38591c4b6c977f125a8c.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/07d34ea6baf78914ef3f104b28297bc7cc17688fb86e216d3ac29e9b37f49cf7.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/0c2a7d8a0d87cbd566e86ebaf41e436bae1185ea88400ae42e37ca32bb437975.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/1339c4df673db2a6842db2cd1b40ac53fc8ecc8018d839f2cbaa3544ae2ccfd9.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/2a844c586d67287cebfcc83a01de32ccc1a2b283d5958ecabf71c591480319d9.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/2d386017d49f8500aa51eb28f61179982761d45cdb5762530e11ec2ee4d98c66.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/35facdb9011a91803e27c0010c84de2f0f6874bed2d5b9204682289ae606b456.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/44abd17042c9ff02d67f13815242c5abf836f2f7c4a68f48410e7dc9ab8e581b.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/4a33ae91220c038da112e94796305d4620b154ed3ac72687a1d07cf03e2995b9.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/4de2533875f0a5f32a0f6d6262ff0497b5a9b663a413f253faa4a33b2278e92e.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/5ac7b4628d4b2ceb8c993b99af406516c1ee747de2bcb886e513e3439642918e.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/65dddbd0a7fcf8b2afe7ba6ed4a566b434bdc047e39928bae250d2d4ac6bb1bb.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/7e914b08c3039aeff44c697b27eeb411c4c25c07c044baeaac8252c16c31a6a1.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/97b884cdb1e6a7c74c8e9a7c445e013bbaa0979f6caea094bb3983c1eed2e07a.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/9e2fc298c5aafa1d6673d6bf5e3ff43d43145c5f40136852fdb684a99512dc7c.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/9fe868c1336f642b61a7ede55f67ee7b8c648f1bdabc134591d99be01767e8c1.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/a3eb43c957648c75bbe352e8392de39b1c00f2c566707090914589d3eda4cd44.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/b0219e20d41691f46f400907cfc9a446f2b705717f59300c77cfcb46a257659f.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/b11d28541547e27fc04f303a6802be6d1117466ed3e8b0e34d8e28da45d22ec7.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/c509c4bfcd53b3e759b6b7a2425e70508a2cce5408b65a63c8ee3c4f0aa9aa15.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/c898c944a685ca3e051c0a2f35d36a0753df1b6783168d1c268084cc224921d0.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/d86d7abec077a75dcbe6085545bc9cb77596f27b791647ba5549cba44c5df3da.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/ed7c01ab8514886206f92e3a920a914eed3624bc046d711e88c682388a8c16a1.jpg)

![images](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/images/f51a8f834fb7ad7bb82bfcf24811c465973a799577d05ca587280193d316737e.jpg)

### Tables

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/29f6cdb207bd70504bcc403c20ea6e9d5f466c972d9f30230d8e3f5824af045c.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/2b28f19481da341c8f70fe9767a7d103f13f0f795db4a7647f6bf25b02853823.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/37b778782c29fe75a8c2e16b2ce8a354c1896cf9b0ecfdc3ff18191a0025b314.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/478e68a89c2aa320a3e093700b093215f5d1313f122b1eb2e0b110cdd9af1d90.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/5a24b4ee07e5a06731958437943e2495c3550219a30ff95532797e8e51edd1cf.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/67b2a086828f01c8ac77363cfe1bfd86d91dc934f842427794c6dbb7b1e3d2d7.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/6e7a6017edf4e7fe5669fdedc1d51e4a2986aa9b4ea8bbcb6679ccd20c07c82c.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/6f59a5d56a6c0e9fc19aa23bb5ac9615dc78aa2604806ac8fa53f204ffbbd274.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/7106d89b32417b0a968fdd69d4a48e459c9e231f8693092adcebe96b47819344.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/853039f1f1a783fa4a42e14b667e64ce9e803c702a5653b03619523661f6c7d6.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/93416a5efeb14ac3dd93098acfbe0cf91aaf69ceff28c3046c5810d537194adf.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/99240a944efda37a986587ef36aa6563f328c356877b4f174a19080b7e433676.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/9d0ea5f3783d1fabed45184a2e3625835534cfbc30876b7fb2ad35f38d76e691.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/9ea2ab89a6f78137596596672d097f03dc32aa2659eee46702874f7a1133c3d6.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/a5b5606730a372ef39ae369ec6634556d1be4299d3f20d3f90904da97e7a95e0.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/a9b78b3390e22f286d514794339149a76682a052010a377159133d33980d31c7.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/ab218418871608e9a0a79f507df78a92ef0ab19e1876e5f64fe5d0e6903bdeaa.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/ab31d1625fd17efaada12a2a45b07e39eb9f1ff3e4f1493dfa4ea77aeff77adf.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/b7087e13e741ac9eb725feb896de4b127b8bfdc177a8bc1bd7af958bce8285be.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/c69b9477707c654a96c9c8b64d9dbfd2fba990b5f19b2df97c15a4fd51476589.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/c852a5f11a5996bea95e07b21c654f7d1b865c566df2239f788895307b4e7561.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/d17a5a40c2c9e9e18deb90e85d53351d9b358f19091a4cd46d71bcf9e09d0239.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/d909c6af29e25640defb8e59c2a90437b2b30f19800c8bb8623eb97e450fb604.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/db914ef5870f50d18bb7193d2515240be5a95252229aeec7ac5c012b5126184b.jpg)

![tables](../iclr_results/3350_QUEST_A_robust_attention_formulation_using_query-modulated_spherical_attention/tables/ea7d5c4d44c3f248406dfaf1517589cffe9977f2a3ded0d03aabed81fd31c455.jpg)

<a id="paper-3351"></a>
## Learning to Be Uncertain: Pre-training World Models with Horizon-Calibrated Uncertainty

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=pZuZWRuPyi)

### Images

![images](../iclr_results/3351_Learning_to_Be_Uncertain_Pre-training_World_Models_with_Horizon-Calibrated_Uncertainty/images/25bb9088f21e625056390bb9f8971dc765a349346608f10cc76867841bba418a.jpg)

![images](../iclr_results/3351_Learning_to_Be_Uncertain_Pre-training_World_Models_with_Horizon-Calibrated_Uncertainty/images/36adf9074f87decbb573b220603312892b2d9248c029b6fa93c2e4a25153c571.jpg)

![images](../iclr_results/3351_Learning_to_Be_Uncertain_Pre-training_World_Models_with_Horizon-Calibrated_Uncertainty/images/835ed997cfc7529befbfca388c0514e12b49436759255a718403012f58467ebd.jpg)

![images](../iclr_results/3351_Learning_to_Be_Uncertain_Pre-training_World_Models_with_Horizon-Calibrated_Uncertainty/images/9c35945aec522e61995f521f6ae10a76b17fd67959a9c2daebc68b6fa1b92ddd.jpg)

![images](../iclr_results/3351_Learning_to_Be_Uncertain_Pre-training_World_Models_with_Horizon-Calibrated_Uncertainty/images/b75ab72fdb82928aed2915240b8662a9b40eb880866d3857dccd3de10f8e6ddd.jpg)

![images](../iclr_results/3351_Learning_to_Be_Uncertain_Pre-training_World_Models_with_Horizon-Calibrated_Uncertainty/images/bb390fddc6e317cc8c527169142c5293911f4d8c3713e819947681ba50cf1a4b.jpg)

### Tables

![tables](../iclr_results/3351_Learning_to_Be_Uncertain_Pre-training_World_Models_with_Horizon-Calibrated_Uncertainty/tables/8bfffdfd5c030257a85360de993f687a5b85b70bf881d59402c4ce72012e97f5.jpg)

![tables](../iclr_results/3351_Learning_to_Be_Uncertain_Pre-training_World_Models_with_Horizon-Calibrated_Uncertainty/tables/9143c616078074253abecdec7aec89719b3c2978952a2eaa4987b63377b9cd31.jpg)

![tables](../iclr_results/3351_Learning_to_Be_Uncertain_Pre-training_World_Models_with_Horizon-Calibrated_Uncertainty/tables/933a5a01d434e62a0546d0c1428e760f20f419b1b153bec5686e979a2f7015f1.jpg)

![tables](../iclr_results/3351_Learning_to_Be_Uncertain_Pre-training_World_Models_with_Horizon-Calibrated_Uncertainty/tables/9b5a6b94ea662f1f273fed79659d60b250a3a32794414e4da9b9f1e24c30431a.jpg)

![tables](../iclr_results/3351_Learning_to_Be_Uncertain_Pre-training_World_Models_with_Horizon-Calibrated_Uncertainty/tables/d333536e0d6d588c9a12c61f7b20f38dc22d015d8d53b05cecc14094d5a85497.jpg)

![tables](../iclr_results/3351_Learning_to_Be_Uncertain_Pre-training_World_Models_with_Horizon-Calibrated_Uncertainty/tables/d65485afb55fa5092501cfb38bedad18b94476339bb2be3cc082c12a1d1c63a9.jpg)
