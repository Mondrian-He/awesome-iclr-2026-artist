# ICLR 2026 Main Papers — Part 13 of 199

27 papers.

- [MoGA: Mixture-of-Groups Attention for End-to-End Long Video Generation](#paper-325)
- [Purifying Generative LLMs from Backdoors without Prior Knowledge or Clean Reference](#paper-326)
- [Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks](#paper-327)
- [KaLM-Embedding-V2: Superior Training Techniques and Data Inspire A Versatile Embedding Model](#paper-328)
- [REAL: Reading Out Transformer Activations for Precise Localization in Language Model Steering](#paper-329)
- [RegionE: Adaptive Region-Aware Generation for Efficient Image Editing](#paper-330)
- [Safety Instincts: LLMs Learn to Trust Their Internal Compass for Self-Defense](#paper-331)
- [Cache-to-Cache: Direct Semantic Communication Between Large Language Models](#paper-332)
- [SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty in TinyML](#paper-333)
- [FastGHA: Generalized Few-Shot 3D Gaussian Head Avatars with Real-Time Animation](#paper-334)
- [FASTer: Toward Powerful and Efficient Autoregressive Vision–Language–Action Models with Learnable Action Tokenizer and Block-wise Decoding](#paper-335)
- [How Stable is the Next Token? A Geometric View of LLM Prediction Stability](#paper-336)
- [FeatureBench: Benchmarking Agentic Coding for Complex Feature Development](#paper-337)
- [How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use](#paper-338)
- [Less Is More: Clustered Cross-Covariance Control for Offline RL](#paper-339)
- [Triangle Multiplication is All You Need for Biomolecular Structure Representations](#paper-340)
- [Biologically Plausible Learning via Bidirectional Spike-Based Distillation](#paper-341)
- [Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification](#paper-342)
- [ODE-GS: Latent ODEs for Dynamic Scene Extrapolation with 3D Gaussian Splatting](#paper-343)
- [BézierFlow: Learning Bézier Stochastic Interpolant Schedulers for Few-Step Generation](#paper-344)
- [PairFlow: Closed-Form Source-Target Coupling for Few-Step Generation in Discrete Flow Models](#paper-345)
- [On the Design of One-step Diffusion via Shortcutting Flow Paths](#paper-346)
- [Chunking the Critic: A Transformer-based Soft Actor-Critic with N-Step Returns](#paper-347)
- [AudioX: A Unified Framework for Anything-to-Audio Generation](#paper-348)
- [Speech-to-LaTeX: New Models and Datasets for Converting Spoken Equations and Sentences](#paper-349)
- [IceCache: Memory-Efficient KV-cache Management for Long-Sequence LLMs](#paper-350)
- [In-Context Algorithm Emulation in Fixed-Weight Transformers](#paper-351)

---

<a id="paper-325"></a>
## MoGA: Mixture-of-Groups Attention for End-to-End Long Video Generation

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=0hy9kJ1ULB)

### Images

![images](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/images/1dbe9c1d73f173d9b626b565711b0d0134243d9a0b67deea8cae5b90e2b8f019.jpg)

![images](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/images/233eb53c50cb820ad9530d35fb166f407415be0fd1e547d191ae0ebacf1aadc4.jpg)

![images](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/images/241bbcea35ce17771a4c96b90910e2d45ea18ac7182f15c3e1eb5d27ba30236f.jpg)

![images](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/images/557441f6348d5fe632de44bdddbc7495e14f0d9ecdc106551715811a12953af3.jpg)

![images](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/images/5bcf518f6350e4d16cd80a9fb6af3df458b40fabe5a5945c09f99b00e1aa91fd.jpg)

![images](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/images/5be02cb94e8481a5187ba6e63e6dfcf16f89a10ea0b574e760780a754bdb7db4.jpg)

![images](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/images/8c2843ccb45c2edde2f099020b3c7554fcb3465eff006adaa4903fe3d9b84565.jpg)

![images](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/images/982ff742643b7a7d0043822bfdb6a482ba4aebe7a2da1755e41f51dee0ad6317.jpg)

![images](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/images/a1b5a9adfa2a81b4f42ee3d804065a209a9c40e5ad6e463607748301a7c72d39.jpg)

![images](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/images/ac94ed4beaa95b484d9ac9b4977c0cddaa124e605631c94b3cffd5a06a2d3992.jpg)

![images](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/images/bd5edee52135eb2143451daf686a366d22eaee0aec0a57d60f18b73eb1c30b3e.jpg)

![images](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/images/bf61a40c50dc07931467f21da74df8124dc7cd9a3c70e1e9b2b81c1beded076a.jpg)

![images](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/images/e03138ae99bd3acb92bfb286f6fdb101b8b7a0f4b66db2c198b6f03079249ae9.jpg)

![images](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/images/f14b6e59c9cc3b9a92f55d59da664a4642913107931464919465287975e35107.jpg)

![images](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/images/f7a2ffc9b37a16f50b990177d977a33a83283803a28ba8cb83e2f9ddbccc45d7.jpg)

### Tables

![tables](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/tables/00f30cb9800906a7344ef79589c60f65d8c69239e165386ee0bdc8513a71ba43.jpg)

![tables](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/tables/0f6b7ac99907df106ce24fb276e3a8adbd748931e9bf9a8e005ccb45deca1220.jpg)

![tables](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/tables/180d6032781941f490729f85224b7436b54cf2cb0bfa88d7da1fd041c5cb3358.jpg)

![tables](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/tables/361ec404f2348c8ddb4a3c227f297b1435824f49a18b70c483d7bb75c4a84cd8.jpg)

![tables](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/tables/62bd9f4359640312031f9c94545db5eb1a8109ef0ec751ef71f47e60052f2e44.jpg)

![tables](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/tables/640e48b3fed86aa5c1ee09e7826d75345b9d05b4c26f2bea11be49384f2f3043.jpg)

![tables](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/tables/aad3eef5a93c64b0173c939c87a0fcad4f592eb992945453762c4b29110a2860.jpg)

![tables](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/tables/b9b8153f6afdac6ea321f744605f549ddc309df998922dd35db3a52efceebcde.jpg)

![tables](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/tables/cb868ba8c34e8a8252e76ed5eafd862d0f0b80666826fd528d38292c3b1024ad.jpg)

![tables](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/tables/df87eff9c1b96db9476040202558e6d67ed24c2f1076284e02b374631d177f62.jpg)

![tables](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/tables/e5fabb79a7a6e0616d8753ced85e2134eba58edf2bd215ff963e02d066c28e37.jpg)

![tables](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/tables/f7bcfb40c485345696b154e867fef11a1a6107980852006f648d052d9276263b.jpg)

![tables](../iclr_results/325_MoGA_Mixture-of-Groups_Attention_for_End-to-End_Long_Video_Generation/tables/fbcc02e5f8ec67cb7371e9616be0572b0d32638065ea18c57f74d20872e4c612.jpg)

<a id="paper-326"></a>
## Purifying Generative LLMs from Backdoors without Prior Knowledge or Clean Reference

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=M7eWB695jp)

### Images

![images](../iclr_results/326_Purifying_Generative_LLMs_from_Backdoors_without_Prior_Knowledge_or_Clean_Reference/images/6c64e78a12e5349cfaa89af2fa0e22f7d1de3de31bbc47525cd8a2a1c23e0fb1.jpg)

![images](../iclr_results/326_Purifying_Generative_LLMs_from_Backdoors_without_Prior_Knowledge_or_Clean_Reference/images/897c22e966c88ad945b0c245456118f70552e215f91c2a33770018121b920f37.jpg)

### Tables

![tables](../iclr_results/326_Purifying_Generative_LLMs_from_Backdoors_without_Prior_Knowledge_or_Clean_Reference/tables/0da2f0545c55472f8a50c820b76c66ac26478af7a59596cf52c7bb0e6155ba75.jpg)

![tables](../iclr_results/326_Purifying_Generative_LLMs_from_Backdoors_without_Prior_Knowledge_or_Clean_Reference/tables/1fb7f8fa313d07b425778cf9a9a783a0af8c8c2fcfa0c06df524cb3363059713.jpg)

![tables](../iclr_results/326_Purifying_Generative_LLMs_from_Backdoors_without_Prior_Knowledge_or_Clean_Reference/tables/222c316a8a60284d37ed8dd4455b19db62c07e0712ca89c6fc114de238d0c9d7.jpg)

![tables](../iclr_results/326_Purifying_Generative_LLMs_from_Backdoors_without_Prior_Knowledge_or_Clean_Reference/tables/246a03bb939e1bd04f02ce537c6b73e7ba329eddc68778aeea8576dee673f48f.jpg)

![tables](../iclr_results/326_Purifying_Generative_LLMs_from_Backdoors_without_Prior_Knowledge_or_Clean_Reference/tables/2627819b24b52aff49bc0db5b80354cd7380188238a4967867f9b35f426dfdb0.jpg)

![tables](../iclr_results/326_Purifying_Generative_LLMs_from_Backdoors_without_Prior_Knowledge_or_Clean_Reference/tables/3aa7cb2132d211364472f897e33d1d5565122a60264336e8b48ff02ee4acbc06.jpg)

![tables](../iclr_results/326_Purifying_Generative_LLMs_from_Backdoors_without_Prior_Knowledge_or_Clean_Reference/tables/4a8a0250cd3b190836ce6c4aafd2cff54fe67e67f91299e8f69d5e1fa027a1f3.jpg)

![tables](../iclr_results/326_Purifying_Generative_LLMs_from_Backdoors_without_Prior_Knowledge_or_Clean_Reference/tables/7d110035e1427cc50f25c78103edf2fb9007a0e4f5529db01420987d92b3256f.jpg)

![tables](../iclr_results/326_Purifying_Generative_LLMs_from_Backdoors_without_Prior_Knowledge_or_Clean_Reference/tables/ac52fdf1d69aacf7430d9add479d476ea945d608ea67a16e22cb9ee728cd5e3f.jpg)

![tables](../iclr_results/326_Purifying_Generative_LLMs_from_Backdoors_without_Prior_Knowledge_or_Clean_Reference/tables/b810c660cee19dde2d40332723beb9e1ced3b6c0561c3b584d3f8d6621aea0c8.jpg)

![tables](../iclr_results/326_Purifying_Generative_LLMs_from_Backdoors_without_Prior_Knowledge_or_Clean_Reference/tables/ee7a4eb3c85847d7c96a263c76f82d86baff4a25f48b91c9d2e09ee2c7ea6881.jpg)

<a id="paper-327"></a>
## Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=b107VY19Id)

### Images

![images](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/images/0785f2613e099679b97663607104eafe8c5d71390d4c19e2e6f628283b7af894.jpg)

![images](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/images/44a276f7136e6502d7f1131244b0b7ec617c2548f1b7655dde62c78bd5f9a196.jpg)

![images](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/images/4737fd981c9f18164980acd93dbf2fc17ac6bb8e9ffb23ebf447c710648fe439.jpg)

![images](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/images/575b80f9c1d621aca21345f5f5a2b4ddf78a17b14c7806b3e26a60c4b59cdb76.jpg)

![images](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/images/8844ce0e26e25ca4ee9572fb93f669a679ae8d0a00db243ef7fd0e8230e2f076.jpg)

![images](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/images/ddc65f406686cc995be82ae63bc091093c5c2c21f979225002fbaed7b7931143.jpg)

### Tables

![tables](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/tables/1446efbf3f4be6bd886582afa60eb36cd633b53641903ce68de29fe3ff090cce.jpg)

![tables](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/tables/22a9f76fa0e7319b4b5370cffec3d31a9a42071b76fb09c20707b4040e0b2976.jpg)

![tables](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/tables/2449a790573324a76e60ec71b7ef7e0a03364a92ef11dd13870063638b7f049e.jpg)

![tables](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/tables/4d7911a591e466a03bd79cb72d259f662ddc67f72cd579e6871a5b7347149941.jpg)

![tables](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/tables/555c3ab589e498bdb36970e2b1b400d46d72adec5d756dfb6241565f83daa85b.jpg)

![tables](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/tables/592bd334d19586edfa7f37dfec063b81ed1b8efe9ec82e7454df8634fb5682de.jpg)

![tables](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/tables/5b4bd43785e615772a0170a40bda69461e82ad1213db90b470ddd2dba6f12306.jpg)

![tables](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/tables/70c3e6282dd49d1830117dddd8e37758233500048d7a6191694c92b604b6c734.jpg)

![tables](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/tables/73bc8959ac45de47aa6433f71c4bb71f4dd5c0876b435082099da17b849f3641.jpg)

![tables](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/tables/8dccb5d735b796ae9d9a012f29d305e82bee9ac42b67a5d488f022fb5c48c3bc.jpg)

![tables](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/tables/8ec6b791cbcff785c5234b8fd005087d47a05bb7aef5571472580ac4591b3fea.jpg)

![tables](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/tables/b55be8522f6e3cc84af2991a550f47b005bdb99b141cfcca34335d363c7d4ec4.jpg)

![tables](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/tables/b74d0c2cdd0f542c8e0cd53813c10c770b8379e64a3ee5e7fcae68cbc3cd74ae.jpg)

![tables](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/tables/d222090b7d6f38e2ed26778cf060dea5111353f0f1b6ee08e5cb7c8ae86bbb4f.jpg)

![tables](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/tables/db3f892ec5c0d044ec4b14a2af459d98ba9687e9a9303a8af67f25352d6f862d.jpg)

![tables](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/tables/e68882e8aeb0bf82b4912ed46ba7dde253161bd59d22ac1bc359a18c115fbb88.jpg)

![tables](../iclr_results/327_Time_Is_All_It_Takes_Spike-Retiming_Attacks_on_Event-Driven_Spiking_Neural_Networks/tables/efe0c8ac9ac35d471528598a8a128caf83208f39096d36b51894dc7f19880f6e.jpg)

<a id="paper-328"></a>
## KaLM-Embedding-V2: Superior Training Techniques and Data Inspire A Versatile Embedding Model

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=Y7qzhvWhcz)

### Images

![images](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/images/1143ce9005f62b26b08e9472dfdacb70a5e56bff58a5fd2ca5df9383e84cc24e.jpg)

![images](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/images/2992c658c0b21bffa15d2e85943fa954cb2e7ce903e28e0eef4d9796845769d7.jpg)

![images](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/images/2da656710c6061dbe7481bdb3a60ac8fb1ec24420bca96fadc290b602f091a54.jpg)

![images](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/images/3fd2aa4859278069f268212e07701706fb51bf77ba8ef7453d7aaae7217bfc8e.jpg)

![images](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/images/6f4285989c535f014a97d279be54dadd0b8003df2ea5ef13f1dd1343a283090f.jpg)

![images](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/images/868d11bf965e63091bde4cf7d8e13c2b43dc8998bf46f46c7f8f9d5f0e2bd114.jpg)

![images](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/images/aa3cd34641f2be5d24b4ae722175945022fb830a1049d42340df19ee62c0397f.jpg)

![images](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/images/b78e2d656250376c6ebc6191b9edbb59a7c551c658e315e920f826f7111539e9.jpg)

### Tables

![tables](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/tables/02a5fce96b9e11b689500ed8b11887980027ceef888f234e1bb38f4dd4274cbd.jpg)

![tables](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/tables/1649f83a8b122742fa2aa7426028b0449558c7f39a106b6dd16158b1c8b3a113.jpg)

![tables](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/tables/18a1336637ab75b63191c84c914b79de33e92df5cef2ff0592093821978cb6c7.jpg)

![tables](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/tables/37eab7fabb6d4d501c1dd049e93b416416bc87f54430aa71ab6a75e28fcd8ac8.jpg)

![tables](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/tables/44a29fe480bb6d07cf253e90f68b37ec89950153940f6f581f31b9b1b9a8bb7c.jpg)

![tables](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/tables/5726f2a7a8051fda43308df25b82df06fd840318593a5fc458a3d0f6ec55c621.jpg)

![tables](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/tables/58f3d4b84fa6d7185418289ae559d3b2f0e6c313addfe281a8558950c3897894.jpg)

![tables](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/tables/5fd2dc8713baf876581ca7dd8c69e257f7e2daa5952531827037eb4a773779de.jpg)

![tables](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/tables/62a8b2e7f6cc18589d3061e5cf9f767ba34b27a3f2dceb519de1c508f357e179.jpg)

![tables](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/tables/67c613e74a2f81c4d00dc7851b4a83917281d114093bcd85838ceb1a9930a1f4.jpg)

![tables](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/tables/67dc3fb0657d968116502a8b1613cee458c7d3ca314da83ff565a59927aa7ed6.jpg)

![tables](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/tables/70dc853e4c696c48975524074ae94a19d1a5b8a50eb86adc6fd6ded44979c23f.jpg)

![tables](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/tables/a075c49cf08ff5a3ebec66d54036c376a90a986779bc5b6964c62e2b4c70c401.jpg)

![tables](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/tables/a26109be973a980cbf8202e009a5e8af4b52f709e84caa21df462703fd43a886.jpg)

![tables](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/tables/b0998eb4a748a5a6e696f5bbf779d69820f52a16a7364a7720e3d96c7a7e10fc.jpg)

![tables](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/tables/b4740fb41792e37ffe66778ff050d166b5e1d965931e431df5396a7a3783bfe4.jpg)

![tables](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/tables/eba1cdf15b04d1ea6fb9900f858503e150dde1d84696ec9b1a6854d7194cb40b.jpg)

![tables](../iclr_results/328_KaLM-Embedding-V2_Superior_Training_Techniques_and_Data_Inspire_A_Versatile_Embedding_Model/tables/fe3c9470b87bc28015210b48c4111677db1eac15b9b156a7576db21616b0161d.jpg)

<a id="paper-329"></a>
## REAL: Reading Out Transformer Activations for Precise Localization in Language Model Steering

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=P38RYdkFLI)

### Images

![images](../iclr_results/329_REAL_Reading_Out_Transformer_Activations_for_Precise_Localization_in_Language_Model_Steering/images/26fbea0c5e4ccf374ffbee242d35a2b3ba5688edd90f37c681c7c8cf65960a21.jpg)

![images](../iclr_results/329_REAL_Reading_Out_Transformer_Activations_for_Precise_Localization_in_Language_Model_Steering/images/5141e0f07021a58158585ff8d77a7a94dc1b2875c9408c10c89dd41b2d071dc3.jpg)

![images](../iclr_results/329_REAL_Reading_Out_Transformer_Activations_for_Precise_Localization_in_Language_Model_Steering/images/5433db0a9e612231226e28a268b80ba8624d4a257cc6a011208a3fe008d51e96.jpg)

![images](../iclr_results/329_REAL_Reading_Out_Transformer_Activations_for_Precise_Localization_in_Language_Model_Steering/images/96ff9acc4b48d19030b88bfe2b2e9017b331a2962d0f13d63fee3a78860f2c6e.jpg)

![images](../iclr_results/329_REAL_Reading_Out_Transformer_Activations_for_Precise_Localization_in_Language_Model_Steering/images/d8a6669a58d875502a7ac943c51684bb3798925a36493e2477036b957678680a.jpg)

### Tables

![tables](../iclr_results/329_REAL_Reading_Out_Transformer_Activations_for_Precise_Localization_in_Language_Model_Steering/tables/03e97706743256cdb44b643eee9441076245c3182e317917ad357f810484eadf.jpg)

![tables](../iclr_results/329_REAL_Reading_Out_Transformer_Activations_for_Precise_Localization_in_Language_Model_Steering/tables/16496e633d11c5f9eca153e62341e2c1e2499c3971d8002b5238edc83900032c.jpg)

![tables](../iclr_results/329_REAL_Reading_Out_Transformer_Activations_for_Precise_Localization_in_Language_Model_Steering/tables/1eeb1618d4412a9ddebee6cbd5e9f0bbddf60e98fda7e09e8490c80cf864dd44.jpg)

![tables](../iclr_results/329_REAL_Reading_Out_Transformer_Activations_for_Precise_Localization_in_Language_Model_Steering/tables/259930d89f9ee2718f21d1170dfd39e31212de6b38ae858a812a46233f7f0b78.jpg)

![tables](../iclr_results/329_REAL_Reading_Out_Transformer_Activations_for_Precise_Localization_in_Language_Model_Steering/tables/2738aa5f9885f1f89e932b8633677959741e337bb31df5342b7df9d3443629fa.jpg)

![tables](../iclr_results/329_REAL_Reading_Out_Transformer_Activations_for_Precise_Localization_in_Language_Model_Steering/tables/90f0ee0947c2b666c2337d39ef212deb618d127e9c9e9ce45affd69b7f0a18ac.jpg)

![tables](../iclr_results/329_REAL_Reading_Out_Transformer_Activations_for_Precise_Localization_in_Language_Model_Steering/tables/93883d589b97b251efd8473b05eff3143b145853f03fc05f7753751d1ba64ee1.jpg)

![tables](../iclr_results/329_REAL_Reading_Out_Transformer_Activations_for_Precise_Localization_in_Language_Model_Steering/tables/a4df7abf4a12b5c46e0bf66130b6211e9db30376c1df5a88ee2169c4ec0f86d6.jpg)

![tables](../iclr_results/329_REAL_Reading_Out_Transformer_Activations_for_Precise_Localization_in_Language_Model_Steering/tables/d19c5e4e9ff3c5620260a3f16dd623a0a8d488e0aa8e6d662faac58acaaaab50.jpg)

![tables](../iclr_results/329_REAL_Reading_Out_Transformer_Activations_for_Precise_Localization_in_Language_Model_Steering/tables/d30c0e760f83bbaf77e5fec39e966e28d9bec4167e4aca52a634e46ebcef7881.jpg)

![tables](../iclr_results/329_REAL_Reading_Out_Transformer_Activations_for_Precise_Localization_in_Language_Model_Steering/tables/e14e8ad943c615349825f4e9d7041242adc1ebcc14c82db9c6b0256d9e4c2f88.jpg)

![tables](../iclr_results/329_REAL_Reading_Out_Transformer_Activations_for_Precise_Localization_in_Language_Model_Steering/tables/e694bf85265be9b3a0c5c0c6dc83dd72e7098b74252a1b9166dcdf44e0b79f76.jpg)

![tables](../iclr_results/329_REAL_Reading_Out_Transformer_Activations_for_Precise_Localization_in_Language_Model_Steering/tables/f7ab390063798c8c9c66ff125d675a8d1972b174e4be912afe761719d28a39d8.jpg)

<a id="paper-330"></a>
## RegionE: Adaptive Region-Aware Generation for Efficient Image Editing

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=I6j5fLdH80)

### Images

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/0638a2ea6834c1fa337e19e2eac6c30659f6de4b0f13e84d44295adb9997229c.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/0b8cef8dfac277ef6161139173b2967cd35e67dac55affa22a7093122071448f.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/113a0d6deeb8bbc9c36831b96c2926f6b7e826806eba2013ee75d1e913ede58e.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/11ab70ea47fc9fcef61010b9bcc27be7bad485f0747f4c5cc77e0e42cafbc92d.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/21155a689745f432015b069e5e5e03c956526bf39e1b4528c9999c34c21d9c65.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/4cab8f92f3c55eae9f81c82f7efa819f258474c9f74addd4e471268d6ab66f66.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/5d5f268b9aa5d5cb8a5268a2466f04371436622a20d91d140ad8d7760a194566.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/6c89efe82735d4356f5d98bb60716e16eeecc010ef46f645950a31244c743d0d.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/70db17aa87225d94ce384946167a4fc6f80379029e2f94928acbe24e25c6ad10.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/7b4695656268de5a733410b2ee7df62a101b32a9af3396f8401142bb9f1190d3.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/7c4125099bfe308caba7852562e6fee061ce1c01ea8f8157872be93d49e56400.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/7dbdb1cf9b0b462823930ef8e4568c7d7db3a45b1ff208d975b1e3bffefdf4ae.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/90dfe271740101c4cc7953baab3936af58c45ec2a29bd2850ec53a4451d75f8a.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/96750e1fa9abf7ed22b022d7ecc0bd814151f1688427861d4d08bad7648965fd.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/ce0a1735bb8b4b8cd291cb775bb29e34fcede64cb005271a195ce5759b0cabc2.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/d116841dc0cb9eb905ec122313242263ae7736a755fc396d94d99015130c67c7.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/d2cd1b05cb69e516d98fff1d1c5fe1098f958b5cb8244d5e46d248fbfe254436.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/dc2f2224c4ccb8452fe7abff086c700c6e43439d93e74387e21edcb31e1d5add.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/dc9ca8651b6fa3dbb4d355f41b8213e9d269e18f427ccaacd98097f9685e6fd7.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/e1f5dd12e70cc60878758894f7ac169e56b140cd4522edf4db9e7d696a3f9bd0.jpg)

![images](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/images/fd1b00a28475a397cbc7ffabb64701c953fd6dfef07d83949ffb979f110b3e0c.jpg)

### Tables

![tables](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/tables/08fc0bb74ff51209b4afd642791495cb3ae380f32a26e02fd870ce013d651261.jpg)

![tables](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/tables/0d8187857c1f133f4faa34ef32ea1b7b55e1af477f4648fffceec3220083e7b1.jpg)

![tables](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/tables/33a93fdbdfd19defd563a5197f9185b91fcd7686682ce116cd53c8087e4d1b4f.jpg)

![tables](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/tables/387a3df13c25ef4fcac30adf10a1d43c0e9c5b6fa0c283612796184f16f67ff1.jpg)

![tables](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/tables/5b3fde0da5fecd05e4c078f8b39988d7af078a987ea582b71c7947624c41a0ea.jpg)

![tables](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/tables/88b9ae36b329c9a93885ba6f649e068627c7885a2e71fe7e724abb02134fc0e7.jpg)

![tables](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/tables/8e40cea96a7730f0cbf31f43b24c121a944625a2b391b07fe1c595a3b430d041.jpg)

![tables](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/tables/8e9feb483740de744b8d733d44fa0b0a12090050d14ab25965251e377a21ca4e.jpg)

![tables](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/tables/93fafab95c4746f211484b84de66af5885dfdc043403ec2a76855b603bf16c3e.jpg)

![tables](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/tables/ad0bc750b80b8e0eee2cfe3bd516415e5b0950743015cd7c7aa41c55bdba2c8d.jpg)

![tables](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/tables/aeb4bbf361afd1b7c7c127646b9c65c2c31bb95bf98f194947f50792040d7ed5.jpg)

![tables](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/tables/b91c94bbda509cd78df6d7d9a13dd5df83907b411c22b67492f33a16250381d3.jpg)

![tables](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/tables/bc90b4c8637e1f1a2cfd490b21f5aa58f855a431557885ecdac194487ec208dc.jpg)

![tables](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/tables/df753c06aab0e16f3ba334cf60f5d38a11b54610bc51437f4297527090b0b12e.jpg)

![tables](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/tables/e155f7e232e47310301620375802d34a1496248927b669e85609af1e71353723.jpg)

![tables](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/tables/e5a75cb84f2044543ba8d799a5ed295888b7469fe4fe3d3db398837d9f3d6743.jpg)

![tables](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/tables/f2428d35eb79768d6a9df30c02d5c7c27f00f092fe62d4dee598d5c58ba8f182.jpg)

![tables](../iclr_results/330_RegionE_Adaptive_Region-Aware_Generation_for_Efficient_Image_Editing/tables/fb7dae989b3d95e642b8df0975fa49dbd990ae553f647d6365937e4363c1ccf1.jpg)

<a id="paper-331"></a>
## Safety Instincts: LLMs Learn to Trust Their Internal Compass for Self-Defense

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=LUiqtv6vrd)

### Images

![images](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/images/0d85fcc1dccbb9549fab09c7bf029342856d286f38f84e7ad4f4e6a6e0d38339.jpg)

![images](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/images/1ca341d5537bc83684fe5721c2a235043a215537a865f35387c2f4fff36769c5.jpg)

![images](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/images/1f68f328de57c41571880c73d7ba0d902b9c5fd3adbae7f358215ca2318c96aa.jpg)

![images](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/images/20f65574e84531b964427b355620cab2cb236124f035b5b45ddebf8fa705b81d.jpg)

![images](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/images/2750f6195aed7a290a62ee3a27e68cd883c2f546363ce5bbbfa17462120a5499.jpg)

![images](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/images/31d7b382050520e3968e34fc386f56203b16fcfa4c1695ea3766a34153e9ab09.jpg)

![images](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/images/4ebf019dda44378fb92fbe0fd225ba47855dde34fadc1d63972e7973675b40e0.jpg)

![images](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/images/4ec80dddcb96725fa579d674c02aba2116ea1f5e60a25cf416300d9babbbbbf1.jpg)

![images](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/images/648c07778cf800dd2cf6ad4ee698c48d24da400d3e38692a09862509fdf1acfc.jpg)

![images](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/images/87eb13b167fa55f8b3f428719276bd24800e52f86abb3cbc1657f8e0551b41c2.jpg)

![images](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/images/c0d1fe657a9d4acc005ae52d19cbfb5284be7c0e50ac065b791880f7eea69108.jpg)

![images](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/images/e2a76947f502f3fc6111a0b1cb32d05c6634ac1da46d896d46d1d3e52212b842.jpg)

![images](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/images/ee0e0ddc22152333ee72ee516d4c62601721e63fd5e94e65ce609f5dae5e5162.jpg)

### Tables

![tables](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/tables/15eceee11201dc17ce5df336f302a85c3656c533fdb699fd4aba10c135661f39.jpg)

![tables](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/tables/1e1802f89e4881980c70e0747d2e28aadd7198090d7eedab0eaa39f41422908e.jpg)

![tables](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/tables/331a0891af2c62d1cd192090a74fa9094ecfd3916912142cf642a6397579c006.jpg)

![tables](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/tables/4b096cb997eedd56718dc487ce0a3f30e425c99651f27eb0d0758a2fe693a245.jpg)

![tables](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/tables/51cf37e66d5dd70363519862da73344c98513c4422276488f8ad8c58784bcc15.jpg)

![tables](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/tables/62f7145e6e49bbaa8fb6726b18d510fbd3db842d6c89a0e00f367937bbb40e54.jpg)

![tables](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/tables/805e105581891e38849550abca5f5e567bb59cb3ee364b9d6d80573f39513589.jpg)

![tables](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/tables/b7aedb3e055a2a0dff4521b1241894ec6f8e4942f8635ff48ae777c73ef7e6b0.jpg)

![tables](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/tables/c429c7a4e54571f03594f724d49b22d1c1be6b3713db2798df7688b724ce6e97.jpg)

![tables](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/tables/e64fd2d357d99cdb2f082d597ea30253f1749a02e7e3d146b6eac1ea33b8d524.jpg)

![tables](../iclr_results/331_Safety_Instincts_LLMs_Learn_to_Trust_Their_Internal_Compass_for_Self-Defense/tables/f5db58956b702b5b5f6e763c5dbc5fa73f0f73e467fa7329f4f02cd35f965d2f.jpg)

<a id="paper-332"></a>
## Cache-to-Cache: Direct Semantic Communication Between Large Language Models

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=LeatkxrBCi)

### Images

![images](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/images/0e5025845067972458f8e33234cfad3065530d06cf5bdcaee3b7850f93314d07.jpg)

![images](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/images/2387d2cac59d24bb91aec52fa90b67efa30b288708d4c8c328e748f85360b567.jpg)

![images](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/images/255288071528b2a525857e39624edae0c66f009dc49ca23cbb3b50a92626425a.jpg)

![images](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/images/2ace5e8c174fc8fa36c7f49fd9d1b02939b81c04d1ed6a452bde453908d3e212.jpg)

![images](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/images/2c2cb8fc596360cd23e7fafed6ec91d43f8943cd419719a9709387cb66b9445d.jpg)

![images](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/images/3eedc758aae072a371b4cbddb3f02a09e1f2c6f63ed15ef8a20a68d2c6a8ad7f.jpg)

![images](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/images/4c90758d30eb3b725b3d084130470407c2c5cc492ceeefa3e7e17a4871d3bf02.jpg)

![images](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/images/583b77b3243f15b3ffa8c83d27a97ffa23a7d97755bb11c5b04c567ccaf71172.jpg)

![images](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/images/69fd7bc61af764a4ee2903912408dd2afbbc25db3bdc568e47a8a018051a7d54.jpg)

![images](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/images/8723e73d90f86a3033778e3c2982cb3a632efda9283ea419d6e4f538fec43126.jpg)

![images](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/images/87609550fa8be33754ecb3c5ca2a74c940e2c4acfafeb43487d437e27485ce6e.jpg)

![images](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/images/af7c147fc9622af78c2c70cc67607b2af8baf183c3ce10f00b662b94a9b8d979.jpg)

![images](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/images/d271621482554ba0139b3a86d5c527447b5b957f7b2e0c5c661e30288ef0fdca.jpg)

![images](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/images/e15bde17a0d8dc75b1a2a4e01ba26553d597feb4ae2c03224ce8ecdfb692726f.jpg)

![images](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/images/faa5164521f3eb407a92b1be26ce136aea4593cca4db2e8e5d371a0a57ba08cf.jpg)

### Tables

![tables](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/tables/111d6f6a87d03881dc6ffadd3747783429fb0c369e25491f0543766f04fddd6a.jpg)

![tables](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/tables/3d5eac140c53e7f94fe52343ee545160e4a9dca76c65adb30a34c7ab1aed61e0.jpg)

![tables](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/tables/43a653b845584a85a7bb4af0c0f7e52ab8f2f38bc6bfd864e6d9e0e7c94925ed.jpg)

![tables](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/tables/560feec32fc84dfbec5d9e088da31608d70525cddc075cf394afbb9be36ea67e.jpg)

![tables](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/tables/589025764152a29bfdf45960b88f5ecbbed584dc3bba11aaef8575137f582bc4.jpg)

![tables](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/tables/7ab54bd5567f7a6e5a2dde58002ec93a3e7ecd4654614a3f5ca3cbf183a2668d.jpg)

![tables](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/tables/81c8990d042f3cb68003969ad451dc1085e55988d7fc444b630e1857254c87c4.jpg)

![tables](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/tables/857b505215557ed3577f5f1aaa13863a407aa5bc4ea5e7edfc2e9d4e3bb57df0.jpg)

![tables](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/tables/8dc3e04b92efb0031f7d8c4f1b4a99ab98356476cf6e9118d17ae92c2765412c.jpg)

![tables](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/tables/90649d2fc31d31bb1a6a5fcaa3aa243c18a650149ca08964ff1ee3abbf550e3d.jpg)

![tables](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/tables/c883f80307f6966284f785185915e8568c46c550275e109336f68ba16e500262.jpg)

![tables](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/tables/ef56b7d7b9f3323013017e7b0b52f645a5a09e266204f12c68b84195047f2e14.jpg)

![tables](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/tables/f3b9fe78df22f667eef73e496305de1fb45649267fb199ab4e2abf7f64d267ad.jpg)

![tables](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/tables/f6e651958e120821c2726a5ee8c37c4ff3cff291f7a779443f6cfe901923ae39.jpg)

![tables](../iclr_results/332_Cache-to-Cache_Direct_Semantic_Communication_Between_Large_Language_Models/tables/f8cd01bb7c613a294f3d624a245fe009304da64a1250e96dfa672e0107a4b5d8.jpg)

<a id="paper-333"></a>
## SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty in TinyML

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=YdK1ZRhrKi)

### Images

![images](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/images/140dd0035d85ee889c7ffc1834597eccf4936818302e74d83a1196ed86a5380f.jpg)

![images](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/images/375d72d194cca7a6f9cb1807e914dfb91939db8fb1151bbf0d52e0457b61027e.jpg)

![images](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/images/5fdb430ee1036cb46e1efbfe54b5a1c0c6a09b3dc9e65a3553ca1412b5bc89cd.jpg)

![images](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/images/720ada8df97cfb640653b5f186eedff3e7a18f1a127dd798fee6739679e858cc.jpg)

![images](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/images/84d3bfa69e743ad0ecae94b2ef8d11020bd8a483cd336c52104f785eaab3e1e9.jpg)

![images](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/images/9da05769042344aae3b4183cd2c292f537a95a197563dc999c1627088b29ea9e.jpg)

![images](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/images/a5302f89c812283549d5d69c2ce7d3d022402dad7606d07b304ed5cf643105a1.jpg)

![images](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/images/a70e25104ffe06ef0a592b9098387f96f62854fac11e8e895b0cc702befb2350.jpg)

![images](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/images/c8378519728d3d5f05ba7e1c0409ddceb4d5d4f658f0e3ec1f7d4bba1ffdf255.jpg)

![images](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/images/cfbd3d7f3737f59896acea38f84cb281877089c3d77dfa4a6c53d95f86a7e9c4.jpg)

![images](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/images/dd680a7f5474880a6cb95e33796b650efab411f0afa76ad97fcab4cf94e1081e.jpg)

![images](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/images/fc581a95f1e048f787b9cbb7a323653604eb93f627571710fcb5fda5f255439d.jpg)

### Tables

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/04f7dc5af152d4157d3f81851aed116ed838564a1339092d1648fee2fd9844ce.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/0dfd1924b2b881fa3f1bf86d9bc6017f1f08c8545157cc3966d3fbcfb762682c.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/1e6eeb4806782105524f50f5eb163d8148def9dea0a9d1ddf4b9e29f1afd190b.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/26b3a2b28bad9e4e85b55e7e92fc1f4d9872cc439afe77cf47192df2f1f9402f.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/2894ece51c532cfc1cabb7f1ab155fb2820bbafe865c0262976a30074bca7c8e.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/28aa6f68666377feed2caafc548541597f4bab231710a4f5d00d3fa109bb343b.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/2b877ff379c7f4cc7dd92b78bf5f0718b07f7b19c84d8906e5621cdce750621f.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/2f9f49ac6eee7c92613fe08ef9d6ff7194bcc3f0ce17529462873dc7b6a2f988.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/38d7cb2138cf99c200d5b3eea5a5734df94f9f203f96740d5812188437868b53.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/4173c302e719525498eff907d74acc80d43ca847c1328f77af75bc00062eb6bf.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/4c523c89ea4690fc70eba8250c917690ea8092ab73db91a7844cd99bd214f4ba.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/5c35b7058502209acfc7a02f08cd45eff4afe12990a38e4465441e6f83756bf3.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/720be6d31dbcf62e39830260938e87ed18f3f4767d716d33790c0df681c4bbd8.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/89e71220b772a119b2eb3e644f3dc7a32a427e4b7137d0e3dbb958baa563f34c.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/8b7501f463d3c6cacdca148fa5d50048949357b6681c344b8fc625e34c055b09.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/915788081389848de1e70b45911250c65d9182d8072bb93f18813b51ce3b8975.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/92f2395dcc0a10a1a130654d02fd63adbe87d38f0d1452494411db8c1b5b9829.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/9f5e0b924a4bf79482c2cfaef9b7e9f93af54d9e73ea5b5f334e90a95306cd85.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/a1eda24ae1169bcbe2b3b310b22081e394a4d83601acd753869964f6368c042e.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/a2c45c6a04d4bd35644d951c143d9be755421bddc19cb8b6fa73bd9226da36e5.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/bb305949e654f769e64cf831bf1c996b238f358567d01b951c359ddf94d137b8.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/c46abaa740ea6b841746828a5b9410c624e32dd48b8d952b113e8c5a3ef6a622.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/c7b1552cb41fda174737a9b4afda03660ad2b74cdaf06da294bf38b4e027f1fd.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/cfbcec3d13e5f7cb4a003c3e8a8eda2f920881417d2d4ced173433fb0c317787.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/d0009d0a1f1bbd2d00ab15410abb4119ef3c0b65800051a4ca1b30632a7970ab.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/d0ea83c1c11c77f10544cfb865beab686584fc10ba8c1dfcb06467f3a2438aef.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/d5a7fcee4b8b236baf97f2bdc064364eb8abfecea1c08240a2441f0dfee94d29.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/d92fff8a96ba1370dea9c816d0fa6fef85482c849aae73bacb89b8fb645adec2.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/db0e7af2e66a7f5fe2d7c6954637fa33663014b1cb23f6941a7caf78eff1ff1f.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/deb8f69cc24003397ab4ba26e0e2196d87b40a8b74299752b587c191d6f055b4.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/e540e4a35e8db501b2907d486122e42f3b11e8fba5075cc2bbcc63b2ca4bfb44.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/e742d03d2a3e8ffb98e23b038f33cac485394fbd3682e3b991b6a6d3e222f5dc.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/ec43130297886a51ef531b1b27a75c511d94beeb6f994290bc9c7d8ba11bf0bf.jpg)

![tables](../iclr_results/333_SNAP-UQ_Self-supervised_Next-Activation_Prediction_for_Single-Pass_Uncertainty_in_TinyML/tables/f26d8430ebd65d0d4b7f30517d1d263bb6133098aebaa7dda979804993c1eb99.jpg)

<a id="paper-334"></a>
## FastGHA: Generalized Few-Shot 3D Gaussian Head Avatars with Real-Time Animation

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=E7VL9Zl1Nc)

### Images

![images](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/images/01af7f7ed99f9acc7286c75162414ca434fb9a5524e3b3a0459085da0d1a305f.jpg)

![images](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/images/08c8622bf99b6fedd7368da8b113d27c5c0296cf35ad3e2d429173e8ed6a0b69.jpg)

![images](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/images/0ef915cb8b7ec0a5841026ca287c436f91eff552d3e59ee77a1a3a7bd5e8e553.jpg)

![images](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/images/1a0c6da2c06388fef260ecfc19488f06696bf7ea9953b7b286e5fbcf71282969.jpg)

![images](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/images/20a7d6683f90214a6ce584541cb0ecaf63903ea0c1ea360cdc6bf667f3f1a2b2.jpg)

![images](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/images/32d49064f319efeb1b3eb61882798f8ad7c51bbf0f9d455ec930ed4c7c0a00e9.jpg)

![images](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/images/40bc04e3fa636c3cc073af0deeeba2e717029f8d401eecb336127047c998e65f.jpg)

![images](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/images/4e3ad4a574f1c61edd14afdd1e9bc0ad9bc56d296f909716e091659abb3e5b07.jpg)

![images](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/images/99557db2f505a59376ab442190b04d2931f33201b66afab39cf8d0c2fcd9ef14.jpg)

![images](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/images/9e20c33fd6fd1b9c079e12880b660ebe77662dcf73dd57a2c9589521fa7fdf1a.jpg)

![images](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/images/b46869de959a1b5519e2d0c1fcb3b698b34e487e340e0cd4679c239ba55a76dd.jpg)

![images](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/images/c43cf77a8c9661d90ef18256e3b13e46938f87cf036f6c2fec0311e76c1aeff9.jpg)

![images](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/images/c554c8d8a1ab23e94a79aeeefad4358aad04d0eb3999046ea6de9b24dec9e0ac.jpg)

![images](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/images/ffc6137a303d477830b249a2b3849ff8f113c2793293c0dcbe3de9058f78fad9.jpg)

### Tables

![tables](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/tables/207c93ffb23e100b90f087bcef930f5b06aa2e4d43112cd0b1b8f660f54a7fb2.jpg)

![tables](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/tables/33599d2c8044b6ee0f10692826758346d405e6fbcb9bcb3164a8fdd658a43707.jpg)

![tables](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/tables/5d2ab77ea874bebbd76da3e9a821da336886cb828f203a377633add548bca413.jpg)

![tables](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/tables/5f2b4318df80c4e367c8df06d896f663308f8a6fcccd1d995cf50b33df95c32c.jpg)

![tables](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/tables/8bc819a31eab9d5a0b7573c8600d525246837834eff3cdb49749b3b5bef36950.jpg)

![tables](../iclr_results/334_FastGHA_Generalized_Few-Shot_3D_Gaussian_Head_Avatars_with_Real-Time_Animation/tables/a24f9c9679db2a7799d24d625dc1e077e7af71320097ce4c36d77a3aa17629fa.jpg)

<a id="paper-335"></a>
## FASTer: Toward Powerful and Efficient Autoregressive Vision–Language–Action Models with Learnable Action Tokenizer and Block-wise Decoding

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=k6nTUFoqeT)

### Images

![images](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/images/1be5c8982e9c56ade7be9a0df7d5247da54078c0ee3b380d5c9b607a2094b1ae.jpg)

![images](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/images/24a97c8597a3079c9ad02dca2dc12b431db2ee5520a40d7cd5f20b3784a3d984.jpg)

![images](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/images/40d638ad2c1a123764bc0051705489bc06296f3508fff70d5a15e4a2635977e9.jpg)

![images](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/images/53bcc55c3dd9f6d939d4256cfeba48dc7ce1fa273242ec9abed142011c643b27.jpg)

![images](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/images/8a42428916f2f7218ab62cd959256824ca5b69bd9c375aa2865e595ef66def31.jpg)

![images](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/images/98ad555d2280ce7edaa32ddc33813ca827f41ff07e4c654dfd10aae6a04101e6.jpg)

![images](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/images/99dabeb1e9d6da91d449b9cb0b6c6b5b41f0b0eebdb300a7c2742f4c1b851937.jpg)

![images](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/images/a8a98028261ad2774d3cb13331e904f357e65f7d9ff11b7041021fba5a4a8f7b.jpg)

![images](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/images/ac96dc73447935f2a565aa3a0112116a7179524ea83bb66228b119ee2ef0f9b6.jpg)

![images](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/images/b061dcff93f636c8073803ec375df464639895c6de8456c00c49cbb4f556c348.jpg)

![images](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/images/bba9cc2631afbe309c434523931a003d10fe33a89c6fcb3bd13c52aa9d735edc.jpg)

![images](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/images/ce32eb730fd16bba9ad81bf90eb5184bec666d169558a31bafcf35c094417b7b.jpg)

![images](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/images/d4d87df5905734bdf960564305e570e3a9b9bf7b450450c699b74d614cafa6af.jpg)

![images](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/images/d8d0f88102ba652dbe921e11132b7134dfed472acb2758d92e39896d52b2238a.jpg)

![images](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/images/ea72a94b754d83cb8ae8a7db4de3cd6018069784a846ed5c8ea1a5c9fa91e39b.jpg)

### Tables

![tables](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/tables/00c4932f342307bab3dde094193cae28932ad9c52d95a4ecfbbe79612e4bbd67.jpg)

![tables](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/tables/117a0469eebe83fee2c80905eb80ace453b55791f9a881b7961aa05defdb5cb9.jpg)

![tables](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/tables/469d80aa779893442290215b7249f277312c2cd6e6ad31491bde08ec8a73b0a2.jpg)

![tables](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/tables/48d751fc924e286355dc96179345efe90773229bcc7316c7a52071498ad8e67e.jpg)

![tables](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/tables/685a631f0c2c6bacbb046909b30b0319dfa5fba07405c3bea70967ca40b574d4.jpg)

![tables](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/tables/77f20680bbd5cb4914020a62e032a42f47ce87038cac8369c33a2832ae4ba79e.jpg)

![tables](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/tables/7b8a03f09d27ef8de5133b57182ba7f4dc6fea4f701d38faa6834b73a10dabb2.jpg)

![tables](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/tables/9d1d47c8aca94b1d321d5c2609b47db4b5934c12adc702327608cb8ec4de3df4.jpg)

![tables](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/tables/adc66e59c5d2dc246955160d8426eb5122c2463f5d36c00f8f1e56de00a1ccaa.jpg)

![tables](../iclr_results/335_FASTer_Toward_Powerful_and_Efficient_Autoregressive_Vision_Language_Action_Models_with_Learnable_Act/tables/fe4d1c536c6000765ca1443c6229cc86717350d89cbc58e8011aabd96675e50b.jpg)

<a id="paper-336"></a>
## How Stable is the Next Token? A Geometric View of LLM Prediction Stability

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=Zjz8F6gdrw)

### Images

![images](../iclr_results/336_How_Stable_is_the_Next_Token_A_Geometric_View_of_LLM_Prediction_Stability/images/2adbcfe59db5830fcc50bda34f9687f7c57dcf11969fb52d3ed708b3a644d8cf.jpg)

![images](../iclr_results/336_How_Stable_is_the_Next_Token_A_Geometric_View_of_LLM_Prediction_Stability/images/2fca03e7cf54ece98f4107c8dcd74e2a6fd1d4da011ebfe087b8766dd57eec2b.jpg)

![images](../iclr_results/336_How_Stable_is_the_Next_Token_A_Geometric_View_of_LLM_Prediction_Stability/images/863b8759e6cdf5a35036d4b5c480df4bc680e839ceee5595bd8c60575a32595e.jpg)

![images](../iclr_results/336_How_Stable_is_the_Next_Token_A_Geometric_View_of_LLM_Prediction_Stability/images/9e2dccd8ffbe04bc7c2933e4beddee871666da31235f774caa0199492a1e30ab.jpg)

![images](../iclr_results/336_How_Stable_is_the_Next_Token_A_Geometric_View_of_LLM_Prediction_Stability/images/ccef41a6da2fe0709bdb502976d4b503682032eaa4c83e98ba8709d2e480196c.jpg)

### Tables

![tables](../iclr_results/336_How_Stable_is_the_Next_Token_A_Geometric_View_of_LLM_Prediction_Stability/tables/23ff0726012e9b1f4ccf9f0c48bc7676a6ea2bd5de42345e714b02ba3627dcc7.jpg)

![tables](../iclr_results/336_How_Stable_is_the_Next_Token_A_Geometric_View_of_LLM_Prediction_Stability/tables/283429fa5adf1c3712970c24c811cbb5b94a58c1252ac1de0eb83829a6234ed6.jpg)

![tables](../iclr_results/336_How_Stable_is_the_Next_Token_A_Geometric_View_of_LLM_Prediction_Stability/tables/7fc2063278797acc3ba2046255a5056cf42a3fcfda9217544606fe5d868e0500.jpg)

![tables](../iclr_results/336_How_Stable_is_the_Next_Token_A_Geometric_View_of_LLM_Prediction_Stability/tables/a210eed68b8ff18c3892f3c60b41cfa2f1006398de4f853e7694aec49c30631b.jpg)

![tables](../iclr_results/336_How_Stable_is_the_Next_Token_A_Geometric_View_of_LLM_Prediction_Stability/tables/cae6f1a42a4a99f66ba45f1d76db3c817dfdc15d2e2eda3f855468f9e88ebf05.jpg)

<a id="paper-337"></a>
## FeatureBench: Benchmarking Agentic Coding for Complex Feature Development

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=41xrZ3uGuI)

### Images

![images](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/images/27facf6b43e2f0dd6307d916bbda7d969a0a61b7420ce8dbaa0da5d014630d2e.jpg)

![images](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/images/385a96f35f5ece75f6c886580df750a0f0394d9aee05776e116d6f30ad8bebac.jpg)

![images](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/images/504f5dc2a067c5b4a8934b50ca6c03432f8a3bec0b4f08fa4373e2a215481334.jpg)

![images](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/images/5a92db27402bdbb8df0f3ccdd96dc8299a46247d1a3cd36f1166e4429944763b.jpg)

![images](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/images/5ee034de4e436a780764db004be7b912ec091278cdae81d6e8585b7fc081b7ba.jpg)

![images](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/images/66e67e22367c43eed008985492b440561f64b60cb1fcca59c40f8747e67a069a.jpg)

![images](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/images/71c1a2abbe72351a4a5c38363c5ddd7cfc50e262fdb12c581ec9e75170b7d25b.jpg)

![images](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/images/76e3c777aa29e5db3348af67cc6ec7c156141602e297f67d72dd78e6424165a3.jpg)

![images](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/images/a317ecdbdb530e04b3a7c7b7951b9c6971fe73328875747b0ee50014b14a4f95.jpg)

![images](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/images/a5a00e14189bfe82d4deff30b8d2d56b7907ed81a935b292464720d4d4756e32.jpg)

![images](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/images/be8999b296ca9b327fede6d94cb42a65e641c7489855acd4aa909517b5e402ba.jpg)

![images](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/images/bfa215d94d50ce190dd57462f320963ec59c8272fbb40b7103f39d4a13e7faa2.jpg)

![images](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/images/cd2040b6ab8fc25de013aa3f5e815218e8c42a17a57b36c0ac50f6526729595d.jpg)

![images](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/images/de710c044378da6f3839062718c10070049196d0b920560830a9b75af742ea38.jpg)

![images](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/images/ed1c90d0dfa2c1b1a4651e1d84541691fd258039053e5170cf9977d122b97fbf.jpg)

![images](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/images/fd6be299331881df59daf4f30e9307b2f2003ad2ba39bd65ae43d3218ca381b0.jpg)

### Tables

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/09771e1b779f61c3b7b6b3465ea38ce4e2b40cccfcec3ef790da20120af0029a.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/2078649d0889b48ca390393ef2a14f3e083d96578ef3d2a3fa5a0851a03d8668.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/2d87a640b14babaeebaacd6a0dd69ab06ac9d5c4acdb7de569ed3b74af0f2426.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/2f98193a2c6521201ed1b4a2aec7ec0a27235df705b2d1f8d0cb05c69984b698.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/52484fa9de8bb27fbcdd6d0f800705a2ca903a2bc5d80ed96fa2e4ba90e4cce5.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/53fe660e304b5dd337ad57cb7b2d90a04e1c6015b9bde91215883d20e0e0a276.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/5d7c3a721b0bd1e2530806a487bcf021313acc2d320fb3180c044509748e3629.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/69e4fa3f99b67b10f26cd86559022f60351fcb9a2f2763d00e2d90552bebc50c.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/70d073c65da810512ed09de10308784af116aab5c4201d83841858419a9f1fed.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/7e2995dcf88221aef19bcc22f51fb56ce99a78576a3a27b30064c11e4e67bb01.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/875619eb1b3238937094965e7d58e3545431f74ac4a3cde381c44258ca0b5229.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/9ac1156b292ca1a03a215863f859674112303fe32e975ba5ad9ecf6ec9b5267b.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/9f3ee8cb7dba6afb31d72b785d1d372a027dc85708e79e8359abffb00962dc48.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/c12b194a366535bbcf071b87199a4a1f2a9496e822bae60ff011eb39ac8f2bcd.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/c9c64a0a1570035ed0b356890d5528e2c818aa0fdb2d56932f454ca6ddd3c01b.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/cce6d075594c12296ee3b93420e57acbb647d7c099731587189a9a210a723d4c.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/d2905ead9d25a7664c9325ee5157d7dbca10864ad35c6f51d0a71ebd4b14e4e0.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/d67615b5cc3584662bd6348708bf2689c58b441f26057a3622ce539f03a7d69c.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/e5263943e8331d28ab01e7ebfb417b5f5cd197c84bb79f77e9f10cac735452ff.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/ef8a9218666c29f6ac39bedaa7a4d72e96c4a6ff1f63b0afe6188f34f50e382e.jpg)

![tables](../iclr_results/337_FeatureBench_Benchmarking_Agentic_Coding_for_Complex_Feature_Development/tables/f0102e3d56e4fce758e04df96c919bf6f2581e2813737fd1c7a11c488b4ae47a.jpg)

<a id="paper-338"></a>
## How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=vV54ShHvGi)

### Images

![images](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/images/8b1a6313a40032c018dd5c98de77634a759b63affe8600676982b5a1e938a21d.jpg)

![images](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/images/9642b1566b0b84e91a01b137db83939b0f07865d27fc768b9725f6b55df7f633.jpg)

![images](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/images/f808111b8ee4e0237c3227b777d2ab6ad52ec54aad2b88245be2df7b8325e1ee.jpg)

### Tables

![tables](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/tables/0c05f7075e4009e675916807f771f7504d1c232ec1e17138e0fc3b6c9979d06f.jpg)

![tables](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/tables/18ccc14ffc0531cd8ff211f48b67606c77e280342be12dfd0bf9af97ce4cd129.jpg)

![tables](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/tables/19dda13b92c608d8b1316f9c2da6410a66765b0ae3d5ce6e1c227a0d1e229c52.jpg)

![tables](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/tables/25a6156f5d07a7f6c378b369fcc6f80c7237b41c18fac2098160dc71d1a5d231.jpg)

![tables](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/tables/2e3fe693121c2317e086ba797aad905bb38a970705569b99797e3f983a786e5e.jpg)

![tables](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/tables/3bd554ee1d594b96143f49306c5b99428d5c71228f40e32e64de1b4d9070872f.jpg)

![tables](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/tables/443f5604d66fdb39f8fd2d153e535cb6c8c8340d3a3c2125d368f37c96d456a4.jpg)

![tables](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/tables/44607637a3bcf63891658b5d8e111dda49c5bbff8e9b82f3756f45fa52717e14.jpg)

![tables](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/tables/5425866237295f58c1da9c33dc25b01655cd3da4726de3f57ee32173f72dee2f.jpg)

![tables](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/tables/54a4d6f9485c00f4d1038738d8beaced1a049fe0d22e51f380c35c213d93d6b2.jpg)

![tables](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/tables/59e6928d70668f81f8249e67ff17e7ad1d6ccfe19f6ed973addff512ffa029e3.jpg)

![tables](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/tables/729dab381e69183a0e3b00fcf618926a4af96ddd4b36a779a642da9641ac4626.jpg)

![tables](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/tables/7a95d21869d51e114dd1f7885721c7fc370e64d0ea1a8be7130e205e4ced5e53.jpg)

![tables](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/tables/870269a5ca8a8c99a172e89f8e5df6451a042959737d1ebad03cc1be631da8ec.jpg)

![tables](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/tables/b158a6df8e592a377b375304065a169c024a27ea93ad4f08170e42c442508115.jpg)

![tables](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/tables/bc4390af2f896e2836c61a850ad6a73a8ec7e759309259ca28d4a81a473b868d.jpg)

![tables](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/tables/deef1271fecf06d18a4ae59fd2d4fe81af99d982ed686a3103db544e9f13bc03.jpg)

![tables](../iclr_results/338_How_Far_Are_LLMs_from_Professional_Poker_Players_Revisiting_Game-Theoretic_Reasoning_with_Agentic_To/tables/ea72bd1c7693ea74afcef6ef7dacfcb5999fa4a37f1ea1e82112b7b58a4a61b8.jpg)

<a id="paper-339"></a>
## Less Is More: Clustered Cross-Covariance Control for Offline RL

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=drOy5wi6Qq)

### Images

![images](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/images/08725c14e33b929bdfd44ab08eac3b9c9211beb6ca22546a8faabd7bf908e1a2.jpg)

![images](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/images/12a97fb2362e69de697e7a56c3238ea94f608216f7d02c256102628b6a7c540b.jpg)

![images](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/images/5b6d5830c41a01c332e078fdc9f05b0aa391e90cf2201307164960d097b6338c.jpg)

![images](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/images/7dddf446500275a8f81e7edbc5fcda881048e5fdf3f4bb965fa863bf344505cb.jpg)

![images](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/images/8c8fcea3c2ef8ac30a34ba5e1812bf9cfbad8c447d87c1a7e34ccbea7db39de0.jpg)

![images](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/images/ac6fc735290724c5f777a0550a019d8089591b8a1c1dd94065fbd53ea5b1e955.jpg)

![images](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/images/b1f6f7999964732ff86d9aeaa178b7e35fa7c016ebf6f2af45c6621448eaf456.jpg)

![images](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/images/bcdddf9c8ee85c871c94f97d89b0cd9a1e989c233eb46647c96ba7c699647efb.jpg)

![images](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/images/c247d67039182ef9bba62b3a9f9e71defcfb786f6bc453f012c7b9a7bf633a9e.jpg)

![images](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/images/d7f3ecb24003d73941c685e5c924bccb1baeabde5f3906ba8638c41fd6f3a87c.jpg)

![images](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/images/e54ce2ce10a9404b07a5d6c04e9134fad17dad90c88f455c92af99b20995b3f2.jpg)

![images](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/images/ef3525cb1460e0373265a306bc44be068237f3bdf728f4e462726211f407023d.jpg)

![images](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/images/f91cb7ab106e857595b95bc1ed8f50f5823156a24d37e9f080cc165838dc943d.jpg)

### Tables

![tables](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/tables/0a7e67b05c3d0cb47fcc2066d7af76c899e92357b5b22e2facd0838159830d9d.jpg)

![tables](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/tables/1423da9d7cc25869a8dd5b41a71a621aba553e721ad73924370755da434f27dc.jpg)

![tables](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/tables/26696736bb2e4f686f007a0f29304dba4b016d59e33ecb6dc99e8dd06112eb22.jpg)

![tables](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/tables/36fba7d819fd8ab3c6c5d7185aaf30feaaf8465c07335354c0c3932595b1612c.jpg)

![tables](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/tables/448e42d41208e3cd1e25ab4afc50f579cc4b1be7d65e38ba4e8a5b4b30bbab5d.jpg)

![tables](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/tables/5c4ab77321c14b2e517c7dc65d05f3e669a2869dfbadcc296a423fedf28cd96d.jpg)

![tables](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/tables/92bf32617ebd1b4cc7de069f82b5976e89e80a205c0aa2e2db76aa0b28c81bbc.jpg)

![tables](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/tables/96889193bbac2b0837a441e77a39ce402b9243f43a18f11f0b4cf9c978830df6.jpg)

![tables](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/tables/b08b4596703928b1377b80f1a9267b50213bfc5ce494c956b54721403106c59b.jpg)

![tables](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/tables/bb9007d79570b8c778a3fc471b4f948ebf5fe4ead8d0bbbc060afff8dd8e78ad.jpg)

![tables](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/tables/da069c472a0117078314f585ecc3d734855662f90515aea76f4304072ea5996a.jpg)

![tables](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/tables/e6ed178109ea24778849d2f5391be024ddd5c9a43d7c967a6af311dcaccbd539.jpg)

![tables](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/tables/ee7a9072b20657e93e826312201dc6998b9f4ddebfe877499d9050d7c12b0326.jpg)

![tables](../iclr_results/339_Less_Is_More_Clustered_Cross-Covariance_Control_for_Offline_RL/tables/ff3af64597bfdc662302c36083ffd360465a0e7a6f408a98809c3f32d9c577fb.jpg)

<a id="paper-340"></a>
## Triangle Multiplication is All You Need for Biomolecular Structure Representations

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=CrXcfMLR9q)

### Images

![images](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/images/01effc9592bcd0f40d855d87ac813b5b6a50761a2ff86f40397cfb4f4db2b659.jpg)

![images](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/images/2e7c1a765dd8e739ad163e4bfb631060b92e8b760bb15347877eafbb672d8da6.jpg)

![images](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/images/2ffe6dd6f90bb65c3d8c501d2888cad0c5b66ed1ae7c2fafc0c2f376e589c8c1.jpg)

![images](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/images/4668f47ff29406fc6bc71a5e51877d914e77df17adb7b5f9a70578885acbc254.jpg)

![images](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/images/4994d10b6edd7ca784510b3eba7505a935edc8eb8e217826a496adfb2990efba.jpg)

![images](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/images/4bae8714dc6249460f46fd91ff7adcfc31aa12db2a1f306d3567cb0951fb60d4.jpg)

![images](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/images/4d80a5a5eb8e0b07d76843d13d704d869a6457705890e3952a4c451a71f6f30b.jpg)

![images](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/images/616f6ce511287289c9704a9ec01c80f97de057054428cf364c313c88c816dfb1.jpg)

![images](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/images/67529135c783c01fe83f5867524bad3e797cb17b4d004082247621c1bd88598c.jpg)

![images](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/images/72c5f27938f0eb3317edb5e2d6fe601630fae21ce4c25a0bddf08addbd344ef0.jpg)

![images](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/images/85351584dd63f419c51ef2f6294ee6ffd13ecff01a5db037965fe8f7d0e4263e.jpg)

![images](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/images/9b55cec80bab593f2f3c717b69f5df82dcdada8dc62f6bf37f499d355e1ae511.jpg)

![images](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/images/b64328fb4e736ee15c26cb09a8994ac56784d0433a0503665daae3f06117fa65.jpg)

![images](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/images/c3b047f7f36b21a38436c726666b82d167ac6efbe485a7ccd8f6fa294233011d.jpg)

![images](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/images/c7d968e758a1ebd23a1cf22ab4ded472465001f97614ebd88d6793f71b9bb2bd.jpg)

![images](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/images/d51f6460b0409c031e4d59ddbf0754874a34106fe32d9ab81d58b8664d09bfe4.jpg)

![images](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/images/e077885781d271270be4cb5ae1f02faa149b970fd228bb14798e632327be6acb.jpg)

### Tables

![tables](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/tables/14acaf81a7752dabb3aa93378c62080c9407ac41a35c6bb851aa46eca8b918e4.jpg)

![tables](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/tables/17443afb885dbf0a01ad695fe7ce3ddc59fd5e289f8e26f21ff3044b5bd3b38f.jpg)

![tables](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/tables/1bdc03ec368efc39690a1402885baef8e53785bc1108bf645a9d41b0e636d345.jpg)

![tables](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/tables/3c03f4b10d09f4adad38392567d513729cfe4f61b55b58e82eadf25113baa11b.jpg)

![tables](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/tables/406763c38d6bed7d32c3c0ab6bce9b59ba17884e289e8a08546b848515c57dd9.jpg)

![tables](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/tables/4e62059f86fcba6279a8e5536322b634c62850fd01c74a86569395280bd64d4e.jpg)

![tables](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/tables/503579b55eee3fa06bfbdd2f698865b1ec3e412aa42f096e41315cf99bbf4de1.jpg)

![tables](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/tables/54668e299d15e2f547d84b5cfeefef3488e8c9b99d9437e2562518d713efcbac.jpg)

![tables](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/tables/85f7f4f880fb8f93dc2e74810336c30b847924ea16baacefd8fe2f490b2703a2.jpg)

![tables](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/tables/8b9afdc061b83b9af599f7bcae2a20b0765fe3e9af5228375590f42ab93a625e.jpg)

![tables](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/tables/b1225d4c0ff98f887c9399f32ce8069c742d898aa96e13a3fb8618ce8ae35f9c.jpg)

![tables](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/tables/b63fe7ae5743c109ce6b137cf1ae2661c3197734fce5a19d15ae4ac7660ddf5e.jpg)

![tables](../iclr_results/340_Triangle_Multiplication_is_All_You_Need_for_Biomolecular_Structure_Representations/tables/dc2406b8b7f59020740a2d59a5649263e4a046a115bbe8529ab441f6d0c602b6.jpg)

<a id="paper-341"></a>
## Biologically Plausible Learning via Bidirectional Spike-Based Distillation

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=MmWZ2xVJ7z)

### Images

![images](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/images/1c63ae4c1d39d89fd615fd7293ee984c229b2b117ac3b06b9cfa7ec8bb5facdf.jpg)

![images](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/images/4aab66cae8ea54306f29371548a0e89f6d44f250ee41294154aec170cd268e18.jpg)

![images](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/images/52ce8da268b9d6592cd10aa104fbe5245d1fcb90f87a2c783311202c1a1a029c.jpg)

![images](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/images/693655001d645cffeaf69dfd937bc1d30e5acd1dc1338255458c0d1ecbe581e5.jpg)

![images](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/images/7dfc4d17776106d554197315b277b4681bddfd105597bd17c5ed413176739249.jpg)

![images](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/images/ae4463085dd29ed7321db3bbbd7b4ffad72201f5267f1abff2b44400cb2a575c.jpg)

![images](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/images/def1820af6992dcfbe80637f7026c439a208888443738a68fc4d016c719bf1ce.jpg)

![images](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/images/e8d3481506a79f2e0cc249a7c5f1f31670c452a003fac69c4840f92d6c4123a5.jpg)

### Tables

![tables](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/tables/2656cd7620e7cd599626c43f7698ec6ceea85bfff518e47fda401aaeca607184.jpg)

![tables](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/tables/2bbfd00c39e42660464dd33ac19a87a5679495639fe8eb149325b64b5d7d401a.jpg)

![tables](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/tables/30a6f32e0bd01ae95676a2c98bbdb270d740f268b9e200f0c868b8507dfc1ceb.jpg)

![tables](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/tables/334139e370e35a5d3fb493f58b94f26e1f989532e2082a24d02a9b9a71e753d6.jpg)

![tables](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/tables/3d421d1418ac8d7eacae153fa55964f389598113ab99720a799d1e97c0f20909.jpg)

![tables](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/tables/51575dc541da35804611f40bae9671e265ca2817ea4740eef43bfc4e9afaf07c.jpg)

![tables](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/tables/5d262befd64323500e435c0b9debce824fa15e04f46f6d32bb2c3c5dab1c2471.jpg)

![tables](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/tables/5eb28472a22131be128622043e53c157cf04ff86b2a4e9b5634157bf788bad16.jpg)

![tables](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/tables/7e9e6abb3c848012749f9099af21fc8a03f07b2a19fc4ad36c6b9827d6a1b74c.jpg)

![tables](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/tables/80bd20b076ce24e181da46b880c85e34e5dff43e695e3c9cacc90f8f7b0115cb.jpg)

![tables](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/tables/8c7f8fcaaf3d048f3da53c414caeac57b02ca6e90a642ac1e282d32461dbf3b8.jpg)

![tables](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/tables/9b6d51bf2a6783a974cc9b0cc49aad7f1fbe00bf58060b68ecf2291204838acc.jpg)

![tables](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/tables/9bb53da3799c4f59f2fadb3ba49cbbfeaa04dbe5fb1254007e485a43c0e1e7af.jpg)

![tables](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/tables/af7efcd0d8d96d2085f378178a8b8b35c2d2fb2f4e5d5644b021b64517476ddc.jpg)

![tables](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/tables/b5a05f75d8933db789f854f2a4453dfdad3f18fdbb821f2626defd98693dd6cb.jpg)

![tables](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/tables/bdf4ca29d14014a2253a80f29c422fa0375c0dfdb80a7f7eacd511b08de0fcce.jpg)

![tables](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/tables/be6b3c7eb09878e51a473fc3a18d365af854b1ff93be3dd4ec4b0009d9922663.jpg)

![tables](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/tables/db8f1f784483ac2a12f7c26f1ea3ecd219ad17079c03f410d4f89567a8d0903a.jpg)

![tables](../iclr_results/341_Biologically_Plausible_Learning_via_Bidirectional_Spike-Based_Distillation/tables/f4f60e40d1f90bbf21a6d0cdf9fc6fee4658eb59207e2f8c992827821b8f3225.jpg)

<a id="paper-342"></a>
## Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=PWhDUWRVhM)

### Images

![images](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/images/0a65fb84fca1965a09765adc89c88eb93ff36c593eacbd54dcbca65863db30c1.jpg)

![images](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/images/10999dfb28b0fd7498631c1c974b29041c37ff675569cb08a2295f67f08f2a79.jpg)

![images](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/images/10ab31d2d30040f1a5b4f71f3dcbf01a6147e62f1ee1211b52298841aacd87db.jpg)

![images](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/images/142ca71afa533066eeb3987171844af9ea93b6c3d546877150b316323ee6e9a4.jpg)

![images](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/images/27dbff8c749d5597d6e1c0b514d339de339864a3b5fb466e932113660996aa25.jpg)

![images](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/images/40049839293777fee78525d6fabd3267762070fddd4ca6d1b26038faf6fd1a4f.jpg)

![images](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/images/4d49b4dab5fba97376d832a31bef0d24c66c13a9dd5861e1e1e963789d860932.jpg)

![images](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/images/6b2e96011fcb2225819f373f8fbf483a287011d32dfc534eba47a5e399d3d85a.jpg)

![images](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/images/7a11def7ff779fd51dd913b498c945b03df222676fc236bf8b6cd151bb5cd4e8.jpg)

![images](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/images/8217548d02f22b738ef266389b3548e6a18e81935b7f28b64319d79e47eb0c63.jpg)

![images](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/images/8a3c5bee554b93a1657fc4d55bb414c7f9f0e8ad1c228837af3fa6c497ce6318.jpg)

![images](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/images/91893fcb6ccfd159aaee729826f3e49e8f6de72a3a8807596019bfffb68c1b6e.jpg)

![images](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/images/a5882b1c22a77a814b4ebfd6392fc4bc766d9b44f72bca41f72f7955f8c82078.jpg)

![images](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/images/b5d2fc79747b832bfde4698575b4540c363a49c7384c8a56fe03eb566b3e20b4.jpg)

### Tables

![tables](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/tables/178b1d9c8d3f2c8115365b8a52869549b98e14d84ee67f42b8fca8a1c777dc63.jpg)

![tables](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/tables/1897aa55a5672cd15d63fe56fea479a1c30f93c17b156e26e81b005459cbb60d.jpg)

![tables](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/tables/21ce82e55f2d7f46ded9343f20640f973fdd15ee40858388580bb84a1564ebed.jpg)

![tables](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/tables/53c4a4d33a18fb1c6f7fd4550f4e4b693dc53addca8245cd0181ef5eafd02a3b.jpg)

![tables](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/tables/b52aa3c69210537432d734891035910a9bc08fc728e9291fd61a7fd846c8514e.jpg)

![tables](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/tables/b820589192b8d4a887a80435af7045687ac548ddc107a5b48b85fc19f951a3e7.jpg)

![tables](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/tables/bee1d868c84451b0bf7c8d9ec41330c518bb75f0aebc36d89e48760f61ac6e7d.jpg)

![tables](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/tables/c820e060e88e20c56536aa7ba031f695969b2eeaac37e63d0df36690fcc083ab.jpg)

![tables](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/tables/e817336620881174379f918b783722ae2d6fca5de5c93be2812ca16b5df26884.jpg)

![tables](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/tables/ebb0466a340328d1b96132f8aee8e7357513fbfcefe05690874583f766df424c.jpg)

![tables](../iclr_results/342_Inference-Time_Dynamic_Modality_Selection_for_Incomplete_Multimodal_Classification/tables/ffc580a1bdf4a11e3548bdbe47038dd9dbe49d99fc5e36118b58a300a5c75104.jpg)

<a id="paper-343"></a>
## ODE-GS: Latent ODEs for Dynamic Scene Extrapolation with 3D Gaussian Splatting

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=XlRbpFj3lJ)

### Images

![images](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/images/19d765e3edb92aa2603b565c342d53cf13e4eba9d1adc720d61f7b39417c1362.jpg)

![images](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/images/3089dd41b80a32d3e4796d69c836c206c25f9e04e2aa761bfe83aa638dab615c.jpg)

![images](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/images/74a3603d8ffccdae52dbdc477023152b6dd1cedecb85747097e15d5f69d5b15e.jpg)

![images](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/images/79243cc2bd1f4cdc1816fcd3b6353b8be2d46181a6a9c323187c1d10f29ee4d3.jpg)

![images](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/images/7acf7c7c6d24f603234f17e3a56c92c2e9374cfe72bb77a078e9703b4c3c698d.jpg)

![images](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/images/80bf4f025522f3076d93f291e868ed5d81cd10f9fefcb571c281d9b2e300bab8.jpg)

![images](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/images/87d4859c4f130f4db8d6d3eb469d817be52427bdc3ad9886cd0005d7a01acc52.jpg)

![images](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/images/929b44778149f601d8edb6d26549eed96126efd8b6758d75269f06102d1fb4d6.jpg)

![images](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/images/932e01be6bb02739dbe0d69f83e1c82bab94ef6f296e25df9d4c134c8a80ffef.jpg)

![images](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/images/a613b95c8be5005a61c21f719aa4eb697c31798f3f9a881c973d117f003ea994.jpg)

![images](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/images/a86cfdc62d047d7d396d7ba94bf417583e8cf1ab6ba36be32286e8c8ba35b5a0.jpg)

![images](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/images/ba94b4e3b2e5cbdfd9fe9f252fbb159b22ccbf49f009fa8cd34d04db12b1ade2.jpg)

![images](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/images/cbcc3265703e7997723fe0179b31dd227fd7c3dddc7bfff02b6b115beb12c100.jpg)

![images](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/images/d22ed238c8115c0c164d849f8b77586b19b018673331783bbc1f55b4a5d6aaae.jpg)

![images](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/images/d86a34951f639d81e54d8afea1626401a8667f661e48a94e6141156b905f1400.jpg)

![images](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/images/da34874caf3afcca6cf01098ee94142efed56d3c60faba7b51e211b9175b72f8.jpg)

![images](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/images/db5b9e6d9438b5bac52f4aae551a8e9326d1e0d2804b25ec9d14311b09d33a88.jpg)

![images](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/images/e79fcd085c551d28a85ce9e0b70585af4a7982625e3245ca613d5976de418e2d.jpg)

![images](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/images/eaf66c39ebdf32e120c784eeb1359e161ccd3db4871306f9c556c3c58133eefe.jpg)

### Tables

![tables](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/tables/03689a15ce74fa8302841bfb4974fba4c89174fa99f1a54e27f4875680c4c185.jpg)

![tables](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/tables/05121957d23ed22170495d1cee86d641edc4e07c319a86714274cbd487d0c9c7.jpg)

![tables](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/tables/0ba28285920c624493e4f9a4510447e3ed256d27fc29033b6d2951f903e664d5.jpg)

![tables](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/tables/0d7b249bd41ad51f1cc2a7417bd34d288b57b53d89261c193ccfb3e145155300.jpg)

![tables](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/tables/517dd904a674b56ab80d592598800bc5452574d33964a0d87afd1c2c52d7c129.jpg)

![tables](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/tables/653e7f2fad60d9604737a129dedc357b02be59d7e75d89829a92f75415c58559.jpg)

![tables](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/tables/697fbac472754485b2dd924ce6d78b033e7c1d71030e29fb6b3db0ed2afdf419.jpg)

![tables](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/tables/7d9f6bb8c23e5df07fdd9662ea82d52b85ec4499f847e82cca2bce804d61fdbc.jpg)

![tables](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/tables/c3b14bb0c5647d954d876a2efb5425dc6727e1cea3674ccdfedf6c689d883df7.jpg)

![tables](../iclr_results/343_ODE-GS_Latent_ODEs_for_Dynamic_Scene_Extrapolation_with_3D_Gaussian_Splatting/tables/ce4bab31c5a026a18d3844fe5bfef870b4a37427b6b1d387dcdbf7f7f918bd2c.jpg)

<a id="paper-344"></a>
## BézierFlow: Learning Bézier Stochastic Interpolant Schedulers for Few-Step Generation

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=PCuDI32xhQ)

### Images

![images](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/images/18fe95053b20ed5347aa87062a9b7b014ce7efdc7437b0cd94b32ef0a8bf5e05.jpg)

![images](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/images/50f743e3bf5c8797130b71824693c090a6472c02e527c34f688d9163424a3087.jpg)

![images](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/images/7098aaafa2cd623a9f888f9e6d488a291d9dd6e982f30c49dbc2b8453ea400d9.jpg)

![images](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/images/7c4efefa4b49f03d8ca7fa0f7a95dfee93c00de2fa42dc340448549cb89c674c.jpg)

![images](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/images/8f6797c026f2f19dc2ad16e6e983abbf87bda0bf267c7e146dafe1b855856928.jpg)

![images](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/images/98e9ca1897ef8e5673096f252ae89fb8bf340de81e2998b05d65242dab128d32.jpg)

![images](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/images/af59e5f18f74e19fcef6a8f3a6dc004c10ba5f265c31fbbea9ca45be7524b8c0.jpg)

![images](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/images/d9c0a472ccfec072132edc4681151ad1acfcf9a7b2078465bffaeb6084904681.jpg)

![images](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/images/ef0c56a3e74571f91be06bf318bc637fce557ec8cfa51ba153ca0241bc368584.jpg)

### Tables

![tables](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/tables/3782e0cb189d63faaf558b1e14f0ab7b8a401c5cab958451d0559bdb26bd717e.jpg)

![tables](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/tables/39e33357b52a8e7148151610e591e2afb350664dab0141cdfbdefa09be06a2e7.jpg)

![tables](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/tables/3a7d718cf92e66c4ca14c08cdffa70b3ec546429d7c262cc68bf62f44b28861e.jpg)

![tables](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/tables/52331c7584ecaa6736268385ec541fbe81229d30ab2b881574269b74eff185a2.jpg)

![tables](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/tables/52455d91e84e9a690491f817a274b74b9c3bf42156d863961ab43ddf80a87df9.jpg)

![tables](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/tables/6f9d6c75ee04e8bb073cc50123969a533386ea47a3ab4df38ad2cce1d64bc6d2.jpg)

![tables](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/tables/70b2a4642cb24e1637e1b6e58982397a414e76bef1ba53421bf85c8c2506c5a4.jpg)

![tables](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/tables/73ff2b0a23d58084e317ec55a00402a9553945ebae85a67e6f0e0ed5b7a7bfac.jpg)

![tables](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/tables/87c751bfb674171adf59aa4a8dcfcc18392f3e5cbb476ddf353c08fce108bb87.jpg)

![tables](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/tables/8a52e9001c1d092b25f7619ccbbdaf69938989956afb48ed1c6461f2b2a1aeb9.jpg)

![tables](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/tables/a421f33e966bebbea5c425dfc4baa95f69d735dc7025603e51c4a5a2e5940bbc.jpg)

![tables](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/tables/ab40317637a3bb011171489f37d0eddc381b3dc68b6cb676b2c4644ae88a30b6.jpg)

![tables](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/tables/dab29a66b7708cb90085cccd7bace3cd96e9407520aa2f8e87effb2e819a063c.jpg)

![tables](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/tables/dd40d49c21af15a79c09c9a6a3d8ce4ac43b7c7b08fb7bcea852e47da90deebc.jpg)

![tables](../iclr_results/344_B_zierFlow_Learning_B_zier_Stochastic_Interpolant_Schedulers_for_Few-Step_Generation/tables/fcb5ab09f6c67a1070cea999476ddaf6d5c3c6bd614cb357e7fc8b9f2c6d046f.jpg)

<a id="paper-345"></a>
## PairFlow: Closed-Form Source-Target Coupling for Few-Step Generation in Discrete Flow Models

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=awEvtKliMC)

### Images

![images](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/images/0e17223d798e658fe4d010cd15483a881bf0b57eae59efb2deda7a6a7ef83c13.jpg)

![images](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/images/2363b00f0edc5c573840d7ef1c4d51b724b175bfca2622447950a86ecd0ab402.jpg)

![images](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/images/2c691b091c8a039bb151c451fb33a80b797d10b5fa47ba803558190860e341a9.jpg)

![images](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/images/2e69859dd73409973a98555308d9bd42328688dd010a861a1123cf1fa3f07b55.jpg)

![images](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/images/66827483992920baff819041f2bd54820bd6d2d5dbaf530fe66a303c7ace0ee7.jpg)

![images](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/images/83653ca734059fc7c5b4d445f006280b79f3dccd2fa4f7f34de05b512026a73b.jpg)

![images](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/images/a804fa010db9a21a872476fd941fedb9a1cf06f7f089d273f400615b2513c057.jpg)

![images](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/images/b7a3f4cb6d270134b75c839c1c02c6b7d5462d652fb823693d628a00c4bfb8ad.jpg)

![images](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/images/c98e081b0e9c12d16cb823c9c9fc8883f664593b3181daa7fded6c097e077de1.jpg)

![images](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/images/e2ed202745c8f64dd045a8a7d4b13e31638511169bba531403cdec86308583f3.jpg)

![images](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/images/fdb14182bac5d0b1608ad7f53261d7f7347aa9e677aa0bd246d145443beab4c9.jpg)

### Tables

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/03238d0e517eefe9fb0c885745c6cef138e75ff9054cea69751168787630d557.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/15e69a1a30885df3f08825ece35086b4fd864c7ff76d27c755475a5f628eafc1.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/18d18846b77e8d2484ffd80afe47e1b3342a667257e4bafc538b1b1600654b0e.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/196b16a74778e965e10ae8e754f8f631df3aaae843deac897b2b87504292fc77.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/256c8d4a5a948c715d2e631fa1034a696b5ff5567e67eb356710bc3ce7f0bd41.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/2b97cc6af5d4d8e45dd65b6426b295b1d5daf8d7af14c25865067453b001691f.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/34203e4cd4cec06f138d5ef23de23447eb02f91bea84293c9a04dddc163eff3d.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/4700cfd6c2453e10f0eb72f35d6675e355ec9c87e169acd78299e99b08222022.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/49c0019133bc3f6ef95a6bbe5a8f07a79c7370bd7570a9ef2995cd2168b21139.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/584d059532c3b8b54185f046bcdb51357c87cccf79a8f9829f2a10041bb1893d.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/6061e81b16b7dc98fa95a28835bd83dbdedec4bfda018160413a6a2ce349c132.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/61bb359ac2b20498a0f4a10b54fecd11887174bd936a4de9aae93c70d959a8bd.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/7872729fe59e19a1e3346cb94eccd1fd535c068e86a9f020afc8e2a8a5c7a3f4.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/81de44549ffa1334de16001fdff3f114e0c6818437882c1ee434514e4ca392b8.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/82d55c04bbbaf2db3566897e88c0d9462229cb49f141722d9c01e846d47365e6.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/870a5343b5d6e3ee5ddc31b59b533bef1233b196c0a9ca7b8957ee193f92ff31.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/8b0fcae93e06afa628ef9b3f7c4bc7dca3c4246c587d571d08a5a57eb8a2d84b.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/8c458d37be2e38c15eb04af8dac141af5f9e510a6e866c87806d07deab118e41.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/9476664e6c58258be9b7943b1d8108a6457d63abf716a2c6b4ed504c85a310e4.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/a1b8d964aa847d6c1470bdcc647ea22f97c8e5f7d0a9ca46ed548f76b4ee9202.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/a2b5d6905b68de3934241b6a07ea8446fd00d3f21a19717e5591cac70f5b9686.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/a60d44597fe16ff478674ee5378bfa9dfaab43390e9ec76857fde0d0321dcc28.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/aff45f5333dafc092d5121425a8f1a2f0a9e68b378b7f12ed90ec56fac0932b2.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/c13a56ba77065cb87921f18f5fef367b332e495eacb7804b80b7c918cdd438d1.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/cad822429b34255e08b98df021ec2a320e06f3e35e53436661b1ac2b3d69be41.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/ef98636fe662a7d459c7e67d3a6bb4e9bf9739f6b409c68c772f414e04d245c4.jpg)

![tables](../iclr_results/345_PairFlow_Closed-Form_Source-Target_Coupling_for_Few-Step_Generation_in_Discrete_Flow_Models/tables/f7254f3fbd8efbdc612ea76192338885702705e6a18a465977c8f745d736db42.jpg)

<a id="paper-346"></a>
## On the Design of One-step Diffusion via Shortcutting Flow Paths

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=k6q8rRYVQR)

### Images

![images](../iclr_results/346_On_the_Design_of_One-step_Diffusion_via_Shortcutting_Flow_Paths/images/0d4d6172dd0afa937a2486ce4b3001ad8a660f2a1c3d18e3ba13196cc141ab8e.jpg)

![images](../iclr_results/346_On_the_Design_of_One-step_Diffusion_via_Shortcutting_Flow_Paths/images/2bd364ffd4eaf33f3aa382b3aff94360dc562e072797e177acf4ccdfbc5bda2f.jpg)

![images](../iclr_results/346_On_the_Design_of_One-step_Diffusion_via_Shortcutting_Flow_Paths/images/663c380d9e11b96837a465e940394f439bdc416e269baea587f9e179c446b708.jpg)

![images](../iclr_results/346_On_the_Design_of_One-step_Diffusion_via_Shortcutting_Flow_Paths/images/886fa44bce5a36716e8789d8c305bcfa91ff8f10e0aa18ae8495796b1c56faae.jpg)

![images](../iclr_results/346_On_the_Design_of_One-step_Diffusion_via_Shortcutting_Flow_Paths/images/adb923662e449c4309a44fd3cca9ae477692e32853a1fecf8d9a1b132dc05eaf.jpg)

![images](../iclr_results/346_On_the_Design_of_One-step_Diffusion_via_Shortcutting_Flow_Paths/images/ae9b51d85399c1774bb49f6ed504b7870e93d77aab37f8f8071a1c83f5fc6a9a.jpg)

![images](../iclr_results/346_On_the_Design_of_One-step_Diffusion_via_Shortcutting_Flow_Paths/images/bbe5e834a294d13909126bc5df87beee61e0ac100975c7c0cfd47eaa66ce01c1.jpg)

![images](../iclr_results/346_On_the_Design_of_One-step_Diffusion_via_Shortcutting_Flow_Paths/images/d3ec8458c777ae2b8ebe791cc08980326a85a9bed66b143516fde7b5bdb213ad.jpg)

![images](../iclr_results/346_On_the_Design_of_One-step_Diffusion_via_Shortcutting_Flow_Paths/images/e6e7ed57c501b564c2565e5a3e43f8a20a2e803ba707229943dd99cc9d305a7d.jpg)

![images](../iclr_results/346_On_the_Design_of_One-step_Diffusion_via_Shortcutting_Flow_Paths/images/f42bcc5a06d13971631819ad92c3cdfcc3733a16c4b4ac06b3cc9887e0a80e4e.jpg)

### Tables

![tables](../iclr_results/346_On_the_Design_of_One-step_Diffusion_via_Shortcutting_Flow_Paths/tables/0060f4bc1d073ac6e99bee43a3772cda89a65ea867fdebe0583c5288fc101ff0.jpg)

![tables](../iclr_results/346_On_the_Design_of_One-step_Diffusion_via_Shortcutting_Flow_Paths/tables/162c26fc1aade5c4d68ac9aa8629c13a283e09f7cb1171438b9eb709ee1f4e61.jpg)

![tables](../iclr_results/346_On_the_Design_of_One-step_Diffusion_via_Shortcutting_Flow_Paths/tables/6022897b8758cf0aaa76dbac5c58e0cabdbfbca2bc9cb8e6cd8f2a69a75b2cb4.jpg)

![tables](../iclr_results/346_On_the_Design_of_One-step_Diffusion_via_Shortcutting_Flow_Paths/tables/864be0138ac25d45b7759bd46c20f8f4c5861d3e4e9167f6b7f4410e7c4f9c89.jpg)

![tables](../iclr_results/346_On_the_Design_of_One-step_Diffusion_via_Shortcutting_Flow_Paths/tables/921b4badf495b3bcedc04a60889143568af9de0cd621ba76cac0e561204b4131.jpg)

![tables](../iclr_results/346_On_the_Design_of_One-step_Diffusion_via_Shortcutting_Flow_Paths/tables/af743ea1ed055e2b122f6cf01db91c2369fda88840e95b24795a12aa25c39645.jpg)

![tables](../iclr_results/346_On_the_Design_of_One-step_Diffusion_via_Shortcutting_Flow_Paths/tables/bc3faef01b04889d4464591377b24d403f2cc5cf182ee069c30ee3d7fb802654.jpg)

<a id="paper-347"></a>
## Chunking the Critic: A Transformer-based Soft Actor-Critic with N-Step Returns

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=rb5eTktqbc)

### Images

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/00d3faa5f163487b957c181e754c35bc38ee593e3f7b3346a757780169d422d9.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/059f3b675e543f9ab078e80f4e9a093f753d9dd77197ba1f41232f09d2f69ed8.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/182aa2da5841308a435b29c215e00079db3b13068d7be46559318309653aa942.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/2f90a2553f1f4874aa85b2c789d0ebae41ae068dc33ced6f7ba17440cdc9a9cc.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/34443207fafc0bd61f71791505ed2f555c6d8186cf4baa5ebf499450b9687b8a.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/3808430241c384f48e40a2507dc8bbd5f97624a427c9dc5f66db64c4db69e8cb.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/3c89a2371bded4c015a0c29789205c110609761d0305bd1f7ce46d2ea7bd6f39.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/4caa6ea53b079c2652c2de366d9733de9bb678e3fd2b7634477ed39d1f78bb72.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/4ef206f2c8cb0e3ff6636b0f37fc874b3d639dfcce276b0c176410cb4202d261.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/69b640ea35d9e3cbdc810f9a90816c61ae1c1d72edd88533ca59335543b8c8e9.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/6d988e5eb4fa095847c495f30c9adf4631e32968489de19359ca01b604bda79f.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/85c8cb1c5d494c775bbf6e0cc20f04cea8846a30f886cad532de7804f46a6534.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/91fe1d9a037c8a9cf07bd87931cffbdb41e4fb9027acdeaa96813cec7130c757.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/984eb19d253d69058152e09bdb379fb95cfbfa32f2bc8f45a14331f286123895.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/a6b8d3a30275927df78f8db845312ff0ac6f0c6f38387ec948c11001462a7b4b.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/bd35f0a52aa04fe55d5d2f5bac983219f3e2b91c8073742e597f55fb33d07dc4.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/c31d4635a1c882b6e85db92b1ad8275b15f94aba336464fe4b07316c21b3fc62.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/caac16e8aa479d187dcfe140af61b8ce3763ac991921cffb964421dbabb2f7ee.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/e4fd81ac9926a47673e705d79f06656dbd83f6ece9f577be68c8330198819942.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/e5273ac20453818811d90c038dd87ef4f20101378106470573c3b218e4ef40cf.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/eec741d84068d9349538ce60010a9118010e3675ba0c3999f8dff13e6a1a4793.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/f0d5b7c9c20fcbb7cfbcc5a11768e367b3eee8044b830b488d9fa528240fa9e6.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/f71b5de034f3d13c8a84005e7a7014e04c00a38d1ca520eaeddac7efa8714fea.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/f71c59857967c5380b17af0844a365043c3d83212d920fb8c315dfdc8f4552dd.jpg)

![images](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/images/fb242b1741ee4cf66d6bcd1febe54922671ffbfee0cbe5b2683fc80d4f5dbfe9.jpg)

### Tables

![tables](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/tables/55b948f4ca5d872e9a2bb6ffe366291de3d845fe1298415e8f6788c8ac22b415.jpg)

![tables](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/tables/6b3994f465df350414c135f40fdf53274d8c2c12e3a4bb3bcf94d4f347e8108b.jpg)

![tables](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/tables/78e565272f9e7db0527521af8464b6584c78c56e72bc8db8ff33089b024f323a.jpg)

![tables](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/tables/893bd8af5760769adcd89576b0e0935b990fa18b0aa5edf4c3fbffeb1dcca863.jpg)

![tables](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/tables/9a1f98dc3648f7245bec7cc1decb23cd8b940d3c66a3c98a66659c3e74efe26e.jpg)

![tables](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/tables/afe20d0c3819d255212a85cb340e846eaaf213e4f9ce11aed13a0267fb25cd66.jpg)

![tables](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/tables/c7eac2648935bf562ae6a38c7428bc2a53706fa27552ef51461e5352ded2c2e1.jpg)

![tables](../iclr_results/347_Chunking_the_Critic_A_Transformer-based_Soft_Actor-Critic_with_N-Step_Returns/tables/cecd4560bc358e251c79b6972326ed8aef59290c34e5d7cc53f57ac0790837db.jpg)

<a id="paper-348"></a>
## AudioX: A Unified Framework for Anything-to-Audio Generation

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=qjJWxK3yWo)

### Images

![images](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/images/1c22c9bf7757b8c28e09c85121198a14c63af4313d7059c8e42d59ddd3692567.jpg)

![images](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/images/30d6a2d01f0cf7e7391fa8dc35d3c2bc965e4c9c7392b5687f02a4a80e6d4502.jpg)

![images](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/images/53361eadf0390771f6147e5f1b2f691ab84a7798b06869cf4a39ca2b16d78b56.jpg)

![images](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/images/5acba4051f7f7e08842bbc2e85ecb9343bff8c0373962d93d62b76515f6bd21e.jpg)

![images](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/images/6d036289a62849b518f8dff5f4bfa33e1d23a3c155433cdfe9f8bbbf5608e1ac.jpg)

![images](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/images/a3c2db1870710ac22311a12a0c7fdb6e202d49be83c0f32407580aec072b8b45.jpg)

![images](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/images/d24dfc3fa9f3b62c33c335906b2b166ee6530ea07891f91500b38301219d8ff6.jpg)

![images](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/images/e01f6c9f9344e84d219296da133e533803e64a9226dae1ad1355eac6fe10a63b.jpg)

![images](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/images/e0822daca6e51de4ee938f3851055aee6ed353eb3bde79d176e2aa89c047fabc.jpg)

![images](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/images/e37f3ce7c7336017fb7a5bd5bbdb994a3b47c84f548867a4c5e727fd1ba07e28.jpg)

![images](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/images/f7865991751f5c3a41895688fe020f35fcfe6925bb02609f5e7652e1b3dc5fc8.jpg)

### Tables

![tables](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/tables/1a1e3ee269d38b5b73a402cd9a2821725d7ede02e8675bf129818f21f6632a26.jpg)

![tables](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/tables/1b63e6224d2e4e63cbe6248c1f54a4b4e71cf895a67856b3a7408c795b9c2549.jpg)

![tables](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/tables/1d5229d1d7161a43e4bcdafc373d8d8e3d54142fac512d117800240cbdbf33a6.jpg)

![tables](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/tables/621bf628d8ee40914170319ee56b3d44c5c3da9382c96b9486706cb1f05b8900.jpg)

![tables](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/tables/813c3b7fc97e6ed7474211130203f9fda987337b9f710cb181c322fc7d66a668.jpg)

![tables](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/tables/84baf2c8f3aac45368239f596dedecbeb7380003f677a92f8e0e1966fe943430.jpg)

![tables](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/tables/b131b039345c82eac91a3c8b113b4319f6e74a8a5354c1f46af7074d874c75c7.jpg)

![tables](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/tables/c73dd7f84beb8cb29004a55f0a45a4fc855ba68c95e2bf6e9eee057b4b4a792d.jpg)

![tables](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/tables/f1c584e736275361cac2da2f2494c3435458aaab8cca1d7f94a448f7bd24790e.jpg)

![tables](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/tables/f1d236689fc6266656bad4471f37c24bc49a18003fbf5cacc34559cf9f53e80d.jpg)

![tables](../iclr_results/348_AudioX_A_Unified_Framework_for_Anything-to-Audio_Generation/tables/f8edefc95223d15737163b98527c9272f08b8c9f35770088e445f3729292191a.jpg)

<a id="paper-349"></a>
## Speech-to-LaTeX: New Models and Datasets for Converting Spoken Equations and Sentences

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=gk8WMxzIQP)

### Images

![images](../iclr_results/349_Speech-to-LaTeX_New_Models_and_Datasets_for_Converting_Spoken_Equations_and_Sentences/images/0ef6f3fc8c66e9f37b74ff709123d31ee1629b6562e0b4772b122af8175db0c3.jpg)

![images](../iclr_results/349_Speech-to-LaTeX_New_Models_and_Datasets_for_Converting_Spoken_Equations_and_Sentences/images/b572eb9e6b32dabd005e5d7e36cf630311c514d68b18d94ad78462d738bf08da.jpg)

### Tables

![tables](../iclr_results/349_Speech-to-LaTeX_New_Models_and_Datasets_for_Converting_Spoken_Equations_and_Sentences/tables/18d97ebcab03bf4a9d36740bd6a888984e55f89fbf06d38296d315b4ac923ca3.jpg)

![tables](../iclr_results/349_Speech-to-LaTeX_New_Models_and_Datasets_for_Converting_Spoken_Equations_and_Sentences/tables/2f8f69f9b27a4652cbb973542c4454c7f19f7fb60ec8119267f1966d58f62790.jpg)

![tables](../iclr_results/349_Speech-to-LaTeX_New_Models_and_Datasets_for_Converting_Spoken_Equations_and_Sentences/tables/4b4b0ef850d35f8c8ebe6f09e0e5bbd1f1a013b8ad7d58af2ae34407a4fd290c.jpg)

![tables](../iclr_results/349_Speech-to-LaTeX_New_Models_and_Datasets_for_Converting_Spoken_Equations_and_Sentences/tables/57020835a00fa6543119a46a58fad8d2e3a55da2690b00d0f2a3d6396166772a.jpg)

![tables](../iclr_results/349_Speech-to-LaTeX_New_Models_and_Datasets_for_Converting_Spoken_Equations_and_Sentences/tables/5923c905316d9afed167388d62bcafd43e5528959e4b358d5da436ed844e46a3.jpg)

![tables](../iclr_results/349_Speech-to-LaTeX_New_Models_and_Datasets_for_Converting_Spoken_Equations_and_Sentences/tables/5fb90d49a0331e532cb225e892fbbe6c0bab6fc25d22a83365172af7abd528c9.jpg)

![tables](../iclr_results/349_Speech-to-LaTeX_New_Models_and_Datasets_for_Converting_Spoken_Equations_and_Sentences/tables/74859c581b774b9109eb75b734e1fddabccc8590ce9d33ae77aed05ddd24649c.jpg)

![tables](../iclr_results/349_Speech-to-LaTeX_New_Models_and_Datasets_for_Converting_Spoken_Equations_and_Sentences/tables/7b6f802a07a33bcb4ed3c5ebb776b403c8a49a2ef42fc34544c5804765067f9c.jpg)

![tables](../iclr_results/349_Speech-to-LaTeX_New_Models_and_Datasets_for_Converting_Spoken_Equations_and_Sentences/tables/862b078a4393ae476a7470e9ec4dccd5ea1061419d85198734449b45c525d118.jpg)

![tables](../iclr_results/349_Speech-to-LaTeX_New_Models_and_Datasets_for_Converting_Spoken_Equations_and_Sentences/tables/88bb61ace64307fef2ea2efd48a8f965e2cbd643180e14649e4ffbc370a865e5.jpg)

![tables](../iclr_results/349_Speech-to-LaTeX_New_Models_and_Datasets_for_Converting_Spoken_Equations_and_Sentences/tables/90d03fcd54dbd8d2132cd74b25898c34bb4832e3a0b112312c8be242b904bc84.jpg)

![tables](../iclr_results/349_Speech-to-LaTeX_New_Models_and_Datasets_for_Converting_Spoken_Equations_and_Sentences/tables/a4b7eec6adc202c22684ad7a2a5882f5bb2d8aa2af5284f530f8b4ca10607eeb.jpg)

![tables](../iclr_results/349_Speech-to-LaTeX_New_Models_and_Datasets_for_Converting_Spoken_Equations_and_Sentences/tables/c1b64109c77575872ed19c384958ae82f85947cabb0761f3921226c5f543b537.jpg)

![tables](../iclr_results/349_Speech-to-LaTeX_New_Models_and_Datasets_for_Converting_Spoken_Equations_and_Sentences/tables/d5b5e1bc78f343a38fe9d87a9bf84bf42e2f4076aa456c92b7a563f60196e1c0.jpg)

![tables](../iclr_results/349_Speech-to-LaTeX_New_Models_and_Datasets_for_Converting_Spoken_Equations_and_Sentences/tables/ee03754b9b32e929a3b4f64f7a19bea400044908a31c14b89e9f9d64ab09e11a.jpg)

![tables](../iclr_results/349_Speech-to-LaTeX_New_Models_and_Datasets_for_Converting_Spoken_Equations_and_Sentences/tables/fe454a2cf0cdd7962100698e29f5b89915020e91d73ea20beef0810b1503a097.jpg)

<a id="paper-350"></a>
## IceCache: Memory-Efficient KV-cache Management for Long-Sequence LLMs

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=yHxSKM9kdr)

### Images

![images](../iclr_results/350_IceCache_Memory-Efficient_KV-cache_Management_for_Long-Sequence_LLMs/images/437f1f85434d445f294b9249da43464e7b7e03f5021cdc36b06d0328d3580a7b.jpg)

![images](../iclr_results/350_IceCache_Memory-Efficient_KV-cache_Management_for_Long-Sequence_LLMs/images/7e6743afc44e827391c862818e12c2efd6f77be5c4c42ae06d1aaaeacb4e49bd.jpg)

![images](../iclr_results/350_IceCache_Memory-Efficient_KV-cache_Management_for_Long-Sequence_LLMs/images/9842fc42285e54bd48f6584e643d6691d8c9e40ed3f3c0a6234ccc05f37659f4.jpg)

![images](../iclr_results/350_IceCache_Memory-Efficient_KV-cache_Management_for_Long-Sequence_LLMs/images/a8fd7139b3f0a372d35b86032b8c7854004af796e862e92f42a1a7915c2a4dbf.jpg)

![images](../iclr_results/350_IceCache_Memory-Efficient_KV-cache_Management_for_Long-Sequence_LLMs/images/cdfe88d6f02d6e5a7576327624fd682dc528cf9e44dcc95d966480d5e4e494bd.jpg)

![images](../iclr_results/350_IceCache_Memory-Efficient_KV-cache_Management_for_Long-Sequence_LLMs/images/edf0e1a998995e90b8ac31c650793e6b1f22cbf7c038290d8c873be386243c49.jpg)

![images](../iclr_results/350_IceCache_Memory-Efficient_KV-cache_Management_for_Long-Sequence_LLMs/images/f7a02f46637cc9f046e211308ccc772c1adcfc22268ede198e91c05e3f7e9a25.jpg)

![images](../iclr_results/350_IceCache_Memory-Efficient_KV-cache_Management_for_Long-Sequence_LLMs/images/f8db9924542b8f759443ab3405553147a57d824026c7b24155bde6de67d161cc.jpg)

### Tables

![tables](../iclr_results/350_IceCache_Memory-Efficient_KV-cache_Management_for_Long-Sequence_LLMs/tables/1504af1d8395bd4bd0f7e07dd946bb6ccf91bcbffa37d5db75d1ae56422b8608.jpg)

![tables](../iclr_results/350_IceCache_Memory-Efficient_KV-cache_Management_for_Long-Sequence_LLMs/tables/3b8d6784cc4447aeda428bb8cec553ae06fac5ba7ae84eee4ea82354f3966573.jpg)

![tables](../iclr_results/350_IceCache_Memory-Efficient_KV-cache_Management_for_Long-Sequence_LLMs/tables/5d46b92853d1f3d2f9fa4e80be968c1fb80d338ef6dfe6a4f7ae975d77840e0c.jpg)

![tables](../iclr_results/350_IceCache_Memory-Efficient_KV-cache_Management_for_Long-Sequence_LLMs/tables/810705b490b6f8b7e9a8627ee198edfde01fd8379fd5dbd02f09a4432d924c94.jpg)

![tables](../iclr_results/350_IceCache_Memory-Efficient_KV-cache_Management_for_Long-Sequence_LLMs/tables/8e1650f2a5ab125a97be8c2bbd88c7e8e39cd028c8f2db6340b04abc64487b37.jpg)

![tables](../iclr_results/350_IceCache_Memory-Efficient_KV-cache_Management_for_Long-Sequence_LLMs/tables/ab5833c5da62f9c913e573ee93ed325b5c1a7c65fe964eaeb5c7d09612a9a196.jpg)

![tables](../iclr_results/350_IceCache_Memory-Efficient_KV-cache_Management_for_Long-Sequence_LLMs/tables/ef435f10320fcf3dd5735490318b3d586742fddadab4214c594b8a0f2c851da9.jpg)

![tables](../iclr_results/350_IceCache_Memory-Efficient_KV-cache_Management_for_Long-Sequence_LLMs/tables/fc1a7bdfe3aca5f5729a2b79f536694408beb430894b4eec49dddf4a822b03ee.jpg)

<a id="paper-351"></a>
## In-Context Algorithm Emulation in Fixed-Weight Transformers

**Type:** Poster | [OpenReview](https://openreview.net/forum?id=BC7YLA0zJ0)

### Images

![images](../iclr_results/351_In-Context_Algorithm_Emulation_in_Fixed-Weight_Transformers/images/49eb145624c3d10d06aed1c51dfd18dcdc7859f206fdc645172e31889c91e019.jpg)

![images](../iclr_results/351_In-Context_Algorithm_Emulation_in_Fixed-Weight_Transformers/images/8f39ded641901b67eeb1e439980166a17781b4e02b1cb575f332cf0f315b6617.jpg)

![images](../iclr_results/351_In-Context_Algorithm_Emulation_in_Fixed-Weight_Transformers/images/d38625da9b6fbee887a468c7b3a91ede621a2bd95013fa2ddab1af9fce99466f.jpg)

![images](../iclr_results/351_In-Context_Algorithm_Emulation_in_Fixed-Weight_Transformers/images/e9616bddd4d4c836284f881975533eb80500acb7363014fe947e6ce865ee2335.jpg)

![images](../iclr_results/351_In-Context_Algorithm_Emulation_in_Fixed-Weight_Transformers/images/eec1153642863c769ae4052284fa47f955f64aecfef8e2b4afcd7c654cb7badd.jpg)

### Tables

![tables](../iclr_results/351_In-Context_Algorithm_Emulation_in_Fixed-Weight_Transformers/tables/2feb6c59bf05e37e308d569735abe426d2f0c7f6e9bf98506356e8eb054ce5fe.jpg)

![tables](../iclr_results/351_In-Context_Algorithm_Emulation_in_Fixed-Weight_Transformers/tables/46c218a025f4b6e9cd99638c52bc06bc3f2dd856513168e2723b74d5f61e7f8b.jpg)

![tables](../iclr_results/351_In-Context_Algorithm_Emulation_in_Fixed-Weight_Transformers/tables/518fea6a46c050b56ad4dff7dceb3a2c3f9f98608290caff7ce33bf54c6aaa42.jpg)

![tables](../iclr_results/351_In-Context_Algorithm_Emulation_in_Fixed-Weight_Transformers/tables/b19e86af03a2186338e11a7abbd0b730bd775504041f7b2b210faac25370f6e9.jpg)
