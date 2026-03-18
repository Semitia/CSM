# Configuration Transition Control of a Continuum Surgical Manipulator for Improved Kinematic Performance

# 连续体手术机械臂为改善运动学性能的构型转换控制

Shu’an Zhang, _Student Member, IEEE_, Qi Li, Haozhe Yang, Jiangran Zhao, and Kai Xu, _Member, IEEE_ Shu’an Zhang（IEEE学生会员），Qi Li，Haozhe Yang，Jiangran Zhao，以及Kai Xu（IEEE会员）

_Abstract_— The use of continuum manipulators in surgical applications has increased recently. **摘要**——近年来，连续体机械臂在手术应用中的使用日益增加。

A continuum surgical manipulator is usually tele-operated after it is fully inserted into a patient’s cavity. 连续体手术机械臂通常在完全插入患者体腔后才进行遥操作控制。

Clearly, it is still possible to control the continuum surgical manipulator while it is not fully inserted, although the mobility might be reduced under its partially-inserted configurations. 显然，在未完全插入时依然可以对连续体手术机械臂进行控制，尽管在部分插入的构型下其灵活性可能会有所降低。

However, such a control scheme for realizing the configuration transition during the manipulator’s gradual insertion is missing. 然而，目前尚缺乏一种能在机械臂逐步插入过程中实现构型转换的控制方案。

This paper hence proposes a novel kinematic framework of controlling a continuum surgical manipulator with configuration transitions for improved kinematic performance in teleoperation. 因此，本文提出了一种新颖的运动学框架，通过构型转换来控制连续体手术机械臂，以改善其在遥操作中的运动学性能。

The kinematic framework includes prioritized formulation of the Jacobian-based inverse kinematics, prediction-based constraint imposition and a set of configuration transition strategies. 该运动学框架包括基于雅可比矩阵逆运动学的优先级公式化、基于预测的约束施加，以及一组构型转换策略。

Both numerical simulations and experimental investigations were carried out to validate the proposed idea. 本文通过数值仿真和实验研究双重方式验证了所提出的理念。

_Index Terms_— Kinematics, motion control, surgical robotics: laparoscopy, continuum mechanism. **索引术语**——运动学，运动控制，手术机器人：腹腔镜检查，连续体机构。

I. INTRODUCTION

## 一、引言

REDUCED trauma in MIS (Minimally Invasive Surgery) benefits patients in terms of less pain, quicker recovery and lower postoperative complication risks [1]. 微创手术（MIS）所带来的创伤减小，在减轻疼痛、加快恢复以及降低术后并发症风险方面使患者受益 [1]。

The operative challenges of MIS stimulated the development of robot-assisted surgical platforms over the past decades [2-4]. 在过去的几十年里，微创手术的操作挑战刺激了机器人辅助手术平台的发展 [2-4]。

While most of the existing robotic surgical platforms use articulated structures, continuum mechanisms provide surgeons with alternative choices [4]. 虽然现有的大多数机器人手术平台都使用关节式结构，但连续体机构为外科医生提供了另一种选择 [4]。

Continuum structures are adopted in the designs of several surgical manipulators [5-10] for distal dexterity and inherent safety. 为了实现远端的灵活性和固有的安全性，一些手术机械臂的设计中采用了连续体结构 [5-10]。

Benefiting from the design compactness, structural compliance, proximal actuation scheme, and potentials in miniaturization, continuum surgical robots are particularly suited for SPL (Single Port Laparoscopy) and NOTES (Natural Orifice Transluminal Endoscopic Surgery). 得益于设计的紧凑性、结构的柔顺性、近端驱动方案以及在微型化方面的潜力，连续体手术机器人特别适用于单孔腹腔镜手术（SPL）和经自然腔道内镜手术（NOTES）。

A multi-segment continuum surgical manipulator is usually fully inserted into a patient’s organ cavity for teleoperation afterwards. 多段连续体手术机械臂通常会完全插入患者器官腔内，随后再进行遥操作。

Under this working pattern, these multi-segment continuum manipulators suffer from an unreachable volume inside their workspace, as analyzed in [11]. 正如文献 [11] 中所分析的那样，在这种工作模式下，这些多段连续体机械臂的工作空间内会存在一个无法到达的盲区（不可达体积）。

Nevertheless, a multi-segment continuum manipulator is in fact capable of being tele-operated even in a partially inserted configuration. 尽管如此，实际上多段连续体机械臂即使在部分插入的构型下也能够被遥操作。

Teleoperation in these partially inserted configurations can expand the manipulators’ kinematic performance, even though the mobility might be reduced in the partially inserted configurations. 虽然在部分插入的构型中机械臂的灵活性可能会下降，但在这些构型下进行遥操作可以扩展其运动学性能。

A kinematics framework for realizing the configuration transition during the gradual insertion of a continuum manipulator is yet missing. 目前尚缺乏一种能在连续体机械臂逐步插入过程中实现构型转换的运动学框架。

This paper hence proposes such a framework to handle the teleoperation and configuration transition when the continuum surgical manipulator is gradually inserted as shown in Fig. 1(a) to Fig. 1(d). 因此，本文提出了这样一个框架，用于处理连续体手术机械臂在逐步插入时的遥操作和构型转换，如图 1(a) 至图 1(d) 所示。

---

**Page 1**

> **图片说明及解析：** Fig. 1. Different configurations of the continuum surgical manipulator: (a) the 1st configuration with the segment #2 partially inserted, (b) the 2nd configuration with the rigid stem partially inserted, (c) the 3rd configuration with the segment #1 partially inserted, and (d) the 4th (fully inserted) configuration; (e) the manipulator's actuation scheme. 图 1. 连续体手术机械臂的不同构型：(a) 第1种构型，第2段（segment #2）部分插入；(b) 第2种构型，刚性杆（rigid stem）部分插入；(c) 第3种构型，第1段（segment #1）部分插入；(d) 第4种构型（完全插入）；(e) 机械臂的驱动方案。 **内容理解：** 图1清晰地展示了机械臂穿过穿刺鞘（Trocar）逐步进入体内的四种不同阶段（构型）。构型1只暴露了最前端的连续体段（#2）；构型2前端连续体段全部进入，中间一段刚性连接杆正穿过穿刺鞘；构型3刚性杆全部进入，近端连续体段（#1）部分穿出；构型4则是所有可动部分全部穿出穿刺鞘进入工作区域。图(e)则展示了体外庞大的控制基座，标注了负责整体送进（Feeding）、轴向旋转（Axial rotation）和控制各段弯曲的驱动单元（Actuation Unit）。

The manipulator in Fig. 1 has two inextensible continuum segments with a rigid stem in between. 图 1 中的机械臂拥有两个不可伸长的连续体段，它们之间由一根刚性杆连接。

The segment #1 is stacked on a base stem, as shown in Fig. 1(d). 第 1 段叠加在一根基座杆上，如图 1(d) 所示。

The rest configurations in Fig. 1(a) to Fig. 1(c) are introduced in detail in Section II. 图 1(a) 至图 1(c) 中的其余构型将在第二节中详细介绍。

The continuum surgical manipulator is mounted onto and actuated by an actuation unit, as shown in Fig. 1(e). 该连续体手术机械臂安装在一个驱动单元上并由其驱动，如图 1(e) 所示。

The actuation unit can realize rotation about its axis as well as be translated by a linear actuator for stem feeding. 该驱动单元可以绕其轴线旋转，也可以通过线性致动器进行平移以实现杆的送进。

A trocar is fixed at the distal end of the linear actuator and guides the insertion of the manipulator. 穿刺鞘固定在线性致动器的远端，用于引导机械臂的插入。

This paper is organized as follows. 本文的组织结构如下。

Section II defines the associated configurations of the manipulator. 第二节定义了机械臂的相关构型。

Kinematics of the manipulator in different configurations is derived in Section III, while the configuration transition strategies are proposed in Section IV. 第三节推导了机械臂在不同构型下的运动学，而第四节提出了构型转换策略。

Numerical and experimental validations are presented in Section V with the conclusions summarized in Section VI. 第五节给出了数值和实验验证，并在第六节中总结了结论。

## II.

CONFIGURATION DEFINITION

## 二、构型定义

The 2-segment continuum surgical manipulator in Fig. 1(d) includes two inextensible bending segments. 图 1(d) 中的两段式连续体手术机械臂包含两个不可伸长的弯曲段。

Each segment possesses two DoFs (Degrees of Freedom). 每个连续体段拥有两个自由度（DoFs）。

The base stem can be fed along and rotated about its axis. 基座杆可以沿着其轴线进行送进和旋转。

Thus the manipulator has 6 DoFs, excluding the actuation of its end effector (a gripper). 因此，在不包括其末端执行器（夹爪）的驱动下，该机械臂共有 6 个自由度。

When the manipulator is fully inserted as in Fig. 1(d), it will not be able to reach anywhere close to the trocar. 当机械臂如图 1(d) 所示完全插入时，它将无法触及穿刺鞘附近的任何位置。

Please refer to the manipulator’s workspace in Fig. 3(a), where an unreachable volume is shown. 请参阅图 3(a) 中机械臂的工作空间，其中显示了一个不可达体积。

This unreachable volume can clearly be reached, if the manipulator can be tele-operated when it is partially inserted as shown in Fig. 1(a). 很明显，如果机械臂能在如图 1(a) 所示的部分插入状态下进行遥操作，这个不可达体积是可以被触及的。

Then the insertion process leads to 4 configurations as follows. 插入过程随之会产生如下 4 种构型。

Please note, when the inextensible segment is being partially inserted, the inserted portion can still bend. 请注意，当不可伸长段被部分插入时，已插入的部分仍然可以弯曲。

While the insertion continues, the feeding motion is kinematically treated as the segment length changing. 在继续插入的同时，从运动学角度来看，送进运动被视为连续体段长度的变化。

In other words, when a segment is being partially inserted, it is treated as it possesses 3 DoFs: 2-DoF bending and 1-DoF length changing. 换言之，当一个连续体段正在被部分插入时，它被视为拥有 3 个自由度：2 个弯曲自由度和 1 个长度变化自由度。

- The $1^{st}$ Configuration (C1): only the segment #2 is being partially inserted as shown in Fig. 1(a).
    
- 第 1 种构型（C1）：如图 1(a) 所示，仅第 2 段处于部分插入状态。
    
- Here, the manipulator has 4 DoFs: axial rotation, length changing and 2-DoF bending of the segment #2.
    
- 在此状态下，机械臂具有 4 个自由度：轴向旋转、长度变化以及第 2 段的 2 个弯曲自由度。
    
- The $2^{nd}$ Configuration (C2): the rigid stem is being partially inserted as in Fig. 1(b).
    
- 第 2 种构型（C2）：如图 1(b) 所示，刚性杆处于部分插入状态。
    
- The manipulator still has 4 DoFs: axial rotation and feeding of the rigid stem, as well as 2-DoF bending of the segment #2.
    
- 机械臂依然具有 4 个自由度：刚性杆的轴向旋转和送进，以及第 2 段的 2 个弯曲自由度。
    
- The $3^{rd}$ Configuration (C3): the segment #1 is being inserted as in Fig. 1(c).
    
- 第 3 种构型（C3）：如图 1(c) 所示，第 1 段正在被插入。
    
- The manipulator has 6 DoFs: 2-DoF bending of the segment #2, as well as axial rotation, length changing and 2-DoF bending of the segment #1.
    
- 此时机械臂具有 6 个自由度：第 2 段的 2 个弯曲自由度，以及轴向旋转、长度变化和第 1 段的 2 个弯曲自由度。
    
- The $4^{th}$ Configuration (C4): the base stem is being fed as in Fig. 1(d).
    
- 第 4 种构型（C4）：如图 1(d) 所示，基座杆正在被送进。
    
- C4 is the configuration in which the manipulator is normally tele-operated.
    
- C4 是机械臂通常进行遥操作的构型。
    

Workspace of the manipulator in different configurations in Fig. 3(b) shows that the workspace in one configuration overlaps with that in its adjacent configuration(s). 图 3(b) 中机械臂在不同构型下的工作空间表明，一种构型下的工作空间与其相邻构型下的工作空间存在重叠。

This leads to the transition strategies between the configurations, which are detailed in Section IV, after the development of the kinematics in Section III. 这引出了构型之间的转换策略，我们将在第三节开展运动学推导之后，于第四节对其进行详细说明。

It is clear from Fig. 3 that, with the configuration transition control, the unreachable volume in the manipulator’s workspace is reduced, indicating better usage of the inherent motion capability of the continuum surgical manipulator. 从图 3 可以清楚地看到，通过构型转换控制，机械臂工作空间内的不可达体积得以减小，这表明连续体手术机械臂固有的运动能力得到了更好的利用。

## III.

KINEMATICS

## 三、运动学

The nomenclature and coordinates are defined in Section III.A, while the kinematics of a single segment is presented in Section III.B. 符号说明和坐标系在第三节 A 部分中定义，而单段的运动学在第三节 B 部分中给出。

Kinematics of the manipulator in various configurations is derived using that of a single segment as in Section III.C to Section III.G. 机械臂在各种构型下的运动学是利用单段运动学推导出来的，见第三节 C 部分至第三节 G 部分。

A. Nomenclature and Coordinates

### A. 符号说明与坐标系

The surgical manipulator includes two structurally similar continuum segments. 该手术机械臂包含两个结构相似的连续体段。

With the coordinates attachment for the $t^{th}$ segment in Fig. 2(a), the coordinate attachments for the entire manipulator is shown in Fig. 2(b). 图 2(a) 设定了第 $t$ 段的坐标系，基于此，整个机械臂的坐标系设定如图 2(b) 所示。

The definitions are as follows, while the nomenclature is defined in Table I. 相关定义如下，而符号说明则在表 I 中定义。

- The Base Ring Coordinate $\{tb\}=[\begin{matrix}\hat{x}_{tb}&\hat{y}_{tb}&\hat{z}_{tb}\end{matrix}]^{T}$ is attached to the base ring of the $t^{th}$ segment at the center.
    
- 基座环坐标系 $\{tb\}=[\begin{matrix}\hat{x}_{tb}&\hat{y}_{tb}&\hat{z}_{tb}\end{matrix}]^{T}$ 设定在第 $t$ 段基座环的中心位置。
    
- $\hat{z}_{tb}$ is perpendicular to the base ring and $\hat{x}_{tb}$ is oriented to the first backbone.
    
- $\hat{z}_{tb}$ 垂直于基座环，且 $\hat{x}_{tb}$ 指向第一根结构骨架（backbone）。
    
- Bending Plane Coordinate #1 $\{t1\}=[\begin{matrix}\hat{x}_{t1}&\hat{y}_{t1}&\hat{z}_{t1}\end{matrix}]^{T}$ shares its origin with $\{tb\}$.
    
- 弯曲平面坐标系 #1 $\{t1\}=[\begin{matrix}\hat{x}_{t1}&\hat{y}_{t1}&\hat{z}_{t1}\end{matrix}]^{T}$ 与 $\{tb\}$ 共用原点。
    
- Its XY plane is aligned with the bending plane of the $t^{th}$ segment.
    
- 其 XY 平面与第 $t$ 段的弯曲平面重合。
    
- Bending Plane Coordinate #2 $\{t2\}=[\begin{matrix}\hat{x}_{t2}&\hat{y}_{t2}&\hat{z}_{t2}\end{matrix}]^{T}$ is attached to the end ring of the $t^{th}$ segment at the ring center.
    
- 弯曲平面坐标系 #2 $\{t2\}=[\begin{matrix}\hat{x}_{t2}&\hat{y}_{t2}&\hat{z}_{t2}\end{matrix}]^{T}$ 设定在第 $t$ 段末端环的环心处。
    
- Its XY plane is aligned with the bending plane.
    
- 其 XY 平面同样与弯曲平面重合。
    
- End Ring Coordinate $\{te\}=[\begin{matrix}\hat{x}_{te}&\hat{y}_{te}&\hat{z}_{te}\end{matrix}]^{T}$ shares its origin with $\{t2\}$.
    
- 末端环坐标系 $\{te\}=[\begin{matrix}\hat{x}_{te}&\hat{y}_{te}&\hat{z}_{te}\end{matrix}]^{T}$ 与 $\{t2\}$ 共用原点。
    
- $\hat{z}_{te}$ is perpendicular to the end ring and $\hat{x}_{te}$ is oriented to the first backbone.
    
- $\hat{z}_{te}$ 垂直于末端环，且 $\hat{x}_{te}$ 指向第一根结构骨架。
    

---

**Page 2**

> **图片说明及解析：** Fig. 2. Coordinates attachement and nomenclature of (a) the $t^{th}$ segment and (b) the continuum surgical manipulator. 图 2. (a) 第 $t$ 段和 (b) 连续体手术机械臂的坐标系设定及术语说明。 **内容理解：** 该图详细展示了论文中所使用的运动学模型坐标系。图 (a) 放大了单个连续体段，标注了各个基准环（如基座环 Base ring、隔离环 Spacer ring、末端环 End ring）、骨架（Backbone）以及假想中心骨架（Imaginary central backbone），同时明确标出了该段的各级局部坐标轴（$\hat{x}_{tb}$, $\hat{y}_{tb}$, $\hat{z}_{tb}$等）及描述弯曲角度的参数（$\theta_t$, $\delta_t$）。图 (b) 则是将这些局部坐标系串联到了拥有两个连续体段的完整机械臂整体结构上，展示了空间坐标在各段及刚性杆传递的关系。图片上的中文手写笔记表明：“X-Y 面与弯曲平面(粉色)重合”，这对应于正文中所述的 `Bending Plane Coordinate` 设定逻辑。

TABLE I

NOMENCLATURE USED IN THE KINEMATICS MODEL

表 I

运动学模型中使用的符号说明

| **Symbol (符号)**                                                                                              | **Definition (定义)**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $t$                                                                                                          | Index of the segments. $t = 1, 2$.<br><br>  <br><br>连续体段的索引。$t = 1, 2$。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| $j$                                                                                                          | Index of the configurations. $j = 1, 2, 3, 4$.<br><br>  <br><br>构型的索引。$j = 1, 2, 3, 4$。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| $L_t, L_{t0}$                                                                                                | Inserted length and the full length of the imaginary central backbone of the $t^{th}$ segment.<br><br>  <br><br>第 $t$ 段假想中心骨架的已插入长度和总长度。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| $\theta_t$                                                                                                   | Rotation angle from $\hat{\mathbf{x}}_{t1}$ to $\hat{\mathbf{x}}_{t2}$ about $\hat{\mathbf{z}}_{t1}$.<br><br>  <br><br>绕 $\hat{\mathbf{z}}_{t1}$ 轴从 $\hat{\mathbf{x}}_{t1}$ 到 $\hat{\mathbf{x}}_{t2}$ 的旋转角。                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| $\delta_t$                                                                                                   | Rotation angle from $\hat{\mathbf{y}}_{t1}$ to $\hat{\mathbf{x}}_{tb}$ along $\hat{\mathbf{z}}_{tb}$.<br><br>  <br><br>沿 $\hat{\mathbf{z}}_{tb}$ 轴从 $\hat{\mathbf{y}}_{t1}$ 到 $\hat{\mathbf{x}}_{tb}$ 的旋转角。                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| $\mathbf{v}_{(t)}, \bm{\omega}_{(t)}$                                                                        | Tip velocity and angular velocity of the $t^{th}$ segment.<br><br>  <br><br>第 $t$ 段的末端速度和角速度。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| $\bm{\psi}_{(t2)}, \bm{\psi}_{(t3)}$                                                                         | Configuration vectors of the $t^{th}$ continuum segment. $\bm{\psi}_{(t2)} = [\theta_t, \delta_t]^T$ when the segment is fully inserted (2-DoF segment), and $\bm{\psi}_{(t3)} = [\theta_t, L_t, \delta_t]^T$ when it is being inserted (3-DoF segment).<br><br>  <br><br>第 $t$ 个连续体段的构型向量。当该段完全插入时（2 自由度段）$\bm{\psi}_{(t2)} = [\theta_t, \delta_t]^T$，当其正在被插入时（3 自由度段）$\bm{\psi}_{(t3)} = [\theta_t, L_t, \delta_t]^T$。                                                                                                                                                                                                                                                           |
| $\mathbf{J}_{(t2)}, \mathbf{J}_{(t3)}$                                                                       | Jacobian matrices of the $t^{th}$ segment when it is fully or partially inserted, respectively.<br><br>  <br><br>第 $t$ 段在完全插入或部分插入时的雅可比矩阵。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| $\mathbf{J}_{(tv2)}, \mathbf{J}_{(t\omega2)}$<br><br>  <br><br>$\mathbf{J}_{(tv3)}, \mathbf{J}_{(t\omega3)}$ | Jacobian matrices of the linear and angular velocities of the $t^{th}$ segment. $\mathbf{v}_{(t)} = \mathbf{J}_{(tv2)}\dot{\bm{\psi}}_{(t2)}$ or $\mathbf{v}_{(t)} = \mathbf{J}_{(tv3)}\dot{\bm{\psi}}_{(t3)}$, while $\bm{\omega}_{(t)} = \mathbf{J}_{(t\omega2)}\dot{\bm{\psi}}_{(t2)}$ or $\bm{\omega}_{(t)} = \mathbf{J}_{(t\omega3)}\dot{\bm{\psi}}_{(t3)}$.<br><br>  <br><br>第 $t$ 段线速度和角速度的雅可比矩阵。$\mathbf{v}_{(t)} = \mathbf{J}_{(tv2)}\dot{\bm{\psi}}_{(t2)}$ 或 $\mathbf{v}_{(t)} = \mathbf{J}_{(tv3)}\dot{\bm{\psi}}_{(t3)}$，且 $\bm{\omega}_{(t)} = \mathbf{J}_{(t\omega2)}\dot{\bm{\psi}}_{(t2)}$ 或 $\bm{\omega}_{(t)} = \mathbf{J}_{(t\omega3)}\dot{\bm{\psi}}_{(t3)}$。 |
| $L_r, L_{r0}$                                                                                                | Inserted length and the full length of the rigid stem.<br><br>  <br><br>刚性连接杆的已插入长度和总长度。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| $L_s, L_{s0}$                                                                                                | Inserted length and the full length of the base stem.<br><br>  <br><br>基座杆的已插入长度和总长度。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| $\varphi$                                                                                                    | Axial rotation realized by the actuation unit.<br><br>  <br><br>由驱动单元实现的轴向旋转。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| $\bm{\psi}_{(r)}, \bm{\psi}_{(s)}$                                                                           | Configuration vectors of the rigid stem and the base stem: $\bm{\psi}_{(r)} = [\varphi \quad L_r]^T$ and $\bm{\psi}_{(s)} = [\varphi \quad L_s]^T$.<br><br>  <br><br>刚性连接杆和基座杆的构型向量：$\bm{\psi}_{(r)} = [\varphi \quad L_r]^T$ 且 $\bm{\psi}_{(s)} = [\varphi \quad L_s]^T$。                                                                                                                                                                                                                                                                                                                                                                                                         |
| $\bm{\psi}_j$                                                                                                | The configuration vector of the manipulator in the $j^{th}$ configuration.<br><br>  <br><br>机械臂在第 $j$ 种构型下的构型向量。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| $\mathbf{J}_j$                                                                                               | The manipulator's Jacobian matrix in its $j^{th}$ configuration.<br><br>  <br><br>机械臂在第 $j$ 种构型下的雅可比矩阵。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| $\mathbf{J}_{jv}, \mathbf{J}_{j\omega}$                                                                      | Jacobian matrices of the tip velocity and angular velocity of the manipulator's end effector in the $j^{th}$ configuration.<br><br>  <br><br>在第 $j$ 种构型下，机械臂末端执行器线速度和角速度的雅可比矩阵。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |

- The World Coordinate $\{w\}=[\hat{\mathbf{x}}_w \quad \hat{\mathbf{y}}_w \quad \hat{\mathbf{z}}_w]^T$ is attached to the trocar and $\hat{\mathbf{z}}_w$ is aligned with the base stem’s axis.
    
- 世界坐标系 $\{w\}=[\hat{\mathbf{x}}_w \quad \hat{\mathbf{y}}_w \quad \hat{\mathbf{z}}_w]^T$ 固定在穿刺鞘上，且 $\hat{\mathbf{z}}_w$ 与基座杆的轴线重合。
    

### _B. Kinematics of the $t^{th}$ Segment_

### _B. 第 $t$ 段的运动学_

A single segment includes of a base ring, an end ring, several spacer rings and several backbones.

单个连续体段包括一个基座环、一个末端环、几个隔离环以及几根结构骨架。

The backbones are attached to the end ring. Pulling and pushing the backbones bends the segment, while the segment can also be entirely fed forward or drawn backwards.

结构骨架连接在末端环上。推拉结构骨架可以使该连续体段弯曲，同时该段也可以被整体向前送进或向后拉回。

An imaginary central backbone characterizes the shape and length of the segment.

一根假想的中心骨架表征了该段的形状和长度。

Following the widely adopted constant curvature bending assumption summarized in [12], kinematics of the $t^{th}$ segment can be derived. The homogenous transformation matrix relating $\{te\}$ and $\{tb\}$ is given as follows.

基于文献 [12] 中总结的被广泛采用的常曲率弯曲假设，可以推导出第 $t$ 段的运动学。关联 $\{te\}$ 和 $\{tb\}$ 的齐次变换矩阵如下所示。

$$^{tb}\mathbf{T}_{te} = \begin{bmatrix} ^{tb}\mathbf{R}_{te} & ^{tb}\mathbf{p}_{te} \\ \mathbf{0}_{1 \times 3} & 1 \end{bmatrix}$$

(1)

Where the expressions of $^{tb}\mathbf{R}_{te}$ and $^{tb}\mathbf{p}_{te}$, involving $L_t$, $\theta_t$ and $\delta_t$, are detailed in [13].

其中 $^{tb}\mathbf{R}_{te}$ 和 $^{tb}\mathbf{p}_{te}$ 的表达式（涉及 $L_t$、$\theta_t$ 和 $\delta_t$）在文献 [13] 中有详细说明。

A single segment is a 3-DoF structure during insertion and a 2-DoF structure once it is fully inserted. The instantaneous kinematics then has two sets of expressions as in (2). Derivation details can be referred to [9, 14].

单个连续体段在插入过程中是一个 3 自由度结构，一旦完全插入则变为 2 自由度结构。因此其瞬态运动学有两套表达式，如公式 (2) 所示。推导细节可参考文献 [9, 14]。

$$\dot{\mathbf{x}}_t = \begin{bmatrix} \mathbf{v}_{(t)} \\ \bm{\omega}_{(t)} \end{bmatrix} = \begin{cases} \mathbf{J}_{(tv3)}\dot{\bm{\psi}}_{(t3)} = \begin{bmatrix} \mathbf{J}_{(tv3)} \\ \mathbf{J}_{(t\omega3)} \end{bmatrix} \begin{bmatrix} \dot{\theta}_t & \dot{L}_t & \dot{\delta}_t \end{bmatrix}^T \\ \mathbf{J}_{(tv2)}\dot{\bm{\psi}}_{(t2)} = \begin{bmatrix} \mathbf{J}_{(tv2)} \\ \mathbf{J}_{(t\omega2)} \end{bmatrix} \begin{bmatrix} \dot{\theta}_t & \dot{\delta}_t \end{bmatrix}^T \end{cases}$$

(2)

$$\mathbf{J}_{(tv3)} = \begin{bmatrix} \cos\delta_t \frac{L_t}{\theta_t}(\sin\theta_t - h(\theta_t)) & \cos\delta_t h(\theta_t) & -L_t\sin\delta_t h(\theta_t) \\ -\sin\delta_t \frac{L_t}{\theta_t}(\sin\theta_t - h(\theta_t)) & -\sin\delta_t h(\theta_t) & -L_t\cos\delta_t h(\theta_t) \\ \frac{L_t}{\theta_t}(\cos\theta_t - \frac{\sin\theta_t}{\theta_t}) & \frac{\sin\theta_t}{\theta_t} & 0 \end{bmatrix}$$

(3)

$$\mathbf{J}_{(t\omega3)} = \begin{bmatrix} \sin\delta_t & 0 & \cos\delta_t \sin\theta_t \\ \cos\delta_t & 0 & -\sin\delta_t \sin\theta_t \\ 0 & 0 & \cos\theta_t - 1 \end{bmatrix}$$

(4)

Where $h(\theta_t) = (1 - \cos\theta_t) / \theta_t$.

其中 $h(\theta_t) = (1 - \cos\theta_t) / \theta_t$。

$$\mathbf{J}_{(tv2)} = \begin{bmatrix} \mathbf{J}_{(tv2)}(:,1) & \mathbf{J}_{(tv2)}(:,3) \end{bmatrix}$$

(5)

$$\mathbf{J}_{(t\omega2)} = \begin{bmatrix} \mathbf{J}_{(t\omega2)}(:,1) & \mathbf{J}_{(t\omega2)}(:,3) \end{bmatrix}$$

(6)

> **批注说明**：你在图片上圈出了公式 (5) 和 (6) 中的列切片提取逻辑，并备注了“_切片_”和“_等式右侧下角标应该是3的_”。你的理解非常准确，原论文印刷时这里出现了笔误。由于这是从完整的 3-DoF 雅可比矩阵降维到 2-DoF，等式右侧本应是提取自 $\mathbf{J}_{(tv3)}$ 和 $\mathbf{J}_{(t\omega3)}$ 的第 1 列和第 3 列（对应弯曲变量 $\theta_t$ 和 $\delta_t$ 的偏导数）。我在翻译和复刻公式时保留了原论文的样貌，但特此确认你的纠错是完全正确的。

> **图片说明及解析：**
> 
> Fig. 3. Translational workspace of the continuum surgical manipulator: (a) in Configuration C4, and (b) in Configurations C1 to C4.
> 
> 图 3. 连续体手术机械臂的平移工作空间：(a) 在构型 C4 下，以及 (b) 在构型 C1 至 C4 下。
> 
> **内容理解：** 该图展示了机械臂在不同阶段能够触及的物理空间范围（点云）。图(a)说明了如果仅在机械臂完全插入后（C4构型）才开始控制，其工作空间虽然很大（灰色部分），但底部穿刺鞘附近存在一个巨大的内部空洞（Unreachable volume，不可达体积）。图(b)则展示了如果将逐渐插入的四个阶段（C1到C4）结合起来控制，C1（红色）、C2（黄色）、C3（青色）的工作空间能够完美填补和重叠，显著缩减了先前的盲区。

### _C. Configuration Vectors_

### _C. 构型向量_

Each configuration vector $\bm{\psi}_j$ consists of the variables that fully describe the manipulator in the corresponding configuration.

每个构型向量 $\bm{\psi}_j$ 由能够完全描述机械臂在相应构型下状态的变量组成。

A parameter may act as a variable in one configuration while becomes constant in other configurations. The variables and constants in the different configurations are summarized in Fig. 4.

某个参数可能在一种构型中作为变量，而在其他构型中变为常量。不同构型下的变量和常量总结在图 4 中。

The only length variable in each configuration vector is the length of the segment or stem being inserted. The lengths of the fully inserted segments and stems become constants.

每个构型向量中唯一的长度变量是正在被插入的连续体段或连接杆的长度。完全插入的连续体段和连接杆的长度则变为常量。

The configuration vectors are defined as follows. $\bm{\psi}_{(t)}$ is used to express $\bm{\psi}_j$ to facilitate expressing the Jacobian matrices.

构型向量定义如下。使用 $\bm{\psi}_{(t)}$ 来表达 $\bm{\psi}_j$，是为了便于表达雅可比矩阵。

- The configuration vector of the manipulator in C4 is:
    
- 机械臂在 C4 构型下的构型向量为：
    
    $$\bm{\psi}_4 = [\varphi \quad L_s \quad \theta_1 \quad \delta_1 \quad \theta_2 \quad \delta_2]^T = [\bm{\psi}_{(s)}^T \quad \bm{\psi}_{(12)}^T \quad \bm{\psi}_{(22)}^T]^T$$
    
    (7)
    
- The configuration vector of the manipulator in C3 is:
    
- 机械臂在 C3 构型下的构型向量为：
    
    $$\bm{\psi}_3 = [\varphi \quad \theta_1 \quad L_1 \quad \delta_1 \quad \theta_2 \quad \delta_2]^T = [\varphi \quad \bm{\psi}_{(13)}^T \quad \bm{\psi}_{(22)}^T]^T$$
    
    (8)
    
- The manipulator's configuration vector in C2 is:
    
- 机械臂在 C2 构型下的构型向量为：
    
    $$\bm{\psi}_2 = [\varphi \quad L_r \quad \theta_2 \quad \delta_2]^T = [\bm{\psi}_{(r)}^T \quad \bm{\psi}_{(22)}^T]^T$$
    
    (9)
    
- The manipulator's configuration vector in C1 is:
    
- 机械臂在 C1 构型下的构型向量为：
    
    $$\bm{\psi}_1 = [\varphi \quad \theta_2 \quad L_2 \quad \delta_2]^T = [\varphi \quad \bm{\psi}_{(23)}^T]^T$$
    
    (10)
    

---

**Page 3**

### _D. Kinematics in the $4^{th}$ Configuration_

### _D. 第 4 种构型下的运动学_

The coordinate system attachment of the manipulator in C4 is shown in Fig. 2(b).

机械臂在 C4 下的坐标系设定如图 2(b) 所示。

- The base stem of the manipulator can be rotated and fed along $\hat{\mathbf{z}}_w$ in $\{w\}$. $\bm{\psi}_{(s)} = [\varphi \quad L_s]^T$ parameterizes the rotation and the feeding accordingly.
    
- 机械臂的基座杆可以在 $\{w\}$ 坐标系中沿着 $\hat{\mathbf{z}}_w$ 旋转和送进。$\bm{\psi}_{(s)} = [\varphi \quad L_s]^T$ 对其旋转和送进运动进行了相应的参数化。
    
- The continuum segment #1 is stacked on the base stem. Thus $\{1b\}$ is obtained by rotating $\{w\}$ by $\varphi$ about $\hat{\mathbf{z}}_s$ and translating $\{w\}$ by a distance of $L_s$ along $\hat{\mathbf{z}}_w$.
    
- 第 1 段连续体叠加在基座杆上。因此，将 $\{w\}$ 绕 $\hat{\mathbf{z}}_s$ 旋转 $\varphi$ 角度，并沿 $\hat{\mathbf{z}}_w$ 平移 $L_s$ 距离，即可得到 $\{1b\}$ 坐标系。
    

> **批注说明**：你在“$\hat{\mathbf{z}}_s$”旁边标注了“_是Z_w_”。考虑到前文明确提到基座杆是绕 $\{w\}$ 坐标系中的 $\hat{\mathbf{z}}_w$ 旋转，且上一个项目符号也是这样写的，这里出现 $\hat{\mathbf{z}}_s$ 确实是论文的又一处行文笔误。我按照你的批注理解保留了翻译。

- The continuum segment #2 and the segment #1 are connected by a rigid stem. $\{2b\}$ is obtained from $\{1e\}$ by a translation of distance $L_{r0}$ along $\hat{\mathbf{z}}_{1e}$.
    
- 第 2 段连续体和第 1 段连续体由刚性杆连接。将 $\{1e\}$ 沿 $\hat{\mathbf{z}}_{1e}$ 平移 $L_{r0}$ 的距离即可得到 $\{2b\}$ 坐标系。
    

The corresponding homogenous transformation matrix for the tip pose of the manipulator in C4 is:

机械臂在 C4 构型下末端位姿对应的齐次变换矩阵为：

$$^w\mathbf{T}_{2e} = ^w\mathbf{T}_{1b} ^{1b}\mathbf{T}_{1e} ^{1e}\mathbf{T}_{2b} ^{2b}\mathbf{T}_{2e}$$

(11)

The instantaneous kinematics is derived as follows.

瞬态运动学推导如下。

$$\dot{\mathbf{x}} = \mathbf{J}_4 \dot{\bm{\psi}}_4$$

(12)

$$\mathbf{J}_4 = \begin{bmatrix} -\left[ ^w\mathbf{p}_{2e}^{1b} \times \right] \cdot \hat{\mathbf{z}}_w & \hat{\mathbf{z}}_w & ^w\mathbf{R}_{1b} \mathbf{W}_1 & ^w\mathbf{R}_{2b} \mathbf{J}_{(tv2)} \\ \hat{\mathbf{z}}_w & \mathbf{0}_{3 \times 1} & ^w\mathbf{R}_{1b} \mathbf{J}_{(1\omega2)} & ^w\mathbf{R}_{2b} \mathbf{J}_{(2\omega2)} \end{bmatrix}$$

(13)

Where $\mathbf{W}_1 = -\left[ ^{1b}\mathbf{p}_{2e}^{1e} \times \right] \cdot \mathbf{J}_{(1\omega2)} + \mathbf{J}_{(1v2)}$, $^A\mathbf{p}_C^B$ is the position vector from the origin of frame B to the origin of frame C, expressed in frame A; $[\mathbf{p} \times]$ is the skew-symmetric matrix of a vector $\mathbf{p}$; $\mathbf{J}_{(tv2)}$ and $\mathbf{J}_{(t\omega2)}$ are obtained from (5) and (6).

其中 $\mathbf{W}_1 = -\left[ ^{1b}\mathbf{p}_{2e}^{1e} \times \right] \cdot \mathbf{J}_{(1\omega2)} + \mathbf{J}_{(1v2)}$，$^A\mathbf{p}_C^B$ 是从坐标系 B 的原点指向坐标系 C 原点的位置向量，且在坐标系 A 中表达；$[\mathbf{p} \times]$ 是向量 $\mathbf{p}$ 的反对称矩阵；$\mathbf{J}_{(tv2)}$ 和 $\mathbf{J}_{(t\omega2)}$ 由公式 (5) 和 (6) 求得。

### _E. Kinematics in the $3^{rd}$ Configuration_

### _E. 第 3 种构型下的运动学_

The manipulator in C3 is shown in Fig. 1(c). With the segment #1 partially constrained by the trocar, the Base Ring Coordinate $\{1b\}$ of the 3-DoF segment #1 is obtained from $\{w\}$ only by a rotation of $\varphi$ about $\hat{\mathbf{z}}_w$.

处于 C3 构型的机械臂如图 1(c) 所示。由于第 1 段受到穿刺鞘的部分约束，3 自由度的第 1 段的基座环坐标系 $\{1b\}$ 仅需将 $\{w\}$ 绕 $\hat{\mathbf{z}}_w$ 旋转 $\varphi$ 角度即可得到。

The instantaneous kinematics is derived as follows.

瞬态运动学推导如下。

$$\dot{\mathbf{x}} = \mathbf{J}_3 \dot{\bm{\psi}}_3$$

(14)

$$\mathbf{J}_3 = \begin{bmatrix} -\left[ ^w\mathbf{p}_{2e}^{1b} \times \right] \cdot \hat{\mathbf{z}}_w & ^w\mathbf{R}_{1b} \mathbf{W}_2 & ^w\mathbf{R}_{2b} \mathbf{J}_{(tv2)} \\ \hat{\mathbf{z}}_w & ^w\mathbf{R}_{1b} \mathbf{J}_{(1\omega3)} & ^w\mathbf{R}_{2b} \mathbf{J}_{(2\omega2)} \end{bmatrix}$$

(15)

Where $\mathbf{W}_2 = -\left[ ^{1b}\mathbf{p}_{2e}^{1e} \times \right] \cdot \mathbf{J}_{(1\omega3)} + \mathbf{J}_{(1v3)}$; $\mathbf{J}_{(1v3)}$ and $\mathbf{J}_{(1\omega3)}$ are from (3) and (4).

其中 $\mathbf{W}_2 = -\left[ ^{1b}\mathbf{p}_{2e}^{1e} \times \right] \cdot \mathbf{J}_{(1\omega3)} + \mathbf{J}_{(1v3)}$；$\mathbf{J}_{(1v3)}$ 和 $\mathbf{J}_{(1\omega3)}$ 来自公式 (3) 和 (4)。

### _F. Kinematics in the $2^{nd}$ Configuration_

### _F. 第 2 种构型下的运动学_

In C2, the 2-DoF segment #2 is the only continuum segment and the manipulator now has only 4 DoFs.

在 C2 构型中，2 自由度的第 2 段是唯一的连续体段，此时机械臂仅具有 4 个自由度。

The homogenous transformation matrix for the tip pose of the manipulator in C2 is as follows.

机械臂在 C2 构型下末端位姿的齐次变换矩阵如下。

$$^w\mathbf{T}_{2e} = ^w\mathbf{T}_{2b} ^{2b}\mathbf{T}_{2e}$$

(16)

In C2, $\{2b\}$ is obtained from $\{w\}$ by a rotation of $\varphi$ about $\hat{\mathbf{z}}_w$ and a translation of $L_r$ along $\hat{\mathbf{z}}_w$.

在 C2 构型中，将 $\{w\}$ 绕 $\hat{\mathbf{z}}_w$ 旋转 $\varphi$ 角度，并沿 $\hat{\mathbf{z}}_w$ 平移 $L_r$ 的距离，即可得到 $\{2b\}$。

The instantaneous kinematics is derived as follows.

瞬态运动学推导如下。

$$\dot{\mathbf{x}} = \mathbf{J}_2 \dot{\bm{\psi}}_2$$

(17)

$$\mathbf{J}_2 = \begin{bmatrix} -\left[ ^w\mathbf{p}_{2e}^{1b} \times \right] \cdot \hat{\mathbf{z}}_w & \hat{\mathbf{z}}_w & ^w\mathbf{R}_{2b} \mathbf{J}_{(tv2)} \\ \hat{\mathbf{z}}_w & \mathbf{0}_{3 \times 1} & ^w\mathbf{R}_{2b} \mathbf{J}_{(2\omega2)} \end{bmatrix}$$

(18)

Where $\mathbf{J}_{(2v2)}$ and $\mathbf{J}_{(2\omega2)}$ are obtained from (5) and (6).

其中 $\mathbf{J}_{(2v2)}$ 和 $\mathbf{J}_{(2\omega2)}$ 由公式 (5) 和 (6) 求得。

### _G. Kinematics in the $1^{st}$ Configuration_

### _G. 第 1 种构型下的运动学_

In the C1 configuration of the continuum manipulator, $\{2b\}$ is obtained from $\{w\}$ only by a rotation of $\varphi$ about $\hat{\mathbf{z}}_w$.

在连续体机械臂的 C1 构型中，$\{2b\}$ 仅需通过将 $\{w\}$ 绕 $\hat{\mathbf{z}}_w$ 旋转 $\varphi$ 角度得到。

The instantaneous kinematics is derived as follows.

瞬态运动学推导如下。

$$\dot{\mathbf{x}} = \mathbf{J}_1 \dot{\bm{\psi}}_1$$

(19)

$$\mathbf{J}_1 = \begin{bmatrix} -\left[ ^{1b}\mathbf{p}_{2e}^{1b} \times \right] \cdot \hat{\mathbf{z}}_w & ^w\mathbf{R}_{2b} \mathbf{J}_{(2v3)} \\ \hat{\mathbf{z}}_w & ^w\mathbf{R}_{2b} \mathbf{J}_{(2\omega3)} \end{bmatrix}$$

(20)

Where $\mathbf{J}_{(2v3)}$ and $\mathbf{J}_{(2\omega3)}$ are from (3) and (4).

其中 $\mathbf{J}_{(2v3)}$ 和 $\mathbf{J}_{(2\omega3)}$ 来自公式 (3) 和 (4)。

## IV. CONFIGURATION TRANSITION

## 四、构型转换

The configuration vectors of the manipulator vary in these different configurations. Thus, the Jacobian-based inverse kinematics approach cannot be directly used to tele-operate the manipulator from a pose in C4 to another pose in C2.

机械臂的构型向量在这些不同的构型中是不断变化的。因此，基于雅可比矩阵的逆运动学方法无法直接用于将机械臂从 C4 中的某个位姿遥操作到 C2 中的另一个位姿。

A kinematics framework for configuration transition control is hence proposed, including i) a prioritized-Jacobian formulation, ii) a prediction-based constraint imposition, and iii) a set of configuration transition strategies.

因此，本文提出了一种用于构型转换控制的运动学框架，包括 i) 优先级雅可比公式化（prioritized-Jacobian formulation），ii) 基于预测的约束施加（prediction-based constraint imposition），以及 iii) 一组构型转换策略。

### _A. Prioritized-Jacobian Formulation_

### _A. 优先级雅可比公式化_

In Configuration C1 and C2, the manipulator only possesses 4 DoFs and it will not always be able to reach a desired position and orientation at the same time.

在构型 C1 和 C2 中，机械臂仅拥有 4 个自由度，因此它无法总是同时达到期望的位置和姿态。

During teleoperation in a surgical procedure, the desired position is considered more important for the surgical manipulator to follow, while orientation errors of the surgical end effector may be tolerated if limited by the kinematic capability of the manipulator.

在外科手术过程的遥操作中，手术机械臂跟随期望位置被认为是更为重要的，而如果受限于机械臂自身的运动学能力，手术末端执行器的姿态误差是可以被容忍的。

The inverse kinematics is hence written using a prioritized-Jacobian formulation in (21) with the derivation details and explanations available in [15], assigning higher priority to the desired linear velocity.

因此，逆运动学采用了公式 (21) 中的优先级雅可比公式来编写，其推导细节和相关解释可见文献 [15]，该公式为期望的线速度分配了更高的优先级。

$$\dot{\bm{\psi}}_j = \mathbf{J}_{jv}^+ \mathbf{v} + (\mathbf{I} - \mathbf{J}_{jv}^+ \mathbf{J}_{jv}) [\mathbf{J}_{j\omega} (\mathbf{I} - \mathbf{J}_{jv}^+ \mathbf{J}_{jv})]^+ (\bm{\omega} - \mathbf{J}_{j\omega} \mathbf{J}_{jv}^+ \mathbf{v})$$

(21)

Where $\mathbf{J}_{jv}$ and $\mathbf{J}_{j\omega}$ are from $\mathbf{J}_j = [\mathbf{J}_{jv}^T \quad \mathbf{J}_{j\omega}^T]^T$ in the $j^{th}$ configuration; and $\mathbf{M}^+$ is the pseudoinverse of a matrix $\mathbf{M}$. To maintain the numerical stability, the damped least-squares formulation for the singularity robust $\mathbf{M}^+$ is used.

其中 $\mathbf{J}_{jv}$ 和 $\mathbf{J}_{j\omega}$ 来自第 $j$ 种构型下的雅可比矩阵分块 $\mathbf{J}_j = [\mathbf{J}_{jv}^T \quad \mathbf{J}_{j\omega}^T]^T$；且 $\mathbf{M}^+$ 是矩阵 $\mathbf{M}$ 的伪逆矩阵。为了保持数值稳定性，这里使用了奇异鲁棒阻尼最小二乘公式来求解 $\mathbf{M}^+$。

### _B. Prediction-Based Constraint Imposition_

### _B. 基于预测的约束施加_

The continuum segment should always be subject to a maximal bending curvature constraint, in order to prevent the structure from being damaged by excessive bending.

为了防止结构因过度弯曲而受损，连续体段应始终受限于最大弯曲曲率约束。

During the Jacobian-based inverse kinematics, every time the configuration vector is updated, the updated $L_t$ or $\theta_t$ values may violate the segment’s bending constraint. Then there are two ways to adjust the $L_t$ or the $\theta_t$ values:

在基于雅可比矩阵的逆运动学计算过程中，每次更新构型向量时，更新后的 $L_t$ 或 $\theta_t$ 值都可能违反连续体段的弯曲约束。此时有两种方法来调整 $L_t$ 或 $\theta_t$ 的值：

- In the _bending-first imposition_, to maintain the updated bending angle $\theta_t$ value, $L_t$ should be adjusted as follows.
    
- 在**优先弯曲约束施加**（bending-first imposition）策略中，为了保持更新后的弯曲角度 $\theta_t$ 值，$L_t$ 应按如下方式调整：
    
    $$L_t = \max(L_t, \theta_t / \kappa_{t0})$$
    
    (22)
    
- In the _length-first imposition_, to maintain the updated length $L_t$ value, $\theta_t$ should be adjusted as follows.
    
- 在**优先长度约束施加**（length-first imposition）策略中，为了保持更新后的长度 $L_t$ 值，$\theta_t$ 应按如下方式调整：
    
    $$\theta_t = \min(\theta_t, \kappa_{t0} \cdot L_t)$$
    
    (23)
    

Where $\kappa_{t0}$ is the maximal bending curvature of the $t^{th}$ segment.

其中 $\kappa_{t0}$ 是第 $t$ 段的最大弯曲曲率。

The prediction-based constraint imposition during a segment’s length change is conducted as follows.

在连续体段长度变化期间，基于预测的约束施加执行如下。

Both the _bending-first imposition_ and the _length-first imposition_ will be carried out.

**优先弯曲约束施加**（bending-first imposition）和**优先长度约束施加**（length-first imposition）都将被执行。

Then two sets of updated configuration vectors will be obtained.

然后将获得两组更新后的构型向量。

The set with smaller position error after the configuration vector update will be adopted.

构型向量更新后位置误差较小的一组将被采用。

### _C. Configuration Transition Strategies_

### _C. 构型转换策略_

When the manipulator is to reach a desired pose in another configuration, configuration vectors should be updated following a set of predefined triggers.

当机械臂要到达另一种构型下的期望位姿时，应按照一组预定义的触发条件来更新构型向量。

From Fig. 1, it is clear that a configuration change is triggered by insertion of the manipulator.

从图 1 可以清楚地看出，构型变化是由机械臂的插入引起的。

Before and after the configuration transition, the manipulator shall have a shared configuration variable.

在构型转换前后，机械臂应具有一个共享的构型变量。

The triggering variables for the configuration transition are hence summarized in Fig. 4, while the strategies are explained as follows.

因此，构型转换的触发变量总结在图 4 中，而具体策略解释如下。

- From C1 to C2: the manipulator configuration with $L_2 = L_{20}$ in C1 is identical to that with $L_r = 0$ in C2.
    
- 从 C1 到 C2：机械臂在 C1 中 $L_2 = L_{20}$ 时的构型与在 C2 中 $L_r = 0$ 时的构型完全相同。
    
- Thus the transition is triggered when $L_2$ is to be updated longer than $L_{20}$.
    
- 因此，当 $L_2$ 将被更新为大于 $L_{20}$ 的值时，就会触发转换。
    
- After the transition, the overshoot of $L_2$ (excess value after one update) in C1 is set to $L_r$ in C2:
    
- 转换后，C1 中 $L_2$ 的超调量（一次更新后的超出值）被设定为 C2 中的 $L_r$：
    
    $$(L_r)_{C2} = (L_2)_{C1} - L_{20}$$
    
    (24)
    
- The values of the rest variables ($\varphi$, $\theta_2$ and $\delta_2$) are inherited directly from C1 to C2.
    
- 其余变量（$\varphi$、$\theta_2$ 和 $\delta_2$）的值直接从 C1 继承到 C2。
    
- From C2 to C1: the transition is triggered when $L_r$ is to be updated less than 0.
    
- 从 C2 到 C1：当 $L_r$ 将被更新为小于 0 的值时，触发转换。
    
- After the transition, the overshoot of $L_r$ (of a negative value) in C2 is set to $L_2$ in C1:
    
- 转换后，C2 中 $L_r$ 的超调量（一个负值）被设定为 C1 中的 $L_2$：
    
    $$(L_2)_{C1} = (L_r)_{C2} + L_{20}$$
    
    (25)
    
- Other variable values are directly inherited.
    
- 其他变量的值被直接继承。
    
- From C2 to C3: the manipulator configuration with $L_r = L_{r0}$ in C2 is identical to that with $L_1 = 0$ in C3.
    
- 从 C2 到 C3：机械臂在 C2 中 $L_r = L_{r0}$ 时的构型与在 C3 中 $L_1 = 0$ 时的构型完全相同。
    
- The transition is triggered when $L_r$ is to be updated longer than $L_{r0}$.
    
- 当 $L_r$ 将被更新为大于 $L_{r0}$ 的值时，触发转换。
    
- After the transition, the overshoot of $L_r$ in C2 is set to $L_1$ in C3:
    
- 转换后，C2 中 $L_r$ 的超调量被设定为 C3 中的 $L_1$：
    
    $$(L_1)_{C3} = (L_r)_{C2} - L_{r0}$$
    
    (26)
    
- The variable values ($\varphi$, $\theta_2$ and $\delta_2$) are inherited from C2 and the $\theta_1$ and $\delta_1$ values are initialized as 0.
    
- 变量（$\varphi$、$\theta_2$ 和 $\delta_2$）的值从 C2 继承，而 $\theta_1$ 和 $\delta_1$ 的值被初始化为 0。
    
- From C3 to C2: the transition is triggered when $L_1$ is to be updated less than 0.
    
- 从 C3 到 C2：当 $L_1$ 将被更新为小于 0 的值时，触发转换。
    
- After the transition, the overshoot of $L_1$ (of a negative value) in C3 is set to $L_r$ in C2:
    
- 转换后，C3 中 $L_1$ 的超调量（一个负值）被设定为 C2 中的 $L_r$：
    
    $$(L_r)_{C2} = (L_1)_{C3} + L_{r0}$$
    
    (27)
    
- To be noted, the bending constraint imposition will assure that $\theta_2$ approaches zero before this transition.
    
- 需要注意的是，弯曲约束施加将确保在进行此转换之前 $\theta_2$ 接近于零。
    
- From C3 to C4: the manipulator configuration with $L_1 = L_{10}$ in C3 is identical to that with $L_s = 0$ in C4.
    
- 从 C3 到 C4：机械臂在 C3 中 $L_1 = L_{10}$ 时的构型与在 C4 中 $L_s = 0$ 时的构型完全相同。
    
- Thus the transition is triggered when $L_1$ is to be updated longer than $L_{10}$.
    
- 因此，当 $L_1$ 将被更新为大于 $L_{10}$ 的值时，触发转换。
    
- After the transition, the overshoot of $L_1$ in C3 is set to $L_s$ in C4:
    
- 转换后，C3 中 $L_1$ 的超调量被设定为 C4 中的 $L_s$：
    
    $$(L_s)_{C4} = (L_1)_{C3} - L_{10}$$
    
    (28)
    
- The $\varphi$, $\theta_1$, $\delta_1$, $\theta_2$ and $\delta_2$ variables are inherited directly.
    
- $\varphi$、$\theta_1$、$\delta_1$、$\theta_2$ 和 $\delta_2$ 变量被直接继承。
    
- From C4 to C3: the transition is triggered when $L_s$ is to be updated less than 0.
    
- 从 C4 到 C3：当 $L_s$ 将被更新为小于 0 的值时，触发转换。
    
- After the transition, the overshoot of $L_s$ (of a negative value) in C4 is set to $L_1$ in C3:
    
- 转换后，C4 中 $L_s$ 的超调量（一个负值）被设定为 C3 中的 $L_1$：
    
    $$(L_1)_{C3} = (L_s)_{C4} + L_{10}$$
    
    (29)
    
- The other variable values are inherited from C4.
    
- 其他变量的值从 C4 继承。
    

In summary, a transition is triggered by the overshoot of a length variable after an update.

综上所述，构型转换是由更新后某个长度变量的超调引起的。

The transition strategies are expressed in (24) to (29).

转换策略由公式 (24) 至 (29) 表达。

---

**Page 4**

> **图片说明及解析：**
> 
> Fig. 4. The variables and constants of the manipulator and the transition conditions between the different configurations.
> 
> 图 4. 机械臂的变量和常量以及不同构型之间的转换条件。
> 
> **内容理解：** 该图直观地展示了C1到C4构型中，哪些参数作为状态变量（绿色框内，如C1的$L_2$），哪些参数固定为常量（黄色框内，如C1的$L_s=0, L_1=0$等）。箭头及其上方的条件（例如 $L_2 > L_{20}$ 或者 $L_r < 0$）正是对应了正文中公式 (24) 到 (29) 描述的长度变量超调触发转换的逻辑。

## V. EXPERIMENTAL VALIDATION

## 五、实验验证

In order to verify the configuration transition control strategy, numerical simulations and experimental validations were carried out.

为了验证构型转换控制策略，本文进行了数值仿真和实验验证。

### _A. Simulation Cases_

### _A. 仿真案例_

The simulations were conducted using the Jacobian-based inverse kinematics approach with the configuration transition incorporated to drive the manipulator from one configuration to a desired pose in another configuration.

仿真实验采用了结合构型转换的、基于雅可比矩阵的逆运动学方法，驱动机械臂从一种构型运动到另一种构型下的期望位姿。

The process can be elaborated as follows, referring to Fig. 5.

参照图 5，该过程可以详细阐述如下。

From a given target pose, the position and orientation errors are first calculated.

首先根据给定的目标位姿计算位置和姿态误差。

Then the desired end effector twist $\dot{\mathbf{x}}$ is obtained from the position and orientation errors, while $\dot{\bm{\psi}}_j$ is obtained from (21).

然后由位置和姿态误差得到期望的末端执行器旋量 $\dot{\mathbf{x}}$，而 $\dot{\bm{\psi}}_j$ 则由公式 (21) 得到。

> **图片说明及解析：**
> 
> Fig. 5. The diagram of the Jacobian-based inverse kinematics process with configuration transition strategies
> 
> 图 5. 结合构型转换策略的基于雅可比逆运动学过程的流程图
> 
> **批注说明：** 你在图中 $\dot{\mathbf{x}} = \begin{bmatrix} \mathbf{v} \\ \bm{\omega} \end{bmatrix} = \begin{bmatrix} \mathbf{v}_{lim} \cdot \mathbf{e}_p / \varepsilon_p \\ \bm{\omega}_{lim} \cdot \mathbf{a}_R \end{bmatrix}$ 旁边做了标注：“_大小固定，方向变化_”。你的理解十分精准：在这个迭代算法步骤中，为了控制单步步长，算法设定了固定的极限速度幅值（线速度 $\mathbf{v}_{lim}$ 和角速度 $\bm{\omega}_{lim}$），然后将其乘上当前误差的单位方向向量（例如 $\mathbf{e}_p / \varepsilon_p$）。因此，每次迭代提供的期望旋量其“大小是固定的”，而“方向会随着误差向量的变化而变化”。

After the configuration vector $\bm{\psi}_j$ is updated, the configuration variables shall be bounded within the limits.

在构型向量 $\bm{\psi}_j$ 更新之后，必须将构型变量限制在极限范围内。

In configurations C1 and C3, the prediction-based constraint imposition shall also be included.

在构型 C1 和 C3 中，还应包括基于预测的约束施加。

If the variable update triggers a configuration transition, the corresponding treatment will be applied to the configuration variables.

如果变量的更新触发了构型转换，则将对构型变量进行相应的处理。

The iterative inverse kinematics process continues, until the errors are smaller than the error thresholds or the number of iterations is larger than a preset value (5000).

逆运动学迭代过程不断持续，直到误差小于误差阈值，或者迭代次数大于预设值（5000 次）。

The maximal iteration number is set mainly to let the inverse kinematics process terminate, when a target pose is not reached.

设置最大迭代次数的主要目的是：当目标位姿不可达时，能够让逆运动学过程终止。

Two simulation cases are shown in Fig. 6.

两个仿真案例展示在图 6 中。

The first one started with a pose in Configuration C4 to reach a pose in Configuration C1, while the second one started with a pose in Configuration C1 to reach a pose in Configuration C4.

第一个案例从构型 C4 下的一个位姿开始，以到达构型 C1 下的位姿为目标；而第二个案例从构型 C1 下的一个位姿开始，目标是构型 C4 下的位姿。

In both cases, the manipulator traverses all the configurations sequentially.

在这两种情况下，机械臂都按顺序遍历了所有的构型。

In all the simulation and experimental cases, an 1-ms time step was used.

在所有的仿真和实验案例中，均使用了 1 毫秒的时间步长。

A slight angular error increase can be seen in Fig. 6(e) from Configuration C3 to Configuration C2. A possible explanation is as follows.

从图 6(e) 中可以看出，从构型 C3 转换到构型 C2 时，角度误差略有增加。一个可能的解释如下。

The manipulator in Configuration C2 only has 4 DoFs.

处于构型 C2 下的机械臂仅有 4 个自由度。

With the prioritized-Jacobian formulation, reducing the position errors is of higher priority and this is a key feature for the inverse kinematics to continue from C3 to C2.

在优先级雅可比公式中，减小位置误差具有更高的优先级，这也是逆运动学能够从 C3 持续向 C2 推进的一个关键特征。

After the manipulator enters Configuration C2, both position and orientation errors continue to decrease till the target is reached.

在机械臂进入构型 C2 之后，位置和姿态误差将继续减小，直到达到目标。

---

**Page 5**

> **图片说明及解析：**
> 
> Fig. 6. Two numerical simulation cases of inverse kinematics across four configurations using the configuration transition strategies. (Unit: mm): (a to e) case #1, and (f to j) case #2
> 
> 图 6. 使用构型转换策略跨越四种构型的逆运动学两个数值仿真案例。（单位：毫米）：(a 到 e) 是案例 #1，(f 到 j) 是案例 #2。

### _B. Extended Validation_

### _B. 扩展验证_

500,000 test cases are designed to further validate the effectiveness of the proposed kinematics framework. Theses test cases are generated as follows.

为了进一步验证所提出运动学框架的有效性，设计了 500,000 个测试用例。这些（注：原文中的“Theses”为拼写错误，应为These）测试用例如下生成：

125,000 poses are generated for each configuration, assigning random values to the corresponding configuration variables (within the allowed ranges).

为每种构型各生成 125,000 个位姿，方法是为相应的构型变量分配（允许范围内的）随机值。

Then the 500,000 poses from the four configurations are randomly arranged twice to generate the initial and the target poses for the test cases.

然后，将来自这四种构型的 500,000 个位姿随机排列两次，以生成测试用例的初始位姿和目标位姿。

In each of the 500,000 test cases, the continuum surgical manipulator tried to reach the target pose from the initial pose, following the proposed inverse kinematic process as in Section V.A.

在每一个测试用例（共 500,000 个）中，连续体手术机械臂都尝试按照第五节 A 部分中提出的逆运动学过程，从初始位姿向目标位姿运动。

In 488,970 (97.79%) test cases, the target poses were successfully reached, among which 487,845 test cases were accomplished within 850 iterations.

在 488,970 个（97.79%）测试用例中，成功达到了目标位姿，其中有 487,845 个测试用例是在 850 次迭代内完成的。

Statistics of the number of consumed iterations is shown in Fig. 7.

所消耗的迭代次数的统计信息如图 7 所示。

> **图片说明及解析：**
> 
> Fig. 7. The statistics for the reached cases
> 
> 图 7. 成功到达案例的迭代次数统计图

Then the 11030 unreached test cases were carefully analyzed and categorized.

随后对未到达目标的 11030 个测试用例进行了仔细的分析和归类。

Two statuses were identified when the proposed iterative inverse kinematics terminated: the iterations were trapped either i) at the _range limits of the configuration variables_ or ii) at _disconnected orientations_.

当所提出的迭代逆运动学终止时，发现了两种状态：迭代陷入了 i) **构型变量的范围限制**处，或者 ii) **不连通的姿态空间**。

There are 6722 unreached test cases in which the configuration variable values were found in proximity to the range limits (e.g., 0.1 mm for the length variables and 0.02 rad for the angle variables), when the inverse kinematics process terminated.

在逆运动学过程终止时，有 6722 个未到达目标的测试用例其构型变量值处于接近范围限制的状态（例如，长度变量接近极限 0.1 毫米，角度变量接近极限 0.02 弧度）。

A summary of these cases is listed in Table II, where the encountered variable range limits include the upper and the lower bounds of $\theta_1$ and $\theta_2$, the upper bound of $L_s$ and the lower bound of $L_2$.

表 II 列出了这些案例的摘要，其中遇到的变量范围限制包括 $\theta_1$ 和 $\theta_2$ 的上下界、$L_s$ 的上界以及 $L_2$ 的下界。

Other encountered length variable limits will trigger configuration transition.

遇到其他长度变量极限时，将会触发构型转换。

Please note that the maximum bending angle for the length changing equals $\kappa_{t0} \cdot L_t$ where $\kappa_{t0}$ is the maximal bending curvature of the $t^{th}$ segment.

请注意，长度变化过程中的最大弯曲角度等于 $\kappa_{t0} \cdot L_t$，其中 $\kappa_{t0}$ 是第 $t$ 段的最大弯曲曲率。

Among the 3535 cases where the $\theta_2$ upper bound was encountered, a representative case is plotted in Fig. 8(a) with the initial pose shown in green and the target pose in red.

在遇到 $\theta_2$ 上界的 3535 个案例中，图 8(a) 绘制了一个代表性案例，其中初始位姿显示为绿色，目标位姿显示为红色。

The initial pose is in Configuration C3 and the target pose is in Configuration C1.

初始位姿处于构型 C3，目标位姿处于构型 C1。

The proposed inverse kinematics framework converged to a pose still in Configuration C3, as shown in blue in Fig. 8(a).

所提出的逆运动学框架收敛于一个仍然处于构型 C3 下的位姿，如图 8(a) 中蓝色部分所示。

> **批注说明：**
> 
> 注意你在图中指出：“_图八图九反了_”。
> 
> 在原论文后续的内容和配图排版中，作者在正文描述“遇到 $\theta_2$ 上界（Segment #2 达到最大弯曲角度）”时指向了 Figure 8(a)，但你在之前的完整PDF和当前截图中可以敏锐地发现，实际带有“Segment #2 at its maximum bending angle”标签的图片被排版成了 Figure 9(a)。作者排版确实搞反了图片标题或引用，你的观察非常仔细。翻译时我仍保持原文引用的字面直译。

Among the 1680 cases where the upper bounds of $\theta_1$ and $\theta_2$ were both encountered, a representative case is plotted in Fig.

在同时遇到 $\theta_1$ 和 $\theta_2$ 上界的 1680 个案例中，代表性案例如图...（当前截图页面到底部截断）绘制。

8(b). The initial pose shown in green is in Configuration C3 and the target pose in red is in Configuration C1.

8(b)。以绿色显示的初始位姿处于构型 C3，红色的目标位姿处于构型 C1。

The converged pose in blue is still in Configuration C3.

以蓝色显示的收敛位姿仍然处于构型 C3。

A discussion for the unreached cases above is as follows.

针对上述未到达目标案例的讨论如下。

When the continuum manipulator tries to move from the initial pose towards the target pose (reaching a gripper tip position closer to the proximal end of the manipulator) as shown in Fig. 8, there are two possible ways to accomplish this goal: i) bend the segment(s) more, or ii) retract the whole manipulator.

当连续体机械臂试图如图 8 所示从初始位姿向目标位姿运动（即到达一个更靠近机械臂近端的夹爪末端位置）时，有两种可能的方法来实现这一目标：i) 进一步弯曲连续体段，或者 ii) 缩回整个机械臂。

However, retracting the manipulator in its Configuration C3 means to straighten Segment #1, otherwise the manipulator cannot be retracted.

然而，在构型 C3 下缩回机械臂意味着要将第 1 段伸直，否则机械臂无法被缩回。

Straightening Segment #1 will then inevitably bring temporary increases in the position and/or orientation errors.

而将第 1 段伸直不可避免地会带来位置和/或姿态误差的暂时增加。

This proposed method is essentially gradient-based and it “chose” to further bend the segments to continuously reduce the errors, leading to the convergence of the poses with the encountered configuration variable limits.

本文提出的方法本质上是基于梯度的，它“选择”了进一步弯曲连续体段以持续减小误差，从而导致位姿在遇到构型变量极限时发生收敛。

This outcome is associated with the inherent gradient-based characteristics of the method.

这一结果与该方法固有的基于梯度的特性有关。

“Intelligently” solving the inverse kinematics for these scenarios might be beyond the scope of this paper.

为这些场景“智能地”求解逆运动学可能超出了本文的范围。

Besides the 6722 cases where the configuration variable limits were encountered, the rest 4308 cases (11030 unreached test cases in total) all reached the target positions.

除了遇到构型变量极限的 6722 个案例之外，其余 4308 个案例（总共 11030 个未到达目标的测试用例）全部到达了目标位置。

These 4308 cases only involve Configurations C1 to C3, where no variable limits were encountered and the target orientations were not reached.

这 4308 个案例仅涉及构型 C1 到 C3，在这些案例中没有遇到变量极限，但未能达到目标姿态。

A detailed categorization is as follows.

详细的分类如下。

TABLE II

STATISTICS OF THE ENCOUNTERED VARIABLE LIMITS IN THE 6722 UNREACHED CASES

表 II

在 6722 个未到达目标案例中遇到的变量极限统计

- 666 cases were founded to have both the converged pose with the target pose in Configuration C2.
    
- 发现有 666 个案例的收敛位姿和目标位姿均处于构型 C2。
    

A representative case is plotted in Fig. 9(a).

图 9(a) 绘制了一个代表性案例。

The continuum surgical manipulator has 4 DoFs in Configuration C2: it can reach a point in 3D and rotates along its gripper axis.

连续体手术机械臂在构型 C2 下具有 4 个自由度：它可以到达三维空间中的一个点，并沿其夹爪轴线旋转。

There might be two inverse kinematics solutions for reaching the same spatial position, with the rigid stem extended at different lengths (a.k.a, $L_r$ has different values).

到达同一个空间位置可能存在两个逆运动学解，此时刚性连接杆伸出的长度不同（即 $L_r$ 具有不同的值）。

This is similar to the “elbow-up” and “elbow-down” solutions of a serial-link manipulator reaching the same position.

这类似于串联机械臂到达同一位置时的“肘部朝上（elbow-up）”和“肘部朝下（elbow-down）”解。

Once converged to one solution, this gradient-based inverse kinematics cannot generate the other inverse kinematics solution.

一旦收敛到一个解，这种基于梯度的逆运动学方法就无法生成另一个逆运动学解。

- 3577 cases were founded to have the converged pose in Configuration C2 with the target pose in Configuration C3.
    
- 发现有 3577 个案例的收敛位姿处于构型 C2，而目标位姿处于构型 C3。
    

A representative case is plotted in Fig. 9(b).

图 9(b) 绘制了一个代表性案例。

Even though the continuum surgical manipulator has 6 DoFs in Configuration C3 and the gripper axis can reach an orientation range (the red patch on the yellow sphere) as shown in the inset of Fig. 9(b), the converged orientation is disconnected from the orientation range at the target pose.

尽管连续体手术机械臂在构型 C3 中具有 6 个自由度，且夹爪轴线可以到达如图 9(b) 插图所示的一个姿态范围（黄色球体上的红色斑块），但收敛的姿态与目标位姿处的姿态范围是不连通的。

For the similar reason as mentioned above, the proposed algorithm did not converge to the target pose, which may involve temporary increases in position and/or orientation errors.

出于与上述类似的原因，所提出的算法未能收敛到目标位姿，因为这可能涉及到位置和/或姿态误差的暂时增加。

- 11 cases were found to have the converged pose in Configuration C1 with the target pose in Configuration C3, while 54 cases were found to have the converged pose in Configuration C1 and the target pose in Configuration C2.
    
- 发现有 11 个案例的收敛位姿处于构型 C1 且目标位姿处于构型 C3，同时发现有 54 个案例的收敛位姿处于构型 C1 且目标位姿处于构型 C2。
    

In all these cases, the manipulator converged to the desired position with a different orientation.

在所有这些案例中，机械臂均以不同的姿态收敛到了期望的位置。

The manipulator has 4 DoFs in the converged pose in Configuration C1.

机械臂在处于构型 C1 的收敛位姿下具有 4 个自由度。

This gradient-based algorithm does not allow the manipulator to “escape” from the converged pose to reach the target pose.

这种基于梯度的算法不允许机械臂从收敛位姿中“逃逸”以到达目标位姿。

In this extended validation, even though the 500,000 test cases were not 100% successful, the proposed kinematics framework might still be sufficient for teleoperation tasks.

在这项扩展验证中，尽管 500,000 个测试用例并未达到 100% 的成功率，但所提出的运动学框架可能仍足以胜任遥操作任务。

During a teleoperation, the desired target positions and orientations are frequently acquired from the master input devices.

在遥操作期间，期望的目标位置和姿态会频繁地从主端输入设备中获取。

No drastic differences will occur between the initial

初始位姿

> **图片说明及解析：**
> 
> Fig. 8. Representative unreached cases due to disconnected target orientations: (a) the “elbow-up” and “elbow-down” scenario, and (b) the disnnected orientation scenario
> 
> 图 8. 目标姿态不连通导致的代表性未到达案例：(a) “肘部朝上”与“肘部朝下”场景，以及 (b) 姿态不连通场景。
> 
> **内容理解：** 图 8 实际上展示了姿态空间断层或多解导致的算法陷入问题。左图 (a) 呈现的是位置达标但收敛姿态错误的案例；右图 (b) 呈现的是虽然末端具有可达的姿态范围（黄色球体上的红点区域），但收敛姿态（蓝线指向区域）与该目标范围在数学空间上断开，导致梯度下降法无法跨越这一断层。值得注意的是，正如您之前圈出的批注，作者在正文排版时错误地将图 8 和图 9 的引用标题放反了，此图内容实质上对应正文中对于 Figure 9 的解释说明。

> **图片说明及解析：**
> 
> Fig. 9. Representative unreached cases due to the encountered configuration variable limits: (a) Segment #2 at the maximum bending angle and (b) both segments at the maximum bending angles
> 
> 图 9. 遇到构型变量极限导致的代表性未到达案例：(a) 第 2 段达到最大弯曲角度，以及 (b) 两段均达到最大弯曲角度。
> 
> **内容理解：** 此图直观展示了由于遭遇物理关节极限导致逆运动学求解失败的情况。机械臂（绿色初始态）为了向后退到目标态（红色），本应优先抽回整个连接杆，但受限于局部梯度最优策略，算法“贪婪地”选择加大末端弯曲度来逼近位置，最终撞上了最大弯曲角度极限（蓝色收敛态），导致无法“自拔”。此图内容实质上对应正文中对于 Figure 8 的解释说明。

---

**Page 8**

poses and the corresponding target poses.

和相应的目标位姿之间不会发生剧烈的差异。

Unless it is beyond the physical motion capabilities of the continuum surgical manipulator, the teleoperation pattern facilitates the convergence to the desired poses.

除非超出了连续体手术机械臂的物理运动能力，否则遥操作模式有助于收敛到期望的位姿。

### _C. Experimental Validations_

### _C. 实验验证_

A peg transfer task was carried out on the experimental setup shown in Fig. 1(e) to further validate the configuration transition strategies.

在图 1(e) 所示的实验装置上执行了移物（peg transfer）任务，以进一步验证构型转换策略。

As shown in Fig. 10, a ring near the trocar entry can be picked up by the manipulator in Configuration C1 and transferred to farther pegs in other configurations.

如图 10 所示，机械臂可以在构型 C1 下拾取穿刺鞘入口附近的一个圆环，并将其转移到其他构型下更远的木钉上。

The task can also be conducted driving the manipulator from Configurations C4 to C1.

该任务也可以通过驱动机械臂从构型 C4 运动到 C1 来完成。

This experiment can also be seen in the multi-media extension.

该实验也可以在多媒体附加材料中看到。

## VI. CONCLUSION

## 六、结论

When a multi-segment continuum surgical manipulator is only tele-operated as it is fully inserted into a patient’s cavity, its end effector often cannot reach a volume close to the manipulator’s trocar entry.

当多段连续体手术机械臂仅在完全插入患者体腔后才进行遥操作时，其末端执行器通常无法到达靠近机械臂穿刺鞘入口的一定体积区域。

Clearly this unreachable volume can be accessed if the manipulator can be tele-operated in its partially inserted configurations.

显然，如果机械臂能在其部分插入的构型下进行遥操作，这个不可达体积是可以被触及的。

In this paper, a novel kinematics framework for configuration transition control is hence proposed to fully utilize the kinematic capability of a continuum surgical manipulator.

因此，本文提出了一种新颖的用于构型转换控制的运动学框架，以充分利用连续体手术机械臂的运动学能力。

This framework includes a prioritized-Jacobian formulation, a prediction-based constraint imposition, and a set of configuration transition strategies.

该框架包括优先级雅可比公式化、基于预测的约束施加以及一组构型转换策略。

Extensive simulation cases and the actual teleoperation experiments show that the proposed kinematics framework for the configuration transition control helps utilize the inherent kinematic capability of the continuum surgical manipulator to generate an expanded workspace.

大量的仿真案例和实际的遥操作实验表明，所提出的用于构型转换控制的运动学框架有助于利用连续体手术机械臂固有的运动学能力，从而生成扩展的工作空间。

Sustained bouncing at the transition points are only detected in the extensive simulation,

在大规模仿真中仅检测到了转换点处的持续跳动，

The proposed framework can also be applied to any other multi-segment continuum manipulator with feeding motions.

所提出的框架也可以应用于任何其他具有送进运动的多段连续体机械臂。

The framework can certainly be further improved, even though the current form may be sufficient for teleoperation tasks.

尽管当前形式可能已足以胜任遥操作任务，但该框架无疑还可以进一步改进。

For example, an error tolerance strategy may be incorporated.

例如，可以结合容错（误差容忍）策略。

Or the orientation may be analytically solved for the inverse kinematics for Configurations C1 and C2 such that the orientation information might be used to design a repulsive potential field.

或者可以通过解析方法求解构型 C1 和 C2 逆运动学中的姿态，从而使姿态信息可用于设计排斥势场。

Chattering issues might be identified around the configuration transitions such that the transition strategies may be further improved.

构型转换周围可能会发现抖振问题，借此可以对转换策略做进一步改进。

These measures then may help the manipulator move to the desired configuration.

这些措施随后可能有助于机械臂运动到期望的构型。

> **图片说明及解析：**
> 
> Fig. 10. (a to h) Experimental validation of the configuration transition in a peg transfer task.
> 
> 图 10. (a 至 h) 移物任务中构型转换的实验验证。
> 
> **内容理解：** 图中展示了真实的连续体手术机械臂实体样机在进行穿孔套环移位（Peg transfer）任务的连续快照。从图 (a) 开始，机械臂处于刚露头的 C1 构型并成功拾取最近端的套环；随后通过构型转换策略，机械臂逐步平滑地向外送进并弯曲变形（跨越 C2、C3 乃至 C4 构型），最终在图 (h) 将套环精准放置在远端的木钉上。实验成功证明了算法在物理应用上的可行性和连贯性。

REFERENCES

参考文献

[1] A. Cuschieri, "Laparoscopic Surgery: Current Status, Issues and Future Developments," _The Surgeon_, vol. 3, no. 3, pp. 125-138, June 2005.

[1] A. Cuschieri, "腹腔镜手术：现状、问题与未来发展", _The Surgeon_, 卷 3, 编号 3, 页码 125-138, 2005年6月.

[2] R. H. Taylor and D. Stoianovici, "Medical Robotics in Computer-Integrated Surgery," _IEEE Transactions on Robotics and Automation_, vol. 19, no. 3, pp. 765-781, 2003.

[2] R. H. Taylor 和 D. Stoianovici, "计算机集成手术中的医疗机器人技术", _IEEE Transactions on Robotics and Automation_, 卷 19, 编号 3, 页码 765-781, 2003年.

[3] C. Bergeles and G.-Z. Yang, "From Passive Tool Holders to Microsurgeons: Safer, Smaller, Smarter Surgical Robots," _IEEE Transactions on Biomedical Engineering_, vol. 61, no. 5, pp. 1565-1576, May 2014.

[3] C. Bergeles 和 G.-Z. Yang, "从被动工具夹持器到显微外科医生：更安全、更小巧、更智能的手术机器人", _IEEE Transactions on Biomedical Engineering_, 卷 61, 编号 5, 页码 1565-1576, 2014年5月.

[4] J. Burgner-Kahrs, D. C. Rucker, and H. Choset, "Continuum Robots for Medical Applications: A Survey," _IEEE Transactions on Robotics_, vol. 31, no. 6, pp. 1261-1280, Dec 2015.

[4] J. Burgner-Kahrs, D. C. Rucker, 和 H. Choset, "医疗应用中的连续体机器人：综述", _IEEE Transactions on Robotics_, 卷 31, 编号 6, 页码 1261-1280, 2015年12月.

[5] J. Ding, K. Xu, R. Goldman, P. K. Allen, D. L. Fowler, and N. Simaan, "Design, Simulation and Evaluation of Kinematic Alternatives for Insertable Robotic Effectors Platforms in Single Port Access Surgery," in _IEEE International Conference on Robotics and Automation (ICRA)_, Anchorage, Alaska, USA, 2010, pp. 1053-1058.

[5] J. Ding, K. Xu, R. Goldman, P. K. Allen, D. L. Fowler, 和 N. Simaan, "单孔入路手术中可插入式机器人执行器平台运动学替代方案的设计、仿真与评估", 载于 _IEEE International Conference on Robotics and Automation (ICRA)_, 安克雷奇, 阿拉斯加, 美国, 2010年, 页码 1053-1058.

[6] P. E. Dupont, J. Lock, B. Itkowitz, and E. Butler, "Design and Control of Concentric-Tube Robots," _IEEE Transactions on Robotics_, vol. 26, no. 2, pp. 209-225, April 2010.

[6] P. E. Dupont, J. Lock, B. Itkowitz, 和 E. Butler, "同心管机器人的设计与控制", _IEEE Transactions on Robotics_, 卷 26, 编号 2, 页码 209-225, 2010年4月.

[7] M. D. M. Kutzer, S. M. Segreti, C. Y. Brown, R. H. Taylor, S. C. Mears, and M. Armand, "Design of a New Cable-Driven Manipulator with a Large Open Lumen: Preliminary Applications in the Minimally-Invasive Removal of Osteolysis," in _IEEE International Conference on Robotics and Automation (ICRA)_, Shanghai, China, 2011, pp. 2913-2920.

[7] M. D. M. Kutzer, S. M. Segreti, C. Y. Brown, R. H. Taylor, S. C. Mears, 和 M. Armand, "具有大开放腔的新型线驱动机械臂的设计：在微创骨质溶解清除术中的初步应用", 载于 _IEEE International Conference on Robotics and Automation (ICRA)_, 上海, 中国, 2011年, 页码 2913-2920.

[8] Y.-J. Kim, S. Cheng, S. Kim, and K. Iagnemma, "A Stiffness-Adjustable Hyperredundant Manipulator Using a Variable Neutral-Line Mechanism for Minimally Invasive Surgery," _IEEE Transactions on Robotics_, vol. 30, no. 2, pp. 382-395, April 2014.

[8] Y.-J. Kim, S. Cheng, S. Kim, 和 K. Iagnemma, "一种用于微创手术的采用可变中性线机构的刚度可调超冗余机械臂", _IEEE Transactions on Robotics_, 卷 30, 编号 2, 页码 382-395, 2014年4月.

[9] K. Xu, J. Zhao, and M. Fu, "Development of the SJTU Unfoldable Robotic System (SURS) for Single Port Laparoscopy," _IEEE/ASME Transactions on Mechatronics_, vol. 20, no. 5, pp. 2133-2145, Oct 2015.

[9] K. Xu, J. Zhao, 和 M. Fu, "用于单孔腹腔镜的上海交通大学可展开机器人系统 (SURS) 的开发", _IEEE/ASME Transactions on Mechatronics_, 卷 20, 编号 5, 页码 2133-2145, 2015年10月.

[10] J. Shang et al., "A Single-Port Robotic System for Transanal Microsurgery - Design and Validation," _IEEE Robotics and Automation Letters_, vol. 2, no. 3, pp. 1510-1517, July 2017.

[10] J. Shang 等人, "一种用于经肛门显微手术的单孔机器人系统——设计与验证", _IEEE Robotics and Automation Letters_, 卷 2, 编号 3, 页码 1510-1517, 2017年7月.

[11] K. Xu, J. Zhao, and X. Zheng, "Configuration Comparison among Kinematically Optimized Continuum Manipulators for Robotic Surgeries through a Single Access Port," _Robotica_, vol. 33, no. 10, pp. 2025-2044, Dec 2015.

[11] K. Xu, J. Zhao, 和 X. Zheng, "用于单通道机器人手术的运动学优化连续体机械臂间的构型比较", _Robotica_, 卷 33, 编号 10, 页码 2025-2044, 2015年12月.

[12] R. J. Webster and B. A. Jones, "Design and Kinematic Modeling of Constant Curvature Continuum Robots: A Review," _International Journal of Robotics Research_, vol. 29, no. 13, pp. 1661-1683, Nov 2010.

[12] R. J. Webster 和 B. A. Jones, "常曲率连续体机器人的设计与运动学建模：综述", _International Journal of Robotics Research_, 卷 29, 编号 13, 页码 1661-1683, 2010年11月.

[13] S. Liu, Z. Yang, Z. Zhu, L. Han, X. Zhu, and K. Xu, "Development of a Dexterous Continuum Manipulator for Exploration and Inspection in Confined Spaces," _Industrial Robot: An International Journal_, vol. 43, no. 3, pp. 284-295, 2016.

[13] S. Liu, Z. Yang, Z. Zhu, L. Han, X. Zhu, 和 K. Xu, "用于受限空间探索和检查的灵巧连续体机械臂的开发", _Industrial Robot: An International Journal_, 卷 43, 编号 3, 页码 284-295, 2016年.

[14] K. Xu and N. Simaan, "An Investigation of the Intrinsic Force Sensing Capabilities of Continuum Robots," _IEEE Transactions on Robotics_, vol. 24, no. 3, pp. 576-587, June 2008.

[14] K. Xu 和 N. Simaan, "连续体机器人内在力传感能力的调查研究", _IEEE Transactions on Robotics_, 卷 24, 编号 3, 页码 576-587, 2008年6月.

[15] D. N. Nenchev, "Restricted Jacobian Matrices of Redundant Manipulators in Constrained Motion Tasks," _The International Journal of Robotics Research_, vol. 11, no. 6, pp. 584-597, 1992.

[15] D. N. Nenchev, "冗余机械臂在受约束运动任务中的受限雅可比矩阵", _The International Journal of Robotics Research_, 卷 11, 编号 6, 页码 584-597, 1992年.