# Figure 1 设计说明和图注

## Figure 1: System Architecture Overview

**图注（Figure Caption）**：
```
Figure 1: System Architecture Overview. The proposed biomedical retrieval framework consists of four core components: (1) Instruction-Aware Query Encoder parses clinical constraints and aligns them with medical ontologies (UMLS); (2) Multi-Source Document Encoder integrates guidelines, trials, case reports, and knowledge graphs into a unified embedding space; (3) Chain-of-Retrieval Reasoning Module performs iterative evidence synthesis through a 5-step loop (initial retrieval → evaluation → refinement → secondary retrieval → triangulation); (4) Safety Constraint Checker validates evidence using rule-based filters, knowledge graph inference, and LLM-based contextual validation. The Fusion and Ranking Layer combines relevance, reasoning, and safety scores to produce the final ranked results.
```

---

## 详细组件说明

### **1. Instruction-Aware Query Encoder（指令感知查询编码器）**

**输入示例**：
```
"Which anticoagulants are safe for AF patients with eGFR <30 mL/min?"
```

**处理流程**：
1. **Constraint Parsing Layer（约束解析层）**
   - 提取约束单元：`eGFR <30 mL/min`
   - 提取临床实体：`anticoagulants`, `AF patients`
   - 提取安全线索：`safe`

2. **Clinical Ontology Alignment（临床本体对齐）**
   - 将 `AF` 映射到 UMLS 概念：`C0004238` (Atrial Fibrillation)
   - 将 `eGFR` 映射到 UMLS 概念：`C3811844` (Estimated Glomerular Filtration Rate)
   - 使用 MetaMap 进行语义标准化

3. **Cross-Constraint Attention（跨约束注意力）**
   - 建模约束之间的交互（如肾功能如何影响药物选择）
   - 使用多头注意力机制

**输出**：
```
Query Embedding q ∈ ℝ^768
```

---

### **2. Multi-Source Document Encoder（多源文档编码器）**

**四种证据源**：

| 证据类型 | 编码策略 | 特殊处理 |
|:---|:---|:---|
| **Clinical Guidelines** | Guidelines Encoder | 强调推荐等级（Class I/II/III）和禁忌症 |
| **Clinical Trials** | Trials Encoder | 强调入排标准和人群约束（通过注意力掩码） |
| **Case Reports** | Case Encoder | 标准 Transformer 编码叙事文本 |
| **Knowledge Graphs** | KG-GNN | 图神经网络编码关系结构（drug-disease, DDI） |

**处理流程**：
1. 每种证据类型通过专门的编码器
2. **Cross-Source Fusion Layer（跨源融合层）**：
   - 使用多头注意力整合异构表示
   - 投影到统一的嵌入空间

**输出**：
```
Unified Document Embeddings d ∈ ℝ^768
```

---

### **3. Chain-of-Retrieval Reasoning Module（检索链推理模块）**

**迭代推理流程（最多 3 次迭代）**：

```
Step 1: Initial Retrieval
├─ 基于 cosine(q, d) 检索 top-k 文档
└─ 输出：候选证据集 E₁

Step 2: Evidence Evaluation
├─ 完整性检查：是否满足所有约束？
├─ 一致性检查：指南和试验是否一致？
└─ 输出：缺失证据类型、矛盾检测

Step 3: Query Refinement
├─ 如果证据不完整或矛盾，调整查询
├─ 添加缺失约束（如"禁忌症"）
└─ 输出：优化后的查询 q'

Step 4: Secondary Retrieval
├─ 使用 q' 检索补充证据
└─ 输出：补充证据集 E₂

Step 5: Cross-Source Triangulation
├─ 验证 Guidelines ↔ Trials ↔ KG 的一致性
├─ 标记矛盾和支持关系
└─ 输出：最终证据集 E_final
```

**推理轨迹记录**：
- 每一步的检索结果
- 检测到的矛盾
- 查询优化路径
- 用于可解释性和审计

---

### **4. Safety Constraint Checker（安全约束检查器）**

**三层混合验证机制**：

#### **Layer 1: Rule-Based Filtering（基于规则的过滤）**
```python
# 示例规则
IF drug == "warfarin" AND patient_status == "pregnancy":
    REJECT (contraindication)

IF drug_metabolism == "CYP3A4" AND patient_eGFR < 30:
    FLAG (high risk)
```

#### **Layer 2: Knowledge Graph Inference（知识图谱推理）**
```sparql
# SPARQL 查询示例：检测 DDI
SELECT ?drug1 ?drug2 ?interaction
WHERE {
    ?drug1 :interactsWith ?drug2 .
    ?interaction :severity "major" .
}
```

#### **Layer 3: LLM-Based Contextual Validation（基于 LLM 的上下文验证）**
```
输入：检索到的证据 + 查询约束
LLM 任务：
1. 检测微妙的矛盾（如人群不匹配）
2. 识别幻觉（未经验证的声明）
3. 验证跨源一致性

输出：Safety Score ∈ [0, 1]
```

**综合安全评分**：
```
Safety(d) = w₁·RuleScore + w₂·KGScore + w₃·LLMScore
```

---

### **5. Fusion and Ranking Layer（融合与排序层）**

**最终评分函数**：
```
Score(q, d) = α·Relevance(q, d) + β·Reasoning(q, d) + γ·Safety(d)
```

**参数设置**（在验证集上调优）：
- α = 0.4（相关性权重）
- β = 0.3（推理权重）
- γ = 0.3（安全性权重）

**输出**：
```
Ranked Results: [d₁, d₂, ..., dₖ]
按综合得分降序排列
```

---

## 图表设计建议

### **配色方案**：
- **主色调**：专业蓝（#2E86AB）和中性灰（#6C757D）
- **强调色**：安全红（#E63946）用于 Safety Checker
- **辅助色**：推理绿（#06A77D）用于 Reasoning Module

### **图标建议**：
- 📋 Clinical Guidelines
- 🔬 Clinical Trials
- 📝 Case Reports
- 🕸️ Knowledge Graphs
- ⚠️ Safety Warning
- 🔄 Iterative Loop

### **排版建议**：
- 使用 **Arial** 或 **Helvetica** 字体（10-12pt）
- 组件边框使用圆角矩形
- 箭头使用实线表示数据流，虚线表示反馈循环
- 添加图例说明不同颜色的含义

---

## 在论文中的引用方式

**在 Section 3.1 中**：
```markdown
We developed a biomedical retrieval framework consisting of four core components: 
(1) an Instruction-Aware Query Encoder for parsing clinical constraints, 
(2) a Multi-Source Document Encoder for unified evidence representation, 
(3) a Chain-of-Retrieval Reasoning Module for iterative evidence synthesis, and 
(4) a Safety Constraint Checker for filtering unsafe content. 
**Figure 1** illustrates the system architecture.
```

**在 Section 4.1 中**：
```markdown
The proposed system (Figure 1) was compared against widely used retrieval baselines...
```

---

## 图片格式要求（JBI 投稿）

根据 JBI 的图片要求：
- **格式**：TIFF 或 EPS（矢量图优先）
- **分辨率**：至少 300 DPI
- **尺寸**：单栏宽度 8.5 cm，双栏宽度 17.5 cm
- **颜色模式**：RGB（在线版）或 CMYK（印刷版）
- **文件大小**：建议 < 10 MB

---

## 可选：使用工具绘制

如果您需要使用专业工具绘制，推荐：
1. **draw.io**（免费，在线）：https://app.diagrams.net/
2. **Lucidchart**（专业版）
3. **Microsoft Visio**
4. **Adobe Illustrator**（矢量图）
5. **Python + Matplotlib**（编程方式）

我已经生成了一个初步的架构图供您参考。您可以基于此图和上述详细说明，使用专业工具创建最终的投稿版本。
