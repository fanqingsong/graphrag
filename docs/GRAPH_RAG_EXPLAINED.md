# GraphRAG 原理详解

本文档详细解释 GraphRAG 项目中 Graph（图）如何增强检索和生成过程。

## 目录

1. [Graph 在项目中的应用](#graph-在项目中的应用)
2. [图数据库中的节点类型](#图数据库中的节点类型)
3. [向量嵌入的作用](#向量嵌入的作用)
4. [实际应用示例](#实际应用示例)
5. [工作流程](#工作流程)

---

## Graph 在项目中的应用

Graph 是 GraphRAG 的核心，用于构建知识图谱并增强检索与推理。主要应用包括：

### 1. 知识图谱存储（Neo4j）

使用 Neo4j 图数据库存储文档知识结构：

- **节点类型**：
  - `Document`：文档节点
  - `Chunk`：文档片段节点（带向量嵌入）
  - `Entity`：实体节点（人物、地点、概念等）

- **关系类型**：
  - `HAS_CHUNK`：文档包含的片段
  - `SIMILAR_TO`：相似片段之间的关系
  - `CONTAINS_ENTITY`：片段包含的实体
  - `RELATED_TO`：实体之间的关系

### 2. 图增强检索（Graph-Enhanced Retrieval）

通过图遍历扩展检索上下文：

- 从初始检索的 Chunk 开始
- 通过图关系找到相关的 Chunk 和 Entity
- 扩展上下文，提供更完整的信息

### 3. 多跳推理（Multi-hop Reasoning）

通过实体关系进行多跳推理，连接跨文档信息：

- 从查询中的实体开始
- 沿着关系路径遍历（最多 2-3 跳）
- 发现间接但相关的信息

### 4. 实体提取与关系建立

文档处理时自动提取实体并建立关系：

- 使用 LLM 提取实体（人物、组织、概念等）
- 建立实体间的关系（如"工作于"、"位于"等）
- 计算关系强度和重要性分数

### 5. 混合检索模式

支持多种检索模式：

- **Chunk-only**：传统向量相似度检索
- **Entity-only**：基于实体的检索
- **Hybrid**：结合向量检索和图遍历
- **Graph-enhanced**：使用图关系扩展上下文

### 核心优势

1. **上下文扩展**：通过图关系找到相关但可能不直接匹配的内容
2. **跨文档连接**：通过实体关系连接不同文档的信息
3. **语义理解**：利用实体和关系提升语义理解
4. **可解释性**：图结构提供推理路径的可视化

---

## 图数据库中的节点类型

图数据库（Neo4j）中存储了以下三种主要类型的节点：

### 1. Document 节点 (`:Document`)

文档节点，代表上传的文档。

**主要属性**：
- `id`: 文档唯一标识符
- `filename`: 文件名
- `original_filename`: 原始文件名
- `summary`: 文档摘要（自动提取）
- `document_type`: 文档类型（如：report, contract, article等）
- `hashtags`: 标签列表（自动提取）
- `created_at`: 创建时间
- `updated_at`: 更新时间
- 其他元数据（文件大小、MIME类型等）

### 2. Chunk 节点 (`:Chunk`)

文档片段节点，代表文档分割后的文本块。

**主要属性**：
- `id`: Chunk唯一标识符
- `content`: 文本内容
- `embedding`: 向量嵌入（用于相似度搜索）
- `chunk_index`: 在文档中的索引位置
- `offset`: 在文档中的偏移量
- 其他元数据（页码、位置等）

### 3. Entity 节点 (`:Entity`)

实体节点，从文档中提取的实体（人物、组织、地点等）。

**主要属性**：
- `id`: 实体唯一标识符
- `name`: 实体名称
- `type`: 实体类型（见下方）
- `description`: 实体描述
- `importance_score`: 重要性分数（0.0-1.0）
- `source_chunks`: 来源chunk ID列表
- `embedding`: 向量嵌入（用于实体相似度搜索）
- `updated_at`: 更新时间

**实体类型**：
- `PERSON`: 人物
- `ORGANIZATION`: 组织
- `LOCATION`: 地点
- `EVENT`: 事件
- `CONCEPT`: 概念
- `TECHNOLOGY`: 技术
- `PRODUCT`: 产品
- `DOCUMENT`: 文档
- `DATE`: 日期
- `MONEY`: 金额

### 节点之间的关系

**关系类型**：

1. **`:HAS_CHUNK`** - Document → Chunk
   - 文档包含的chunk

2. **`:CONTAINS_ENTITY`** - Chunk → Entity
   - Chunk中包含的实体

3. **`:SIMILAR_TO`** - Chunk ↔ Chunk 或 Entity ↔ Entity
   - 相似度关系（带相似度分数）

4. **`:RELATED_TO`** - Entity ↔ Entity
   - 实体之间的关系（带类型、强度、描述）

### 图结构示例

```
(Document) --[:HAS_CHUNK]--> (Chunk) --[:CONTAINS_ENTITY]--> (Entity)
                                      |
                                      |--[:SIMILAR_TO]--> (Chunk)
                                      
(Entity) --[:RELATED_TO]--> (Entity)
         |
         |--[:SIMILAR_TO]--> (Entity)
```

---

## 向量嵌入的作用

### 什么是向量嵌入（Embedding）？

向量嵌入是将文本转换为数值向量（浮点数数组），用于表示文本的语义。语义相近的文本，其向量在空间中更接近。

**示例**：
```
文本1: "人工智能的发展"
文本2: "AI技术的进步"
文本3: "今天的天气真好"

向量1: [0.2, 0.8, 0.1, ..., 0.5]  (1536维)
向量2: [0.21, 0.79, 0.12, ..., 0.48]  (非常接近向量1)
向量3: [0.9, 0.1, 0.8, ..., 0.2]  (与向量1、2差异很大)
```

### 如何生成向量嵌入？

在项目中，使用 OpenAI 的 embedding 模型（如 `text-embedding-3-small`）生成：

1. 文档处理时，每个 Chunk 的文本内容会被转换为向量
2. 向量存储在 Chunk 节点的 `embedding` 属性中
3. 查询时，用户查询也会被转换为向量进行比较

### 向量嵌入的作用

#### 1. 语义相似度搜索

用户查询时，将查询转换为向量，然后与所有 Chunk 的向量比较，找出最相似的：

**工作流程**：
1. 用户查询："什么是机器学习？"
2. 将查询转换为向量：`query_embedding = [0.3, 0.7, ...]`
3. 计算与所有 Chunk 的余弦相似度
4. 返回相似度最高的 top_k 个 Chunk

#### 2. 计算 Chunk 之间的相似度

用于建立 Chunk 之间的 `SIMILAR_TO` 关系：

- 计算同一文档内所有 Chunk 之间的相似度
- 如果相似度超过阈值，创建 `SIMILAR_TO` 关系
- 这些关系用于图扩展检索

#### 3. 在检索流程中的应用

在 chunk-based 检索中使用：

- 生成查询的向量嵌入
- 与所有 Chunk 的向量进行相似度计算
- 返回最相关的 Chunk

### 核心优势

#### 1. 语义理解，而非关键词匹配

**传统搜索**：
- 查询："AI技术"
- 只能找到包含"AI技术"的文本
- 可能遗漏"人工智能"、"机器学习"等相关内容

**向量搜索**：
- 查询："AI技术"
- 能找到语义相关的内容，即使没有完全相同的词
- 例如："机器学习算法"、"深度学习模型"等

#### 2. 相似度量化

每个结果都有相似度分数（0-1），便于排序和过滤：
- 相似度 0.9：非常相关
- 相似度 0.7：相关
- 相似度 0.3：不太相关

#### 3. 支持图扩展

向量相似度与图关系结合，实现更智能的检索：
- 先用向量找到初始 Chunk
- 再通过图关系扩展找到相关内容

### 实际例子

假设你有这些 Chunk：

```
Chunk A: "机器学习是人工智能的一个分支"
Chunk B: "深度学习使用神经网络"
Chunk C: "今天的天气很好"
```

用户查询："什么是AI？"

1. **查询向量化**：`query_vec = [0.3, 0.7, ...]`
2. **计算相似度**：
   - Chunk A: 0.85（高相似度，因为"AI"和"人工智能"语义相近）
   - Chunk B: 0.72（中等相似度）
   - Chunk C: 0.15（低相似度）
3. **返回结果**：Chunk A 和 Chunk B（按相似度排序）

---

## 实际应用示例

### 场景设置

假设你上传了三个文档：

**文档1：Microsoft投资新闻**
```
Microsoft宣布向OpenAI投资100亿美元，这是AI领域最大的投资之一。
OpenAI的CEO Sam Altman表示这次合作将加速AI技术的发展。
```

**文档2：OpenAI技术报告**
```
OpenAI开发了GPT-4模型，在多个领域取得突破。
公司总部位于旧金山，由Sam Altman领导。
```

**文档3：AI行业分析**
```
AI领域的投资正在快速增长。Microsoft、Google等科技巨头都在加大投入。
旧金山湾区是AI创新的中心。
```

### 第一步：文档处理时构建 Graph

系统处理这些文档时，会：

1. **提取实体（Entity）**：
   - `Microsoft` (COMPANY, 重要性: 0.9)
   - `OpenAI` (COMPANY, 重要性: 0.9)
   - `Sam Altman` (PERSON, 重要性: 0.8)
   - `旧金山` (LOCATION, 重要性: 0.6)
   - `GPT-4` (TECHNOLOGY, 重要性: 0.7)

2. **建立关系（Relationship）**：
   - `Microsoft` --[投资于]--> `OpenAI` (强度: 0.9)
   - `Sam Altman` --[领导]--> `OpenAI` (强度: 0.95)
   - `OpenAI` --[位于]--> `旧金山` (强度: 0.8)
   - `OpenAI` --[开发了]--> `GPT-4` (强度: 0.85)

3. **创建图结构（在Neo4j中）**：
```
(Microsoft:COMPANY) --[投资于]--> (OpenAI:COMPANY)
(OpenAI:COMPANY) <--[领导]-- (Sam Altman:PERSON)
(OpenAI:COMPANY) --[位于]--> (旧金山:LOCATION)
(OpenAI:COMPANY) --[开发了]--> (GPT-4:TECHNOLOGY)
```

### 第二步：用户查询

用户问：**"Sam Altman领导的公司在AI领域有什么重要投资？"**

### 第三步：Graph 如何帮助检索

#### 传统向量检索（没有Graph）

可能只返回：
- 文档1中直接提到"Sam Altman"和"投资"的片段
- 得分：0.75

**问题**：可能遗漏间接相关的信息。

#### Graph增强检索（有Graph）

系统会：

1. **初始检索**：找到包含"Sam Altman"的chunk
   ```
   Chunk A: "OpenAI的CEO Sam Altman表示这次合作..."
   ```

2. **图遍历扩展**：
   ```python
   # 从Sam Altman实体开始
   Sam Altman --[领导]--> OpenAI (找到!)
   
   # 从OpenAI继续扩展
   OpenAI <--[投资于]-- Microsoft (找到!)
   
   # 通过图关系找到相关chunk
   Chunk B: "Microsoft宣布向OpenAI投资100亿美元..."
   Chunk C: "OpenAI开发了GPT-4模型..."
   ```

3. **多跳推理路径**：
   ```
   查询: Sam Altman领导的公司的投资
   ↓
   路径1: Sam Altman → 领导 → OpenAI → 被投资于 → Microsoft
   路径2: Sam Altman → 领导 → OpenAI → 开发了 → GPT-4
   ```

4. **最终返回的增强上下文**：
   ```
   Chunk A (直接匹配): "OpenAI的CEO Sam Altman..."
   Chunk B (图扩展): "Microsoft宣布向OpenAI投资100亿美元..."
   Chunk C (图扩展): "OpenAI开发了GPT-4模型..."
   ```

### 第四步：LLM生成答案

有了这些通过graph连接的上下文，LLM可以生成更完整的答案：

> "Sam Altman领导的OpenAI公司获得了Microsoft的100亿美元投资，这是AI领域最大的投资之一。同时，OpenAI还开发了GPT-4等重要的AI技术。"

### 关键优势

1. **跨文档连接**：即使"Microsoft投资"和"Sam Altman"不在同一文档，graph也能连接它们
2. **语义理解**：理解"领导"关系，而不仅仅是关键词匹配
3. **可解释性**：可以看到推理路径：Sam Altman → OpenAI → Microsoft
4. **上下文扩展**：找到间接相关但重要的信息

---

## 工作流程

### 文档处理流程

```
1. 文档上传
   ↓
2. 文档分割成 Chunks
   ↓
3. 为每个 Chunk 生成向量嵌入
   ↓
4. 创建 Document 和 Chunk 节点
   ↓
5. 提取实体（Entity）
   ↓
6. 建立实体关系（Relationship）
   ↓
7. 创建 Entity 节点和关系
   ↓
8. 计算 Chunk 之间的相似度
   ↓
9. 创建 SIMILAR_TO 关系
```

### 查询处理流程

```
1. 用户查询
   ↓
2. 查询分析（分析查询类型、复杂度等）
   ↓
3. 生成查询向量嵌入
   ↓
4. 向量相似度搜索（找到初始 Chunks）
   ↓
5. 图扩展（通过图关系找到相关 Chunks）
   ↓
6. 多跳推理（可选，通过实体关系进行多跳遍历）
   ↓
7. 合并和排序结果
   ↓
8. 传递给 LLM 生成答案
```

### 代码实现详解

#### 步骤1-2：用户查询和查询分析

**实现位置**：`rag/nodes/query_analysis.py`

查询分析通过 `analyze_query()` 函数实现：

```python
def analyze_query(query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    分析用户查询，提取意图和关键概念
    
    返回：
    - query_type: 查询类型（factual, analytical, comparative）
    - complexity: 复杂度（simple, complex）
    - key_concepts: 关键概念列表
    - requires_reasoning: 是否需要推理
    - multi_hop_recommended: 是否推荐多跳推理
    """
```

**主要功能**：
- 检测是否为后续问题（follow-up question）
- 使用 LLM 分析查询意图
- 提取关键概念
- 判断查询类型和复杂度
- 决定是否使用多跳推理

**代码位置**：
```13:230:rag/nodes/query_analysis.py
def analyze_query(
    query: str, chat_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Analyze user query to extract intent and key concepts.

    Args:
        query: User query string
        chat_history: Optional list of previous messages [{"role": "user/assistant", "content": "..."}]

    Returns:
        Dictionary containing query analysis
    """
    try:
        # Check if this is a follow-up question
        is_follow_up = False
        needs_context = False
        context_query = query  # The enriched query with context if needed

        if chat_history and len(chat_history) >= 2:
            # Detect follow-up questions using LLM
            follow_up_detection = _detect_follow_up_question(query, chat_history)
            is_follow_up = follow_up_detection.get("is_follow_up", False)
            needs_context = follow_up_detection.get("needs_context", False)

            if is_follow_up and needs_context:
                # Create a contextualized version of the query
                context_query = _create_contextualized_query(query, chat_history)
                logger.info(
                    f"Follow-up question detected. Original: '{query}' -> Contextualized: '{context_query}'"
                )

        # Use LLM to analyze the query (using contextualized version if needed)
        analysis_result = llm_manager.analyze_query(context_query)

        # Extract key information (simplified version)
        analysis = {
            "original_query": query,
            "contextualized_query": context_query,
            "is_follow_up": is_follow_up,
            "needs_context": needs_context,
            "query_type": "factual",  # Default type
            "key_concepts": [],
            "intent": "information_seeking",
            "complexity": "simple",
            "analysis_text": analysis_result.get("analysis", ""),
            "requires_reasoning": False,
            "requires_multiple_sources": False,
        }

        # Simple heuristics to enhance analysis (use contextualized query for better analysis)
        query_lower = context_query.lower()

        # Detect question types
        if any(
            word in query_lower
            for word in ["compare", "difference", "vs", "versus", "contrast"]
        ):
            analysis["query_type"] = "comparative"
            analysis["requires_multiple_sources"] = True
            analysis["requires_reasoning"] = True
        elif any(
            word in query_lower
            for word in [
                "why",
                "how",
                "explain",
                "reason",
                "analyze",
                "relationship",
                "connection",
            ]
        ):
            analysis["query_type"] = "analytical"
            analysis["requires_reasoning"] = True
        elif any(word in query_lower for word in ["what", "who", "when", "where"]):
            analysis["query_type"] = "factual"

        # Detect complexity
        if len(query.split()) > 10 or "and" in query_lower or "or" in query_lower:
            analysis["complexity"] = "complex"
            analysis["requires_multiple_sources"] = True

        # Extract potential key concepts (simple keyword extraction)
        # Skip common words
        stop_words = {
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "which",
            "that",
            "this",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
        }

        words = query_lower.replace("?", "").replace("!", "").replace(",", "").split()
        key_concepts = [
            word for word in words if len(word) > 2 and word not in stop_words
        ]
        analysis["key_concepts"] = key_concepts[:5]  # Limit to top 5 concepts

        # Determine if multi-hop reasoning would be beneficial
        multi_hop_beneficial = False

        # Multi-hop is beneficial for:
        # 1. Comparative queries (need to connect multiple entities)
        if analysis["query_type"] == "comparative":
            multi_hop_beneficial = True

        # 2. Analytical queries that need reasoning (relationships, explanations)
        elif analysis["query_type"] == "analytical" and analysis["requires_reasoning"]:
            multi_hop_beneficial = True

        # 3. Complex queries with multiple concepts
        elif analysis["complexity"] == "complex" and len(key_concepts) >= 3:
            multi_hop_beneficial = True

        # 4. Queries explicitly asking for relationships or connections
        elif any(
            word in query_lower
            for word in [
                "relationship",
                "connection",
                "related",
                "link",
                "connect",
                "between",
            ]
        ):
            multi_hop_beneficial = True

        # 5. Queries asking about trends, patterns, or implications
        elif any(
            word in query_lower
            for word in [
                "trend",
                "pattern",
                "impact",
                "effect",
                "influence",
                "implication",
            ]
        ):
            multi_hop_beneficial = True

        # Multi-hop is NOT beneficial for:
        # 1. Simple factual lookups (addresses, names, single facts)
        # 2. Direct "what is X" questions about specific entities
        # 3. Simple definition requests
        if (
            analysis["query_type"] == "factual"
            and analysis["complexity"] == "simple"
            and len(key_concepts) <= 2
            and not analysis["requires_multiple_sources"]
        ):
            multi_hop_beneficial = False

        analysis["multi_hop_recommended"] = multi_hop_beneficial

        logger.info(
            f"Query analysis completed: {analysis['query_type']}, {len(key_concepts)} concepts, "
            f"multi-hop recommended: {multi_hop_beneficial}, is_follow_up: {is_follow_up}"
        )
        return analysis

    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        return {
            "original_query": query,
            "contextualized_query": query,
            "is_follow_up": False,
            "needs_context": False,
            "query_type": "factual",
            "key_concepts": [],
            "intent": "information_seeking",
            "complexity": "simple",
            "analysis_text": "",
            "requires_reasoning": False,
            "requires_multiple_sources": False,
            "error": str(e),
        }
```

#### 步骤3：生成查询向量嵌入

**实现位置**：`rag/retriever.py` 和 `core/embeddings.py`

在检索过程中，查询文本会被转换为向量嵌入：

```python
# 在 chunk_based_retrieval 中
query_embedding = embedding_manager.get_embedding(query)
```

**代码位置**：
```89:111:rag/retriever.py
    async def chunk_based_retrieval(
        self,
        query: str,
        top_k: int = 5,
        allowed_document_ids: Optional[List[str]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Traditional chunk-based retrieval using vector similarity.

        Args:
            query: User query
            top_k: Number of similar chunks to retrieve
            allowed_document_ids: Optional list of document IDs to restrict retrieval
            query_embedding: Pre-computed query embedding (to avoid recomputation)

        Returns:
            List of similar chunks with metadata
        """
        try:
            # Generate query embedding if not provided
            if query_embedding is None:
                query_embedding = embedding_manager.get_embedding(query)
```

#### 步骤4：向量相似度搜索 ⭐

**实现位置**：`core/graph_db.py` 的 `vector_similarity_search()` 方法

这是关键步骤！**Neo4j 本身不直接支持向量搜索，但通过 GDS (Graph Data Science) 插件可以实现**。

**重要说明**：
- Neo4j 使用 **GDS (Graph Data Science)** 插件的 `gds.similarity.cosine()` 函数来计算余弦相似度
- 向量数据存储在 Chunk 节点的 `embedding` 属性中（一个浮点数数组）
- 查询向量通过参数传入，与数据库中所有 Chunk 的向量进行比较

**代码实现**：
```486:503:core/graph_db.py
    def vector_similarity_search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search using cosine similarity."""
        with self.driver.session() as session:  # type: ignore
            result = session.run(
                """
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
                WITH c, d, gds.similarity.cosine(c.embedding, $query_embedding) AS similarity
                RETURN c.id as chunk_id, c.content as content, similarity,
                       coalesce(d.original_filename, d.filename) as document_name, d.id as document_id
                ORDER BY similarity DESC
                LIMIT $top_k
                """,
                query_embedding=query_embedding,
                top_k=top_k,
            )
            return [record.data() for record in result]
```

**工作原理**：
1. **Cypher 查询**：使用 `MATCH` 找到所有 Document 和它们的 Chunk
2. **余弦相似度计算**：`gds.similarity.cosine(c.embedding, $query_embedding)` 
   - `c.embedding`：Chunk 节点存储的向量（在文档处理时生成）
   - `$query_embedding`：用户查询的向量（在检索时生成）
   - 返回相似度分数（0-1之间）
3. **排序和限制**：按相似度降序排序，返回 top_k 个结果

**GDS 插件说明**：
- GDS (Graph Data Science) 是 Neo4j 的官方插件
- 提供各种图算法和相似度计算函数
- 需要在 Neo4j 中安装并启用（项目中的 docker-compose.yml 已配置）

**配置位置**（docker-compose.yml）：
```yaml
neo4j:
  environment:
    - NEO4J_PLUGINS=["graph-data-science"]
```

#### 步骤5-6：图扩展和多跳推理

**实现位置**：`rag/nodes/graph_reasoning.py`

在向量搜索找到初始 Chunks 后，通过图关系扩展上下文：

```13:90:rag/nodes/graph_reasoning.py
def reason_with_graph(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    query_analysis: Dict[str, Any],
    retrieval_mode: str = "graph_enhanced",
) -> List[Dict[str, Any]]:
    """
    Perform graph-based reasoning to enhance context.

    Args:
        query: User query string
        retrieved_chunks: Initially retrieved chunks
        query_analysis: Query analysis results

    Returns:
        Enhanced list of chunks with graph context
    """
    try:
        if not retrieved_chunks:
            logger.warning("No retrieved chunks for graph reasoning")
            return []

        enhanced_chunks = list(retrieved_chunks)  # Start with original chunks

        # If retrieval mode explicitly requests simple retrieval, skip reasoning.
        if retrieval_mode == "simple":
            logger.info("Retrieval mode is 'simple' - skipping graph reasoning")
            return enhanced_chunks

        # For chunk_only mode, skip graph reasoning entirely
        if retrieval_mode == "chunk_only":
            logger.info("Chunk-only mode selected - skipping graph reasoning")
            return enhanced_chunks

        # For entity_only and hybrid modes, always run graph reasoning to respect user's choice
        logger.info(f"Running graph reasoning for {retrieval_mode} mode")

        # Find related chunks through graph traversal
        seen_chunk_ids = {chunk.get("chunk_id") for chunk in retrieved_chunks}

        for chunk in retrieved_chunks[:3]:  # Only expand from top 3 chunks
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue

            try:
                # Get chunks related through graph relationships
                related_chunks = graph_db.get_related_chunks(
                    chunk_id=chunk_id,
                    relationship_types=["SIMILAR_TO", "HAS_CHUNK"],
                    max_depth=2,  # Keep it shallow for performance
                )

                # Add unique related chunks
                for related_chunk in related_chunks:
                    related_id = related_chunk.get("chunk_id")
                    if related_id and related_id not in seen_chunk_ids:
                        # Add relationship context to metadata
                        related_chunk["reasoning_context"] = {
                            "related_to": chunk_id,
                            "relationship_type": "graph_expansion",
                            "distance": related_chunk.get("distance", 1),
                        }
                        enhanced_chunks.append(related_chunk)
                        seen_chunk_ids.add(related_id)

                        # Limit total chunks to prevent overwhelming the LLM
                        if len(enhanced_chunks) >= 10:
                            break

            except Exception as e:
                logger.warning(f"Failed to get related chunks for {chunk_id}: {e}")
                continue

        logger.info(
            f"Graph reasoning: {len(retrieved_chunks)} -> {len(enhanced_chunks)} chunks"
        )
        return enhanced_chunks

    except Exception as e:
        logger.error(f"Graph reasoning failed: {e}")
        return retrieved_chunks  # Return original chunks on error
```

#### 完整流程调用链

**入口**：`rag/graph_rag.py` 的 `query()` 方法
```python
graph_rag.query(
    user_query="什么是机器学习？",
    retrieval_mode="hybrid",
    top_k=5
)
```

**流程节点**：
1. `_analyze_query_node()` → 调用 `analyze_query()`
2. `_retrieve_documents_node()` → 调用 `retrieve_documents()`
3. `_reason_with_graph_node()` → 调用 `reason_with_graph()`
4. `_generate_response_node()` → 调用 LLM 生成答案

**代码位置**：
```65:120:rag/graph_rag.py
    def _analyze_query_node(self, state) -> Any:
        """Analyze the user query (dict-based state for LangGraph)."""
        try:
            query = state.get("query", "")
            chat_history = state.get("chat_history", [])
            logger.info(f"Analyzing query: {query}")
            
            # Initialize stages list if not present
            if "stages" not in state:
                state["stages"] = []
            
            # Track stage
            state["stages"].append("query_analysis")
            logger.info(f"Stage query_analysis completed, current stages: {state['stages']}")
            
            state["query_analysis"] = analyze_query(query, chat_history)
            return state
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            state["query_analysis"] = {"error": str(e)}
            return state

    def _retrieve_documents_node(self, state) -> Any:
        """Retrieve relevant documents (dict-based state for LangGraph)."""
        try:
            logger.info("Retrieving relevant documents")
            
            # Initialize stages list if not present
            if "stages" not in state:
                state["stages"] = []
            
            # Track retrieval stage
            state["stages"].append("retrieval")
            logger.info(f"Stage retrieval completed, current stages: {state['stages']}")
            
            # Pass additional retrieval tuning parameters from state
            chunk_weight = state.get("chunk_weight", 0.5)
            graph_expansion = state.get("graph_expansion", True)
            use_multi_hop = state.get("use_multi_hop", False)

            state["retrieved_chunks"] = retrieve_documents(
                state.get("query", ""),
                state.get("query_analysis", {}),
                state.get("retrieval_mode", "graph_enhanced"),
                state.get("top_k", 5),
                chunk_weight=chunk_weight,
                graph_expansion=graph_expansion,
                use_multi_hop=use_multi_hop,
                context_documents=state.get("context_documents", []),
            )
            
            return state
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            state["retrieved_chunks"] = []
            return state
```

### 关于 Neo4j 向量搜索的说明

**重要理解**：
1. **Neo4j 本身不直接支持向量搜索**，但可以通过 GDS 插件实现
2. **向量数据存储**：向量作为节点属性存储在 `c.embedding` 中
3. **相似度计算**：使用 `gds.similarity.cosine()` 函数在查询时计算
4. **性能考虑**：对于大规模数据，可能需要使用专门的向量数据库（如 Pinecone、Weaviate）或 Neo4j 的向量索引功能（如果可用）

**当前实现**：
- ✅ 使用 GDS 插件的余弦相似度函数
- ✅ 向量存储在节点属性中
- ✅ 查询时实时计算相似度
- ⚠️ 对于超大规模数据，可能需要优化（如使用向量索引）

### 检索模式对比

| 模式 | 向量搜索 | 图扩展 | 多跳推理 | 适用场景 |
|------|---------|--------|---------|---------|
| Chunk-only | ✅ | ❌ | ❌ | 简单查询，快速响应 |
| Entity-only | ✅ | ✅ | ❌ | 实体相关的查询 |
| Hybrid | ✅ | ✅ | ❌ | 平衡性能和准确性 |
| Graph-enhanced | ✅ | ✅ | ✅ | 复杂查询，需要深度推理 |

---

## 总结

GraphRAG 的核心优势在于：

1. **双重检索**：结合向量相似度搜索和图关系遍历
2. **语义理解**：通过向量嵌入理解语义，而非简单关键词匹配
3. **关系推理**：通过图结构进行多跳推理，发现间接关系
4. **上下文扩展**：自动扩展检索上下文，提供更完整的信息
5. **可解释性**：图结构提供清晰的推理路径

这就是为什么这个项目叫 **GraphRAG**：它使用图（Graph）来增强检索增强生成（RAG），使系统能够理解实体之间的关系，而不仅仅是文本相似度。

