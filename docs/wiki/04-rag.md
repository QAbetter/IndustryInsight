# 04 · RAG 知识库（app/rag）

## 1. 总览

```mermaid
flowchart LR
    A["PDF / DOCX / MD / TXT"] --> B["loaders<br/>解析为 LoadedDocument"]
    B --> C["splitter<br/>递归字符切片"]
    C --> D["embeddings<br/>GLM / OpenAI / Hashing"]
    D --> E["store<br/>LocalVectorStore / Chroma"]
    E --> F["knowledge.search<br/>向量 + 词法混合排序"]
    F --> G["kb_search 工具<br/>返回 JSON 给 Agent"]
```

| 模块 | 职责 |
| --- | --- |
| [loaders.py](../../app/rag/loaders.py) | 文件类型识别、文本清洗、按页/段落解析 |
| [splitter.py](../../app/rag/splitter.py) | 递归切片，LangChain 优先，本地实现兜底 |
| [embeddings.py](../../app/rag/embeddings.py) | 文档/查询向量化 |
| [store.py](../../app/rag/store.py) | 向量持久化与 Top-K 检索 |
| [knowledge.py](../../app/rag/knowledge.py) | 入库编排、混合检索、阈值、统计 |
| [models.py](../../app/rag/models.py) | `LoadedDocument` / `Chunk` / `RetrievedChunk` / `IngestStats` |

## 2. 文档解析（loaders.py）

### 支持类型

`SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".markdown"}`

### `load_file(path, base_dir=None) -> list[LoadedDocument]`

- `source` 默认取文件名；传入 `base_dir` 时取相对路径（保留子目录信息）。
- 分发：
  - `.pdf` → `_load_pdf`
  - `.docx` → `_load_docx`
  - 其余 → `_load_text`

### 各格式行为

| 格式 | 解析方式 | 粒度 |
| --- | --- | --- |
| PDF | `pypdf.PdfReader`，逐页 `extract_text()` | **每页一个 `LoadedDocument`**，`page` 从 1 开始；单页异常跳过 |
| DOCX | `python-docx`，段落 + 表格 | 全文一个 `LoadedDocument`，`page=None`；表格转为 Markdown 行（`\| cell \| cell \|`），并加 `[表格 N]` 前缀 |
| MD / TXT | 文本读取，编码依次尝试 `utf-8-sig` → `utf-8` → `gb18030` | 全文一个 `LoadedDocument` |

### 文本清洗 `_clean()`

- 去除 `\x00`；
- 统一换行（`\r\n` / `\r` → `\n`）；
- 去掉行尾空格；
- 连续 3 个以上换行压缩为 2 个；
- 首尾 `strip()`。

### `iter_files(paths)`

- 目录递归收集支持的后缀（`rglob`）；
- 文件直接收集；
- 按解析后的绝对路径小写去重。

## 3. 递归切片（splitter.py）

### `RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120, separators=None)`

- 校验：`chunk_size > 0`，`0 <= chunk_overlap < chunk_size`。
- 默认分隔符（中文优先）：

```python
["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""]
```

- 若安装了 `langchain-text-splitters`，内部使用其 `RecursiveCharacterTextSplitter(keep_separator=True)`。
- 未安装则使用等价本地实现：
  - `_split()`：找到第一个出现在文本中的分隔符，切分后递归处理超长片段；
  - `_hard_split()`：无可细分隔符时按 `chunk_size - chunk_overlap` 步长硬切；
  - `_merge()`：贪心合并小块，并按 `chunk_overlap` 回退保留重叠。
- `split_text()` 返回去除空白的非空片段。

> 切片参数在 `config/settings.yaml`：`chunk_size: 900`、`chunk_overlap: 150`（dataclass 默认值 700/120 会被 YAML 覆盖）。

## 4. 向量化（embeddings.py）

### 抽象

```python
class Embedder(ABC):
    name: str
    dim: int
    embed_documents(texts) -> np.ndarray     # (n, dim)，已归一化
    embed_query(text) -> np.ndarray = embed_documents([text])[0]
```

### `GLMEmbedder(model="embedding-3")` — 默认

- 通过智谱 OpenAI 兼容接口调用：`https://open.bigmodel.cn/api/paas/v4/`；
- 需要 `ZHIPU_API_KEY`；
- 默认模型 `embedding-3`，输出 **2048 维**向量，L2 归一化；
- `name = "zhipu:embedding-3"`，写入索引 `meta.json`，用于防止换模型后误用旧索引；
- 按 64 条一批请求，并按响应 `index` 排序，避免批次乱序；
- 维度自动探测：本地索引不需要预先写死 `dim`。

### `OpenAIEmbedder(model, dim=0)` — 可选

- 需要 `OPENAI_API_KEY`（可选 `OPENAI_BASE_URL`）；
- 与 GLM 共用同一实现，`name_prefix` 区分来源；
- 按 64 条一批调用 `embeddings.create`；
- 返回归一化数组；首次调用后自动确定 `dim`。

### `HashingEmbedder(dim=768)` — 离线回退

字符 n-gram 哈希，离线、确定性、对中文友好，`demo` 与单测默认使用它。

| 特征 | 权重 |
| --- | --- |
| 单个汉字 | 0.35 |
| 汉字 bigram | 1.0 |
| 汉字 trigram | 0.6 |
| ASCII 单词（`[a-z][a-z0-9_-]{1,}`） | 1.0 |
| 数字（形如 `#123#`） | 1.2 |

哈希与符号：

```python
digest = blake2b(feature, digest_size=8).digest()
index  = int.from_bytes(digest[:4], "big") % dim
sign   = 1.0 if digest[4] & 1 else -1.0
vector[index] += sign * weight
```

最后 L2 归一化。查询侧先经过 `_clean_query()` 去掉疑问词与标点（“是多少 / 是什么 / 有哪些 / 为什么 / 如何 / 是否”等），提升短查询质量。

**已知限制**：哈希向量没有语义泛化能力，跨公司/近义表述容易混淆；因此只作为离线演示与 CI 回退。

### `create_embedder(settings)`

```python
hashing → HashingEmbedder(settings.dim)
zhipu   → GLMEmbedder(settings.embedding_model or "embedding-3", dim=0,
                      api_key_env=settings.embedding_api_key_env or "ZHIPU_API_KEY",
                      base_url=settings.embedding_base_url or 智谱默认端点)
openai  → OpenAIEmbedder(settings.embedding_model or "text-embedding-3-small", dim=0,
                         api_key_env=settings.embedding_api_key_env or "OPENAI_API_KEY",
                         base_url=settings.embedding_base_url or OPENAI_BASE_URL)
其他    → ConfigError
```

## 5. 向量库（store.py）

### `VectorStore`（ABC）

```python
add(chunks, vectors) -> None
search(query_vector, top_k) -> list[tuple[Chunk, float]]   # 相似度越大越相关
count() -> int
all_ids() -> set[str]
reset() -> None
save() -> None      # 默认空实现
```

### `LocalVectorStore(path, embedder_name, dim)` — 默认

持久化文件（`data/index/`）：

| 文件 | 内容 |
| --- | --- |
| `vectors.npy` | `(n, dim)` float32，已归一化 |
| `chunks.jsonl` | 每行一个 `Chunk.to_dict()` |
| `meta.json` | `{backend, embedder, dim, count}` |

关键行为：

- `load()`：
  - 三文件齐全才加载；
  - `meta.embedder != 当前 embedder_name` → `IndexMismatchError`，记录 `load_error` 并清空内存索引（**不自动删除磁盘文件**）；
  - `chunks` 与 `vectors` 数量不一致 → `IndexMismatchError`。
- `add()`：新向量归一化后 `vstack`，chunk 追加。
- `search()`：`self.vectors @ query`（归一化后即余弦相似度），`argsort` 取 Top-K。
- `save()`：全量覆写三个文件。
- `reset()`：清空内存并 `save()` 空索引。

### `ChromaVectorStore(path, collection, embedder_name, dim)` — 可选

- 延迟导入 `chromadb`，未安装时抛 `RuntimeError` 并提示安装 `requirements-optional.txt`。
- `PersistentClient` + `get_or_create_collection(metadata={"hnsw:space": "cosine"})`。
- `add()` 用 `upsert`，metadata 存 `source/page/chunk_index/doc_type/embedder/company/report_year`（`page` 缺失存 -1）。
- `search()` 返回 `1.0 - distance` 作为相似度；`company` / `report_year` 通过 Chroma `where` 下推。
- `all_chunks()` 兼容 `collection.get()` 的不同版本签名。
- `reset()` 删除并重建 collection。

### `create_store(settings, root, embedder_name, dim)`

- `backend == "local"` → `root / rag.persist_dir`
- `backend == "chroma"` → `root / rag.chroma_dir`
- 其他 → `ValueError`

## 6. KnowledgeBase（knowledge.py）

### 构造

```python
KnowledgeBase(settings)
# embedder = create_embedder(settings.rag)
# store    = create_store(settings.rag, settings.root, embedder.name, embedder.dim)
# splitter = RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)
```

若 store 加载失败，`self.load_error` 记录原因；后续 `search()` 直接返回空列表，`ingest()` 会自动 `reset()` 重建。

### 入库 `ingest(paths, *, reset=False) -> IngestStats`

流程：

1. 若存在 `load_error`，先 `store.reset()`；
2. `reset=True` 时清空索引；
3. 收集文件并记录 `files`；
4. 对每个文件：
   - 解析为 `LoadedDocument`（失败则记入 `skipped_files`）；
   - 对每个 document 切片；
   - 生成 chunk id：

```python
sha1(f"{source}|{page}|{index}|{text[:48]}").hexdigest()[:20]
```

   - 已存在则 `chunks_skipped += 1`，否则加入 `new_chunks`；
5. 批量向量化并 `store.add()` + `store.save()`；
6. 统计 `chunks_added`、`dim`、`elapsed_ms`。

**幂等性**：chunk id 由内容决定，重复导入同一文件不会重复入库（被跳过）。

### 检索 `search(query, *, top_k=None, threshold=None, company=None, report_year=None) -> list[RetrievedChunk]`

```python
if load_error or store.count() == 0: return []
search_query = strip_report_context(query) if (company or report_year) else query
query_vector = embedder.embed_query(search_query)
limit  = top_k or settings.rag.top_k
cutoff = threshold if threshold is not None else settings.rag.score_threshold
candidate_limit = min(max(limit * 8, 30), 200)
rows = store.search(query_vector, candidate_limit,
                    company=company, report_year=report_year)
```

**元数据过滤**：入库时按文件名写入 `company` / `report_year` / `report_name`（如 `estun_2024_annual_report.pdf` → 埃斯顿 / 2024）。检索时先按这两个字段裁剪候选集，再做向量 + 词法排序。

- 目的：四家年报的“研发投入/净利润/现金流”措辞几乎相同，不筛公司会把别家同名科目排到第 1；
- 传了过滤条件时，`strip_report_context()` 会去掉题干里的“在埃斯顿2024年年报中，”前缀，只用核心问题排序，避免模板词稀释切片区分度；
- 实测 100 题（`qa_robotics_v2` 前 100，全部可回答）：原始题目 93% / 0.8201；补全公司/年份但不过滤 88% / 0.7416；加元数据过滤后 **97% / 0.8889**。501 题是题库总量（493 可答 + 8 不可答），全量 501 的 `97.16% / 0.884` 为历史运行、产物未保留，不再作为当前指标；
- 未传过滤条件时行为与旧版一致。

混合打分：

```python
lexical = |query_terms ∩ chunk_terms| / |query_terms|
score   = 0.55 * max(0, vector_score) + 0.45 * lexical
```

- 词法项 `_lexical_terms()`：
  - 先去掉疑问噪声词；
  - 中文连续片段生成 bigram + trigram；
  - 英文单词（长度 ≥2）；
  - 数字（含小数）。
- 排序后按 `score >= cutoff` 过滤，按 chunk id 去重，截断到 `limit`。
- **为什么混合**：长研报的向量容易被主题词稀释，词法重合能把包含具体数字/公司名的切片拉回来。

### 其他方法

| 方法 | 说明 |
| --- | --- |
| `stats()` | `backend / embedder / dim / chunks / sources / path / load_error` |
| `list_sources()` | 仅 local 后端；按 source 聚合 `doc_type / chunks / pages` |
| `all_chunks()` | 返回全部切片（评测集构建使用） |
| `reset()` | 清空索引并清除 `load_error` |

## 7. 领域对象（models.py）

```python
@dataclass LoadedDocument:  text, source, page=None, doc_type="txt", metadata={}
@dataclass Chunk:           id, text, source, page, chunk_index, doc_type="txt", metadata={}
@dataclass RetrievedChunk:  chunk, score, vector_score=0.0, lexical_score=0.0
@dataclass IngestStats:     files, documents, chunks_added, chunks_skipped,
                            elapsed_ms, backend, embedder, dim, skipped_files=[]
```

`Chunk` 提供 `to_dict()` / `from_dict()`；`RetrievedChunk.to_dict()` 会 round 到 4 位；`IngestStats.to_dict()` 会 round 耗时。

## 8. 配置项与调参

当前仓库索引（GLM 配置）：6 份公开研报 / 992 页 / **1896 切片 / 9 sources**（含 3 份演示语料，仅 6 个切片）；Hashing 评测快照为 **1890 切片 / 6 sources**。切换 Embedding 后切片数与 meta 会变化，比较指标时需先对齐索引快照。

| 配置 | 默认（YAML） | 影响 |
| --- | --- | --- |
| `rag.backend` | `local` | `local` / `chroma` |
| `rag.embedding` | `zhipu` | `zhipu` / `openai` / `hashing` |
| `rag.embedding_model` | `embedding-3` | 远程 Embedding 模型名 |
| `rag.embedding_api_key_env` | `ZHIPU_API_KEY` | 远程 Embedding 的 Key 环境变量 |
| `rag.embedding_base_url` | `null` | 留空使用智谱默认端点 |
| `rag.dim` | `2048` | 仅 Hashing 生效；远程 Embedding 自动探测维度 |
| `rag.chunk_size` | `900` | 切片长度 |
| `rag.chunk_overlap` | `150` | 重叠长度 |
| `rag.top_k` | `5` | 默认返回条数 |
| `rag.score_threshold` | `0.15` | 相似度阈值 |
| `rag.persist_dir` | `data/index` | 本地索引目录 |
| `rag.chroma_dir` | `data/chroma` | Chroma 目录 |
| `rag.collection` | `industry_reports` | Chroma collection |

调参建议（评测见 [05-pipeline-and-eval.md](./05-pipeline-and-eval.md#3-检索评测)）：

1. 先调 `chunk_size` / `chunk_overlap`（财务表格建议 500–900，overlap 100–150）；
2. 再扫 `top_k ∈ {2,4,6,8}` 与 `threshold ∈ {0, 0.15, 0.3}`；
3. 若语义召回仍不足，可换 OpenAI Embedding 或领域向量模型；
4. 切换 Embedding 后必须重建索引（meta 校验会阻止不匹配的旧索引）。

## 9. 已知限制

- GLM `embedding-3` 需要网络与 `ZHIPU_API_KEY`，单题检索延迟约百毫秒级；Hashing 仅作离线回退，语义泛化弱；
- PDF 解析依赖文本层，扫描件（图片 PDF）无法抽取；
- DOCX 只取段落与表格，不处理页眉页脚、脚注、批注；
- 本地向量库为全量暴力检索（矩阵乘法），适合万级切片，不适合百万级；
- 词法项为字符级 n-gram，不做分词与同义词扩展。

继续阅读：[05-pipeline-and-eval.md](./05-pipeline-and-eval.md)
