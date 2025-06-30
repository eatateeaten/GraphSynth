This project covers a low-to-no-code visual compiler for tracing, visualizing and editing DNNs
┌────────────────────┐
│  1. Front‑End UI   │ – React + React‑Flow canvas
└────────┬───────────┘
         │ Electron IPC bridge (GraphDef + live tensor‑shapes in‑process)
         ▼
┌────────────────────┐
│ 4. Compiler        │ – OpType‑based shape checker & SSA code‑gen
│
│  ── Graph model (connect) ─────────────────────────────────────────
│  • connect(sourceId, sinkId, sPort, tPort)
│        1. Lookup nodes; throw if missing
│        2. Ensure source.outShapes[sPort] is defined
│        3. If sink.inShapes[tPort] is null and !inShapeInferred(sink) ⇒ error
│        4. If both shapes exist and !shapeMatch ⇒ ShapeMatchError
│        5. sink.addPrev(...) & source.addNext(...)
│           • If inShapeInferred(sink) == true, sink.addPrev triggers
│             validateInShape + computeOutShape (currently concrete dims; TODO: lift to symbolic shape solver)
│        6. shapeMatch already uses symbolic shape solver; TODO: port the other two routines
│        7. Push {edgeId, sourceId, sinkId, sPort, tPort} to _edges
│
│  ── Passes ────────────────────────────────────────────────────────
│  ① Hierarchical walk  (ModuleOp nodes stay opaque for PyTorch target)
│  ② shapeInference   – per edge
│  ③ validateInShape – per edge
│  ④ emit(target) ⇒ our‑IR | PyTorch‑FX | StableHLO MLIR¹
│     • After each Tracer import, regenerates code and maintains visual↔source line map (opaque block → code block)
└────────┬───────────┘
         │ trait queries
         ▼
┌────────────────────┐
│ 3. Op Hierarchy    │ – tag‑based hierarchy (parent ➜ child)
│
│   ModuleOp (N‑in / N‑out, may wrap nn.Module)
│   └─► UnaryOp (single‑in / single‑out)
│         ├─ ModuleUnaryOp – single‑in/out specialisation
│         ├─ UnaryElementwiseOp (shape‑preserving)
│         │     • abs, negate, log, log1p, exp, expm1, cosine, sine, tan, tanh,
│         │       cbrt, sqrt, rsqrt, sign, count_leading_zeros
│         └─ Transformations (data‑layout, no math)
│               • reshape, transpose, reverse, pad, iota
│
│   MergeOp (multiple‑in / single‑out)
│         ├─ BinaryOp
│         │     ├─ BinaryElementwiseOp (shape‑preserving)
│         │     │     • add, subtract, multiply, divide, power, max, min, atan2,
│         │     │       compare, and, or, xor, shifts
│         │     └─ BinaryLinAlg – dot, dot_general, cross, bilinear
│         └─ ReduceOp (commutative + associative)
│               ├─ NaryElementwiseOp – addN, maxN, minN, orN
│               └─ ReduceWindowOp (Slice + Reduce)
│
│   BranchOp (single‑in / multiple‑out)
│         • broadcast, slice, dynamic_slice, concatenate, reshape, iota …
│
│   Tag `Op` — primitive op with one‑to‑one StableHLO correspondence
│   *Still deciding: pure hierarchy vs tag‑based*

└────────┬───────────┘
         │ look‑ups / IR resolve
         ▼
┌────────────────────┐
│ 2. ModuleDB        │ – stores modules, ops, params, IR, shape flags
│  • emit + opType + target      → code & IR generation
│  • validateInShape + opType    → input shape check
│  • shapeInference + opType     → output shape inference
└────────┬───────────┘
         ▲                                       ▲
         │                                       │
         │  look‑ups                             │  IR
         │                                       │
┌────────┴───────────┐                           │
│ 5. Tracer          │ ──────────────── IR ──────┘
│  • torch.fx capture² (nn.Module ∪ nn.functional leaves)
│  • imports .pt / ONNX → IR
│  • loops → vmap; torch.cond & random‑noise → opaque block (auto‑converted to ModuleOp with static shapes)
│  • feeds IR to Compiler (for editing) & ModuleDB (for registering)
└────────────────────┘
