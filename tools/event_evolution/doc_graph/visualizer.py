from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from utils.io_utils import ensure_parent_dir, read_json

DOC_ID_PREFIX_PATTERN = re.compile(r"^(?P<prefix>[A-Za-z]+)\d{3}")


def _prepare_nodes(graph_payload: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for node in graph_payload.get("nodes", []):
        payload = dict(node)
        match = DOC_ID_PREFIX_PATTERN.match(str(payload.get("doc_id", "")))
        payload["category"] = match.group("prefix") if match else "UNKNOWN"
        nodes.append(payload)
    return nodes


def render_document_graph_threshold_html(
    graph_json_path: str | Path,
    output_html_path: str | Path,
    *,
    default_threshold: float = 0.0,
    default_center_doc: str | None = None,
    default_center_hops: int = 1,
) -> Path:
    """把有向文档图渲染为带阈值和中心文档 k 跳邻域过滤的自包含 HTML。"""

    graph_payload = read_json(graph_json_path)
    graph_data = {
        "meta": graph_payload.get("meta", {}),
        "nodes": _prepare_nodes(graph_payload),
        "edges": graph_payload.get("edges", []),
    }

    template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>事件演化文档图阈值可视化</title>
  <style>
    :root {
      --bg: #f3efe5;
      --panel: rgba(255, 250, 242, 0.92);
      --ink: #1f2b2d;
      --muted: #6b7a7d;
      --accent: #0d6b6b;
      --accent-soft: rgba(13, 107, 107, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Noto Serif SC", "Source Han Serif SC", "Songti SC", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(13, 107, 107, 0.12), transparent 28%),
        linear-gradient(180deg, #f8f3e8 0%, var(--bg) 100%);
    }
    .shell {
      max-width: 1560px;
      margin: 0 auto;
      padding: 24px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid rgba(31, 43, 45, 0.08);
      border-radius: 20px;
      box-shadow: 0 18px 40px rgba(31, 43, 45, 0.08);
      padding: 18px 20px 22px;
      backdrop-filter: blur(10px);
    }
    .toolbar {
      display: grid;
      grid-template-columns: 1fr repeat(4, minmax(110px, auto));
      gap: 16px;
      align-items: center;
      margin-bottom: 16px;
    }
    .toolbar h1 {
      margin: 0;
      font-size: 28px;
    }
    .toolbar p {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }
    .stat {
      padding: 10px 14px;
      border-radius: 14px;
      background: var(--accent-soft);
      min-width: 110px;
      text-align: center;
    }
    .stat strong {
      display: block;
      font-size: 20px;
      line-height: 1.1;
    }
    .controls {
      display: flex;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }
    .controls label {
      font-size: 14px;
      color: var(--muted);
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .controls input[type="range"] {
      width: 260px;
      accent-color: var(--accent);
    }
    .controls input[type="number"],
    .controls input[type="text"] {
      border: 1px solid rgba(31, 43, 45, 0.14);
      border-radius: 10px;
      padding: 8px 10px;
      background: #fffdf8;
      color: var(--ink);
      min-width: 130px;
    }
    .controls input[type="text"] {
      width: 240px;
    }
    .controls button {
      border: 0;
      border-radius: 10px;
      padding: 8px 14px;
      cursor: pointer;
      background: var(--ink);
      color: #fffaf2;
      font-family: inherit;
    }
    .controls button.secondary {
      background: rgba(31, 43, 45, 0.12);
      color: var(--ink);
    }
    .legend {
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 10px;
      line-height: 1.5;
    }
    canvas {
      width: 100%;
      height: 920px;
      border-radius: 18px;
      background:
        radial-gradient(circle at top, rgba(13, 107, 107, 0.06), transparent 40%),
        rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(31, 43, 45, 0.08);
    }
    .tooltip {
      position: fixed;
      pointer-events: none;
      display: none;
      max-width: 360px;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(31, 43, 45, 0.92);
      color: #f6f0e4;
      font-size: 13px;
      line-height: 1.5;
      box-shadow: 0 12px 28px rgba(0, 0, 0, 0.25);
    }
    @media (max-width: 1200px) {
      .toolbar {
        grid-template-columns: 1fr 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="panel">
      <div class="toolbar">
        <div>
          <h1>事件演化文档图</h1>
          <p>只显示当前阈值下存在连边的节点。指定中心文档后，按当前阈值下的有向边构造无向邻接关系，出度边和入度边都参与 k 跳扩展，但图中仍保留原始箭头方向。</p>
        </div>
        <div class="stat"><strong id="visibleNodeCount">0</strong><span>显示节点</span></div>
        <div class="stat"><strong id="visibleEdgeCount">0</strong><span>显示边数</span></div>
        <div class="stat"><strong id="thresholdText">0.00</strong><span>当前阈值</span></div>
        <div class="stat"><strong id="viewModeText">全局</strong><span>当前视图</span></div>
      </div>
      <div class="controls">
        <label>边阈值
          <input id="thresholdRange" type="range" min="-1" max="1" step="0.01" />
        </label>
        <label>精确输入
          <input id="thresholdInput" type="number" min="-1" max="1" step="0.01" />
        </label>
        <label>中心文档
          <input id="centerDocInput" type="text" placeholder="输入 doc_id，例如 M016" />
        </label>
        <label>邻域跳数
          <input id="centerHopInput" type="number" min="1" max="12" step="1" />
        </label>
        <button id="applyCenterButton" type="button">应用中心文档</button>
        <button id="clearCenterButton" class="secondary" type="button">返回全局视图</button>
      </div>
      <div class="legend">网络使用宽松的力导向布局，节点数量会随阈值变化而变化。正权边为青绿色，负权边为棕红色；中心文档节点会被高亮，距离中心更近的节点会放在更内层。</div>
      <canvas id="graphCanvas" width="1460" height="920"></canvas>
    </div>
  </div>
  <div id="tooltip" class="tooltip"></div>
  <script>
    const graph = __GRAPH_DATA__;
    const defaultThreshold = __DEFAULT_THRESHOLD__;
    const defaultCenterDoc = __DEFAULT_CENTER_DOC__;
    const defaultCenterHops = __DEFAULT_CENTER_HOPS__;
    const canvas = document.getElementById('graphCanvas');
    const context = canvas.getContext('2d');
    const thresholdRange = document.getElementById('thresholdRange');
    const thresholdInput = document.getElementById('thresholdInput');
    const centerDocInput = document.getElementById('centerDocInput');
    const centerHopInput = document.getElementById('centerHopInput');
    const applyCenterButton = document.getElementById('applyCenterButton');
    const clearCenterButton = document.getElementById('clearCenterButton');
    const visibleNodeCount = document.getElementById('visibleNodeCount');
    const visibleEdgeCount = document.getElementById('visibleEdgeCount');
    const thresholdText = document.getElementById('thresholdText');
    const viewModeText = document.getElementById('viewModeText');
    const tooltip = document.getElementById('tooltip');
    const nodeRadius = 7;
    const centerRadius = 10;
    const categoryPalette = ['#213033', '#0d6b6b', '#b55d3f', '#3f6d2a', '#825f99', '#23528c', '#8d7b1f'];
    const nodeByLowerId = new Map();
    graph.nodes.forEach((node) => {
      nodeByLowerId.set(String(node.doc_id).toLowerCase(), node);
    });
    const layoutCache = new Map();
    let currentState = { nodes: [], edges: [], centerId: '', centerHops: 1 };

    function clamp(value, minValue, maxValue) {
      return Math.min(maxValue, Math.max(minValue, value));
    }

    function hashString(value) {
      let hash = 2166136261;
      for (let index = 0; index < value.length; index += 1) {
        hash ^= value.charCodeAt(index);
        hash = Math.imul(hash, 16777619);
      }
      return Math.abs(hash >>> 0);
    }

    function normalizeDocId(value) {
      return String(value || '').trim().toLowerCase();
    }

    function sanitizeHopCount(value) {
      const numeric = Number.parseInt(String(value), 10);
      if (!Number.isFinite(numeric)) {
        return 1;
      }
      return clamp(Math.floor(numeric), 1, 12);
    }

    function currentThreshold() {
      return Number(thresholdRange.value);
    }

    function currentCenterDoc() {
      return normalizeDocId(centerDocInput.value);
    }

    function currentCenterHops() {
      return sanitizeHopCount(centerHopInput.value || defaultCenterHops);
    }

    function syncThreshold(value) {
      const numeric = Number(value);
      thresholdRange.value = String(numeric);
      thresholdInput.value = numeric.toFixed(2);
      thresholdText.textContent = numeric.toFixed(2);
      draw();
    }

    function syncCenterHops(value) {
      centerHopInput.value = String(sanitizeHopCount(value));
      draw();
    }

    function setCenterDoc(docId) {
      const normalized = normalizeDocId(docId);
      if (!normalized) {
        centerDocInput.value = '';
        draw();
        return;
      }
      const node = nodeByLowerId.get(normalized);
      centerDocInput.value = node ? node.doc_id : String(docId).trim();
      draw();
    }

    function colorForCategory(category) {
      const hash = hashString(category || 'UNKNOWN');
      return categoryPalette[hash % categoryPalette.length];
    }

    function buildHopNeighborhood(edges, centerId, hopLimit) {
      const adjacency = new Map();
      edges.forEach((edge) => {
        if (!adjacency.has(edge.source)) {
          adjacency.set(edge.source, new Set());
        }
        if (!adjacency.has(edge.target)) {
          adjacency.set(edge.target, new Set());
        }
        adjacency.get(edge.source).add(edge.target);
        adjacency.get(edge.target).add(edge.source);
      });

      const levels = new Map([[centerId, 0]]);
      const queue = [centerId];
      for (let head = 0; head < queue.length; head += 1) {
        const current = queue[head];
        const depth = levels.get(current) || 0;
        if (depth >= hopLimit) {
          continue;
        }
        const neighbors = adjacency.get(current) || new Set();
        neighbors.forEach((neighbor) => {
          if (!levels.has(neighbor)) {
            levels.set(neighbor, depth + 1);
            queue.push(neighbor);
          }
        });
      }
      return levels;
    }

    function buildFilteredGraph(threshold, centerKey, hopLimit) {
      const centerNode = centerKey ? nodeByLowerId.get(centerKey) || null : null;
      const visibleEdges = graph.edges.filter((edge) => Number(edge.weight) >= threshold);
      const nodeIds = new Set();
      let hopLevels = new Map();

      if (centerKey && !centerNode) {
        return {
          centerNode: null,
          centerMissing: true,
          nodes: [],
          edges: [],
          hopLevels,
        };
      }

      let edges = visibleEdges;
      if (centerNode) {
        hopLevels = buildHopNeighborhood(visibleEdges, centerNode.doc_id, hopLimit);
        edges = visibleEdges.filter(
          (edge) => hopLevels.has(edge.source) && hopLevels.has(edge.target)
        );
        if (!edges.length) {
          return {
            centerNode,
            centerMissing: false,
            nodes: [],
            edges: [],
            hopLevels,
          };
        }
        hopLevels.forEach((_depth, docId) => {
          nodeIds.add(docId);
        });
      } else {
        edges.forEach((edge) => {
          nodeIds.add(edge.source);
          nodeIds.add(edge.target);
        });
      }

      const nodes = graph.nodes
        .filter((node) => nodeIds.has(node.doc_id))
        .map((node) => ({
          ...node,
          hop_distance: hopLevels.has(node.doc_id) ? hopLevels.get(node.doc_id) : null,
        }))
        .sort((left, right) => {
          const leftHop = Number.isFinite(left.hop_distance) ? left.hop_distance : 99;
          const rightHop = Number.isFinite(right.hop_distance) ? right.hop_distance : 99;
          if (leftHop !== rightHop) {
            return leftHop - rightHop;
          }
          return String(left.doc_id).localeCompare(String(right.doc_id));
        });

      return {
        centerNode,
        centerMissing: false,
        nodes,
        edges,
        hopLevels,
      };
    }

    function computeLayout(nodes, edges, centerNode, cacheKey) {
      if (layoutCache.has(cacheKey)) {
        return layoutCache.get(cacheKey);
      }

      const positions = new Map();
      const width = canvas.width;
      const height = canvas.height;
      const padding = 60;
      const centerX = width / 2;
      const centerY = height / 2;
      const centerId = centerNode ? centerNode.doc_id : '';

      nodes.forEach((node, index) => {
        if (centerId && node.doc_id === centerId) {
          positions.set(node.doc_id, { x: centerX, y: centerY, vx: 0, vy: 0 });
          return;
        }

        const hash = hashString(node.doc_id);
        const angle = ((hash % 360) / 180) * Math.PI;
        let ring = 130 + (hash % 11) * 18 + Math.floor(index / 12) * 12;
        if (centerId) {
          const hopDistance = Number.isFinite(node.hop_distance) ? node.hop_distance : 1;
          ring = 120 + (hopDistance - 1) * 130 + (hash % 7) * 22 + Math.floor(index / 10) * 8;
        }
        const offsetX = Math.cos(angle) * ring + ((hash % 29) - 14);
        const offsetY = Math.sin(angle) * ring + (((hash >> 4) % 29) - 14);
        positions.set(node.doc_id, {
          x: clamp(centerX + offsetX, padding, width - padding),
          y: clamp(centerY + offsetY, padding, height - padding),
          vx: 0,
          vy: 0,
        });
      });

      if (nodes.length <= 1) {
        layoutCache.set(cacheKey, positions);
        return positions;
      }

      const iterations = centerId ? Math.min(180, 80 + nodes.length) : Math.min(140, 70 + nodes.length);
      const repulsion = centerId ? 3000 : 3400;
      const gravity = centerId ? 0.0024 : 0.0015;
      const springBaseLength = centerId ? 170 : 150;
      const maxSpeed = 18;

      for (let iteration = 0; iteration < iterations; iteration += 1) {
        for (let leftIndex = 0; leftIndex < nodes.length; leftIndex += 1) {
          const leftNode = nodes[leftIndex];
          const left = positions.get(leftNode.doc_id);
          if (!left) {
            continue;
          }
          for (let rightIndex = leftIndex + 1; rightIndex < nodes.length; rightIndex += 1) {
            const rightNode = nodes[rightIndex];
            const right = positions.get(rightNode.doc_id);
            if (!right) {
              continue;
            }
            let dx = right.x - left.x;
            let dy = right.y - left.y;
            let distanceSquared = dx * dx + dy * dy;
            if (distanceSquared < 0.01) {
              dx = 0.5 - Math.random();
              dy = 0.5 - Math.random();
              distanceSquared = dx * dx + dy * dy;
            }
            const distance = Math.sqrt(distanceSquared);
            const repel = repulsion / distanceSquared;
            const collisionDistance = 22;
            const collisionPush = distance < collisionDistance
              ? (collisionDistance - distance) * 0.08
              : 0;
            const forceX = (dx / distance) * (repel + collisionPush);
            const forceY = (dy / distance) * (repel + collisionPush);
            left.vx -= forceX;
            left.vy -= forceY;
            right.vx += forceX;
            right.vy += forceY;
          }
        }

        edges.forEach((edge) => {
          const source = positions.get(edge.source);
          const target = positions.get(edge.target);
          if (!source || !target) {
            return;
          }
          let dx = target.x - source.x;
          let dy = target.y - source.y;
          let distance = Math.sqrt(dx * dx + dy * dy);
          if (distance < 0.01) {
            distance = 0.01;
          }
          const involvesCenter = centerId && (edge.source === centerId || edge.target === centerId);
          const preferredLength = involvesCenter ? 120 : springBaseLength;
          const spring = (distance - preferredLength) * 0.0034 * (0.55 + Math.min(Math.abs(Number(edge.weight)), 1));
          const forceX = (dx / distance) * spring;
          const forceY = (dy / distance) * spring;
          source.vx += forceX;
          source.vy += forceY;
          target.vx -= forceX;
          target.vy -= forceY;
        });

        nodes.forEach((node) => {
          const position = positions.get(node.doc_id);
          if (!position) {
            return;
          }
          if (centerId && node.doc_id === centerId) {
            position.x = centerX;
            position.y = centerY;
            position.vx = 0;
            position.vy = 0;
            return;
          }
          position.vx += (centerX - position.x) * gravity;
          position.vy += (centerY - position.y) * gravity;
          position.vx *= 0.82;
          position.vy *= 0.82;
          position.vx = clamp(position.vx, -maxSpeed, maxSpeed);
          position.vy = clamp(position.vy, -maxSpeed, maxSpeed);
          position.x = clamp(position.x + position.vx, padding, width - padding);
          position.y = clamp(position.y + position.vy, padding, height - padding);
        });
      }

      layoutCache.set(cacheKey, positions);
      return positions;
    }

    function edgeStyle(weight) {
      const numeric = Number(weight);
      return numeric >= 0
        ? { color: 'rgba(13, 107, 107, 0.55)', alpha: 0.16 + Math.min(Math.abs(numeric), 1) * 0.56 }
        : { color: 'rgba(176, 94, 65, 0.45)', alpha: 0.14 + Math.min(Math.abs(numeric), 1) * 0.42 };
    }

    function drawArrow(x1, y1, x2, y2, color, alpha, radius) {
      const angle = Math.atan2(y2 - y1, x2 - x1);
      const endX = x2 - Math.cos(angle) * radius;
      const endY = y2 - Math.sin(angle) * radius;
      context.strokeStyle = color;
      context.globalAlpha = alpha;
      context.beginPath();
      context.moveTo(x1, y1);
      context.lineTo(endX, endY);
      context.stroke();
      context.beginPath();
      context.moveTo(endX, endY);
      context.lineTo(endX - Math.cos(angle - Math.PI / 7) * 9, endY - Math.sin(angle - Math.PI / 7) * 9);
      context.lineTo(endX - Math.cos(angle + Math.PI / 7) * 9, endY - Math.sin(angle + Math.PI / 7) * 9);
      context.closePath();
      context.fillStyle = color;
      context.fill();
      context.globalAlpha = 1;
    }

    function drawEmptyState(message) {
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.save();
      context.fillStyle = 'rgba(31, 43, 45, 0.68)';
      context.textAlign = 'center';
      context.font = '18px serif';
      context.fillText(message, canvas.width / 2, canvas.height / 2 - 12);
      context.font = '14px serif';
      context.fillStyle = 'rgba(31, 43, 45, 0.46)';
      context.fillText('可以降低阈值、提高跳数，或切换/清空中心文档。', canvas.width / 2, canvas.height / 2 + 18);
      context.restore();
    }

    function draw() {
      const threshold = currentThreshold();
      const centerKey = currentCenterDoc();
      const hopCount = currentCenterHops();
      const filtered = buildFilteredGraph(threshold, centerKey, hopCount);
      const centerId = filtered.centerNode ? filtered.centerNode.doc_id : '';
      const cacheKey = `${threshold.toFixed(2)}::${centerId || '__global__'}::${hopCount}`;
      const positions = computeLayout(filtered.nodes, filtered.edges, filtered.centerNode, cacheKey);
      const drawnNodes = filtered.nodes.map((node) => {
        const position = positions.get(node.doc_id);
        return {
          ...node,
          x: position ? position.x : canvas.width / 2,
          y: position ? position.y : canvas.height / 2,
        };
      });
      const drawnNodeById = new Map(drawnNodes.map((node) => [node.doc_id, node]));
      currentState = { nodes: drawnNodes, edges: filtered.edges, centerId, centerHops: hopCount };

      visibleNodeCount.textContent = String(drawnNodes.length);
      visibleEdgeCount.textContent = String(filtered.edges.length);
      viewModeText.textContent = centerId ? `${centerId} · ${hopCount} 跳` : '全局';
      thresholdText.textContent = threshold.toFixed(2);

      if (filtered.centerMissing) {
        drawEmptyState(`未找到中心文档 ${centerKey.toUpperCase()}`);
        return;
      }
      if (!drawnNodes.length) {
        drawEmptyState(centerId
          ? `阈值 ${threshold.toFixed(2)} 下，${centerId} 的 ${hopCount} 跳邻域内没有可显示边`
          : '当前阈值下没有可显示的连通节点');
        return;
      }

      context.clearRect(0, 0, canvas.width, canvas.height);
      context.save();
      context.fillStyle = 'rgba(13, 107, 107, 0.03)';
      context.fillRect(24, 24, canvas.width - 48, canvas.height - 48);
      context.restore();

      const sortedEdges = [...filtered.edges].sort((left, right) => Math.abs(Number(left.weight)) - Math.abs(Number(right.weight)));
      sortedEdges.forEach((edge) => {
        const source = drawnNodeById.get(edge.source);
        const target = drawnNodeById.get(edge.target);
        if (!source || !target) {
          return;
        }
        const style = edgeStyle(edge.weight);
        const targetRadius = edge.target === centerId ? centerRadius + 2 : nodeRadius + 2;
        drawArrow(source.x, source.y, target.x, target.y, style.color, style.alpha, targetRadius);
      });

      drawnNodes.forEach((node) => {
        const isCenter = node.doc_id === centerId;
        const hopDistance = Number.isFinite(node.hop_distance) ? node.hop_distance : null;
        const baseRadius = isCenter ? centerRadius : hopDistance === 1 ? nodeRadius + 1 : nodeRadius;
        const categoryColor = colorForCategory(node.category);
        context.beginPath();
        context.arc(node.x, node.y, baseRadius + 3, 0, Math.PI * 2);
        context.fillStyle = hopDistance === 1 && !isCenter ? 'rgba(198, 90, 55, 0.14)' : 'rgba(31, 43, 45, 0.12)';
        context.fill();

        context.beginPath();
        context.arc(node.x, node.y, baseRadius, 0, Math.PI * 2);
        context.fillStyle = isCenter ? '#c65a37' : categoryColor;
        context.fill();

        const shouldLabel = isCenter || hopDistance === 1 || drawnNodes.length <= 24;
        if (shouldLabel) {
          context.fillStyle = '#1f2b2d';
          context.font = isCenter ? 'bold 13px serif' : '12px serif';
          context.fillText(node.doc_id, node.x + baseRadius + 5, node.y - baseRadius - 2);
        }
      });
    }

    function hoveredNode(event) {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const x = (event.clientX - rect.left) * scaleX;
      const y = (event.clientY - rect.top) * scaleY;
      return currentState.nodes.find((node) => {
        const radius = node.doc_id === currentState.centerId ? centerRadius + 4 : nodeRadius + 4;
        return Math.hypot(node.x - x, node.y - y) <= radius;
      }) || null;
    }

    function updateHover(event) {
      const hovered = hoveredNode(event);
      if (!hovered) {
        tooltip.style.display = 'none';
        return;
      }
      tooltip.style.display = 'block';
      tooltip.style.left = `${event.clientX + 16}px`;
      tooltip.style.top = `${event.clientY + 16}px`;
      const hopText = Number.isFinite(hovered.hop_distance) ? `${hovered.hop_distance} 跳` : '全局';
      tooltip.innerHTML = `<strong>${hovered.doc_id}</strong><br />${hovered.title}<br />结束时间：${hovered.time_normalized || 'UNKNOWN_TIME'}<br />句子数：${hovered.sentence_count}<br />类别：${hovered.category}<br />邻域层级：${hopText}`;
    }

    function handleNodeClick(event) {
      const clicked = hoveredNode(event);
      if (!clicked) {
        return;
      }
      setCenterDoc(clicked.doc_id);
    }

    thresholdRange.addEventListener('input', (event) => syncThreshold(event.target.value));
    thresholdInput.addEventListener('change', (event) => syncThreshold(event.target.value));
    centerHopInput.addEventListener('change', (event) => syncCenterHops(event.target.value));
    centerHopInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        syncCenterHops(event.target.value);
      }
    });
    centerDocInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        setCenterDoc(event.target.value);
      }
    });
    applyCenterButton.addEventListener('click', () => setCenterDoc(centerDocInput.value));
    clearCenterButton.addEventListener('click', () => setCenterDoc(''));
    canvas.addEventListener('mousemove', updateHover);
    canvas.addEventListener('mouseleave', () => {
      tooltip.style.display = 'none';
    });
    canvas.addEventListener('click', handleNodeClick);

    thresholdRange.value = String(defaultThreshold);
    thresholdInput.value = Number(defaultThreshold).toFixed(2);
    centerDocInput.value = defaultCenterDoc;
    centerHopInput.value = String(sanitizeHopCount(defaultCenterHops));
    syncThreshold(defaultThreshold);
  </script>
</body>
</html>
"""

    html = template.replace(
        "__GRAPH_DATA__", json.dumps(graph_data, ensure_ascii=False)
    )
    html = html.replace("__DEFAULT_THRESHOLD__", json.dumps(float(default_threshold)))
    html = html.replace(
        "__DEFAULT_CENTER_DOC__",
        json.dumps((default_center_doc or "").strip(), ensure_ascii=False),
    )
    html = html.replace(
        "__DEFAULT_CENTER_HOPS__",
        json.dumps(max(1, int(default_center_hops))),
    )
    path = ensure_parent_dir(output_html_path)
    path.write_text(html, encoding="utf-8")
    return path
