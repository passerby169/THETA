"use client"

import { useState, useEffect, useCallback } from "react"
import type { CSSProperties, DragEventHandler, MouseEventHandler } from "react"
import { createPortal } from "react-dom"
import { apiFetch, API_BASE } from "@/lib/api/config"
import { Loader2, AlertCircle, Download, ChevronDown, ChevronUp, X, ZoomIn, ZoomOut, Send, Maximize2, RotateCcw } from "lucide-react"
import { ETMAgentAPI } from "@/lib/api/etm-agent"

interface VisualizationTabProps {
  dataset: string
  mode: string
  modelName?: string
  shouldLoad: boolean
  selectedModel?: string
}

interface VisualizationFile {
  name: string
  path: string
  url: string
  size: number
  type: string
}

interface VisualizationData {
  dataset: string
  model: string
  global_files: VisualizationFile[]
  topic_files: Record<string, VisualizationFile[]>
}

// Global 文件名映射
const GLOBAL_FILE_LABELS: Record<string, string> = {
  "7核心指标.png": "7 核心指标",
  "主题交互式可视化.html": "主题交互式可视化",
  "主题占比饼图.png": "主题占比饼图",
  "主题网络图.png": "主题网络图",
  "主题相似度.png": "主题相似度",
  "文档聚类图.png": "文档聚类图",
  "训练损失图.png": "训练损失图",
}

// CSV 预览组件（通过后端代理获取内容，解决 CORS 和 Content-Disposition 问题）
function CsvPreview({ dataset, path }: { dataset: string; path: string }) {
  const [csvData, setCsvData] = useState<{ headers: string[]; rows: string[][] } | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const token = localStorage.getItem("access_token")
    fetch(
      `${API_BASE}/api/results/${encodeURIComponent(dataset)}/visualizations/file?path=${encodeURIComponent(path)}`,
      {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      }
    )
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch CSV")
        return res.text()
      })
      .then((text) => {
        const lines = text.trim().split("\n")
        if (lines.length > 0) {
          const headers = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""))
          const rows = lines.slice(1, 101).map((line) =>
            line.split(",").map((cell) => cell.trim().replace(/^"|"$/g, ""))
          )
          setCsvData({ headers, rows })
        }
        setLoading(false)
      })
      .catch(() => {
        setError("无法加载 CSV")
        setLoading(false)
      })
  }, [dataset, path])

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-slate-500 py-4">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span className="text-sm">加载 CSV...</span>
      </div>
    )
  }

  if (error || !csvData) {
    return <span className="text-sm text-slate-400 py-4">{error || "无法预览"}</span>
  }

  return (
    <div className="overflow-auto w-full">
      <table className="text-xs border-collapse w-full">
        <thead>
          <tr className="bg-slate-200">
            {csvData.headers.map((h, i) => (
              <th key={i} className="px-2 py-1 text-left font-medium text-slate-600 border border-slate-300 whitespace-nowrap">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {csvData.rows.map((row, i) => (
            <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-slate-50"}>
              {row.map((cell, j) => (
                <td key={j} className="px-2 py-1 text-slate-700 border border-slate-200 whitespace-nowrap max-w-[200px] overflow-hidden text-ellipsis">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {csvData.rows.length === 100 && (
        <p className="text-xs text-slate-400 text-center py-2">只显示前 100 行</p>
      )}
    </div>
  )
}

// HTML 预览组件（通过 iframe srcdoc 加载，可执行 JavaScript）
function HtmlPreview({ dataset, model, path }: { dataset: string; model: string; path: string }) {
  const [htmlContent, setHtmlContent] = useState<string>("")
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const proxyUrl = `${API_BASE}/api/results/${encodeURIComponent(dataset)}/visualizations/file?model=${encodeURIComponent(model)}&path=${encodeURIComponent(path)}`

  useEffect(() => {
    const token = localStorage.getItem("access_token")
    setLoading(true)
    setError(null)
    fetch(proxyUrl, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => {
        if (!res.ok) throw new Error("无法加载 HTML")
        return res.text()
      })
      .then((text) => {
        setHtmlContent(text)
        setLoading(false)
      })
      .catch(() => {
        setError("加载失败")
        setLoading(false)
      })
  }, [proxyUrl])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[400px] bg-slate-100">
        <Loader2 className="w-6 h-6 animate-spin text-slate-400" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-[400px] bg-slate-100 text-slate-400">
        {error}
      </div>
    )
  }

  // 构建 srcdoc，注入 base 标签使相对路径资源正确加载
  const baseUrl = proxyUrl.replace(/\?.*$/, "")
  const srcdoc = htmlContent.replace("<head>", `<head><base href="${baseUrl}">`)

  return (
    <iframe
      srcDoc={srcdoc}
      title="HTML Preview"
      className="w-full h-full border-0"
      sandbox="allow-scripts allow-same-origin"
    />
  )
}

function AuthenticatedImage({
  dataset,
  model,
  path,
  name,
  className,
  style,
  loading = "lazy",
  draggable,
  onClick,
  onDoubleClick,
  onDragStart,
}: {
  dataset: string
  model: string
  path: string
  name: string
  className?: string
  style?: CSSProperties
  loading?: "lazy" | "eager"
  draggable?: boolean
  onClick?: MouseEventHandler<HTMLImageElement>
  onDoubleClick?: MouseEventHandler<HTMLImageElement>
  onDragStart?: DragEventHandler<HTMLImageElement>
}) {
  const [objectUrl, setObjectUrl] = useState<string | null>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    let cancelled = false
    let nextObjectUrl: string | null = null
    const token = localStorage.getItem("access_token")

    setObjectUrl(null)
    setError(false)

    fetch(
      `${API_BASE}/api/results/${encodeURIComponent(dataset)}/visualizations/file?model=${encodeURIComponent(model)}&path=${encodeURIComponent(path)}`,
      {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      }
    )
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch image")
        return res.blob()
      })
      .then((blob) => {
        if (cancelled) return
        nextObjectUrl = URL.createObjectURL(blob)
        setObjectUrl(nextObjectUrl)
      })
      .catch(() => {
        if (!cancelled) setError(true)
      })

    return () => {
      cancelled = true
      if (nextObjectUrl) URL.revokeObjectURL(nextObjectUrl)
    }
  }, [dataset, model, path])

  if (error) {
    return (
      <div className="flex h-full min-h-[120px] w-full flex-col items-center justify-center gap-2 text-xs text-slate-400">
        <AlertCircle className="h-5 w-5" />
        <span>图片加载失败</span>
      </div>
    )
  }

  if (!objectUrl) {
    return (
      <div className="flex h-full min-h-[120px] w-full items-center justify-center gap-2 text-xs text-slate-400">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span>加载图片...</span>
      </div>
    )
  }

  return (
    <img
      src={objectUrl}
      alt={name}
      className={className}
      style={style}
      loading={loading}
      draggable={draggable}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      onDragStart={onDragStart}
    />
  )
}

export function VisualizationTab({ dataset, mode, shouldLoad, selectedModel = "theta" }: VisualizationTabProps) {
  const [vizData, setVizData] = useState<VisualizationData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set())
  const [expandedTopics, setExpandedTopics] = useState<Set<string>>(new Set(["1"]))
  const [enlargedImage, setEnlargedImage] = useState<{ url: string; name: string; path: string; dataset: string } | null>(null)
  const [selectedImages, setSelectedImages] = useState<Set<string>>(new Set())
  // AI 解读结果缓存
  const [aiInterpretations, setAiInterpretations] = useState<Record<string, { status: 'loading' | 'done' | 'error'; text: string }>>({})
  // 图片预览缩放和拖拽状态
  const [zoomLevel, setZoomLevel] = useState(1)
  const [panPosition, setPanPosition] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 })

  useEffect(() => {
    if (!shouldLoad) return
    setLoading(true)
    setError(null)

    apiFetch<VisualizationData>(API_BASE, `/api/results/${encodeURIComponent(dataset)}/visualizations?model=${encodeURIComponent(selectedModel)}`)
      .then((data) => {
        setVizData(data)
      })
      .catch((e) => {
        console.error("Failed to load visualizations:", e)
        setError(e.message ?? "加载失败")
        setVizData(null)
      })
      .finally(() => setLoading(false))
  }, [dataset, shouldLoad, selectedModel])

  // 自动触发所有图片的 AI 解读，在客户端 useEffect 中执行避免 SSR 401
  useEffect(() => {
    if (!vizData) return
    return

    // 遍历所有全局图片
    vizData.global_files.forEach((file) => {
      const isHtml = file.name.endsWith(".html")
      if (isHtml || file.name.endsWith(".csv")) return

      // 在函数式更新中检查是否已经加载过，避免依赖外部 aiInterpretations 导致 stale closure
      setAiInterpretations(prev => {
        if (prev[file.path]) return prev // 已经存在，跳过
        return {
          ...prev,
          [file.path]: { status: 'loading', text: '' }
        }
      })

      ETMAgentAPI.analyzeChart('', file.name, 'general', 'zh', vizData.dataset, file.path, selectedModel || "theta")
        .then(result => {
          // 如果后端返回了分析内容（不管 success 是否 true），都显示出来
          const analysis = result.data?.analysis || result.analysis || '无法生成解读'
          const success = result.success ?? true
          setAiInterpretations(prev => ({
            ...prev,
            [file.path]: {
              status: success ? 'done' : 'error',
              text: analysis
            }
          }))
        })
        .catch((error) => {
          setAiInterpretations(prev => ({
            ...prev,
            [file.path]: {
              status: 'error',
              text: `请求失败: ${error.message || 'AI 解读暂时无法获取'}`
            }
          }))
        })
    })

    // 遍历所有主题图表图片
    Object.values(vizData.topic_files).forEach((files) => {
      files.forEach((file) => {
        const isHtml = file.name.endsWith(".html")
        if (isHtml || file.name.endsWith(".csv")) return

        // 在函数式更新中检查是否已经加载过
        setAiInterpretations(prev => {
          if (prev[file.path]) return prev // 已经存在，跳过
          return {
            ...prev,
            [file.path]: { status: 'loading', text: '' }
          }
        })

        ETMAgentAPI.analyzeChart('', file.name, 'general', 'zh', vizData.dataset, file.path, selectedModel || "theta")
          .then(result => {
            // 如果后端返回了分析内容（不管 success 是否 true），都显示出来
            const analysis = result.data?.analysis || result.analysis || '无法生成解读'
            const success = result.success ?? true
            setAiInterpretations(prev => ({
              ...prev,
              [file.path]: {
                status: success ? 'done' : 'error',
                text: analysis
              }
            }))
          })
          .catch((error) => {
            setAiInterpretations(prev => ({
              ...prev,
              [file.path]: {
                status: 'error',
                text: `请求失败: ${error.message || 'AI 解读暂时无法获取'}`
              }
            }))
          })
      })
    })
  }, [vizData, selectedModel, setAiInterpretations])

  const generateChartAnalysis = useCallback((file: VisualizationFile) => {
    if (!vizData) return

    setAiInterpretations(prev => ({
      ...prev,
      [file.path]: { status: 'loading', text: '' }
    }))

    ETMAgentAPI.analyzeChart('', file.name, 'general', 'zh', vizData.dataset, file.path, selectedModel || "theta")
      .then(result => {
        const analysis = result.data?.analysis || result.analysis || '无法生成解读'
        const success = result.success ?? true
        setAiInterpretations(prev => ({
          ...prev,
          [file.path]: {
            status: success ? 'done' : 'error',
            text: analysis
          }
        }))
      })
      .catch((error) => {
        setAiInterpretations(prev => ({
          ...prev,
          [file.path]: {
            status: 'error',
            text: `请求失败: ${error.message || 'AI 解读暂时无法获取'}`
          }
        }))
      })
  }, [vizData, selectedModel])

  const toggleFileSelection = (filePath: string) => {
    setSelectedFiles((prev) => {
      const next = new Set(prev)
      if (next.has(filePath)) {
        next.delete(filePath)
      } else {
        next.add(filePath)
      }
      return next
    })
  }

  const toggleImageSelection = (filePath: string) => {
    setSelectedImages((prev) => {
      const next = new Set(prev)
      if (next.has(filePath)) {
        next.delete(filePath)
      } else {
        next.add(filePath)
      }
      return next
    })
  }

  const sendSelectedToAI = () => {
    if (selectedImages.size === 0 || !vizData) return
    // Send to AI via custom event - send image URLs for AI to analyze
    const imagesToSend = vizData.global_files.filter((f) => selectedImages.has(f.path))
    window.dispatchEvent(new CustomEvent("theta:images-to-chat", {
      detail: imagesToSend.map((f) => ({ url: f.url, name: f.name, path: f.path, dataset: vizData.dataset }))
    }))
    setSelectedImages(new Set())
  }

  const toggleSelectAll = (files: VisualizationFile[]) => {
    const allSelected = files.every((f) => selectedFiles.has(f.path))
    setSelectedFiles((prev) => {
      const next = new Set(prev)
      if (allSelected) {
        files.forEach((f) => next.delete(f.path))
      } else {
        files.forEach((f) => next.add(f.path))
      }
      return next
    })
  }

  const fetchFileBlob = async (path: string) => {
    const token = localStorage.getItem("access_token")
    const response = await fetch(getProxyUrl(path), {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })
    if (!response.ok) {
      throw new Error("Failed to fetch file")
    }
    return response.blob()
  }

  const downloadFile = async (file: VisualizationFile) => {
    const blob = await fetchFileBlob(file.path)
    const objectUrl = URL.createObjectURL(blob)
    const anchor = document.createElement("a")
    anchor.href = objectUrl
    anchor.download = file.name
    document.body.appendChild(anchor)
    anchor.click()
    document.body.removeChild(anchor)
    URL.revokeObjectURL(objectUrl)
  }

  const openImageFullscreen = async (image: { path: string; name: string }) => {
    const blob = await fetchFileBlob(image.path)
    const objectUrl = URL.createObjectURL(blob)
    const win = window.open("")
    if (win) {
      win.document.write(`<html><head><title>${image.name}</title><style>body{margin:0;display:flex;justify-content:center;align-items:center;height:100vh;background:#000}img{max-width:100%;max-height:100%}#close{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);padding:10px 20px;background:#333;color:#fff;border:none;border-radius:5px;cursor:pointer}</style></head><body><img src="${objectUrl}" alt="${image.name}" /><button onclick="window.close()" id="close">关闭</button></body></html>`)
      win.addEventListener("beforeunload", () => URL.revokeObjectURL(objectUrl), { once: true })
    } else {
      URL.revokeObjectURL(objectUrl)
    }
  }

  const downloadSelected = () => {
    selectedFiles.forEach((path) => {
      const file = findFileByPath(path)
      if (file) downloadFile(file)
    })
  }

  const findFileByPath = (path: string): VisualizationFile | undefined => {
    if (!vizData) return undefined
    for (const f of vizData.global_files) {
      if (f.path === path) return f
    }
    for (const files of Object.values(vizData.topic_files)) {
      for (const f of files) {
        if (f.path === path) return f
      }
    }
    return undefined
  }

  const toggleTopicExpand = (topicId: string) => {
    setExpandedTopics((prev) => {
      const next = new Set(prev)
      if (next.has(topicId)) {
        next.delete(topicId)
      } else {
        next.add(topicId)
      }
      return next
    })
  }

  const getDisplayName = (filename: string): string => {
    return GLOBAL_FILE_LABELS[filename] || filename
  }

  // 通过后端代理获取文件内容（解决 CORS 和 Content-Disposition 问题）
  const fetchFileContent = useCallback(async (path: string): Promise<string> => {
    const token = localStorage.getItem("access_token")
    const response = await fetch(
      `${API_BASE}/api/results/${encodeURIComponent(dataset)}/visualizations/file?path=${encodeURIComponent(path)}`,
      {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      }
    )
    if (!response.ok) {
      throw new Error("Failed to fetch file content")
    }
    return response.text()
  }, [dataset])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-6 h-6 animate-spin text-slate-400 mr-2" />
        <span className="text-slate-500">加载可视化...</span>
      </div>
    )
  }

  if (error || !vizData) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-2 text-slate-500">
        <AlertCircle className="w-8 h-8 text-amber-400" />
        <p className="text-sm">{error ?? "暂无可视化数据"}</p>
      </div>
    )
  }

  const { global_files, topic_files } = vizData
  const topicIds = Object.keys(topic_files).sort((a, b) => parseInt(a) - parseInt(b))

  // 分离图片文件和非图片文件（HTML、CSV）
  const imageFiles = global_files.filter((f) => !f.name.endsWith(".html") && !f.name.endsWith(".csv"))
  const htmlFiles = global_files.filter((f) => f.name.endsWith(".html"))
  const csvFiles = global_files.filter((f) => f.name.endsWith(".csv"))

  return (
    <div className="space-y-6">
      {/* 顶部操作栏 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm text-slate-500">
            已选择 {selectedFiles.size} 个文件
          </span>
          {selectedFiles.size > 0 && (
            <button
              onClick={downloadSelected}
              className="inline-flex items-center gap-1 text-sm px-3 py-1.5 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              <Download className="w-4 h-4" />
              下载所选
            </button>
          )}
          {selectedImages.size > 0 && (
            <button
              onClick={sendSelectedToAI}
              className="inline-flex items-center gap-1 text-sm px-3 py-1.5 bg-emerald-600 text-white rounded-md hover:bg-emerald-700"
            >
              <Send className="w-4 h-4" />
              发送给 AI ({selectedImages.size})
            </button>
          )}
        </div>
      </div>

      {/* Global 全局图表 */}
      <section>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">全局图表</h3>

        {/* 图片文件网格 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {imageFiles.map((file) => {
            const isDownloadSelected = selectedFiles.has(file.path)
            const isSendSelected = selectedImages.has(file.path)

            return (
              <div
                key={file.path}
                className={`rounded-xl border overflow-hidden ${
                  isSendSelected ? "border-emerald-400 ring-2 ring-emerald-100" : isDownloadSelected ? "border-blue-300 ring-2 ring-blue-100" : "border-slate-200"
                }`}
              >
                {/* 头部工具栏 */}
                <div className="flex items-center gap-2 px-3 py-2 bg-slate-50 border-b border-slate-100">
                  <input
                    type="checkbox"
                    checked={isSendSelected}
                    onClick={(e) => e.stopPropagation()}
                    onMouseDown={(e) => e.stopPropagation()}
                    onChange={(e) => {
                      e.stopPropagation()
                      toggleImageSelection(file.path)
                    }}
                    className="h-4 w-4 rounded border-emerald-300 text-emerald-600 cursor-pointer"
                    title="选择发送给 AI"
                  />
                  <span className="text-sm font-medium text-slate-700 truncate flex-1">
                    {getDisplayName(file.name)}
                  </span>
                  <button
                    onClick={() => downloadFile(file)}
                    className="inline-flex items-center gap-1 text-xs px-2 py-1 text-slate-600 hover:bg-slate-100 rounded"
                  >
                    <Download className="w-3.5 h-3.5" />
                  </button>
                </div>

                {/* 预览区 */}
                <div
                  className="min-h-[180px] flex items-center justify-center bg-slate-100 max-h-[400px] overflow-auto relative group cursor-zoom-in"
                  draggable
                  onDragStart={(e) => {
                    e.dataTransfer.setData("application/x-theta-ai-drop", JSON.stringify({ type: "image", url: file.url, name: file.name, path: file.path, dataset: vizData.dataset }))
                    e.dataTransfer.setData("text/uri-list", file.url)
                    e.dataTransfer.setData("text/plain", file.url)
                  }}
                  onClick={() => setEnlargedImage({ url: file.url, name: file.name, path: file.path, dataset: vizData.dataset })}
                >
                  <AuthenticatedImage
                    dataset={vizData.dataset}
                    model={selectedModel || "theta"}
                    path={file.path}
                    name={file.name}
                    className="w-full h-full object-contain"
                    loading="lazy"
                  />
                  <div className="absolute inset-0 flex items-center justify-center bg-black/0 group-hover:bg-black/10 transition-colors">
                    <ZoomIn className="w-6 h-6 text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow-lg" />
                  </div>
                </div>
                {/* AI 解读结果 */}
                <div className="px-3 py-3 border-t border-slate-100 bg-emerald-50/50">
                  {!aiInterpretations[file.path] && (
                    <button
                      type="button"
                      onClick={() => generateChartAnalysis(file)}
                      className="text-sm font-medium text-emerald-700 hover:text-emerald-800"
                    >
                      生成 AI 解读
                    </button>
                  )}
                  {aiInterpretations[file.path]?.status === 'loading' && (
                    <div className="flex items-center justify-center gap-2 text-sm text-emerald-700 py-1">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>AI 正在分析图表...</span>
                    </div>
                  )}
                  {aiInterpretations[file.path]?.status === 'done' && (
                    <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap break-words">
                      <span className="font-semibold text-emerald-700">AI 解读：</span> {aiInterpretations[file.path].text}
                    </p>
                  )}
                  {aiInterpretations[file.path]?.status === 'error' && (
                    <div className="space-y-2">
                      <p className="text-sm text-slate-500">
                        {aiInterpretations[file.path].text}
                      </p>
                      <button
                        type="button"
                        onClick={() => generateChartAnalysis(file)}
                        className="text-sm font-medium text-emerald-700 hover:text-emerald-800"
                      >
                        重试解读
                      </button>
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>

        {/* HTML 文件（独立一行） */}
        {htmlFiles.map((file) => {
          const isSelected = selectedFiles.has(file.path)
          return (
            <div
              key={file.path}
              className={`mt-4 rounded-xl border overflow-hidden ${
                isSelected ? "border-blue-300 ring-2 ring-blue-100" : "border-slate-200"
              }`}
            >
              <div className="flex items-center gap-2 px-3 py-2 bg-slate-50 border-b border-slate-100">
                <input
                  type="checkbox"
                  checked={isSelected}
                  onClick={(e) => e.stopPropagation()}
                  onMouseDown={(e) => e.stopPropagation()}
                  onChange={(e) => {
                    e.stopPropagation()
                    toggleFileSelection(file.path)
                  }}
                  className="h-4 w-4 rounded border-slate-300 text-blue-600 cursor-pointer"
                />
                <span className="text-sm font-medium text-slate-700 truncate flex-1">
                  {getDisplayName(file.name)}
                </span>
                <button
                  onClick={() => downloadFile(file)}
                  className="inline-flex items-center gap-1 text-xs px-2 py-1 text-slate-600 hover:bg-slate-100 rounded"
                >
                  <Download className="w-3.5 h-3.5" />
                </button>
              </div>
              <div
                className="bg-slate-100 relative cursor-pointer group h-[60vh] min-h-[400px]"
                onClick={() => window.open(`/preview?dataset=${encodeURIComponent(dataset)}&path=${encodeURIComponent(file.path)}`, "_blank")}
              >
                <HtmlPreview dataset={dataset} model={selectedModel || "theta"} path={file.path} />
                {/* 悬停遮罩层 */}
                <div className="absolute inset-0 flex items-center justify-center bg-black/0 group-hover:bg-black/40 transition-colors">
                  <div className="text-white text-lg font-medium opacity-0 group-hover:opacity-100 transition-opacity bg-black/50 px-4 py-2 rounded-lg">
                    进入预览
                  </div>
                </div>
              </div>
            </div>
          )
        })}

        {/* CSV 文件（独立一行） */}
        {csvFiles.map((file) => {
          const isSelected = selectedFiles.has(file.path)
          return (
            <div
              key={file.path}
              className={`mt-4 rounded-xl border overflow-hidden ${
                isSelected ? "border-blue-300 ring-2 ring-blue-100" : "border-slate-200"
              }`}
            >
              <div className="flex items-center gap-2 px-3 py-2 bg-slate-50 border-b border-slate-100">
                <input
                  type="checkbox"
                  checked={isSelected}
                  onClick={(e) => e.stopPropagation()}
                  onMouseDown={(e) => e.stopPropagation()}
                  onChange={(e) => {
                    e.stopPropagation()
                    toggleFileSelection(file.path)
                  }}
                  className="h-4 w-4 rounded border-slate-300 text-blue-600 cursor-pointer"
                />
                <span className="text-sm font-medium text-slate-700 truncate flex-1">
                  {getDisplayName(file.name)}
                </span>
                <button
                  onClick={() => downloadFile(file)}
                  className="inline-flex items-center gap-1 text-xs px-2 py-1 text-slate-600 hover:bg-slate-100 rounded"
                >
                  <Download className="w-3.5 h-3.5" />
                </button>
              </div>
              <div className="bg-slate-100 max-h-[400px] overflow-auto">
                <CsvPreview dataset={dataset} path={file.path} />
              </div>
            </div>
          )
        })}
      </section>

      {/* 图片放大模态框 */}
      {enlargedImage && createPortal(
        <div
          className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/95"
          onClick={() => {
            setEnlargedImage(null)
            setZoomLevel(1)
            setPanPosition({ x: 0, y: 0 })
          }}
          onWheel={(e) => {
            e.preventDefault()
            e.stopPropagation()
            const delta = e.deltaY > 0 ? -0.1 : 0.1
            setZoomLevel((z) => Math.min(4, Math.max(0.5, z + delta)))
          }}
        >
          {/* 顶部工具栏 */}
          <div className="absolute top-4 left-4 flex items-center gap-3 z-10">
            <div className="bg-black/60 backdrop-blur-sm px-4 py-2 rounded-lg text-white text-sm font-medium">
              {enlargedImage.name}
            </div>
            <div className="bg-black/60 backdrop-blur-sm px-3 py-2 rounded-lg text-white text-xs">
              {Math.round(zoomLevel * 100)}%
            </div>
          </div>

          {/* 右侧工具栏 */}
          <div className="absolute top-4 right-4 flex items-center gap-2 z-10">
            {/* 缩小按钮 */}
            <button
              onClick={(e) => {
                e.stopPropagation()
                setZoomLevel((z) => Math.max(0.5, z - 0.25))
              }}
              className="p-2 text-white hover:bg-white/20 rounded-full transition-colors"
              title="缩小"
            >
              <ZoomOut className="w-5 h-5" />
            </button>
            {/* 放大按钮 */}
            <button
              onClick={(e) => {
                e.stopPropagation()
                setZoomLevel((z) => Math.min(4, z + 0.25))
              }}
              className="p-2 text-white hover:bg-white/20 rounded-full transition-colors"
              title="放大"
            >
              <ZoomIn className="w-5 h-5" />
            </button>
            {/* 重置按钮 */}
            <button
              onClick={(e) => {
                e.stopPropagation()
                setZoomLevel(1)
                setPanPosition({ x: 0, y: 0 })
              }}
              className="p-2 text-white hover:bg-white/20 rounded-full transition-colors"
              title="适应屏幕"
            >
              <RotateCcw className="w-5 h-5" />
            </button>
            <div className="w-px h-6 bg-white/30" />
            {/* 下载按钮 */}
            <button
              onClick={(e) => {
                e.stopPropagation()
                const file = findFileByPath(enlargedImage.path)
                if (file) void downloadFile(file)
              }}
              className="p-2 text-white hover:bg-white/20 rounded-full transition-colors"
              title="下载图片"
            >
              <Download className="w-5 h-5" />
            </button>
            {/* 全屏按钮 */}
            <button
              onClick={(e) => {
                e.stopPropagation()
                void openImageFullscreen(enlargedImage)
              }}
              className="p-2 text-white hover:bg-white/20 rounded-full transition-colors"
              title="全屏查看"
            >
              <Maximize2 className="w-5 h-5" />
            </button>
            {/* 发送给 AI 按钮 */}
            <button
              onClick={(e) => {
                e.stopPropagation()
                window.dispatchEvent(new CustomEvent("theta:images-to-chat", {
                  detail: [{ url: enlargedImage.url, name: enlargedImage.name, path: enlargedImage.path, dataset: enlargedImage.dataset }]
                }))
                setEnlargedImage(null)
                setZoomLevel(1)
                setPanPosition({ x: 0, y: 0 })
              }}
              className="p-2 text-white hover:bg-white/20 rounded-full transition-colors"
              title="发送给 AI"
            >
              <Send className="w-5 h-5" />
            </button>
            {/* 关闭按钮 */}
            <button
              onClick={() => {
                setEnlargedImage(null)
                setZoomLevel(1)
                setPanPosition({ x: 0, y: 0 })
              }}
              className="p-2 text-white hover:bg-white/20 rounded-full transition-colors"
              title="关闭"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* 图片容器 - 处理拖拽 */}
          <div
            className="relative overflow-hidden"
            style={{
              maxWidth: "90vw",
              maxHeight: "90vh",
              cursor: zoomLevel > 1 ? (isDragging ? "grabbing" : "grab") : "default",
            }}
            onMouseDown={(e) => {
              if (zoomLevel > 1) {
                e.preventDefault()
                setIsDragging(true)
                setDragStart({ x: e.clientX - panPosition.x, y: e.clientY - panPosition.y })
              }
            }}
            onMouseMove={(e) => {
              if (isDragging && zoomLevel > 1) {
                setPanPosition({
                  x: e.clientX - dragStart.x,
                  y: e.clientY - dragStart.y,
                })
              }
            }}
            onMouseUp={() => setIsDragging(false)}
            onMouseLeave={() => setIsDragging(false)}
            onClick={(e) => e.stopPropagation()}
          >
            <AuthenticatedImage
              dataset={enlargedImage.dataset}
              model={selectedModel || "theta"}
              path={enlargedImage.path}
              name={enlargedImage.name}
              className="object-contain select-none max-w-[90vw] max-h-[90vh]"
              loading="eager"
              style={{
                transform: zoomLevel !== 1 ? `translate(${panPosition.x}px, ${panPosition.y}px) scale(${zoomLevel})` : undefined,
                transformOrigin: "center center",
                transition: isDragging ? "none" : "transform 0.15s ease-out",
              }}
              onDoubleClick={(e) => {
                e.stopPropagation()
                if (zoomLevel > 1) {
                  setZoomLevel(1)
                  setPanPosition({ x: 0, y: 0 })
                } else {
                  setZoomLevel(2)
                }
              }}
              onDragStart={(e) => {
                e.dataTransfer.setData("application/x-theta-ai-drop", JSON.stringify({ type: "image", url: enlargedImage.url, name: enlargedImage.name, path: enlargedImage.path, dataset: enlargedImage.dataset }))
                e.dataTransfer.setData("text/uri-list", enlargedImage.url)
              }}
              draggable={zoomLevel === 1}
            />
          </div>

          {/* 操作提示 */}
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-4 text-white/50 text-xs">
            <span>双击切换缩放</span>
            <span className="w-px h-3 bg-white/30" />
            <span>拖拽平移</span>
            <span className="w-px h-3 bg-white/30" />
            <span>滚轮缩放</span>
            <span className="w-px h-3 bg-white/30" />
            <span>点击空白处关闭</span>
          </div>
        </div>,
        document.body
      )}

      {/* Topic 主题图表 */}
      <section>
        <h3 className="text-sm font-semibold text-slate-700 mb-3">主题图表</h3>
        <div className="space-y-3">
          {topicIds.map((topicId) => {
            const files = topic_files[topicId] || []
            const isExpanded = expandedTopics.has(topicId)

            return (
              <div key={topicId} className="rounded-xl border border-slate-200 overflow-hidden">
                {/* 主题折叠头 */}
                <button
                  onClick={() => toggleTopicExpand(topicId)}
                  className="w-full flex items-center gap-2 px-4 py-3 bg-slate-50 hover:bg-slate-100 transition-colors text-left"
                >
                  {isExpanded ? (
                    <ChevronUp className="w-4 h-4 text-slate-400 shrink-0" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-slate-400 shrink-0" />
                  )}
                  <span className="text-sm font-medium text-slate-700">主题 {topicId}</span>
                  <span className="text-xs text-slate-400">({files.length} 个图表)</span>
                </button>

                {/* 展开内容 */}
                {isExpanded && (
                  <div className="p-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {files.map((file) => {
                        const isHtml = file.name.endsWith(".html")
                        const isSelected = selectedFiles.has(file.path)
                        const displayName = file.name.replace(".png", "")

                        return (
                          <div
                            key={file.path}
                            className={`rounded-lg border overflow-hidden ${
                              isSelected ? "border-blue-300 ring-2 ring-blue-100" : "border-slate-200"
                            }`}
                          >
                            {/* 卡片头部 */}
                            <div className="flex items-center gap-2 px-3 py-2 bg-slate-50 border-b border-slate-100">
                              <input
                                type="checkbox"
                                checked={isSelected}
                                onClick={(e) => e.stopPropagation()}
                                onMouseDown={(e) => e.stopPropagation()}
                                onChange={(e) => {
                                  e.stopPropagation()
                                  toggleFileSelection(file.path)
                                }}
                                className="h-4 w-4 rounded border-slate-300 text-blue-600 cursor-pointer"
                              />
                              <span className="text-sm text-slate-700 truncate flex-1">
                                {displayName}
                              </span>
                              <button
                                onClick={() => downloadFile(file)}
                                className="inline-flex items-center gap-1 text-xs px-2 py-1 text-slate-600 hover:bg-slate-100 rounded"
                              >
                                <Download className="w-3.5 h-3.5" />
                              </button>
                            </div>

                            {/* 预览 */}
                            <div className="min-h-[140px] max-h-[240px] flex items-center justify-center bg-slate-100 overflow-auto relative group">
                              {isHtml ? (
                                <HtmlPreview dataset={dataset} model={selectedModel || "theta"} path={file.path} />
                              ) : file.name.endsWith(".csv") ? (
                                <CsvPreview dataset={dataset} path={file.path} />
                              ) : (
                                <>
                                  <AuthenticatedImage
                                    dataset={vizData.dataset}
                                    model={selectedModel || "theta"}
                                    path={file.path}
                                    name={file.name}
                                    className="w-full h-full object-contain cursor-zoom-in"
                                    loading="lazy"
                                    onClick={() => setEnlargedImage({ url: file.url, name: file.name, path: file.path, dataset: vizData.dataset })}
                                  />
                                  <button
                                    onClick={() => setEnlargedImage({ url: file.url, name: file.name, path: file.path, dataset: vizData.dataset })}
                                    className="absolute inset-0 flex items-center justify-center bg-black/0 group-hover:bg-black/10 transition-colors"
                                  >
                                    <ZoomIn className="w-5 h-5 text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow-lg" />
                                  </button>
                                </>
                              )}
                            </div>
                            {/* AI 解读结果 - 只对图片添加 */}
                            {!isHtml && !file.name.endsWith(".csv") && (
                              <div className="px-3 py-2 border-t border-slate-100 bg-emerald-50/50">
                                {!aiInterpretations[file.path] && (
                                  <button
                                    type="button"
                                    onClick={() => generateChartAnalysis(file)}
                                    className="text-sm font-medium text-emerald-700 hover:text-emerald-800"
                                  >
                                    生成 AI 解读
                                  </button>
                                )}
                                {aiInterpretations[file.path]?.status === 'loading' && (
                                  <div className="flex items-center justify-center gap-2 text-sm text-emerald-700 py-1">
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                    <span>AI 正在分析图表...</span>
                                  </div>
                                )}
                                {aiInterpretations[file.path]?.status === 'done' && (
                                  <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap break-words">
                                    <span className="font-semibold text-emerald-700">AI 解读：</span> {aiInterpretations[file.path].text}
                                  </p>
                                )}
                                {aiInterpretations[file.path]?.status === 'error' && (
                                  <div className="space-y-2">
                                    <p className="text-sm text-slate-500">
                                      {aiInterpretations[file.path].text}
                                    </p>
                                    <button
                                      type="button"
                                      onClick={() => generateChartAnalysis(file)}
                                      className="text-sm font-medium text-emerald-700 hover:text-emerald-800"
                                    >
                                      重试解读
                                    </button>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </section>
    </div>
  )
}
