"use client"

import { useState, useRef, useCallback, useEffect } from "react"
import { createPortal } from "react-dom"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Send, Loader2, User, Info, Sparkles, ChevronDown, ChevronUp, GripHorizontal } from "lucide-react"
import { ETMAgentAPI } from "@/lib/api/etm-agent"
import { MarkdownRenderer } from "@/components/markdown-renderer"
import { cn } from "@/lib/utils"

interface Message {
  id: string
  role: "user" | "ai"
  content: string
  timestamp: string
}

interface ConfigAssistantFloatingPanelProps {
  config: object
  visible: boolean
  onClose: () => void
}

const CONFIG_HELP_CONTEXT = `
用户正在配置主题模型分析，当前配置项包括：

1. **数据语言 (language)**: 选择数据集的语言类型（中文/英文/德语/西班牙语），影响分词和预处理方式。

2. **模型选择 (models)**:
   - **神经模型**: THETA（推荐，融合主题模型与语言模型）、NVDM、GSM、ProdLDA、CTM、ETM、DTM、BERTopic
   - **传统模型**: LDA、HDP、STM、BTM
   - 可多选组合，如 THETA + LDA 进行对比实验

3. **Qwen 尺寸 (modelSize)**: 仅 THETA 模型使用，影响 0.6B/4B/8B 不同规模的 Qwen 语言模型

4. **嵌入模式 (mode)**:
   - Zero-shot: 不训练嵌入，最快
   - Unsupervised: 无监督嵌入训练
   - Supervised: 需指定标签列，有监督嵌入

5. **主题数 (numTopics)**: 5-100，主题越多越细粒度，但可能导致过拟合。HDP/BERTopic 会自动确定主题数。

6. **训练轮数 (epochs)**: 10-500，神经模型训练迭代次数，太多容易过拟合，太少欠拟合。

7. **批大小 (batchSize)**: 8-512，每批次处理的文档数，影响显存和训练稳定性。

8. **学习率 (learningRate)**: 1e-5 到 0.1，控制参数更新步长，太高不稳定，太低收敛慢。

9. **隐藏层维度 (hiddenDim)**: 128-1024，模型内部表示维度，更大可能捕捉更复杂模式但耗算力。

10. **词汇表大小 (vocabSize)**: 1000-20000，保留的最高频词数，过小丢失信息，过大引入噪声。

请根据用户问题，用简洁易懂的语言解释这些配置项的作用和调参建议。
`

function generateId() {
  return `config-msg-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
}

function getTimestamp() {
  return new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
}

export function ConfigAssistantFloatingPanel({
  config,
  visible,
  onClose,
}: ConfigAssistantFloatingPanelProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: generateId(),
      role: "ai",
      content: "您好！我是配置助手，可以帮您理解各项分析参数的含义。\n\n您可以问我：\n- 「主题数是什么？」\n- 「学习率要怎么调？」\n- 「THETA 和 LDA 有什么区别？」\n- 「嵌入模式选哪个好？」",
      timestamp: getTimestamp(),
    },
  ])
  const [inputValue, setInputValue] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isExpanded, setIsExpanded] = useState(false)
  const [isMinimized, setIsMinimized] = useState(false)
  const [position, setPosition] = useState({ x: 24, y: 100 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
  const scrollRef = useRef<HTMLDivElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // 拖拽相关 - 只在点击header时开始拖拽
  const handleHeaderMouseDown = useCallback((e: React.MouseEvent) => {
    // 只在点击 header 时开始拖拽，不阻止其他事件
    const target = e.target as HTMLElement
    if (target.closest("button")) return

    e.preventDefault()
    setIsDragging(true)
    setDragOffset({
      x: e.clientX - position.x,
      y: e.clientY - position.y,
    })
  }, [position])

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging) return
    setPosition({
      x: e.clientX - dragOffset.x,
      y: e.clientY - dragOffset.y,
    })
  }, [isDragging, dragOffset])

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleSend = useCallback(async () => {
    const trimmed = inputValue.trim()
    if (!trimmed || isLoading) return

    const userMessage: Message = {
      id: generateId(),
      role: "user",
      content: trimmed,
      timestamp: getTimestamp(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInputValue("")
    setIsLoading(true)

    try {
      const response = await ETMAgentAPI.chat(trimmed, {
        current_page: "analysis_config",
        current_view_name: "分析配置",
        current_view: "config_dialog",
        config_context: CONFIG_HELP_CONTEXT,
        actual_config: config,
      })

      const text = response.message ?? (response as { response?: string }).response ?? "抱歉，暂时无法回答您的问题。"

      const aiMessage: Message = {
        id: generateId(),
        role: "ai",
        content: text,
        timestamp: getTimestamp(),
      }

      setMessages((prev) => [...prev, aiMessage])
    } catch {
      const errorMessage: Message = {
        id: generateId(),
        role: "ai",
        content: "抱歉，连接失败，请稍后再试。",
        timestamp: getTimestamp(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }, [inputValue, isLoading, config])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // 点击展开/收起
  const handleToggleExpand = () => {
    if (!isExpanded) {
      setIsExpanded(true)
      setIsMinimized(false)
    } else {
      setIsExpanded(false)
    }
  }

  // 点击最小化为猫图标
  const handleMinimize = () => {
    setIsMinimized(true)
    setIsExpanded(false)
  }

  // 阻止事件冒泡到 Dialog（Radix 在 document 级别检测点击）
  const handlePanelClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    e.nativeEvent.stopImmediatePropagation()
  }

  useEffect(() => {
    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove)
      document.addEventListener("mouseup", handleMouseUp)
      return () => {
        document.removeEventListener("mousemove", handleMouseMove)
        document.removeEventListener("mouseup", handleMouseUp)
      }
    }
  }, [isDragging, handleMouseMove, handleMouseUp])

  if (!visible) return null

  // 最小化状态：只显示猫图标，可拖动
  if (isMinimized) {
    const minimizedContent = (
      <div
        ref={panelRef}
        className="fixed z-[9999]"
        style={{ left: position.x, top: position.y }}
        onClick={handlePanelClick}
      >
        <div
          className="w-14 h-14 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 shadow-lg border-2 border-white flex items-center justify-center cursor-grab active:cursor-grabbing hover:scale-105 transition-transform"
          onMouseDown={handleHeaderMouseDown}
          onClick={(e) => {
            e.stopPropagation()
            handleToggleExpand()
          }}
        >
          <img
            src="/ai-avatar.png"
            alt="配置助手"
            className="w-10 h-10 rounded-full object-contain"
          />
        </div>
      </div>
    )

    if (typeof document === "undefined") return null
    return createPortal(minimizedContent, document.body)
  }

  const panelContent = (
    <div
      ref={panelRef}
      className="fixed z-[9999] flex flex-col bg-white rounded-xl shadow-2xl border border-slate-200/60 overflow-hidden"
      style={{
        left: position.x,
        top: position.y,
        width: "360px",
      }}
      onClick={handlePanelClick}
    >
      {/* Header - 可拖动 */}
      <div
        className="flex-shrink-0 flex items-center justify-between px-4 py-3 border-b border-slate-100 bg-gradient-to-r from-slate-50 to-white cursor-grab select-none"
        onMouseDown={handleHeaderMouseDown}
      >
        <div className="flex items-center gap-2">
          <img
            src="/ai-avatar.png"
            alt="猫咪科学家"
            className="h-8 w-8 rounded-lg object-contain"
          />
          <div>
            <p className="text-sm font-semibold text-slate-800">配置助手</p>
            {isExpanded && (
              <p className="text-[10px] text-slate-500">帮您理解分析参数</p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-slate-400 hover:text-slate-600 hover:bg-slate-100"
            onClick={(e) => {
              e.stopPropagation()
              handleToggleExpand()
            }}
            title={isExpanded ? "收起" : "展开"}
          >
            {isExpanded ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-slate-400 hover:text-slate-600 hover:bg-slate-100"
            onClick={(e) => {
              e.stopPropagation()
              handleMinimize()
            }}
            title="最小化为图标"
          >
            <GripHorizontal className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-slate-400 hover:text-red-500 hover:bg-red-50"
            onClick={(e) => {
              e.stopPropagation()
              onClose()
            }}
            title="关闭"
          >
            <ChevronDown className="h-4 w-4 rotate-45" />
          </Button>
        </div>
      </div>

      {/* 收起状态提示 */}
      {!isExpanded && (
        <div
          className="p-4 bg-slate-50 cursor-pointer hover:bg-slate-100 transition-colors"
          onClick={(e) => {
            e.stopPropagation()
            handleToggleExpand()
          }}
        >
          <div className="flex items-center gap-2 text-sm text-slate-600">
            <Info className="h-4 w-4" />
            <span>点击展开配置助手</span>
          </div>
        </div>
      )}

      {/* 展开状态：显示完整内容 */}
      {isExpanded && (
        <>
          <div className="flex-1 min-h-0 max-h-[400px] overflow-hidden flex flex-col">
            {/* Messages */}
            <ScrollArea className="flex-1 min-h-0 px-3 py-3" ref={scrollRef}>
              <div className="space-y-4">
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={cn(
                      "flex gap-2.5",
                      msg.role === "user" ? "flex-row-reverse" : "flex-row"
                    )}
                  >
                    {/* Avatar */}
                    <div
                      className={cn(
                        "h-7 w-7 rounded-full flex items-center justify-center shrink-0 overflow-hidden",
                        msg.role === "ai" ? "bg-slate-100" : "bg-blue-100"
                      )}
                    >
                      {msg.role === "ai" ? (
                        <img src="/ai-avatar.png" alt="AI" className="h-full w-full object-contain" />
                      ) : (
                        <User className="h-4 w-4 text-blue-600" />
                      )}
                    </div>

                    {/* Bubble */}
                    <div
                      className={cn(
                        "max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm",
                        msg.role === "ai"
                          ? "bg-white border border-slate-200/80 shadow-sm text-slate-700"
                          : "bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-md shadow-blue-200/40"
                      )}
                    >
                      {msg.role === "ai" ? (
                        <div className="prose prose-xs prose-slate max-w-none">
                          <MarkdownRenderer content={msg.content} />
                        </div>
                      ) : (
                        <p className="leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                      )}
                      <p
                        className={cn(
                          "text-[10px] mt-1",
                          msg.role === "ai" ? "text-slate-400" : "text-blue-200"
                        )}
                      >
                        {msg.timestamp}
                      </p>
                    </div>
                  </div>
                ))}

                {/* Loading indicator */}
                {isLoading && (
                  <div className="flex gap-2.5">
                    <div className="h-7 w-7 rounded-full flex items-center justify-center shrink-0 overflow-hidden bg-slate-100">
                      <img src="/ai-avatar.png" alt="AI" className="h-full w-full object-contain" />
                    </div>
                    <div className="bg-white border border-slate-200/80 rounded-2xl px-3.5 py-2.5 shadow-sm">
                      <div className="flex items-center gap-2">
                        <Loader2 className="h-3.5 w-3.5 text-blue-500 animate-spin" />
                        <span className="text-xs text-slate-500">思考中...</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>

            {/* Input */}
            <div className="flex-shrink-0 p-3 border-t border-slate-100 bg-white/80">
              <div className="flex items-end gap-2">
                <Textarea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="输入您的问题..."
                  className="min-h-[50px] max-h-[80px] resize-none text-sm"
                  disabled={isLoading}
                />
                <Button
                  size="sm"
                  className="h-[50px] px-4 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-md disabled:opacity-50 shrink-0"
                  onClick={handleSend}
                  disabled={!inputValue.trim() || isLoading}
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
              <div className="flex items-center gap-1 mt-2">
                <Sparkles className="h-3 w-3 text-amber-500" />
                <p className="text-[10px] text-slate-400">
                  试试问：「主题数设多少合适？」
                </p>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )

  if (typeof document === "undefined") return null
  return createPortal(panelContent, document.body)
}
