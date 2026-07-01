"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Send, X, Loader2, Bot, User, Info, Sparkles } from "lucide-react"
import { ETMAgentAPI } from "@/lib/api/etm-agent"
import { MarkdownRenderer } from "@/components/markdown-renderer"
import { cn } from "@/lib/utils"

interface Message {
  id: string
  role: "user" | "ai"
  content: string
  timestamp: string
}

interface ConfigAssistantPanelProps {
  config: object
  onClose?: () => void
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

export function ConfigAssistantPanel({ config, onClose }: ConfigAssistantPanelProps) {
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
  const scrollRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

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

  return (
    <div className="flex flex-col h-full bg-gradient-to-b from-slate-50 to-white border-l border-slate-200/60">
      {/* Header */}
      <div className="flex-shrink-0 flex items-center justify-between px-4 py-3 border-b border-slate-100 bg-white/80 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-md shadow-blue-200/50">
            <Bot className="h-4 w-4 text-white" />
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-800">配置助手</p>
            <p className="text-[10px] text-slate-500">帮您理解分析参数</p>
          </div>
        </div>
        {onClose && (
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-slate-400 hover:text-slate-600 hover:bg-slate-100"
            onClick={onClose}
          >
            <X className="h-4 w-4" />
          </Button>
        )}
      </div>

      {/* Info Banner */}
      <div className="flex-shrink-0 mx-3 mt-3 p-3 bg-blue-50 border border-blue-100 rounded-lg">
        <div className="flex items-start gap-2">
          <Info className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
          <p className="text-xs text-blue-700 leading-relaxed">
            左侧 AI 助手可以帮您理解各项配置的作用。点击参数名称可直接跳转到对应设置。
          </p>
        </div>
      </div>

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
                  "h-7 w-7 rounded-full flex items-center justify-center shrink-0",
                  msg.role === "ai"
                    ? "bg-gradient-to-br from-blue-500 to-indigo-600"
                    : "bg-slate-200"
                )}
              >
                {msg.role === "ai" ? (
                  <Bot className="h-3.5 w-3.5 text-white" />
                ) : (
                  <User className="h-3.5 w-3.5 text-slate-600" />
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
              <div className="h-7 w-7 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shrink-0">
                <Bot className="h-3.5 w-3.5 text-white" />
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
      <div className="flex-shrink-0 p-3 border-t border-slate-100 bg-white/80 backdrop-blur-sm">
        <div className="flex items-end gap-2">
          <Textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入您的问题..."
            className="min-h-[60px] max-h-[120px] resize-none text-sm"
            disabled={isLoading}
          />
          <Button
            size="sm"
            className="h-[60px] px-4 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-md disabled:opacity-50 shrink-0"
            onClick={handleSend}
            disabled={!inputValue.trim() || isLoading}
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
        <div className="flex items-center gap-1 mt-2">
          <Sparkles className="h-3 w-3 text-amber-500" />
          <p className="text-[10px] text-slate-400">
            试试问：「主题数设多少合适？」或「THETA 相比 LDA 有什么优势？」
          </p>
        </div>
      </div>
    </div>
  )
}
