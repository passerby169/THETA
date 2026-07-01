"use client"

import { useState } from "react"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Info } from "lucide-react"
import { cn } from "@/lib/utils"

// ==================== 模型配置 ====================

const LANGUAGES = [
  { value: "zh", label: "中文" },
  { value: "en", label: "English" },
] as const

const MODEL_LIST = [
  { id: "theta", name: "THETA", group: "neural" as const },
  { id: "nvdm", name: "NVDM", group: "neural" as const },
  { id: "gsm", name: "GSM", group: "neural" as const },
  { id: "prodlda", name: "ProdLDA", group: "neural" as const },
  { id: "ctm", name: "CTM", group: "neural" as const },
  { id: "etm", name: "ETM", group: "neural" as const },
  { id: "dtm", name: "DTM", group: "neural" as const },
  { id: "bertopic", name: "BERTopic", group: "neural" as const },
  { id: "lda", name: "LDA", group: "traditional" as const },
  { id: "hdp", name: "HDP", group: "traditional" as const },
  { id: "stm", name: "STM", group: "traditional" as const },
  { id: "btm", name: "BTM", group: "traditional" as const },
] as const

const MODEL_GROUPS = [
  {
    label: "神经",
    ids: ["theta", "nvdm", "gsm", "prodlda", "ctm", "etm", "dtm", "bertopic"],
  },
  {
    label: "传统",
    ids: ["lda", "hdp", "stm", "btm"],
  },
] as const

const QWEN_SIZES = [
  { value: "0.6B", label: "0.6B（默认）" },
  { value: "4B", label: "4B" },
  { value: "8B", label: "8B" },
] as const

const EMBEDDING_MODES = [
  { value: "zero_shot", label: "Zero-shot（默认）", tip: "不训练嵌入，最快" },
  { value: "unsupervised", label: "Unsupervised", tip: "无监督嵌入" },
  { value: "supervised", label: "Supervised", tip: "需额外指定标签列" },
] as const

// ==================== 类型 ====================

export interface AnalysisConfig {
  language: string
  models: string[]
  // THETA specific
  modelSize: string
  mode: "zero_shot" | "unsupervised" | "supervised"
  // Shared
  vocabSize: number
  // Per-model parameters
  theta: {
    numTopics: number
    epochs: number
    batchSize: number
    learningRate: number
    hiddenDim: number
    patience: number
  }
  nvdm: {
    numTopics: number
    epochs: number
    batchSize: number
    learningRate: number
    hiddenDim: number
  }
  gsm: {
    numTopics: number
    epochs: number
    batchSize: number
    learningRate: number
    hiddenDim: number
  }
  prodlda: {
    numTopics: number
    epochs: number
    batchSize: number
    learningRate: number
    hiddenDim: number
  }
  ctm: {
    numTopics: number
    epochs: number
    batchSize: number
    learningRate: number
    hiddenDim: number
  }
  etm: {
    numTopics: number
    epochs: number
    batchSize: number
    learningRate: number
    hiddenDim: number
  }
  dtm: {
    numTopics: number
    epochs: number
    batchSize: number
    learningRate: number
    hiddenDim: number
  }
  bertopic: {
    // auto-determines number of topics
  }
  lda: {
    numTopics: number
    maxIter: number
    dropout: number
  }
  hdp: {
    // auto-determines number of topics
  }
  stm: {
    numTopics: number
    maxIter: number
  }
  btm: {
    numTopics: number
    maxIter: number
  }
}

const DEFAULT_PER_MODEL: Omit<AnalysisConfig, 'language' | 'models' | 'vocabSize'> = {
  modelSize: "0.6B",
  mode: "zero_shot",
  theta: {
    numTopics: 20,
    epochs: 100,
    batchSize: 64,
    learningRate: 0.002,
    hiddenDim: 512,
    patience: 10,
  },
  nvdm: {
    numTopics: 20,
    epochs: 100,
    batchSize: 64,
    learningRate: 0.002,
    hiddenDim: 512,
  },
  gsm: {
    numTopics: 20,
    epochs: 100,
    batchSize: 64,
    learningRate: 0.002,
    hiddenDim: 512,
  },
  prodlda: {
    numTopics: 20,
    epochs: 100,
    batchSize: 64,
    learningRate: 0.002,
    hiddenDim: 512,
  },
  ctm: {
    numTopics: 20,
    epochs: 100,
    batchSize: 64,
    learningRate: 0.002,
    hiddenDim: 512,
  },
  etm: {
    numTopics: 20,
    epochs: 100,
    batchSize: 64,
    learningRate: 0.002,
    hiddenDim: 512,
  },
  dtm: {
    numTopics: 20,
    epochs: 100,
    batchSize: 64,
    learningRate: 0.002,
    hiddenDim: 512,
  },
  bertopic: {},
  lda: {
    numTopics: 20,
    maxIter: 1000,
    dropout: 0.1,
  },
  hdp: {},
  stm: {
    numTopics: 20,
    maxIter: 1000,
  },
  btm: {
    numTopics: 20,
    maxIter: 1000,
  },
}

const DEFAULT_CONFIG: AnalysisConfig = {
  language: "zh",
  models: ["theta"],
  vocabSize: 5000,
  ...DEFAULT_PER_MODEL,
}

// ==================== 组件 ====================

interface AnalysisConfigPanelProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onConfirm: (config: AnalysisConfig) => void
  datasetName: string
}

// Get model display name
function getModelName(modelId: string): string {
  return MODEL_LIST.find(m => m.id === modelId)?.name || modelId
}

export function AnalysisConfigPanel({
  open,
  onOpenChange,
  onConfirm,
  datasetName,
}: AnalysisConfigPanelProps) {
  const [config, setConfig] = useState<AnalysisConfig>({ ...DEFAULT_CONFIG })
  const [activeTab, setActiveTab] = useState<string>("theta")

  const handleModelToggle = (modelId: string, checked: boolean) => {
    setConfig(prev => {
      const current = new Set(prev.models)
      if (checked) {
        current.add(modelId)
        // Switch to the newly added model's tab
        if (current.size === 1 || prev.models.length === 0) {
          setActiveTab(modelId)
        }
      } else {
        current.delete(modelId)
        // If we're removing the active tab, switch to the first remaining
        if (modelId === activeTab && current.size > 0) {
          setActiveTab(Array.from(current)[0])
        }
      }
      const next = Array.from(current)
      return {
        ...prev,
        models: next.length ? next : ["theta"],
      }
    })
  }

  // Update a nested parameter for a specific model
  const updateModelParam = <M extends keyof AnalysisConfig, P extends keyof AnalysisConfig[M]>(
    modelId: M,
    param: P,
    value: AnalysisConfig[M][P]
  ) => {
    setConfig(prev => ({
      ...prev,
      [modelId]: {
        ...prev[modelId],
        [param]: value,
      },
    }))
  }

  const isNeural = (modelId: string): boolean => {
    return MODEL_LIST.find(m => m.id === modelId)?.group === "neural"
  }

  const handleConfirm = () => {
    if (config.models.length === 0) {
      setConfig(prev => ({ ...prev, models: ["theta"] }))
    }
    onConfirm(config)
    onOpenChange(false)
  }

  return (
    <TooltipProvider>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto" overlayPointerEvents="none">
          <DialogHeader>
            <DialogTitle>分析配置</DialogTitle>
            <DialogDescription>
              上传完成，请选择模型并配置参数。数据集：{datasetName}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-6 py-4">
            {/* 基础配置 - 语言和模型选择 */}
            <div className="space-y-4">
              {/* 数据语言 */}
              <div className="space-y-2">
                <Label>数据语言</Label>
                <RadioGroup
                  value={config.language}
                  onValueChange={v => setConfig(prev => ({ ...prev, language: v }))}
                  className="flex flex-wrap gap-4"
                >
                  {LANGUAGES.map(lang => {
                  return (
                    <div key={lang.value} className="flex items-center space-x-2">
                      <RadioGroupItem value={lang.value} id={`lang-${lang.value}`} />
                      <Label htmlFor={`lang-${lang.value}`} className="font-normal cursor-pointer">
                        {lang.label}
                      </Label>
                    </div>
                  );
                })}
                </RadioGroup>
              </div>

              {/* 模型选择 */}
              <div className="space-y-3">
                <Label>选择模型（可多选）</Label>
                {MODEL_GROUPS.map(group => (
                  <div key={group.label}>
                    <p className="text-xs text-slate-500 mb-2">{group.label}</p>
                    <div className="flex flex-wrap gap-3">
                      {group.ids.map(modelId => {
                        const model = MODEL_LIST.find(m => m.id === modelId)!
                        return (
                          <div key={modelId} className="flex items-center space-x-2">
                            <Checkbox
                              id={`model-${modelId}`}
                              checked={config.models.includes(modelId)}
                              onCheckedChange={checked =>
                                handleModelToggle(modelId, !!checked)
                              }
                            />
                            <Label
                              htmlFor={`model-${modelId}`}
                              className={cn(
                                "font-normal cursor-pointer text-sm",
                                config.models.includes(modelId) && "font-medium text-blue-700"
                              )}
                            >
                              {model.name}
                            </Label>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                ))}
              </div>

              {/* 全局词汇表大小 */}
              <div className="space-y-2">
                <Label className="text-sm">全局词汇表大小</Label>
                <Input
                  type="number"
                  min={1000}
                  max={20000}
                  value={config.vocabSize}
                  onChange={e =>
                    setConfig(prev => ({
                      ...prev,
                      vocabSize: parseInt(e.target.value) || 5000,
                    }))
                  }
                />
                <p className="text-xs text-slate-500">所有模型共用，推荐: 3000-10000</p>
              </div>
            </div>

            {/* 每个模型单独的参数配置选项卡 */}
            {config.models.length > 0 && (
              <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                <TabsList className="w-full flex-wrap h-auto p-1 bg-slate-100">
                  {config.models.map(modelId => (
                    <TabsTrigger
                      key={modelId}
                      value={modelId}
                      className="flex-1 min-w-[80px] data-[state=active]:bg-white"
                    >
                      {getModelName(modelId)}
                    </TabsTrigger>
                  ))}
                </TabsList>

                {/* THETA 配置 */}
                <TabsContent value="theta" className="mt-4">
                  <div className="space-y-6 p-6 bg-blue-50 rounded-lg border border-blue-100">
                    {/* THETA 专属配置 */}
                    <div className="space-y-4">
                      <h4 className="font-semibold text-blue-800">THETA 专属配置</h4>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label className="text-sm">Qwen 模型尺寸</Label>
                          <RadioGroup
                            value={config.modelSize}
                            onValueChange={v =>
                              setConfig(prev => ({ ...prev, modelSize: v }))
                            }
                            className="flex gap-3"
                          >
                            {QWEN_SIZES.map(s => {
                            return (
                              <div key={s.value} className="flex items-center space-x-2">
                                <RadioGroupItem value={s.value} id={`size-${s.value}`} />
                                <Label htmlFor={`size-${s.value}`} className="font-normal text-sm">
                                  {s.label}
                                </Label>
                              </div>
                            );
                          })}
                          </RadioGroup>
                        </div>
                        <div className="space-y-2">
                          <Label className="text-sm">嵌入模式</Label>
                          <RadioGroup
                            value={config.mode}
                            onValueChange={v =>
                              setConfig(prev => ({ ...prev, mode: v as any }))
                            }
                            className="flex flex-wrap gap-3"
                          >
                            {EMBEDDING_MODES.map(mode => {
                            return (
                              <Tooltip key={mode.value}>
                                <TooltipTrigger asChild>
                                  <div className="flex items-center space-x-2">
                                    <RadioGroupItem value={mode.value} id={`mode-${mode.value}`} />
                                    <Label htmlFor={`mode-${mode.value}`} className="font-normal text-sm cursor-pointer">
                                      {mode.label}
                                    </Label>
                                    <Info className="w-3.5 h-3.5 text-slate-400" />
                                  </div>
                                </TooltipTrigger>
                                <TooltipContent>{mode.tip}</TooltipContent>
                              </Tooltip>
                            );
                          })}
                          </RadioGroup>
                        </div>
                      </div>
                    </div>

                    {/* 通用超参数 */}
                    <div className="space-y-4">
                      <h4 className="font-semibold text-blue-800">训练超参数</h4>
                      <div className="grid grid-cols-2 gap-4">
                        {/* 主题数 */}
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Label className="text-sm">主题数</Label>
                            <span className="text-xs text-slate-500">5–100</span>
                          </div>
                          <Slider
                            value={[config.theta.numTopics]}
                            onValueChange={([v]) =>
                              updateModelParam("theta", "numTopics", v)
                            }
                            min={5}
                            max={100}
                            step={1}
                          />
                          <p className="text-xs text-slate-500">{config.theta.numTopics}</p>
                        </div>

                        {/* 训练轮数 */}
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Label className="text-sm">训练轮数</Label>
                            <span className="text-xs text-slate-500">10–500</span>
                          </div>
                          <Input
                            type="number"
                            min={10}
                            max={500}
                            value={config.theta.epochs}
                            onChange={e =>
                              updateModelParam("theta", "epochs", parseInt(e.target.value) || 100)
                            }
                          />
                        </div>

                        {/* 批大小 */}
                        <div className="space-y-2">
                          <Label className="text-sm">批大小</Label>
                          <Input
                            type="number"
                            min={8}
                            max={512}
                            value={config.theta.batchSize}
                            onChange={e =>
                              updateModelParam("theta", "batchSize", parseInt(e.target.value) || 64)
                          }
                          />
                        </div>

                        {/* 学习率 */}
                        <div className="space-y-2">
                          <Label className="text-sm">学习率</Label>
                          <Input
                            type="number"
                            min={1e-5}
                            max={0.1}
                            step={0.001}
                            value={config.theta.learningRate}
                            onChange={e =>
                              updateModelParam("theta", "learningRate", parseFloat(e.target.value) || 0.002)
                            }
                          />
                        </div>

                        {/* 隐藏层维度 */}
                        <div className="space-y-2">
                          <Label className="text-sm">隐藏层维度</Label>
                          <Input
                            type="number"
                            min={128}
                            max={1024}
                            value={config.theta.hiddenDim}
                            onChange={e =>
                              updateModelParam("theta", "hiddenDim", parseInt(e.target.value) || 512)
                            }
                          />
                        </div>

                        {/* Early Stopping 耐心值 */}
                        <div className="space-y-2">
                          <Label className="text-sm">Early Stopping 耐心值</Label>
                          <Input
                            type="number"
                            min={1}
                            max={100}
                            value={config.theta.patience}
                            onChange={e =>
                              updateModelParam("theta", "patience", parseInt(e.target.value) || 10)
                            }
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </TabsContent>

                {/* Generic Neural Model Content (NVDM, GSM, etc.) */}
                {["nvdm", "gsm", "prodlda", "ctm", "etm", "dtm"].map(modelId => (
                  <TabsContent key={modelId} value={modelId} className="mt-4">
                    <div className="space-y-6 p-6 bg-slate-50 rounded-lg border border-slate-200">
                      <div className="space-y-4">
                        <h4 className="font-semibold text-slate-800">{getModelName(modelId)} 训练超参数</h4>
                        <div className="grid grid-cols-2 gap-4">
                          {/* 主题数 */}
                          <div className="space-y-2">
                            <div className="flex justify-between">
                              <Label className="text-sm">主题数</Label>
                              <span className="text-xs text-slate-500">5–100</span>
                            </div>
                            <Slider
                              value={[config[modelId].numTopics]}
                              onValueChange={([v]) =>
                                updateModelParam(modelId as any, "numTopics", v)
                              }
                              min={5}
                              max={100}
                              step={1}
                            />
                            <p className="text-xs text-slate-500">{config[modelId].numTopics}</p>
                          </div>

                          {/* 训练轮数 */}
                          <div className="space-y-2">
                            <div className="flex justify-between">
                              <Label className="text-sm">训练轮数</Label>
                              <span className="text-xs text-slate-500">10–500</span>
                            </div>
                            <Input
                              type="number"
                              min={10}
                              max={500}
                              value={config[modelId].epochs}
                              onChange={e =>
                                updateModelParam(modelId as any, "epochs", parseInt(e.target.value) || 100)
                              }
                            />
                          </div>

                          {/* 批大小 */}
                          <div className="space-y-2">
                            <Label className="text-sm">批大小</Label>
                            <Input
                              type="number"
                              min={8}
                              max={512}
                              value={config[modelId].batchSize}
                              onChange={e =>
                                updateModelParam(modelId as any, "batchSize", parseInt(e.target.value) || 64)
                              }
                            />
                          </div>

                          {/* 学习率 */}
                          <div className="space-y-2">
                            <Label className="text-sm">学习率</Label>
                            <Input
                              type="number"
                              min={1e-5}
                              max={0.1}
                              step={0.001}
                              value={config[modelId].learningRate}
                              onChange={e =>
                                updateModelParam(modelId as any, "learningRate", parseFloat(e.target.value) || 0.002)
                              }
                            />
                          </div>

                          {/* 隐藏层维度 */}
                          <div className="space-y-2">
                            <Label className="text-sm">隐藏层维度</Label>
                            <Input
                              type="number"
                              min={128}
                              max={1024}
                              value={config[modelId].hiddenDim}
                              onChange={e =>
                                updateModelParam(modelId as any, "hiddenDim", parseInt(e.target.value) || 512)
                              }
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  </TabsContent>
                ))}

                {/* BERTopic */}
                <TabsContent value="bertopic" className="mt-4">
                  <div className="space-y-6 p-6 bg-slate-50 rounded-lg border border-slate-200">
                    <div className="space-y-4">
                      <h4 className="font-semibold text-slate-800">BERTopic 配置</h4>
                      <p className="text-sm text-slate-600">
                        BERTopic 自动确定主题数量，无需手动配置。使用默认参数即可获得较好效果。
                      </p>
                    </div>
                  </div>
                </TabsContent>

                {/* LDA */}
                <TabsContent value="lda" className="mt-4">
                  <div className="space-y-6 p-6 bg-slate-50 rounded-lg border border-slate-200">
                    <div className="space-y-4">
                      <h4 className="font-semibold text-slate-800">LDA 配置</h4>
                      <div className="grid grid-cols-2 gap-4">
                        {/* 主题数 */}
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Label className="text-sm">主题数</Label>
                            <span className="text-xs text-slate-500">5–100</span>
                          </div>
                          <Slider
                            value={[config.lda.numTopics]}
                            onValueChange={([v]) =>
                              updateModelParam("lda", "numTopics", v)
                            }
                            min={5}
                            max={100}
                            step={1}
                          />
                          <p className="text-xs text-slate-500">{config.lda.numTopics}</p>
                        </div>

                        {/* 最大迭代次数 */}
                        <div className="space-y-2">
                          <Label className="text-sm">最大迭代次数</Label>
                          <Input
                            type="number"
                            min={100}
                            max={5000}
                            value={config.lda.maxIter}
                            onChange={e =>
                              updateModelParam("lda", "maxIter", parseInt(e.target.value) || 1000)
                            }
                          />
                        </div>

                        {/* dropout */}
                        <div className="space-y-2">
                          <Label className="text-sm">Dropout 概率</Label>
                          <Input
                            type="number"
                            min={0}
                            max={0.5}
                            step={0.05}
                            value={config.lda.dropout}
                            onChange={e =>
                              updateModelParam("lda", "dropout", parseFloat(e.target.value) || 0.1)
                            }
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </TabsContent>

                {/* HDP */}
                <TabsContent value="hdp" className="mt-4">
                  <div className="space-y-6 p-6 bg-slate-50 rounded-lg border border-slate-200">
                    <div className="space-y-4">
                      <h4 className="font-semibold text-slate-800">HDP 配置</h4>
                      <p className="text-sm text-slate-600">
                        HDP 是非参数贝叶斯模型，自动确定主题数量，无需手动配置。
                      </p>
                    </div>
                  </div>
                </TabsContent>

                {/* STM */}
                <TabsContent value="stm" className="mt-4">
                  <div className="space-y-6 p-6 bg-slate-50 rounded-lg border border-slate-200">
                    <div className="space-y-4">
                      <h4 className="font-semibold text-slate-800">STM 配置</h4>
                      <div className="grid grid-cols-2 gap-4">
                        {/* 主题数 */}
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Label className="text-sm">主题数</Label>
                            <span className="text-xs text-slate-500">5–100</span>
                          </div>
                          <Slider
                            value={[config.stm.numTopics]}
                            onValueChange={([v]) =>
                              updateModelParam("stm", "numTopics", v)
                            }
                            min={5}
                            max={100}
                            step={1}
                          />
                          <p className="text-xs text-slate-500">{config.stm.numTopics}</p>
                        </div>

                        {/* 最大迭代次数 */}
                        <div className="space-y-2">
                          <Label className="text-sm">最大迭代次数</Label>
                          <Input
                            type="number"
                            min={100}
                            max={5000}
                            value={config.stm.maxIter}
                            onChange={e =>
                              updateModelParam("stm", "maxIter", parseInt(e.target.value) || 1000)
                            }
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </TabsContent>

                {/* BTM */}
                <TabsContent value="btm" className="mt-4">
                  <div className="space-y-6 p-6 bg-slate-50 rounded-lg border border-slate-200">
                    <div className="space-y-4">
                      <h4 className="font-semibold text-slate-800">BTM 配置</h4>
                      <div className="grid grid-cols-2 gap-4">
                        {/* 主题数 */}
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <Label className="text-sm">主题数</Label>
                            <span className="text-xs text-slate-500">5–100</span>
                          </div>
                          <Slider
                            value={[config.btm.numTopics]}
                            onValueChange={([v]) =>
                              updateModelParam("btm", "numTopics", v)
                            }
                            min={5}
                            max={100}
                            step={1}
                          />
                          <p className="text-xs text-slate-500">{config.btm.numTopics}</p>
                        </div>

                        {/* 最大迭代次数 */}
                        <div className="space-y-2">
                          <Label className="text-sm">最大迭代次数</Label>
                          <Input
                            type="number"
                            min={100}
                            max={5000}
                            value={config.btm.maxIter}
                            onChange={e =>
                              updateModelParam("btm", "maxIter", parseInt(e.target.value) || 1000)
                            }
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            )}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              取消
            </Button>
            <Button
              onClick={handleConfirm}
              disabled={config.models.length === 0}
              className="bg-gradient-to-r from-blue-600 to-indigo-600"
            >
              开始分析
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </TooltipProvider>
  )
}
