"use client"

import { useState, useCallback, useEffect, useRef } from "react"
import { useRouter } from "next/navigation"
import { AppShell, type Tab } from "@/components/layout/app-shell"
import { ProjectHub, type Project } from "@/components/dashboard/project-hub"
import { NewProjectDialog, type NewProjectData } from "@/components/dashboard/new-project-dialog"
import { AutoPipeline } from "@/components/project/auto-pipeline"
import type { ChatMessage, SuggestionCard, SendMessagePayload } from "@/components/chat/ai-sidebar"
import { ProtectedRoute } from "@/components/protected-route"
import { apiFetch, API_BASE } from "@/lib/api/config"
import { ETMAgentAPI, DatasetInfo } from "@/lib/api/etm-agent"
import { PROMPTS } from "@/lib/config"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { TopicWordsTab } from "@/components/results/topic-words-tab"
import { MetricsTab } from "@/components/results/metrics-tab"
import { VisualizationTab } from "@/components/results/visualization-tab"
import { ExportTab } from "@/components/results/export-tab"

/** 指标展示名与方向说明：↑ 越高越好 | ↓ 越低越好 | → 越接近 0 越好 */
// Helper to generate timestamp
function getTimestamp() {
  return new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
}

// Generate unique ID
function generateId() {
  return `msg-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`
}

// Extended project type with additional fields
interface WorkspaceProject extends Project {
  description?: string
  datasetName?: string
  mode?: "zero_shot" | "unsupervised" | "supervised"
  models?: string[]
  numTopics?: number
  pipelineStatus?: "running" | "completed" | "error"
  dbProjectId?: number  // 数据库项目 ID，用于更新/删除
  taskId?: string | null  // 关联的训练任务 ID
}

// SessionStorage keys for tab state persistence
const STORAGE_KEYS = {
  TABS: "theta_dashboard_tabs",
  ACTIVE_TAB: "theta_dashboard_active_tab",
} as const

function DashboardContent() {
  const router = useRouter()

  // Initialize tabs from sessionStorage
  const [tabs, setTabs] = useState<Tab[]>(() => {
    if (typeof window !== "undefined") {
      const saved = sessionStorage.getItem(STORAGE_KEYS.TABS)
      if (saved) {
        try {
          const parsed = JSON.parse(saved)
          if (Array.isArray(parsed) && parsed.length > 0) return parsed
        } catch { /* ignore */ }
      }
    }
    return [{ id: "hub", title: "项目中心", closable: false }]
  })

  // Initialize activeTabId from sessionStorage
  const [activeTabId, setActiveTabId] = useState<string>(() => {
    if (typeof window !== "undefined") {
      const saved = sessionStorage.getItem(STORAGE_KEYS.ACTIVE_TAB)
      if (saved) return saved
    }
    return "hub"
  })
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([])
  const [isNewProjectDialogOpen, setIsNewProjectDialogOpen] = useState(false)
  const [projects, setProjects] = useState<WorkspaceProject[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [dynamicSuggestions, setDynamicSuggestions] = useState<SuggestionCard[]>([])
  const [projectTransitionName, setProjectTransitionName] = useState<string | null>(null)
  /** 用于强制重新渲染 renderContent（PLC 完成切换到结果视图时使用） */
  const [renderKey, setRenderKey] = useState(0)

  const transitionTimerRef = useRef<number | null>(null)
  const pollingTimerRef = useRef<number | null>(null)
  const syncTimerRef = useRef<number | null>(null)
  const refreshProjectsRef = useRef<() => Promise<void>>(() => {})

  const handleSendMessageRef = useRef<(payload: string | SendMessagePayload) => void | Promise<void>>(() => {})

  useEffect(() => {
    return () => {
      if (transitionTimerRef.current) {
        window.clearTimeout(transitionTimerRef.current)
      }
      if (syncTimerRef.current) {
        window.clearInterval(syncTimerRef.current)
      }
    }
  }, [])

  // Load projects: 优先数据库（用户关联），再合并 datasets + tasks
  useEffect(() => {
    const loadProjects = async () => {
      try {
        console.log("[Dashboard] Loading projects...");
        const [dbProjects, datasets, tasks, ossInfo] = await Promise.all([
          ETMAgentAPI.getProjects(),
          ETMAgentAPI.getDatasets(),
          ETMAgentAPI.getTasks({ limit: 100 }).catch(() => []),
          ETMAgentAPI.listOssDatasets().catch(() => ({ datasets: [] as { name: string; chart_count: number }[] })),
        ])

        // 构建有结果的 dataset 名称集合（OSS 上有图表文件即为有结果）
        const datasetsWithResults = new Set(ossInfo.datasets.map((d: { name: string }) => d.name))

        console.log("[Dashboard] Results - datasets:", datasets);
        console.log("[Dashboard] OSS datasets with results:", datasetsWithResults);
        const seen = new Set<string>()
        const list: WorkspaceProject[] = []

        // 预构建 dataset→task 映射：优先已完成的任务
        const taskByDataset = new Map<string, { task_id: string; status: string; pipeline_status?: string }>()
        for (const t of tasks) {
          const ds = t.dataset || (t as any).dataset_name
          if (!ds) continue
          const existing = taskByDataset.get(ds)
          if (!existing || (t.status === "completed" && existing.status !== "completed")) {
            taskByDataset.set(ds, { task_id: t.task_id, status: t.status, pipeline_status: t.status === "completed" ? "completed" : t.status === "failed" ? "error" : "running" })
          }
        }

        // 1. 数据库中的项目（用户关联，跨设备同步）
        for (const p of dbProjects) {
          const key = p.dataset_name || `db-${p.id}`
          seen.add(key)
          // 如果项目没有 task_id，尝试从任务列表中匹配
          let effectiveTaskId = p.task_id ?? null
          let effectivePipelineStatus = p.pipeline_status
          if (!effectiveTaskId && p.dataset_name) {
            const matched = taskByDataset.get(p.dataset_name)
            if (matched) {
              effectiveTaskId = matched.task_id
              effectivePipelineStatus = effectivePipelineStatus || matched.pipeline_status
            }
          }
          // 如果数据库已标记完成，即使 OSS API 返回空也认为有结果（解决未登录时 OSS API 401 问题）
          const hasResults = p.dataset_name ? (datasetsWithResults.has(p.dataset_name) || p.pipeline_status === "completed") : false
          const derivedPipelineStatus = effectivePipelineStatus === "completed" ? "completed"
            : effectivePipelineStatus === "error" ? "error"
            : effectivePipelineStatus === "running" ? "running"
            : effectiveTaskId ? "running"
            : p.dataset_name && datasetsWithResults.has(p.dataset_name) ? "completed"
            : undefined
          list.push({
            id: `proj-db-${p.id}`,
            name: p.name,
            rows: 0,
            createdAt: p.created_at ? "已保存" : "刚刚",
            status: derivedPipelineStatus === "completed" ? "completed"
              : derivedPipelineStatus === "error" ? "no_result"
              : derivedPipelineStatus === "running" ? "vectorizing"
              : p.dataset_name ? "draft"
              : "draft" as const,
            datasetName: p.dataset_name ?? undefined,
            mode: (p.mode as any) ?? "zero_shot",
            models: ["theta"],
            numTopics: p.num_topics ?? 20,
            pipelineStatus: derivedPipelineStatus as any,
            hasResults,
            dbProjectId: p.id,
            taskId: effectiveTaskId,
          })
        }

        // 2. 数据集（未在 DB 中的）
        for (const ds of datasets) {
          if (seen.has(ds.name)) continue
          seen.add(ds.name)
          const hasResults = datasetsWithResults.has(ds.name)
          // 检查是否有正在运行的任务，如果有，使用任务状态
          let effectivePipelineStatus: "running" | "completed" | "error" | undefined =
            hasResults ? "completed" : undefined
          let effectiveTaskId: string | null = null
          const matchedTask = taskByDataset.get(ds.name)
          if (matchedTask) {
            effectivePipelineStatus = matchedTask.pipeline_status as any
            effectiveTaskId = matchedTask.task_id
          }
          list.push({
            id: `proj-${ds.name}`,
            name: ds.name,
            rows: ds.size ?? (ds as any).file_count ?? 0,
            createdAt: "已上传",
            status: hasResults && !effectivePipelineStatus ? "completed" as const : "draft" as const,
            pipelineStatus: effectivePipelineStatus,
            hasResults,
            datasetName: ds.name,
            taskId: effectiveTaskId,
            models: ["theta"],
          })
        }

        // 3. OSS 上已有结果但不在 DB files 中的数据集
        for (const ossDatasetName of Array.from(datasetsWithResults)) {
          if (seen.has(ossDatasetName)) continue
          seen.add(ossDatasetName)
          list.push({
            id: `proj-${ossDatasetName}`,
            name: ossDatasetName,
            rows: 0,
            createdAt: "已分析",
            status: "completed" as const,
            pipelineStatus: "completed" as const,
            hasResults: true,
            datasetName: ossDatasetName,
            models: ["theta"],
          })
        }

        // 3. 任务中的数据集
        for (const t of tasks) {
          const ds = t.dataset || (t as any).dataset_name
          if (ds && !seen.has(ds)) {
            seen.add(ds)
            const hasResults = datasetsWithResults.has(ds)
            list.push({
              id: `proj-${ds}`,
              name: ds,
              rows: 0,
              createdAt: "已分析",
              status: hasResults ? "completed" as const : (t.status === "completed" ? "no_result" as const : "vectorizing" as const),
              pipelineStatus: t.status === "completed" ? "completed" : t.status === "failed" ? "error" : "running",
              hasResults,
              datasetName: ds,
              models: ["theta"],
              taskId: t.task_id || null,
            })
          }
        }

        setProjects(list)
      } catch (error) {
        console.error("Failed to load projects:", error)
      } finally {
        setIsLoading(false)
      }
    }
    loadProjects()
  }, [])

  // Validate activeTabId after projects load: if active tab no longer exists, fallback to hub
  useEffect(() => {
    if (isLoading) return
    const tabExists = tabs.some(t => t.id === activeTabId)
    if (!tabExists && tabs.length > 0) {
      setActiveTabId("hub")
    }
  }, [isLoading, projects, tabs, activeTabId])

  useEffect(() => {
    if (isLoading || activeTabId === "hub") return
    const currentProject = projects.find(p => p.id === activeTabId)
    if (!currentProject) {
      setActiveTabId("hub")
    }
  }, [isLoading, activeTabId, projects])

  // 轮询训练状态
  useEffect(() => {
    const pollTrainingStatus = async () => {
      const runningJobs = projects.filter(p => p.pipelineStatus === "running" || p.status === "vectorizing");
      if (runningJobs.length === 0) return;

      const jobIds: number[] = [];
      for (const job of runningJobs) {
        if (job.taskId) {
          const numId = parseInt(job.taskId.replace("job-", ""), 10);
          if (!isNaN(numId)) jobIds.push(numId);
        }
      }
      if (jobIds.length === 0) return;

      try {
        const results = await Promise.all(jobIds.map(id => ETMAgentAPI.getTrainStatusByJobId(id)));
        setProjects(prev => prev.map(p => {
          if (p.taskId) {
            const numId = parseInt(p.taskId.replace("job-", ""), 10);
            const result = results.find((r, i) => jobIds[i] === numId);
            if (result) {
              const newStatus = result.status === "completed" || result.status === "succeeded" ? "completed"
                : result.status === "failed" ? "error"
                : result.status === "running" || result.status === "training" ? "running"
                : p.pipelineStatus;
              return {
                ...p,
                pipelineStatus: newStatus,
                status: newStatus === "completed" ? "completed"
                  : newStatus === "error" ? "no_result"
                  : newStatus === "running" ? "vectorizing"
                  : p.status,
                hasResults: newStatus === "completed" ? true : p.hasResults,
              };
            }
          }
          return p;
        }));
      } catch (err) {
        console.error("[Polling] Error:", err);
      }
    };

    pollingTimerRef.current = window.setInterval(pollTrainingStatus, 10000);
    return () => { if (pollingTimerRef.current) window.clearInterval(pollingTimerRef.current); };
  }, [projects]);

  // 定期同步项目列表已移除，避免打断上传操作
  // 需要刷新请手动点击刷新按钮
  syncTimerRef.current = null;

  // 刷新项目列表（与 load 相同逻辑，保留正在运行的项目）
  const refreshProjects = useCallback(async () => {
    try {
      console.log("[Dashboard] Refreshing projects...");
      const [dbProjects, datasets, tasks, ossInfo] = await Promise.all([
        ETMAgentAPI.getProjects(),
        ETMAgentAPI.getDatasets(),
        ETMAgentAPI.getTasks({ limit: 100 }).catch(() => []),
        ETMAgentAPI.listOssDatasets().catch(() => ({ datasets: [] as { name: string; chart_count: number }[] })),
      ])

      // 构建有结果的 dataset 名称集合（OSS 上有图表文件即为有结果）
      const datasetsWithResults = new Set(ossInfo.datasets.map((d: { name: string }) => d.name))
      const seen = new Set<string>()
      const list: WorkspaceProject[] = []

      // 预构建 dataset→task 映射：优先已完成的任务，也包含失败的任务
      const taskByDataset = new Map<string, { task_id: string; status: string; pipeline_status?: string; error_message?: string }>()
      for (const t of tasks) {
        const ds = t.dataset || (t as any).dataset_name
        if (!ds) continue
        const existing = taskByDataset.get(ds)
        if (!existing || (t.status === "completed" && existing.status !== "completed")) {
          taskByDataset.set(ds, { task_id: t.task_id, status: t.status, pipeline_status: t.status === "completed" ? "completed" : t.status === "failed" ? "error" : "running" })
        }
      }

      for (const p of dbProjects) {
        const key = p.dataset_name || `db-${p.id}`
        seen.add(key)
        let effectiveTaskId = p.task_id ?? null
        let effectivePipelineStatus = p.pipeline_status
        if (!effectiveTaskId && p.dataset_name) {
          const matched = taskByDataset.get(p.dataset_name)
          if (matched) {
            effectiveTaskId = matched.task_id
            effectivePipelineStatus = effectivePipelineStatus || matched.pipeline_status
          }
        }
        // 如果数据库已标记完成，即使 OSS API 返回空也认为有结果（解决未登录时 OSS API 401 问题）
        const hasResults = p.dataset_name ? (datasetsWithResults.has(p.dataset_name) || p.pipeline_status === "completed") : false
        const derivedPipelineStatus = effectivePipelineStatus === "completed" ? "completed"
          : effectivePipelineStatus === "error" ? "error"
          : effectivePipelineStatus === "running" ? "running"
          : effectiveTaskId ? "running"
          : p.dataset_name && datasetsWithResults.has(p.dataset_name) ? "completed"
          : undefined
        list.push({
          id: `proj-db-${p.id}`,
          name: p.name,
          rows: 0,
          createdAt: p.created_at ? "已保存" : "刚刚",
          status: derivedPipelineStatus === "completed" ? "completed"
            : derivedPipelineStatus === "error" ? "no_result"
            : derivedPipelineStatus === "running" ? "vectorizing"
            : p.dataset_name ? "draft"
            : "draft" as const,
          datasetName: p.dataset_name ?? undefined,
          mode: (p.mode as any) ?? "zero_shot",
          models: ["theta"],
          numTopics: p.num_topics ?? 20,
          pipelineStatus: derivedPipelineStatus as any,
          hasResults,
          dbProjectId: p.id,
          taskId: effectiveTaskId,
        })
      }
      for (const ds of datasets) {
        if (seen.has(ds.name)) continue
        seen.add(ds.name)
        const hasResults = datasetsWithResults.has(ds.name)
        list.push({
          id: `proj-${ds.name}`,
          name: ds.name,
          rows: ds.size ?? (ds as any).file_count ?? 0,
          createdAt: "已上传",
          status: hasResults ? "completed" as const : "draft" as const,
          pipelineStatus: hasResults ? "completed" as any : undefined,
          hasResults,
          datasetName: ds.name,
          models: ["theta"],
        })
      }
      for (const t of tasks) {
        const ds = t.dataset || (t as any).dataset_name
        if (ds && !seen.has(ds)) {
          seen.add(ds)
          const hasResults = datasetsWithResults.has(ds)
          list.push({
            id: `proj-${ds}`,
            name: ds,
            rows: 0,
            createdAt: "已分析",
            status: hasResults ? "completed" as const : (t.status === "completed" ? "no_result" as const : "vectorizing" as const),
            pipelineStatus: t.status === "completed" ? "completed" : t.status === "failed" ? "error" : "running",
            hasResults,
            datasetName: ds,
            models: ["theta"],
            taskId: t.task_id || null,
          })
        }
      }
      setProjects(prev => {
        const previousByStableKey = new Map<string, WorkspaceProject>()
        for (const project of prev) {
          const stableKey = project.dbProjectId ? `db:${project.dbProjectId}` : `ds:${project.datasetName || project.name}`
          previousByStableKey.set(stableKey, project)
        }

        const normalizedList = list.map(project => {
          const stableKey = project.dbProjectId ? `db:${project.dbProjectId}` : `ds:${project.datasetName || project.name}`
          const previous = previousByStableKey.get(stableKey)
          return previous?.models?.length ? { ...project, models: previous.models } : project
        })

        // 构建 dbProjectId → 旧项目 ID 的映射，用于迁移 temp ID
        const oldIdByDbId = new Map<number, string>()
        for (const p of prev) {
          if (p.dbProjectId) oldIdByDbId.set(p.dbProjectId, p.id)
        }

        // 迁移：如果旧列表中有 temp ID（如 new-xxx）指向同一个 dbProjectId，迁移 tab
        for (const np of normalizedList) {
          if (np.dbProjectId && oldIdByDbId.has(np.dbProjectId)) {
            const oldId = oldIdByDbId.get(np.dbProjectId)!
            if (oldId !== np.id) {
              setTabs(t => t.map(tab => tab.id === oldId ? { ...tab, id: np.id } : tab))
              setActiveTabId(a => a === oldId ? np.id : a)
            }
          }
        }

        // 保留正在运行的项目，但用新 ID 替换旧 temp ID
        const runningProjects = prev
          .filter(p => p.pipelineStatus === "running")
          .map(rp => {
            if (rp.dbProjectId) {
              const newVersion = normalizedList.find(np => np.dbProjectId === rp.dbProjectId)
              if (newVersion) return { ...rp, id: newVersion.id }
            }
            return rp
          })
        const runningIds = new Set(runningProjects.map(p => p.id))
        const runningDbIds = new Set(runningProjects.filter(p => p.dbProjectId).map(p => p.dbProjectId))
        const newProjects = normalizedList.filter(np =>
          !runningIds.has(np.id) && !(np.dbProjectId && runningDbIds.has(np.dbProjectId))
        )
        return [...runningProjects, ...newProjects]
      })
    } catch (error) {
      console.error("Failed to refresh projects:", error)
    }
  }, [])

  // 同步 refreshProjects 到 ref，确保定时器能访问最新版本
  refreshProjectsRef.current = refreshProjects

  const handleOpenProject = (projectId: string) => {
    const existingTab = tabs.find((tab) => tab.id === projectId)
    if (existingTab) {
      setActiveTabId(projectId)
    } else {
      const project = projects.find(p => p.id === projectId)
      const projectName = project?.name || "Project"
      const newTab: Tab = {
        id: projectId,
        title: projectName,
        closable: true,
      }
      setTabs([...tabs, newTab])
      setActiveTabId(projectId)
    }
  }

  // 创建新项目：保存到数据库（需登录），并打开工作台
  const handleCreateProject = useCallback(async (data: NewProjectData) => {
    const datasetName = data.name.trim().replace(/\s+/g, "_").replace(/[^\w\u4e00-\u9fa5-]/g, "").toLowerCase() || "dataset"
    const tempProjectId = `new-${Date.now()}`
    const optimisticProject: WorkspaceProject = {
      id: tempProjectId,
      name: data.name,
      datasetName,
      mode: "zero_shot",
      models: ["theta"],
      numTopics: 20,
      rows: 0,
      createdAt: "刚刚",
      status: "draft",
      pipelineStatus: undefined,
    }

    setProjects(prev => [optimisticProject, ...prev])
    setTabs(prev => [...prev, { id: tempProjectId, title: data.name, closable: true }])
    setActiveTabId(tempProjectId)
    setProjectTransitionName(data.name)

    if (transitionTimerRef.current) {
      window.clearTimeout(transitionTimerRef.current)
    }
    transitionTimerRef.current = window.setTimeout(() => {
      setProjectTransitionName(null)
      transitionTimerRef.current = null
    }, 900)

    try {
      const created = await ETMAgentAPI.createProject({
        name: data.name,
        dataset_name: datasetName,
        mode: "zero_shot",
        num_topics: 20,
      })

      setProjects(prev => prev.map(p => {
        if (p.id !== tempProjectId) return p
        return {
          ...p,
          name: created.name,
          datasetName: created.dataset_name ?? datasetName,
          mode: (created.mode as any) ?? "zero_shot",
          numTopics: created.num_topics ?? 20,
          dbProjectId: created.id,
        }
      }))
    } catch {
      // 未登录或 API 不可用时，继续使用本地项目
    }
  }, [])

  // Pipeline 完成回调：更新本地状态，并同步到数据库（若有 dbProjectId）
  const handlePipelineComplete = useCallback(async (
    projectId: string,
    result?: { dataset?: string; taskId?: string } | null,
    dbProjectId?: number,
  ) => {
    const updates = {
      status: "completed" as const,
      pipelineStatus: "completed" as const,
      ...(result?.dataset && { datasetName: result.dataset }),
    }
    setProjects(prev => prev.map(p => (p.id === projectId ? { ...p, ...updates } : p)))

    if (dbProjectId && result) {
      try {
        await ETMAgentAPI.updateProject(dbProjectId, {
          dataset_name: result.dataset,
          status: "completed",
          pipeline_status: "completed",
          task_id: result.taskId,
        })
      } catch {
        // 忽略同步失败
      }
    }
    // 刷新项目列表，确保 UI 立即从 AutoPipeline 切换到 ProjectResultView
    refreshProjects()
    // 强制重新渲染 renderContent，切换到结果页面
    setRenderKey(k => k + 1)
  }, [refreshProjects])

  const handlePipelineError = useCallback((projectId: string) => {
    setProjects(prev => prev.map(p =>
      p.id === projectId ? { ...p, status: "completed" as const, pipelineStatus: "error" } : p
    ))
  }, [])

  // Persist tabs to sessionStorage
  useEffect(() => {
    sessionStorage.setItem(STORAGE_KEYS.TABS, JSON.stringify(tabs))
  }, [tabs])

  // Persist activeTabId to sessionStorage
  useEffect(() => {
    sessionStorage.setItem(STORAGE_KEYS.ACTIVE_TAB, activeTabId)
  }, [activeTabId])

  const handleTabChange = (tabId: string) => {
    setActiveTabId(tabId)
  }

  const handleTabClose = (tabId: string) => {
    const tab = tabs.find((t) => t.id === tabId)
    if (!tab?.closable) return

    const newTabs = tabs.filter((t) => t.id !== tabId)
    setTabs(newTabs)

    if (activeTabId === tabId) {
      setActiveTabId("hub")
    }
  }

  // 删除项目：删除 OSS 数据集目录
  const handleDeleteProject = useCallback(async (projectId: string) => {
    const project = projects.find((p) => p.id === projectId)
    if (!project) return

    const datasetName = project.datasetName || (projectId.startsWith("proj-") && !projectId.startsWith("proj-db-") ? projectId.replace(/^proj-/, "") : null)
    if (datasetName) {
      try {
        await ETMAgentAPI.deleteDataset(datasetName)
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error)
        if (!msg.includes("404") && !msg.includes("not found")) {
          console.error("删除数据集失败:", error)
        }
      }
    }

    setProjects((prev) => prev.filter((p) => p.id !== projectId))
    const newTabs = tabs.filter((t) => t.id !== projectId)
    setTabs(newTabs)
    if (activeTabId === projectId) {
      setActiveTabId("hub")
    }
  }, [projects, tabs, activeTabId])

  // 批量删除：并发执行，部分失败不阻断其他项目
  const handleBatchDelete = useCallback(async (projectIds: string[]) => {
    const results = await Promise.allSettled(
      projectIds.map(async (projectId) => {
        const project = projects.find((p) => p.id === projectId)
        if (!project) return
        const datasetName = project.datasetName || (projectId.startsWith("proj-") && !projectId.startsWith("proj-db-") ? projectId.replace(/^proj-/, "") : null)
        if (datasetName) {
          try {
            await ETMAgentAPI.deleteDataset(datasetName)
          } catch (error) {
            const msg = error instanceof Error ? error.message : String(error)
            if (!msg.includes("404") && !msg.includes("not found")) throw error
          }
        }
      })
    )
    const deletedIds = new Set(
      projectIds.filter((_, i) => results[i].status === "fulfilled")
    )
    if (deletedIds.size === 0) return
    setProjects((prev) => prev.filter((p) => !deletedIds.has(p.id)))
    setTabs((prev) => prev.filter((t) => !deletedIds.has(t.id)))
    if (deletedIds.has(activeTabId)) setActiveTabId("hub")
  }, [projects, activeTabId])

  // Load chat history on mount
  useEffect(() => {
    const loadHistory = async () => {
      try {
        const response = await ETMAgentAPI.getConversationHistory("dashboard")
        const history = (response as any)?.messages || []
        if (history && history.length > 0) {
          setChatHistory(history.map((m: any) => ({
            id: String(m.id),
            role: m.role as "user" | "ai",
            content: m.content,
            type: "text" as const,
            timestamp: m.created_at,
            shouldAnimateTyping: false,
          })))
        }
      } catch (e) {
        // Silently ignore chat history load errors (user may not be authenticated)
        console.log("Chat history not available:", e)
      }
    }
    loadHistory()
  }, [])

  // Chat handlers — 使用 SSE 流式对话，fallback 到普通请求
  const handleSendMessage = useCallback(async (payload: string | SendMessagePayload) => {
    const content = (typeof payload === "string" ? payload : payload.content).trim()
    const images = typeof payload === "string" ? [] : (payload.images || [])
    const files = typeof payload === "string" ? [] : (payload.files || [])

    if (!content && images.length === 0 && files.length === 0) {
      return
    }

    const userMessage: ChatMessage = {
      id: generateId(),
      role: "user",
      content: content || (images.length > 0 ? "[图片]" : "[附件]"),
      type: "text",
      timestamp: getTimestamp(),
      data: (images.length > 0 || files.length > 0) ? {
        imageAttachments: images.map((img) => ({
          name: img.name,
          mimeType: img.mimeType,
          size: img.size,
          url: img.dataUrl,
        })),
        fileAttachments: files.map((f) => ({
          name: f.name,
          mimeType: f.mimeType,
          size: f.size,
        })),
      } : undefined,
    }
    setChatHistory((prev) => [...prev, userMessage])

    const aiMsgId = generateId()

    // 先添加一条空的 AI 消息，后续流式追加内容
    setChatHistory((prev) => [
      ...prev,
      { id: aiMsgId, role: "ai", content: "", type: "text", timestamp: getTimestamp(), isThinking: true, shouldAnimateTyping: true },
    ])

    // 延迟显示思考效果
    await new Promise(resolve => setTimeout(resolve, 800))

    try {
      let fullText = ""
      let streamed = false

      // 尝试 SSE 流式对话
      try {
        for await (const chunk of ETMAgentAPI.chatStream(content || PROMPTS.CHAT.default_message_with_attachments, "dashboard", {
          current_view_name: "项目中心",
          current_view: activeTabId === "hub" ? "hub" : "workspace",
          app_state: "workspace",
          projects: projects.map(p => ({ name: p.name, status: p.pipelineStatus, mode: p.mode })),
          current_project: activeTabId !== "hub" ? projects.find(p => p.id === activeTabId) : null,
        }, images, files)) {
          streamed = true
          if (chunk.type === "text" && chunk.content) {
            fullText = chunk.content
            setChatHistory((prev) =>
              prev.map((m) => (m.id === aiMsgId ? { ...m, content: fullText, isThinking: false, shouldAnimateTyping: true } : m))
            )
          }
        }
      } catch {
        // SSE 不可用，fallback 到普通对话
        if (!streamed) {
            const response = await ETMAgentAPI.chat(content || PROMPTS.CHAT.default_message_with_attachments, {
            current_view_name: "项目中心",
            current_view: activeTabId === "hub" ? "hub" : "workspace",
            app_state: "workspace",
            session_id: "dashboard",
            projects: projects.map(p => ({ name: p.name, status: p.pipelineStatus, mode: p.mode })),
            current_project: activeTabId !== "hub" ? projects.find(p => p.id === activeTabId) : null,
            }, { images, files, sessionId: "dashboard" })
          fullText = response.message ?? (response as { response?: string }).response ?? "（无回复）"
        }
      }

      if (fullText) {
        setChatHistory((prev) =>
            prev.map((m) => (m.id === aiMsgId ? { ...m, content: fullText, isThinking: false, shouldAnimateTyping: false } : m))
        )
      }
    } catch (error) {
      setChatHistory((prev) =>
        prev.map((m) =>
          m.id === aiMsgId
            ? {
                ...m,
                content: "无法连接 AI 服务，请确认后端已启动。",
                followUpQuestions: ["如何开始？", "支持哪些格式？"],
                  shouldAnimateTyping: false,
              }
            : m
        )
      )
    }
  }, [activeTabId, projects])

  handleSendMessageRef.current = handleSendMessage

  // 当查看已完成项目时，拉取 interpret API 生成动态建议
  useEffect(() => {
    const project = projects.find(p => p.id === activeTabId)
    if (!project || project.pipelineStatus !== "completed" || !project.datasetName) {
      setDynamicSuggestions([])
      return
    }
    const jobId = project.datasetName
    const send = (q: string) => handleSendMessageRef.current(q)
    const fallbacks: SuggestionCard[] = PROMPTS.DASHBOARD.dynamic_suggestions.map(s => ({
      title: s.title,
      description: s.description,
      onClick: () => send(s.prompt),
    }))
    Promise.allSettled([
      ETMAgentAPI.interpretMetrics(jobId, "zh").catch(() => null),
      ETMAgentAPI.interpretTopics(jobId, "zh", true).catch(() => null),
      ETMAgentAPI.generateSummary(jobId, "zh").catch(() => null),
    ]).then(([m, t, s]) => {
      const cards: SuggestionCard[] = [
        m.status === "fulfilled" && (m.value as any)?.summary
          ? { title: "指标解读", description: String((m.value as any).summary).slice(0, 55) + "...", onClick: () => send("请解读评估指标。") }
          : fallbacks[0],
        t.status === "fulfilled" && (t.value as any)?.summary
          ? { title: "主题解读", description: String((t.value as any).summary).slice(0, 55) + "...", onClick: () => send("请解读各主题含义。") }
          : fallbacks[1],
        s.status === "fulfilled" && (s.value as any)?.summary
          ? { title: "分析报告", description: String((s.value as any).summary).slice(0, 55) + "...", onClick: () => send("请生成分析报告。") }
          : fallbacks[2],
      ]
      setDynamicSuggestions(cards)
    }).catch(() => setDynamicSuggestions(fallbacks))
  }, [activeTabId, projects])

  const handleDataUploaded = useCallback(async (file: File) => {
    console.log("Data uploaded:", file.name)
  }, [])

  const handleFocusChart = useCallback((chartId: string) => {
    console.log("Focus chart:", chartId)
  }, [])

  const handleClearChat = useCallback(() => {
    setChatHistory([])
  }, [])

  // 渲染内容
  const renderContent = () => {
    if (activeTabId === "hub") {
      return (
        <ProjectHub
          onProjectSelect={handleOpenProject}
          onNewProject={() => setIsNewProjectDialogOpen(true)}
          onDeleteProject={handleDeleteProject}
          onBatchDelete={handleBatchDelete}
          onRefresh={refreshProjects}
          projects={projects}
          isLoading={isLoading}
        />
      )
    }

    // 查找当前项目
    const currentProject = projects.find(p => p.id === activeTabId)

    if (isLoading) {
      return (
        <div className="p-8 text-center">
          <div className="flex items-center justify-center mb-4">
            <Spinner className="h-8 w-8 text-slate-600" />
          </div>
          <h2 className="text-xl font-semibold text-slate-900 mb-2">正在加载</h2>
          <p className="text-slate-500">正在获取项目数据，请稍候...</p>
        </div>
      )
    }

    // 如果项目找不到，等待 effect 切回项目中心，避免在 render 中触发状态更新
    if (!currentProject) {
      return null
    }

    // 新建项目 / 运行中 / 有关联任务 / 已上传但未训练：显示配置页面
    // "draft" 状态表示已上传文件但未开始训练，需要显示配置页面
    // 如果项目已有结果 (hasResults=true) 且状态不是 running/error，显示结果页面
    // 即使 OSS API 认证失败（未登录），只要项目状态标记为 completed 就显示结果
    const shouldShowConfig =
      currentProject.pipelineStatus === "running" ||
      currentProject.pipelineStatus === "error" ||
      (currentProject.status === "draft" && !currentProject.hasResults) ||
      currentProject.pipelineStatus === undefined && !currentProject.hasResults;

    if (shouldShowConfig) {
      return (
        <AutoPipeline
          projectName={currentProject.name}
          mode={currentProject.mode || "zero_shot"}
          numTopics={currentProject.numTopics || 20}
          initialTaskId={currentProject.taskId}
          pipelineStatus={currentProject.pipelineStatus}
          onConfigConfirmed={async (config) => {
            setProjects(prev => prev.map(p =>
              p.id === currentProject.id
                ? {
                    ...p,
                    mode: config.mode,
                    numTopics: config.numTopics,
                    models: config.models.length > 0 ? config.models : ["theta"],
                  }
                : p
            ))
            if (currentProject.dbProjectId) {
              try {
                await ETMAgentAPI.updateProject(currentProject.dbProjectId, {
                  mode: config.mode,
                  num_topics: config.numTopics,
                })
              } catch { /* skip */ }
            }
          }}
          onComplete={(result) => handlePipelineComplete(currentProject.id, result, currentProject.dbProjectId)}
          onError={() => handlePipelineError(currentProject.id)}
          onTaskCreated={async (tid) => {
            setProjects(prev => prev.map(p =>
              p.id === currentProject.id
                ? { ...p, taskId: tid, pipelineStatus: "running" }
                : p
            ))
            if (currentProject.dbProjectId) {
              try {
                await ETMAgentAPI.updateProject(currentProject.dbProjectId, {
                  task_id: tid,
                  pipeline_status: "running",
                })
              } catch { /* skip */ }
            }
          }}
          onUploadComplete={async (datasetName) => {
            if (currentProject.dbProjectId) {
              try {
                await ETMAgentAPI.updateProject(currentProject.dbProjectId, {
                  dataset_name: datasetName,
                  status: "uploading",
                })
                setProjects(prev => prev.map(p =>
                  p.id === currentProject.id ? { ...p, datasetName } : p
                ))
              } catch { /* skip */ }
            }
          }}
        />
      )
    }

    // 已完成的项目显示结果概览
    return (
      <ProjectResultView project={currentProject} />
    )
  }

  return (
    <>
      <AppShell
        tabs={tabs}
        activeTabId={activeTabId}
        onTabChange={handleTabChange}
        onTabClose={handleTabClose}
        onTabsReorder={(fromIdx, toIdx) => {
          const newTabs = [...tabs]
          const [moved] = newTabs.splice(fromIdx, 1)
          newTabs.splice(toIdx, 0, moved)
          setTabs(newTabs)
        }}
        chatHistory={chatHistory}
        onSendMessage={handleSendMessage}
        onDataUploaded={handleDataUploaded}
        onFocusChart={handleFocusChart}
        onClearChat={handleClearChat}
        dynamicSuggestions={dynamicSuggestions}
      >
        <div className="relative min-h-[360px]">
          <div
            key={renderKey}
            className={`transition-all duration-500 ${projectTransitionName ? "opacity-70 scale-[0.995]" : "opacity-100 scale-100"}`}
          >
            {renderContent()}
          </div>

          {projectTransitionName && (
            <div className="absolute inset-0 z-20 flex items-center justify-center bg-white/70 backdrop-blur-[1px]">
              <div className="rounded-xl border border-slate-200 bg-white px-5 py-4 shadow-sm">
                <div className="flex items-center gap-3">
                  <div className="h-5 w-5 rounded-full border-2 border-blue-600 border-t-transparent animate-spin" />
                  <div>
                    <p className="text-sm font-medium text-slate-800">正在创建项目</p>
                    <p className="text-xs text-slate-500">{projectTransitionName} 初始化中，请稍候...</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </AppShell>
      
      <NewProjectDialog
        open={isNewProjectDialogOpen}
        onOpenChange={setIsNewProjectDialogOpen}
        onSubmit={handleCreateProject}
      />
    </>
  )
}


// 项目结果视图 — 使用统一结果 Tab 组件
function ProjectResultView({ project }: { project: WorkspaceProject }) {
  const dataset = project.datasetName || project.name
  const mode    = project.mode || "zero_shot"
  const [activeResultTab, setActiveResultTab] = useState("topics")
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [loadingModels, setLoadingModels] = useState(true)
  const [availableModels, setAvailableModels] = useState<string[]>([])

  // 获取用户选择的模型列表
  const userSelectedModels = project.models || []

  // 获取可用的模型列表（信任 /models API 的结果，不再二次验证 topic-words）
  useEffect(() => {
    const loadAvailableModels = async () => {
      setLoadingModels(true)
      try {
        const availableResponse = await apiFetch<{ models?: string[] }>(
          API_BASE,
          `/api/results/${encodeURIComponent(dataset)}/models`
        )
        if (availableResponse && availableResponse.models && availableResponse.models.length > 0) {
          const models = availableResponse.models
          setAvailableModels(models)
          // 始终使用后端返回的第一个模型，避免 project.models 与实际不符
          setSelectedModel(models[0])
        } else {
          setAvailableModels([])
        }
      } catch {
        // 后端 API 不可用时回退
        setAvailableModels(userSelectedModels.length > 0 ? userSelectedModels : [])
        if (userSelectedModels.length > 0) {
          setSelectedModel(userSelectedModels[0])
        }
      } finally {
        setLoadingModels(false)
      }
    }

    loadAvailableModels()
  }, [dataset])

  // 模型标签映射
  const MODEL_LABELS: Record<string, string> = {
    theta: "THETA",
    lda: "LDA",
    bertopic: "BERTopic",
    ctm: "CTM",
    prodlda: "ProdLDA",
    btm: "BTM",
    hdp: "HDP",
    nvdm: "NVDM",
    gsm: "GSM",
    dtm: "DTM",
    stm: "STM",
  }

  if (loadingModels) {
    return (
      <div className="p-6 lg:p-8">
        <div className="flex items-center justify-center py-20">
          <div className="text-slate-500">加载模型结果...</div>
        </div>
      </div>
    )
  }

  if (availableModels.length === 0) {
    return (
      <div className="p-6 lg:p-8">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-slate-900 mb-1">{project.name}</h1>
          <p className="text-slate-500 text-sm">
            数据集: {dataset} · 模式: {mode} · 主题数: {project.numTopics || 20}
          </p>
        </div>
        <div className="flex flex-col items-center justify-center py-20 gap-2 text-slate-500">
          <p className="text-sm">暂无模型结果</p>
          <p className="text-xs text-slate-400">请先运行训练任务生成结果</p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 lg:p-8">
      <div className="mb-4">
        <h1 className="text-2xl font-bold text-slate-900 mb-1">{project.name}</h1>
        <p className="text-slate-500 text-sm">
          数据集: {dataset} · 模式: {mode} · 主题数: {project.numTopics || 20}
        </p>
      </div>

      {/* 模型选择器 - 只显示用户选择的且有结果的模型 */}
      <div className="flex items-center gap-3 mb-4 pb-4 border-b border-slate-200">
        <span className="text-sm text-slate-500">模型:</span>
        <div className="flex flex-wrap gap-2">
          {availableModels.map((m) => (
            <button
              key={m}
              onClick={() => setSelectedModel(m)}
              className={`text-xs px-3 py-1.5 rounded-full border transition-all ${
                selectedModel === m
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-white text-slate-600 border-slate-200 hover:border-blue-300 hover:text-blue-600"
              }`}
            >
              {MODEL_LABELS[m] || m.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      <Tabs value={activeResultTab} onValueChange={setActiveResultTab}>
        <TabsList>
          <TabsTrigger value="topics">主题词</TabsTrigger>
          <TabsTrigger value="metrics">评估指标</TabsTrigger>
          <TabsTrigger value="viz">可视化</TabsTrigger>
          <TabsTrigger value="export">导出</TabsTrigger>
        </TabsList>

        <TabsContent value="topics" className="mt-4">
          {selectedModel && (
            <TopicWordsTab dataset={dataset} mode={mode} shouldLoad={activeResultTab === "topics"} selectedModel={selectedModel} />
          )}
        </TabsContent>

        <TabsContent value="metrics" className="mt-4">
          {selectedModel && (
            <MetricsTab dataset={dataset} mode={mode} shouldLoad={activeResultTab === "metrics"} selectedModel={selectedModel} />
          )}
        </TabsContent>

        <TabsContent value="viz" className="mt-4">
          {selectedModel && (
            <VisualizationTab dataset={dataset} mode={mode} selectedModel={selectedModel} shouldLoad={activeResultTab === "viz"} />
          )}
        </TabsContent>

        <TabsContent value="export" className="mt-4">
          <ExportTab dataset={dataset} mode={mode} selectedModel={selectedModel} />
        </TabsContent>
      </Tabs>
    </div>
  )
}


// Dashboard page with auth protection
export default function DashboardPage() {
  return (
    <ProtectedRoute>
      <DashboardContent />
    </ProtectedRoute>
  )
}
