"use client"

import { useState, useRef, useCallback, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Camera,
  CameraOff,
  Play,
  Square,
  CornerDownLeft,
  Delete,
  Trash2,
  Volume2,
  Sparkles,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Brain,
} from "lucide-react"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
type DetectionMode = "static" | "dynamic"
type SentenceAction = "enter" | "delete" | "clear"

const LABEL_ALIASES: Record<string, string> = {
  ily: "I Love You",
  thank_you: "Thank You",
}

function formatPredictionLabel(raw: string): string {
  const normalized = raw.trim().toLowerCase()
  const aliased = LABEL_ALIASES[normalized]
  if (aliased) return aliased

  return normalized
    .replace(/_/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ")
}

interface ModelStatus {
  model_loaded: boolean
  is_trained_weights: boolean
  checkpoint_source: string | null
  num_classes: number
  classes: string[]
  device: string
}

interface SentenceActionResponse {
  mode: DetectionMode
  session_id: string
  sentence: string
}

export default function DetectionPage() {
  const [detectionMode, setDetectionMode] = useState<DetectionMode>("static")
  const [cameraEnabled, setCameraEnabled] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [isDetecting, setIsDetecting] = useState(false)
  const [prediction, setPrediction] = useState<string | null>(null)
  const [confidence, setConfidence] = useState<number | null>(null)
  const [predictionHistory, setPredictionHistory] = useState<{ label: string; conf: number }[]>([])
  const [dynamicProgress, setDynamicProgress] = useState<{ collected: number; required: number } | null>(null)
  const [detecting, setDetecting] = useState(false)
  const [predictionError, setPredictionError] = useState<string | null>(null)
  const [sentenceByMode, setSentenceByMode] = useState<Record<DetectionMode, string>>({ static: "", dynamic: "" })
  const [sentenceActionLoading, setSentenceActionLoading] = useState(false)

  // ── Model status ─────────────────────────────────────────────────────────────
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null)
  const [modelStatusLoading, setModelStatusLoading] = useState(true)
  const [modelStatusError, setModelStatusError] = useState<string | null>(null)

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const inFlightRef = useRef(false)
  const dynamicSessionRef = useRef(
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `dynamic-${Date.now()}`
  )
  const staticSentenceSessionRef = useRef(
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `sentence-static-${Date.now()}`
  )
  const dynamicSentenceSessionRef = useRef(
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `sentence-dynamic-${Date.now()}`
  )

  const sentence = sentenceByMode[detectionMode]

  const getSentenceSessionId = useCallback(() => {
    return detectionMode === "dynamic" ? dynamicSentenceSessionRef.current : staticSentenceSessionRef.current
  }, [detectionMode])

  const applySentenceAction = useCallback(
    async (action: SentenceAction, token?: string | null) => {
      setSentenceActionLoading(true)
      try {
        const res = await fetch(`${API}/predict/sentence/${action}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            mode: detectionMode,
            session_id: getSentenceSessionId(),
            token: token ?? undefined,
          }),
        })

        if (!res.ok) {
          const errorData = await res.json().catch(() => null)
          throw new Error(errorData?.detail ?? "Sentence action failed.")
        }

        const data: SentenceActionResponse = await res.json()
        setSentenceByMode((prev) => ({
          ...prev,
          [detectionMode]: data.sentence ?? "",
        }))
      } catch (error) {
        console.error("Sentence action error:", error)
        setPredictionError(error instanceof Error ? error.message : "Sentence action failed.")
      } finally {
        setSentenceActionLoading(false)
      }
    },
    [detectionMode, getSentenceSessionId]
  )

  const resetDynamicSession = useCallback(async () => {
    if (detectionMode !== "dynamic") return
    const sessionId = dynamicSessionRef.current
    try {
      await fetch(`${API}/predict/dynamic/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      })
    } catch {
      // No-op: if reset fails, a fresh session ID is generated below.
    }
    dynamicSessionRef.current =
      typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
        ? crypto.randomUUID()
        : `dynamic-${Date.now()}`
  }, [detectionMode])

  // ── Fetch model status on mount ───────────────────────────────────────────────
  useEffect(() => {
    const fetchModelStatus = async () => {
      setModelStatusLoading(true)
      setModelStatusError(null)
      try {
        const res = await fetch(`${API}/model/status?mode=${detectionMode}`)
        if (!res.ok) throw new Error(`Server responded ${res.status}`)
        const data: ModelStatus = await res.json()
        setModelStatus(data)
      } catch (e) {
        setModelStatusError("Cannot reach backend. Make sure the server is running on port 8000.")
      } finally {
        setModelStatusLoading(false)
      }
    }
    fetchModelStatus()
  }, [detectionMode])

  const enableCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } })
      streamRef.current = stream
      setCameraEnabled(true)
      setCameraError(null)
      setPredictionError(null)
    } catch (e) {
      console.error("Camera denied:", e)
      setCameraError("Unable to access camera. Please allow permission or try a different device.")
      setCameraEnabled(false)
    }
  }, [])

  const disableCamera = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setCameraEnabled(false)
    setCameraError(null)
    setPredictionError(null)
    setDynamicProgress(null)
    setIsDetecting(false)
  }, [])

  useEffect(() => {
    if (!cameraEnabled || !streamRef.current || !videoRef.current) return
    const video = videoRef.current
    video.srcObject = streamRef.current
    const handleLoaded = () => {
      video.play().catch((err) => {
        console.error("Video playback failed:", err)
        setCameraError(
          "Camera stream started but video playback was blocked. Please allow autoplay or refresh the page."
        )
      })
    }
    video.addEventListener("loadedmetadata", handleLoaded)
    return () => {
      video.removeEventListener("loadedmetadata", handleLoaded)
    }
  }, [cameraEnabled])

  const captureAndPredict = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || inFlightRef.current) return
    const video = videoRef.current
    const canvas = canvasRef.current
    canvas.width = video.videoWidth || 320
    canvas.height = video.videoHeight || 240
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    ctx.drawImage(video, 0, 0)
    try {
      inFlightRef.current = true
      setDetecting(true)
      setPredictionError(null)

      const blob = await new Promise<Blob | null>((resolve) => {
        canvas.toBlob(resolve, "image/jpeg", 0.8)
      })
      if (!blob) {
        setPredictionError("Unable to capture the current frame.")
        return
      }

      const formData = new FormData()
      formData.append("file", blob, "frame.jpg")

      const endpoint =
        detectionMode === "dynamic"
          ? `${API}/predict/dynamic?session_id=${encodeURIComponent(dynamicSessionRef.current)}`
          : `${API}/predict`

      const res = await fetch(endpoint, {
        method: "POST",
        body: formData,
      })

      if (!res.ok) {
        const errorData = await res.json().catch(() => null)
        setPredictionError(errorData?.detail ?? "Prediction failed.")
        return
      }

      const data = await res.json()
      if (detectionMode === "dynamic") {
        if (!data.ready) {
          setDynamicProgress({ collected: data.frames_collected ?? 0, required: data.frames_required ?? 30 })
          return
        }

        setDynamicProgress({ collected: data.frames_required ?? 30, required: data.frames_required ?? 30 })
      }

      if (data.prediction) {
        setPrediction(data.prediction)
        setConfidence(typeof data.confidence === "number" ? data.confidence : null)
        setPredictionHistory((prev) => {
          const nextItem = { label: data.prediction, conf: typeof data.confidence === "number" ? data.confidence : 1 }
          if (prev[0]?.label === nextItem.label) {
            return [{ ...prev[0], conf: nextItem.conf }, ...prev.slice(1)]
          }
          return [nextItem, ...prev].slice(0, 10)
        })
      }
    } catch (e) {
      console.error("Prediction error:", e)
      setPredictionError("Prediction failed. Is the backend running?")
    } finally {
      inFlightRef.current = false
      setDetecting(false)
    }
  }, [detectionMode])

  const startDetection = useCallback(() => {
    setIsDetecting(true)
    setDynamicProgress(null)

    if (detectionMode === "dynamic") {
      resetDynamicSession()
      intervalRef.current = setInterval(captureAndPredict, 80)
      return
    }

    intervalRef.current = setInterval(captureAndPredict, 1500)
  }, [captureAndPredict, detectionMode, resetDynamicSession])

  const stopDetection = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    setIsDetecting(false)
    setDynamicProgress(null)
    if (detectionMode === "dynamic") {
      resetDynamicSession()
    }
  }, [detectionMode, resetDynamicSession])

  const speakPrediction = () => {
    if (prediction && "speechSynthesis" in window) {
      const u = new SpeechSynthesisUtterance(prediction)
      u.text = formatPredictionLabel(prediction)
      u.rate = 0.8
      window.speechSynthesis.speak(u)
    }
  }

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
      streamRef.current?.getTracks().forEach((t) => t.stop())
      if (detectionMode === "dynamic") {
        resetDynamicSession()
      }
    }
  }, [detectionMode, resetDynamicSession])

  useEffect(() => {
    if (isDetecting && intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = setInterval(captureAndPredict, detectionMode === "dynamic" ? 80 : 1500)
    }

    setPrediction(null)
    setConfidence(null)
    setPredictionHistory([])
    setPredictionError(null)
    setDynamicProgress(null)

    if (detectionMode === "dynamic") {
      resetDynamicSession()
    }
  }, [captureAndPredict, detectionMode, isDetecting, resetDynamicSession])

  // ── Model status banner ───────────────────────────────────────────────────────
  const renderModelBanner = () => {
    if (modelStatusLoading) {
      return (
        <div className="flex items-center gap-2 rounded-xl border border-border/50 bg-muted/40 px-4 py-3 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          Checking model connection…
        </div>
      )
    }
    if (modelStatusError || !modelStatus) {
      return (
        <div className="flex items-center gap-2 rounded-xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          <XCircle className="h-4 w-4 shrink-0" />
          <span>
            <strong>Backend unreachable.</strong> {modelStatusError}
          </span>
        </div>
      )
    }
    if (!modelStatus.model_loaded) {
      return (
        <div className="flex items-center gap-2 rounded-xl border border-yellow-500/40 bg-yellow-500/10 px-4 py-3 text-sm text-yellow-600 dark:text-yellow-400">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span>
            <strong>Model not loaded.</strong> The backend is running but the model failed to initialise.
          </span>
        </div>
      )
    }
    if (!modelStatus.is_trained_weights && detectionMode === "static") {
      return (
        <div className="flex items-center gap-2 rounded-xl border border-yellow-500/40 bg-yellow-500/10 px-4 py-3 text-sm text-yellow-600 dark:text-yellow-400">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <span>
            <strong>Random placeholder weights active.</strong> Place&nbsp;
            <code className="rounded bg-yellow-500/20 px-1">asl_resnet50.pth</code> or{" "}
            <code className="rounded bg-yellow-500/20 px-1">asl_resnet50.pth.zip</code> inside{" "}
            <code className="rounded bg-yellow-500/20 px-1">MPR_STATIC_M/</code> and restart the backend.
          </span>
        </div>
      )
    }
    return (
      <div className="flex flex-wrap items-center gap-3 rounded-xl border border-green-500/40 bg-green-500/10 px-4 py-3 text-sm">
        <div className="flex items-center gap-2 text-green-600 dark:text-green-400">
          <CheckCircle2 className="h-4 w-4 shrink-0" />
          <span>
            <strong>{detectionMode === "dynamic" ? "Dynamic model active" : "Trained model active"}</strong> — predictions are from{" "}
            <code className="rounded bg-green-500/20 px-1">{modelStatus.checkpoint_source ?? "trained checkpoint"}</code>
          </span>
        </div>
        <div className="ml-auto flex flex-wrap gap-2">
          <Badge variant="secondary" className="gap-1">
            <Brain className="h-3 w-3" />
            {modelStatus.num_classes} classes
          </Badge>
          <Badge variant="secondary">{modelStatus.device.toUpperCase()}</Badge>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen pt-24 pb-16">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-8">
          <div className="inline-flex items-center gap-2 rounded-full bg-primary/10 px-4 py-1.5 text-sm font-medium text-primary mb-4">
            <Sparkles className="h-4 w-4" /> AI-Powered Detection
          </div>
          <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">Live Sign Detection</h1>
          <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
            Perform sign language in front of your camera and let our AI detect and translate your signs in real-time
          </p>
          <div className="mt-5 inline-flex rounded-xl border border-border bg-card p-1">
            <Button
              type="button"
              variant={detectionMode === "static" ? "default" : "ghost"}
              className="h-9"
              onClick={() => setDetectionMode("static")}
            >
              Static Signs
            </Button>
            <Button
              type="button"
              variant={detectionMode === "dynamic" ? "default" : "ghost"}
              className="h-9"
              onClick={() => setDetectionMode("dynamic")}
            >
              Dynamic Signs
            </Button>
          </div>
        </motion.div>

        {/* Model status banner */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
          className="mb-6"
        >
          {renderModelBanner()}
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Camera */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-2"
          >
            <Card className="bg-card/50 border-border/50 h-full">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="text-lg">Camera Preview</CardTitle>
                <div className="flex items-center gap-2">
                  {detecting && <Loader2 className="h-4 w-4 animate-spin text-primary" />}
                  {isDetecting && (
                    <span className="flex items-center gap-2 text-sm text-primary">
                      <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-primary" />
                      </span>
                      Detecting
                    </span>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                <div className="aspect-video rounded-xl bg-muted overflow-hidden relative">
                  {cameraEnabled ? (
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="w-full h-full object-cover scale-x-[-1]"
                    />
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center">
                        <Camera className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                        <p className="text-muted-foreground font-medium">Camera not enabled</p>
                        <p className="text-sm text-muted-foreground mt-1">Click below to enable your camera</p>
                      </div>
                    </div>
                  )}
                </div>
                {cameraError && <p className="mt-3 text-sm text-destructive">{cameraError}</p>}
                {/* Hidden canvas for frame capture */}
                <canvas ref={canvasRef} className="hidden" />

                <div className="flex flex-wrap gap-3 mt-6">
                  {!cameraEnabled ? (
                    <Button onClick={enableCamera} size="lg">
                      <Camera className="mr-2 h-4 w-4" /> Enable Camera
                    </Button>
                  ) : (
                    <>
                      {!isDetecting ? (
                        <Button
                          onClick={startDetection}
                          size="lg"
                          disabled={!modelStatus?.model_loaded}
                        >
                          <Play className="mr-2 h-4 w-4" /> Start Detection
                        </Button>
                      ) : (
                        <Button onClick={stopDetection} variant="destructive" size="lg">
                          <Square className="mr-2 h-4 w-4" /> Stop Detection
                        </Button>
                      )}
                      <Button variant="outline" size="lg" onClick={disableCamera} disabled={isDetecting}>
                        <CameraOff className="mr-2 h-4 w-4" /> Disable Camera
                      </Button>
                      <Button
                        variant="outline"
                        size="lg"
                        onClick={async () => {
                          setPrediction(null)
                          setConfidence(null)
                          setPredictionError(null)
                          setPredictionHistory([])
                          await applySentenceAction("clear")
                        }}
                      >
                        <Trash2 className="mr-2 h-4 w-4" /> Clear
                      </Button>
                    </>
                  )}
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Predictions */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="space-y-6"
          >
            <Card className="bg-card/50 border-border/50">
              <CardHeader>
                <CardTitle className="text-lg">Current Prediction</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="p-6 rounded-xl bg-linear-to-br from-primary/10 to-accent/10 border border-primary/20 text-center min-h-30 flex flex-col items-center justify-center">
                  <AnimatePresence mode="wait">
                    {prediction ? (
                      <motion.div
                        key={prediction}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        className="text-center"
                      >
                        <p className="text-5xl font-bold text-primary">{formatPredictionLabel(prediction)}</p>
                        {confidence !== null && (
                          <p className="text-sm text-muted-foreground mt-2">
                            {(confidence * 100).toFixed(0)}% confident
                          </p>
                        )}
                      </motion.div>
                    ) : (
                      <motion.p
                        key="idle"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="text-muted-foreground"
                      >
                        {isDetecting ? "Analyzing your signs…" : "Start detection to see predictions"}
                      </motion.p>
                    )}
                  </AnimatePresence>
                </div>
                {prediction && (
                  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mt-4">
                    <Button onClick={speakPrediction} className="w-full" variant="secondary">
                      <Volume2 className="mr-2 h-4 w-4" /> Speak
                    </Button>
                  </motion.div>
                )}
                {predictionError && <p className="mt-4 text-sm text-destructive">{predictionError}</p>}
                {detectionMode === "dynamic" && isDetecting && dynamicProgress && !prediction && (
                  <p className="mt-4 text-sm text-muted-foreground">
                    Building sequence: {dynamicProgress.collected}/{dynamicProgress.required} frames
                  </p>
                )}
              </CardContent>
            </Card>

            <Card className="bg-card/50 border-border/50">
              <CardHeader>
                <CardTitle className="text-lg">Sentence Box</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="rounded-xl border border-primary/20 bg-linear-to-br from-primary/10 to-accent/10 p-4">
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">{detectionMode} sentence</p>
                  <p className="mt-2 min-h-12 text-lg font-medium wrap-break-word whitespace-pre-wrap">
                    {sentence || "Build your sentence using Enter"}
                  </p>
                </div>

                <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-3">
                  <Button
                    onClick={() => applySentenceAction("enter", prediction)}
                    disabled={!prediction || sentenceActionLoading}
                  >
                    <CornerDownLeft className="mr-2 h-4 w-4" /> Enter
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={() => applySentenceAction("delete")}
                    disabled={sentenceActionLoading || !sentence}
                  >
                    <Delete className="mr-2 h-4 w-4" /> Delete
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() => applySentenceAction("clear")}
                    disabled={sentenceActionLoading || !sentence}
                  >
                    <Trash2 className="mr-2 h-4 w-4" /> Clear
                  </Button>
                </div>

                <p className="mt-3 text-xs text-muted-foreground">
                  {detectionMode === "static"
                    ? 'Use Enter to append letters. Signing "space" and pressing Enter inserts a space.'
                    : 'Use Enter to append words with spaces automatically. "ily" is expanded to "i love you".'}
                </p>
              </CardContent>
            </Card>

            <Card className="bg-card/50 border-border/50">
              <CardHeader>
                <CardTitle className="text-lg">Recent Predictions</CardTitle>
              </CardHeader>
              <CardContent>
                {predictionHistory.length > 0 ? (
                  <div className="space-y-2">
                    {predictionHistory.map((p, i) => (
                      <motion.div
                        key={`${p.label}-${i}`}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.04 }}
                        className={`flex items-center justify-between px-4 py-2 rounded-lg ${
                          i === 0 ? "bg-primary/10 border border-primary/20" : "bg-muted/50"
                        }`}
                      >
                        <span className={i === 0 ? "text-primary font-medium" : "text-muted-foreground"}>
                          {formatPredictionLabel(p.label)}
                        </span>
                        <span className="text-xs text-muted-foreground">{(p.conf * 100).toFixed(0)}%</span>
                      </motion.div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground text-center py-8">No predictions yet</p>
                )}
              </CardContent>
            </Card>

            {/* Class list */}
            {modelStatus && modelStatus.model_loaded && (
              <Card className="bg-card/50 border-border/50">
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Brain className="h-4 w-4 text-primary" />
                    Detectable Signs
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-1.5">
                    {modelStatus.classes.map((c) => (
                      <Badge
                        key={c}
                        variant={prediction === c ? "default" : "secondary"}
                        className="text-xs transition-colors"
                      >
                        {formatPredictionLabel(c)}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  )
}
