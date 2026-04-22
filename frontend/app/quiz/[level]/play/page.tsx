"use client"

import { useEffect, useState, useRef, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import Link from "next/link"
import { useParams, useSearchParams, useRouter } from "next/navigation"
import {
  ArrowLeft, CheckCircle, XCircle, Hand, Play, RotateCcw,
  Camera, Square, Circle, Volume2, Loader2,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { useAuth } from "@/contexts/auth-context"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

interface Question {
  id: number; type: string; sign: string
  options?: string[] | null; correct?: number | null
  image_url?: string | null
  video_url?: string | null
}

export default function QuizPlayPage() {
  const params = useParams()
  const level = (params.level as string) ?? "beginner"
  const searchParams = useSearchParams()
  const subQuiz = parseInt(searchParams.get("sub") ?? "1")
  const router = useRouter()
  const { user, token } = useAuth()

  const [questions, setQuestions] = useState<Question[]>([])
  const [loadingQ, setLoadingQ] = useState(true)
  const [currentIdx, setCurrentIdx] = useState(0)
  const [answers, setAnswers] = useState<number[]>([])
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null)
  const [isAnswered, setIsAnswered] = useState(false)
  const [score, setScore] = useState(0)
  const [isComplete, setIsComplete] = useState(false)
  const [finalResult, setFinalResult] = useState<any>(null)

  // Hard mode camera
  const [cameraEnabled, setCameraEnabled] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [hasRecording, setHasRecording] = useState(false)
  const [cameraScore, setCameraScore] = useState<number | null>(null)
  const [hardVerdict, setHardVerdict] = useState<{ expected: string; predicted: string; correct: boolean; confidence: number } | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const captureIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const hardInFlightRef = useRef(false)
  const hardPredictionsRef = useRef<{ label: string; conf: number }[]>([])
  const hardSessionRef = useRef(
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `hard-${Date.now()}`
  )

  // Load questions
  useEffect(() => {
    fetch(`${API}/quiz/questions?level=${level}&sub_quiz=${subQuiz}`)
      .then((r) => r.json())
      .then((d) => { setQuestions(d.questions ?? []); setLoadingQ(false) })
      .catch(() => setLoadingQ(false))
  }, [level, subQuiz])

  const question = questions[currentIdx]
  const progress = questions.length > 0 ? ((currentIdx + 1) / questions.length) * 100 : 0

  // ── MCQ answer ────────────────────────────────────────────────────────────
  const handleAnswer = (index: number) => {
    if (isAnswered) return
    setSelectedAnswer(index)
    setIsAnswered(true)
    setAnswers((prev) => { const a = [...prev]; a[currentIdx] = index; return a })
    if (index === question?.correct) setScore((s) => s + 1)
  }

  const handleNext = () => {
    if (currentIdx < questions.length - 1) {
      setCurrentIdx((i) => i + 1)
      setSelectedAnswer(null)
      setIsAnswered(false)
      setCameraScore(null)
      setHasRecording(false)
    } else {
      submitQuiz()
    }
  }

  const submitQuiz = async () => {
    const finalScore = level === "hard"
      ? Math.round(answers.reduce((s, v) => s + (v ?? 75), 0) / questions.length)
      : Math.round((score / questions.length) * 100)

    if (user) {
      try {
        const res = await fetch(`${API}/quiz/submit`, {
          method: "POST",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
          body: JSON.stringify({
            user_id: user.id,
            level,
            sub_quiz: subQuiz,
            answers,
            total_questions: questions.length,
          }),
        })
        const data = await res.json()
        setFinalResult(data)
      } catch (_) { /* offline fallback */ }
    }
    setIsComplete(true)
  }

  // ── Camera (hard mode) ────────────────────────────────────────────────────
  const enableCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } })
      streamRef.current = stream
      setCameraEnabled(true)
      setCameraError(null)
    } catch (e) {
      console.error("Camera denied", e)
      setCameraError("Unable to access camera. Please allow permission or try a different device.")
      setCameraEnabled(false)
    }
  }, [])

  const disableCamera = useCallback(() => {
    if (captureIntervalRef.current) clearInterval(captureIntervalRef.current)
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setCameraEnabled(false); setIsRecording(false); setCameraError(null)
  }, [])

  const normalizeTargetLabel = useCallback((text: string) => {
    const normalized = text
      .trim()
      .toLowerCase()
      .replace(/[\.,!?'-]/g, "")
      .replace(/\s+/g, " ")
    const collapsed = normalized.replace(/\s+/g, "")

    if (["i love you", "i_love_you", "iloveyou", "ily"].includes(normalized) || collapsed === "iloveyou") {
      return "ily"
    }
    if (["thank you", "thank_you", "thankyou", "thanks"].includes(normalized) || collapsed === "thankyou") {
      return "thank_you"
    }
    return normalized.replace(/ /g, "_")
  }, [])

  const captureAndEvaluateHardFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || hardInFlightRef.current) return
    const video = videoRef.current
    const canvas = canvasRef.current
    canvas.width = video.videoWidth || 320
    canvas.height = video.videoHeight || 240
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    ctx.drawImage(video, 0, 0)

    try {
      hardInFlightRef.current = true
      const blob = await new Promise<Blob | null>((resolve) => {
        canvas.toBlob(resolve, "image/jpeg", 0.8)
      })
      if (!blob) return

      const formData = new FormData()
      formData.append("file", blob, "frame.jpg")

      const res = await fetch(
        `${API}/predict/dynamic?session_id=${encodeURIComponent(hardSessionRef.current)}`,
        { method: "POST", body: formData }
      )
      if (!res.ok) return
      const data = await res.json()
      if (data.ready && data.prediction && typeof data.confidence === "number") {
        hardPredictionsRef.current.push({ label: data.prediction, conf: data.confidence })
      }
    } catch {
      // Keep quiz running even if a frame upload fails.
    } finally {
      hardInFlightRef.current = false
    }
  }, [])

  useEffect(() => {
    if (!cameraEnabled || !streamRef.current || !videoRef.current) return
    const video = videoRef.current
    video.srcObject = streamRef.current
    const handleLoaded = () => {
      video.play().catch((err) => {
        console.error("Video playback failed:", err)
        setCameraError("Camera stream started but video playback was blocked. Please allow autoplay or refresh the page.")
      })
    }
    video.addEventListener("loadedmetadata", handleLoaded)
    return () => {
      video.removeEventListener("loadedmetadata", handleLoaded)
    }
  }, [cameraEnabled])

  const startRecording = async () => {
    setIsRecording(true)
    setCameraScore(null)
    setHardVerdict(null)
    setHasRecording(false)
    hardPredictionsRef.current = []
    hardSessionRef.current =
      typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
        ? crypto.randomUUID()
        : `hard-${Date.now()}`

    try {
      await fetch(`${API}/predict/dynamic/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: hardSessionRef.current }),
      })
    } catch {
      // Non-blocking; session id is still unique.
    }

    if (captureIntervalRef.current) clearInterval(captureIntervalRef.current)
    captureIntervalRef.current = setInterval(captureAndEvaluateHardFrame, 120)
  }

  const stopRecording = async () => {
    setIsRecording(false)
    setHasRecording(true)
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current)
      captureIntervalRef.current = null
    }

    const expected = normalizeTargetLabel(question?.sign ?? "")
    const collected = hardPredictionsRef.current
    const totalReady = collected.length

    if (totalReady < 3) {
      setCameraError("Not enough model predictions captured. Record a bit longer while holding the sign steady.")
      setHasRecording(false)
      setCameraScore(null)
      return
    }

    const normalizedPredictions = collected.map((item) => ({
      label: normalizeTargetLabel(item.label),
      conf: item.conf,
    }))

    const labelCounts = normalizedPredictions.reduce<Record<string, number>>((acc, item) => {
      acc[item.label] = (acc[item.label] ?? 0) + 1
      return acc
    }, {})
    const majorityLabel = Object.keys(labelCounts).sort((a, b) => labelCounts[b] - labelCounts[a])[0]
    const majorityRate = (labelCounts[majorityLabel] ?? 0) / totalReady

    const expectedMatches = normalizedPredictions.filter((item) => item.label === expected)
    const matchRate = expectedMatches.length / totalReady
    const avgMatchConfidence =
      expectedMatches.length > 0
        ? expectedMatches.reduce((sum, item) => sum + item.conf, 0) / expectedMatches.length
        : 0
    const bestMatchConfidence = expectedMatches.length > 0 ? Math.max(...expectedMatches.map((m) => m.conf)) : 0

    let computedScore = 0
    if (majorityLabel === expected) {
      computedScore = Math.round((0.5 * avgMatchConfidence + 0.35 * majorityRate + 0.15 * bestMatchConfidence) * 100)
    } else {
      computedScore = Math.round((0.5 * matchRate + 0.35 * avgMatchConfidence + 0.15 * bestMatchConfidence) * 100)
    }
    computedScore = Math.max(0, Math.min(100, computedScore))

    setHardVerdict({
      expected,
      predicted: majorityLabel,
      correct: majorityLabel === expected,
      confidence: majorityRate > 0 ? Math.round(majorityRate * 100) / 100 : 0,
    })

    setCameraScore(computedScore)
    setAnswers((prev) => {
      const a = [...prev]
      a[currentIdx] = computedScore
      return a
    })

    try {
      await fetch(`${API}/predict/dynamic/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: hardSessionRef.current }),
      })
    } catch {
      // No-op
    }
  }

  const speakSign = (text: string) => {
    if ("speechSynthesis" in window) {
      const u = new SpeechSynthesisUtterance(text)
      u.rate = 0.85
      window.speechSynthesis.speak(u)
    }
  }

  const handleRestart = () => {
    setCurrentIdx(0); setAnswers([]); setSelectedAnswer(null)
    setIsAnswered(false); setScore(0); setIsComplete(false)
    setFinalResult(null); setCameraScore(null); setHasRecording(false)
  }

  useEffect(() => {
    return () => {
      if (captureIntervalRef.current) clearInterval(captureIntervalRef.current)
      fetch(`${API}/predict/dynamic/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: hardSessionRef.current }),
      }).catch(() => undefined)
    }
  }, [])

  if (loadingQ) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  // ── Completion screen ─────────────────────────────────────────────────────
  if (isComplete) {
    const pct = finalResult?.score ?? Math.round((score / questions.length) * 100)
    return (
      <div className="min-h-screen pt-24 pb-16 flex items-center justify-center">
        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="max-w-md w-full mx-4">
          <Card className="bg-card/50 border-border/50 text-center">
            <CardContent className="p-8">
              <div className="mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-primary/10 mx-auto">
                <CheckCircle className="h-10 w-10 text-primary" />
              </div>
              <h2 className="text-2xl font-bold text-foreground mb-1">Quiz Complete!</h2>
              <p className="text-muted-foreground mb-2">
                {finalResult?.message ?? (pct >= 70 ? "Great job!" : "Keep practicing!")}
              </p>
              {level !== "hard" && (
                <p className="text-sm text-muted-foreground mb-6">
                  You got {finalResult?.correct_answers ?? score} out of {questions.length} correct
                </p>
              )}
              <div className="mb-6">
                <div className="text-5xl font-bold text-primary mb-2">{pct}%</div>
                <Progress value={pct} className="h-2" />
              </div>
              <div className="flex gap-3">
                <Button variant="outline" className="flex-1" onClick={handleRestart}>
                  <RotateCcw className="mr-2 h-4 w-4" /> Try Again
                </Button>
                <Button asChild className="flex-1">
                  <Link href={user ? "/dashboard" : `/quiz/${level}`}>
                    {user ? "Dashboard" : "More Quizzes"}
                  </Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    )
  }

  // ── Hard (camera) question ────────────────────────────────────────────────
  if (level === "hard" && question) {
    return (
      <div className="min-h-screen pt-24 pb-16">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between mb-6">
            <Button variant="ghost" size="sm" asChild onClick={disableCamera}>
              <Link href={`/quiz/${level}`}><ArrowLeft className="mr-2 h-4 w-4" />Back</Link>
            </Button>
            <span className="text-sm text-muted-foreground">
              {currentIdx + 1} / {questions.length}
            </span>
          </div>
          <Progress value={progress} className="h-2 mb-8" />

          <div className="grid lg:grid-cols-2 gap-6">
            {/* Camera */}
            <Card className="bg-card/50 border-border/50">
              <CardHeader><CardTitle>Camera Preview</CardTitle></CardHeader>
              <CardContent>
                <div className="aspect-video rounded-xl bg-muted overflow-hidden relative">
                  {cameraEnabled ? (
                    <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover scale-x-[-1]" />
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center">
                        <Camera className="h-12 w-12 text-muted-foreground mx-auto mb-3" />
                        <p className="text-sm text-muted-foreground">Camera not enabled</p>
                      </div>
                    </div>
                  )}
                  {isRecording && (
                    <div className="absolute top-3 left-3 flex items-center gap-2 px-3 py-1 bg-red-500 text-white rounded-full text-xs font-medium">
                      <Circle className="h-2 w-2 fill-current animate-pulse" /> Recording
                    </div>
                  )}
                </div>
                {cameraError && <p className="mt-3 text-sm text-destructive">{cameraError}</p>}
                <canvas ref={canvasRef} className="hidden" />
                <div className="flex gap-2 mt-4">
                  {!cameraEnabled ? (
                    <Button onClick={enableCamera} className="flex-1">
                      <Camera className="mr-2 h-4 w-4" /> Enable Camera
                    </Button>
                  ) : (
                    <>
                      {!isRecording ? (
                        <Button onClick={startRecording} className="flex-1" disabled={hasRecording}>
                          <Circle className="mr-2 h-4 w-4" /> Record
                        </Button>
                      ) : (
                        <Button onClick={stopRecording} variant="destructive" className="flex-1">
                          <Square className="mr-2 h-4 w-4" /> Stop
                        </Button>
                      )}
                      <Button variant="outline" onClick={disableCamera} disabled={isRecording}>Disable</Button>
                    </>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Sentence + score */}
            <div className="space-y-4">
              <Card className="bg-card/50 border-border/50">
                <CardHeader><CardTitle>Sign This</CardTitle></CardHeader>
                <CardContent>
                  <div className="p-6 rounded-xl bg-linear-to-br from-primary/5 to-accent/5 border border-primary/10">
                    <p className="text-xl font-medium text-foreground text-center">"{question.sign}"</p>
                  </div>
                  <Button variant="ghost" size="sm" className="mt-3 w-full" onClick={() => speakSign(question.sign)}>
                    <Volume2 className="mr-2 h-4 w-4" /> Hear it spoken
                  </Button>
                </CardContent>
              </Card>

              {cameraScore !== null && (
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                  <Card className="bg-linear-to-br from-primary/10 to-accent/10 border-primary/20">
                    <CardContent className="p-6 text-center">
                      <p className="text-sm text-muted-foreground mb-1">Your Score</p>
                      <div className="text-4xl font-bold text-primary mb-3">{cameraScore}%</div>
                      <Progress value={cameraScore} className="h-2 mb-4" />
                      {hardVerdict && (
                        <div className="mb-4 rounded-xl border border-border/60 bg-background/60 p-4 text-left text-sm">
                          <p><span className="font-medium">Expected:</span> {hardVerdict.expected}</p>
                          <p><span className="font-medium">Predicted:</span> {hardVerdict.predicted}</p>
                          <p>
                            <span className="font-medium">Result:</span> {hardVerdict.correct ? "Correct" : "Incorrect"}
                          </p>
                        </div>
                      )}
                      <Button className="w-full" onClick={handleNext}>
                        {currentIdx < questions.length - 1 ? "Next Sentence" : "See Results"}
                      </Button>
                    </CardContent>
                  </Card>
                </motion.div>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

  // ── MCQ question (beginner / medium) ──────────────────────────────────────
  return (
    <div className="min-h-screen pt-24 pb-16">
      <div className="mx-auto max-w-2xl px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between mb-6">
          <Button variant="ghost" size="sm" asChild>
            <Link href={`/quiz/${level}`}><ArrowLeft className="mr-2 h-4 w-4" />Back</Link>
          </Button>
          <div className="flex items-center gap-3">
            {question?.type === "video" && (
              <span className="text-xs px-2 py-1 rounded-full bg-amber-500/10 text-amber-600 dark:text-amber-400 font-medium">Video</span>
            )}
            <span className="text-sm text-muted-foreground">{currentIdx + 1} / {questions.length}</span>
          </div>
        </div>
        <Progress value={progress} className="h-2 mb-8" />

        <AnimatePresence mode="wait">
          <motion.div
            key={currentIdx}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.25 }}
          >
            <Card className="bg-card/50 border-border/50 mb-6">
              <CardContent className="p-8">
                <h2 className="text-lg font-medium text-center text-muted-foreground mb-6">
                  What {level === "beginner" ? "letter" : "word"} does this sign represent?
                </h2>

                {/* Sign display */}
                <div className="aspect-square max-w-xs mx-auto rounded-2xl bg-linear-to-br from-primary/10 to-accent/10 flex items-center justify-center mb-8">
                  {question?.video_url ? (
                    <video
                      src={question.video_url}
                      title={`Sign video for ${question.sign}`}
                      className="h-full w-full object-cover rounded-2xl bg-black"
                      autoPlay
                      loop
                      muted
                      playsInline
                      controls
                    />
                  ) : question?.type === "video" ? (
                    <div className="text-center">
                      <Button size="lg" variant="secondary" className="rounded-full h-16 w-16 p-0 shadow-lg">
                        <Play className="h-7 w-7 ml-1" />
                      </Button>
                      <p className="mt-3 text-sm text-muted-foreground">Video for "{question.sign}"</p>
                    </div>
                  ) : question?.image_url ? (
                    <img
                      src={question.image_url}
                      alt={`ASL sign for ${question.sign}`}
                      className="h-full w-full object-contain p-4 dark:brightness-0 dark:invert"
                      loading="lazy"
                    />
                  ) : (
                    <div className="text-center">
                      <Hand className="h-20 w-20 text-primary/60 mx-auto mb-2" />
                      <span className="text-sm text-muted-foreground">Sign for "{question?.sign}"</span>
                    </div>
                  )}
                </div>

                {/* Options */}
                <div className="grid grid-cols-2 gap-4">
                  {question?.options?.map((opt, i) => {
                    const isCorrect = i === question.correct
                    const isSelected = selectedAnswer === i
                    let cls = "h-14 text-lg font-medium transition-all duration-200"
                    if (isAnswered) {
                      if (isCorrect) cls += " bg-green-500/20 border-green-500 text-green-700 dark:text-green-400"
                      else if (isSelected) cls += " bg-red-500/20 border-red-500 text-red-700 dark:text-red-400"
                    }
                    return (
                      <Button key={i} variant="outline" className={cls} onClick={() => handleAnswer(i)} disabled={isAnswered}>
                        {opt}
                        {isAnswered && isCorrect && <CheckCircle className="ml-2 h-5 w-5" />}
                        {isAnswered && isSelected && !isCorrect && <XCircle className="ml-2 h-5 w-5" />}
                      </Button>
                    )
                  })}
                </div>
              </CardContent>
            </Card>

            {isAnswered && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <Button className="w-full" size="lg" onClick={handleNext}>
                  {currentIdx < questions.length - 1 ? "Next Question" : "See Results"}
                </Button>
              </motion.div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}
