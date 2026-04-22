"use client"

import { useEffect, useState } from "react"
import { motion } from "framer-motion"
import Link from "next/link"
import { useRouter } from "next/navigation"
import {
  Trophy, Target, TrendingUp, Flame, Hand,
  BookOpen, Zap, Brain, CheckCircle2, Lock,
  ArrowRight, Activity,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { useAuth } from "@/contexts/auth-context"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

const LEVEL_META = {
  beginner: { label: "Beginner", icon: BookOpen, color: "from-green-500/20 to-emerald-500/20", text: "text-green-600 dark:text-green-400", border: "border-green-500/30" },
  medium:   { label: "Medium",   icon: Zap,      color: "from-amber-500/20 to-orange-500/20",  text: "text-amber-600 dark:text-amber-400",  border: "border-amber-500/30"  },
  hard:     { label: "Hard",     icon: Brain,    color: "from-red-500/20 to-rose-500/20",      text: "text-red-600 dark:text-red-400",      border: "border-red-500/30"    },
}

interface SubQuizStatus { sub_quiz: number; completed: boolean; best_score: number | null }
interface LevelStatus   { level: string; completed_count: number; total_count: number; sub_quizzes: SubQuizStatus[] }

interface DashboardData {
  user: { id: number; name: string; email: string }
  overall_progress: number
  streak_days: number
  total_signs_detected: number
  level_statuses: LevelStatus[]
  recent_activity: any[]
}

export default function DashboardPage() {
  const { user, token, loading } = useAuth()
  const router = useRouter()
  const [data, setData] = useState<DashboardData | null>(null)
  const [fetching, setFetching] = useState(true)

  useEffect(() => {
    if (!loading && !user) router.replace("/")
  }, [loading, user, router])

  useEffect(() => {
    if (!user) return
    fetch(`${API}/dashboard/${user.id}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((r) => r.json())
      .then(setData)
      .catch(console.error)
      .finally(() => setFetching(false))
  }, [user, token])

  if (loading || fetching || !data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <div className="h-10 w-10 rounded-full border-4 border-primary border-t-transparent animate-spin" />
          <p className="text-muted-foreground">Loading your dashboard…</p>
        </div>
      </div>
    )
  }

  const statCards = [
    { label: "Overall Progress", value: `${data.overall_progress}%`, icon: TrendingUp, progress: data.overall_progress },
    { label: "Day Streak",        value: `${data.streak_days} 🔥`,   icon: Flame,      progress: null },
    { label: "Signs Detected",    value: data.total_signs_detected,  icon: Hand,       progress: null },
  ]

  return (
    <div className="min-h-screen pt-24 pb-16">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 space-y-10">

        {/* Header */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="text-3xl font-bold text-foreground">
            Welcome back, {data.user.name.split(" ")[0]} 👋
          </h1>
          <p className="mt-1 text-muted-foreground">Here's your learning progress at a glance.</p>
        </motion.div>

        {/* Stat Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid gap-6 sm:grid-cols-3"
        >
          {statCards.map((s) => (
            <Card key={s.label} className="bg-card/50 border-border/50">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                    <s.icon className="h-5 w-5 text-primary" />
                  </div>
                  <span className="text-2xl font-bold text-foreground">{s.value}</span>
                </div>
                <p className="text-sm text-muted-foreground">{s.label}</p>
                {s.progress !== null && (
                  <Progress value={s.progress} className="mt-3 h-1.5" />
                )}
              </CardContent>
            </Card>
          ))}
        </motion.div>

        {/* Level Progress Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <h2 className="text-xl font-semibold text-foreground mb-5">Your Quiz Progress</h2>
          <div className="grid gap-6 md:grid-cols-3">
            {data.level_statuses.map((ls) => {
              const meta = LEVEL_META[ls.level as keyof typeof LEVEL_META]
              const pct = Math.round((ls.completed_count / ls.total_count) * 100)
              return (
                <Card key={ls.level} className={`bg-card/50 border-border/50 hover:border-primary/30 transition-all`}>
                  <CardHeader>
                    <div className={`mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br ${meta.color}`}>
                      <meta.icon className="h-6 w-6 text-foreground" />
                    </div>
                    <CardTitle className="text-lg flex items-center justify-between">
                      {meta.label}
                      <span className={`text-sm font-normal ${meta.text}`}>
                        {ls.completed_count}/{ls.total_count}
                      </span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <Progress value={pct} className="h-2" />

                    {/* Sub-quiz grid */}
                    <div className="grid grid-cols-5 gap-1.5">
                      {ls.sub_quizzes.map((sq) => (
                        <Link
                          key={sq.sub_quiz}
                          href={`/quiz/${ls.level}?sub=${sq.sub_quiz}`}
                          title={sq.completed ? `Sub-quiz ${sq.sub_quiz} — Best: ${sq.best_score}%` : `Sub-quiz ${sq.sub_quiz}`}
                        >
                          <div className={`aspect-square rounded-lg flex items-center justify-center text-xs font-semibold border transition-all hover:scale-110 ${
                            sq.completed
                              ? `bg-primary/20 border-primary/40 ${meta.text}`
                              : "bg-muted/50 border-border/50 text-muted-foreground"
                          }`}>
                            {sq.completed ? (
                              <CheckCircle2 className="h-4 w-4" />
                            ) : (
                              sq.sub_quiz
                            )}
                          </div>
                        </Link>
                      ))}
                    </div>

                    <Button asChild className="w-full" variant="outline" size="sm">
                      <Link href={`/quiz/${ls.level}`}>
                        {ls.completed_count === 0 ? "Start" : ls.completed_count < 5 ? "Continue" : "Review"}
                        <ArrowRight className="ml-2 h-3 w-3" />
                      </Link>
                    </Button>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </motion.div>

        {/* Recent Activity */}
        {data.recent_activity.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <h2 className="text-xl font-semibold text-foreground mb-5 flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              Recent Activity
            </h2>
            <Card className="bg-card/50 border-border/50">
              <CardContent className="p-0">
                {data.recent_activity.map((rec: any, i: number) => (
                  <div
                    key={i}
                    className={`flex items-center justify-between px-6 py-4 ${i !== data.recent_activity.length - 1 ? "border-b border-border/50" : ""}`}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`h-8 w-8 rounded-full flex items-center justify-center bg-gradient-to-br ${LEVEL_META[rec.level as keyof typeof LEVEL_META]?.color}`}>
                        <Target className="h-4 w-4" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-foreground capitalize">
                          {rec.level} — Sub-quiz {rec.sub_quiz}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {new Date(rec.completed_at).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                    <div className={`text-sm font-bold ${rec.score >= 80 ? "text-green-500" : rec.score >= 50 ? "text-amber-500" : "text-red-500"}`}>
                      {rec.score}%
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="flex flex-wrap gap-3"
        >
          <Button asChild>
            <Link href="/quiz">Go to Quiz</Link>
          </Button>
          <Button asChild variant="outline">
            <Link href="/detection">Live Detection</Link>
          </Button>
          <Button asChild variant="outline">
            <Link href="/learning">Learning Guide</Link>
          </Button>
        </motion.div>
      </div>
    </div>
  )
}
