"use client"

import { useEffect, useState } from "react"
import { motion } from "framer-motion"
import Link from "next/link"
import { useParams, useSearchParams } from "next/navigation"
import { ArrowLeft, CheckCircle2, Lock, Play, Star } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { useAuth } from "@/contexts/auth-context"

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

const LEVEL_META = {
  beginner: { label: "Beginner", color: "from-green-500/20 to-emerald-500/20", text: "text-green-600 dark:text-green-400" },
  medium:   { label: "Medium",   color: "from-amber-500/20 to-orange-500/20",  text: "text-amber-600 dark:text-amber-400" },
  hard:     { label: "Hard",     color: "from-red-500/20 to-rose-500/20",      text: "text-red-600 dark:text-red-400"     },
}

export default function QuizLevelPage() {
  const params = useParams()
  const level = (params.level as string) ?? "beginner"
  const meta = LEVEL_META[level as keyof typeof LEVEL_META] ?? LEVEL_META.beginner

  const { user, token } = useAuth()
  const [subData, setSubData] = useState<Record<number, { completed: boolean; best_score: number | null }>>({})

  useEffect(() => {
    if (!user) return
    fetch(`${API}/dashboard/${user.id}`, { headers: { Authorization: `Bearer ${token}` } })
      .then((r) => r.json())
      .then((d) => {
        const ls = d.level_statuses?.find((s: any) => s.level === level)
        if (ls) {
          const map: Record<number, any> = {}
          ls.sub_quizzes.forEach((sq: any) => { map[sq.sub_quiz] = sq })
          setSubData(map)
        }
      })
      .catch(console.error)
  }, [user, token, level])

  const subQuizzes = [1, 2, 3, 4, 5]

  return (
    <div className="min-h-screen pt-24 pb-16">
      <div className="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <Button variant="ghost" size="sm" asChild>
            <Link href="/quiz">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Link>
          </Button>
          <div>
            <h1 className="text-2xl font-bold text-foreground">{meta.label} Level</h1>
            <p className="text-sm text-muted-foreground">Choose a sub-quiz to practice</p>
          </div>
        </div>

        {/* Sub-quiz cards */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {subQuizzes.map((n, i) => {
            const sq = subData[n]
            const completed = sq?.completed ?? false
            const bestScore = sq?.best_score ?? null

            return (
              <motion.div
                key={n}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.07 }}
              >
                <Card className={`bg-card/50 border-border/50 hover:border-primary/30 hover:shadow-md transition-all group`}>
                  <CardContent className="p-5">
                    {/* Sub-quiz number badge */}
                    <div className="flex items-start justify-between mb-4">
                      <div className={`h-10 w-10 rounded-xl flex items-center justify-center bg-gradient-to-br ${meta.color} text-lg font-bold text-foreground`}>
                        {n}
                      </div>
                      {completed && (
                        <div className="flex items-center gap-1">
                          <CheckCircle2 className="h-4 w-4 text-green-500" />
                          {bestScore !== null && (
                            <span className="text-xs font-semibold text-green-500">{bestScore}%</span>
                          )}
                        </div>
                      )}
                    </div>

                    <h3 className="font-semibold text-foreground mb-1">
                      Sub-quiz {n}
                    </h3>
                    <p className="text-xs text-muted-foreground mb-4">
                      5 questions • {level === "hard" ? "Camera required" : "Multiple choice"}
                    </p>

                    {bestScore !== null && (
                      <div className="mb-3">
                        <Progress value={bestScore} className="h-1.5" />
                        <p className="text-xs text-muted-foreground mt-1">Best: {bestScore}%</p>
                      </div>
                    )}

                    <Button asChild size="sm" className="w-full">
                      <Link href={`/quiz/${level}/play?sub=${n}`}>
                        <Play className="mr-2 h-3 w-3" />
                        {completed ? "Retry" : "Start"}
                      </Link>
                    </Button>
                  </CardContent>
                </Card>
              </motion.div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
