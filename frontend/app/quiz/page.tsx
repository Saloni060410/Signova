"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import Link from "next/link"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import {
  Trophy,
  Target,
  TrendingUp,
  Smile,
  Zap,
  Brain,
  ArrowRight,
} from "lucide-react"
import { useAuth } from "@/contexts/auth-context"

interface Stats {
  label: string
  value: number
  icon: any
  suffix: string
}

const defaultStats: Stats[] = [
  {
    label: "Overall Progress",
    value: 0,
    icon: TrendingUp,
    suffix: "%",
  },
  {
    label: "Quizzes Completed",
    value: 0,
    icon: Target,
    suffix: "",
  },
  {
    label: "Accuracy Score",
    value: 0,
    icon: Trophy,
    suffix: "%",
  },
]

const quizLevels = [
  {
    level: "Beginner",
    description: "Test your knowledge of basic alphabets and numbers with static images",
    icon: Smile,
    href: "/quiz/beginner",
    color: "from-green-500/20 to-emerald-500/20",
    borderColor: "hover:border-green-500/50",
  },
  {
    level: "Medium",
    description: "Challenge yourself with static signs and dynamic video-based questions",
    icon: Zap,
    href: "/quiz/medium",
    color: "from-amber-500/20 to-orange-500/20",
    borderColor: "hover:border-amber-500/50",
  },
  {
    level: "Hard",
    description: "Perform sign language sentences in front of your camera for AI evaluation",
    icon: Brain,
    href: "/quiz/hard",
    color: "from-red-500/20 to-rose-500/20",
    borderColor: "hover:border-red-500/50",
  },
]

export default function QuizPage() {
  const { user } = useAuth()
  const [stats, setStats] = useState<Stats[]>(defaultStats)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!user) {
      setLoading(false)
      return
    }

    const fetchStats = async () => {
      try {
        const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
        const res = await fetch(`${API}/progress/${user.id}`)
        if (!res.ok) throw new Error("Failed to fetch progress")
        
        const data = await res.json()
        setStats([
          {
            label: "Overall Progress",
            value: Math.round(data.overall_progress),
            icon: TrendingUp,
            suffix: "%",
          },
          {
            label: "Quizzes Completed",
            value: data.quizzes_completed,
            icon: Target,
            suffix: "",
          },
          {
            label: "Accuracy Score",
            value: Math.round(data.accuracy_score),
            icon: Trophy,
            suffix: "%",
          },
        ])
      } catch (error) {
        console.error("Error fetching progress:", error)
        setStats(defaultStats)
      } finally {
        setLoading(false)
      }
    }

    fetchStats()
    
    // Refetch stats every 5 seconds to reflect updates
    const interval = setInterval(fetchStats, 5000)
    return () => clearInterval(interval)
  }, [user])

  return (
    <div className="min-h-screen pt-24 pb-16">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Page Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12"
        >
          <h1 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            Quiz Dashboard
          </h1>
          <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
            Track your progress and challenge yourself with quizzes at different
            levels
          </p>
        </motion.div>

        {/* Stats Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="grid gap-6 sm:grid-cols-3 mb-12"
        >
          {stats.map((stat, index) => (
            <Card
              key={stat.label}
              className="bg-card/50 border-border/50"
            >
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                    <stat.icon className="h-5 w-5 text-primary" />
                  </div>
                  <span className="text-2xl font-bold text-foreground">
                    {stat.value}
                    {stat.suffix}
                  </span>
                </div>
                <p className="text-sm text-muted-foreground">{stat.label}</p>
                {stat.suffix === "%" && (
                  <Progress value={stat.value} className="mt-3 h-1.5" />
                )}
              </CardContent>
            </Card>
          ))}
        </motion.div>

        {/* Quiz Levels */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <h2 className="text-xl font-semibold text-foreground mb-6 text-center">
            Choose a Quiz Level
          </h2>

          <div className="grid gap-6 md:grid-cols-3">
            {quizLevels.map((quiz, index) => (
              <motion.div
                key={quiz.level}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.3 + index * 0.1 }}
              >
                <Card
                  className={`group h-full bg-card/50 border-border/50 ${quiz.borderColor} hover:shadow-lg transition-all duration-300`}
                >
                  <CardHeader>
                    <div
                      className={`mb-4 flex h-14 w-14 items-center justify-center rounded-xl bg-gradient-to-br ${quiz.color}`}
                    >
                      <quiz.icon className="h-7 w-7 text-foreground" />
                    </div>
                    <CardTitle className="text-xl">{quiz.level}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground mb-6 leading-relaxed">
                      {quiz.description}
                    </p>
                    <Button asChild className="w-full group/btn">
                      <Link href={quiz.href}>
                        Start Quiz
                        <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover/btn:translate-x-1" />
                      </Link>
                    </Button>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  )
}
