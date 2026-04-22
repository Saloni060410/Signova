"use client"

import { motion } from "framer-motion"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { AlphabetsGrid } from "@/components/learning/alphabets-grid"
import { NumbersGrid } from "@/components/learning/numbers-grid"
import { WordsGrid } from "@/components/learning/words-grid"

export default function LearningPage() {
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
            Learning Guide
          </h1>
          <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
            Master sign language step by step with our comprehensive learning
            materials
          </p>
        </motion.div>

        {/* Learning Tabs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <Tabs defaultValue="alphabets" className="w-full">
            <TabsList className="grid w-full max-w-md mx-auto grid-cols-3 mb-8">
              <TabsTrigger value="alphabets">Alphabets</TabsTrigger>
              <TabsTrigger value="numbers">Numbers</TabsTrigger>
              <TabsTrigger value="words">Words</TabsTrigger>
            </TabsList>

            <TabsContent value="alphabets">
              <AlphabetsGrid />
            </TabsContent>

            <TabsContent value="numbers">
              <NumbersGrid />
            </TabsContent>

            <TabsContent value="words">
              <WordsGrid />
            </TabsContent>
          </Tabs>
        </motion.div>
      </div>
    </div>
  )
}
