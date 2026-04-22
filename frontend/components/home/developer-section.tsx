"use client"

import { motion } from "framer-motion"
import { useInView } from "framer-motion"
import { useRef } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { User } from "lucide-react"

const developers = [
  {
    name: "Saloni",
    title: "Developer",
  },
  {
    name: "Yachna",
    title: "Developer",
  },
  {
    name: "Krishna",
    title: "Developer",
  },
]

function DeveloperCard({
  developer,
  index,
}: {
  developer: (typeof developers)[0]
  index: number
}) {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: "-50px" })

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 30 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
    >
      <Card className="group h-full bg-card/50 backdrop-blur-sm border-border/50 hover:border-primary/50 hover:shadow-lg transition-all duration-300">
        <CardContent className="p-6 text-center">
          {/* Avatar Placeholder */}
          <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-primary/20 to-accent/20 group-hover:from-primary/30 group-hover:to-accent/30 transition-colors">
            <User className="h-10 w-10 text-primary" />
          </div>

          {/* Name & Title */}
          <h3 className="text-lg font-semibold text-foreground">
            {developer.name}
          </h3>
          <p className="mt-1 text-sm text-muted-foreground">{developer.title}</p>
        </CardContent>
      </Card>
    </motion.div>
  )
}

export function DeveloperSection() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: "-100px" })

  return (
    <section className="py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            Meet Our Team
          </h2>
          <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
            The talented developers behind this platform
          </p>
        </motion.div>

        {/* Developers Grid */}
        <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-3 max-w-3xl mx-auto">
          {developers.map((developer, index) => (
            <DeveloperCard
              key={developer.name}
              developer={developer}
              index={index}
            />
          ))}
        </div>
      </div>
    </section>
  )
}
